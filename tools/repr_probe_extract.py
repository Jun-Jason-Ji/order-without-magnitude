#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hidden-state extraction for the representation probe (results/courtdyn/repr_probe/PREREGISTRATION.json).

One forward pass per item on the explicit METRE prompt of the 2026-09-12 control protocol, with the
frozen evaluation settings of eval/run_bench.load_model (nf4, max_pixels 200704, enable_thinking=False,
question + GENERATION_SUFFIX).  Saves, per (model, clip): hidden states at layers 4, 8, ..., 32 pooled
as the mean over image tokens ("img") and the last prompt token ("last"), float16; the metric and
pixel references; the item keys; and the greedy metre answer (max_new_tokens 16, descriptive only).

Resumable: one .npz per (model, clip), written atomically (tmp + fsync + os.replace); finished files
are skipped.  All GPU use goes through gpuq (see tools/repr_probe_queue.sh).

Usage: python tools/repr_probe_extract.py --model v3 [--clips Q2_top_480-510 ...] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "results" / "courtdyn" / "repr_probe" / "features"
BASE = (r"E:\models\hf\hub\models--Qwen--Qwen3.5-4B\snapshots"
        r"\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
MODELS = {"base": None,
          "native": ROOT / "models" / "courtdyn-native-sft",
          "v3": ROOT / "models" / "courtdyn-revision-v3-eventholdout-s42",
          "mixunit": ROOT / "models" / "courtdyn-mixunit-v3-s42"}
CLIPS = ["Q2_top_480-510", "Q1_top_0-30", "Q1_side_0-30", "Q2_side_300-330", "Q4_side_570-600",
         "Q2_top_480-510@static4"]
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
SPEED, PATH = "dynamics_speed_player", "dynamics_path_player"
PRIOR = "Assume a typical player on this court is 1.93 m tall. "
CONTEXT = ("The 4 frames are consecutive samples from one basketball clip spanning "
           "2.0 seconds, in chronological order. {prior}"
           "Each supplied frame is 960 pixels wide and 540 pixels high. "
           "Pixel answers refer to this original canvas before any resizing. "
           "Physical units describe floor-plane motion; pixels describe projected "
           "image-plane motion. ")


def metre_question(category):
    if category == SPEED:
        q = "What is the average speed of the player marked with the red box, in meters per second? "
    else:
        q = "What is the path length of the player marked with the red box over the whole clip, in meters? "
    return CONTEXT.format(prior=PRIOR) + q + "Output only the number, one decimal place."


def build_items(clip):
    """Items of one clip with metric + pixel references (the protocol's own computation)."""
    import build_courtdyn_ruler as R
    from engine.court_homography import CourtPlane, recompute_answer
    seq, _, mode = clip.partition("@")
    items = json.loads(Path(R.seq_dir(seq), "qa_dyn_v1.json").read_text(encoding="utf-8"))
    tracks, info = R.DQ.load_tracks(R.tt_split(seq)), R.DQ.load_seqinfo(R.tt_split(seq))
    plane, fps = CourtPlane.load(seq), float(info["fps"])
    frame_root = ROOT / "data" / "courtdyn" / f"frames_{seq}"
    out = []
    for it in items:
        if it["category"] not in (SPEED, PATH):
            continue
        r = recompute_answer(plane, it, tracks, fps)
        px = R.px_answer(it, tracks, 960 / float(info["width"]), fps)
        if r is None or px is None:
            continue
        metric = r[1]["speed_mps" if it["category"] == SPEED else "path_m"]
        frames = [str(frame_root / i) for i in it["image_ids"]]
        if mode == "static4":
            frames = [frames[0]] * len(frames)
        m = it["meta"]
        out.append(dict(key=f"{seq}|{it['category']}|{m['track']}|{m['window'][0]}|{m['window'][1]}",
                        category=it["category"], window0=int(m["window"][0]), metric=float(metric),
                        px=float(px), frames=frames, question=metre_question(it["category"])))
    assert len(out) == 280, (clip, len(out))
    return out


def load(model_key):
    """Model + processor exactly as eval/run_bench.load_model(load_4bit=True, max_pixels=200704)."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from train.model_family import assert_processor_policy, inspect_local_model, processor_policy
    fam, _ = inspect_local_model(BASE)
    cap = float(os.environ.get("VRAM_CAP_FRACTION", "0.93"))
    if cap > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(cap, 0)
    proc_kwargs = dict(trust_remote_code=fam.trust_remote_code)
    if fam.image_budget_kind != "max_pixels" or 200704:
        proc_kwargs.update(processor_policy(fam, 200704))
    processor = AutoProcessor.from_pretrained(BASE, **proc_kwargs)
    assert_processor_policy(fam, processor)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", trust_remote_code=fam.trust_remote_code,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True))
    if MODELS[model_key] is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(MODELS[model_key]))
        assert any("lora_" in n for n, _ in model.named_parameters()), "adapter not attached"
    model.eval()
    return model, processor


def atomic_savez(path: Path, **arrays):
    import numpy as np
    tmp = path.with_name(path.name + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def extract(model_key, clips, limit=None):
    import numpy as np
    import torch
    from PIL import Image
    from eval.run_bench import GENERATION_SUFFIX
    todo = [c for c in clips if not (OUT / model_key / f"{c.replace('@', '__')}.npz").is_file()]
    if not todo:
        print(f"[repr] {model_key}: all clips done"); return
    model, processor = load(model_key)
    image_token = model.config.image_token_id if hasattr(model.config, "image_token_id") \
        else model.base_model.model.config.image_token_id
    (OUT / model_key).mkdir(parents=True, exist_ok=True)
    for clip in todo:
        items = build_items(clip)[:limit] if limit else build_items(clip)
        feats = {f"{p}_L{l}": [] for p in ("img", "last") for l in LAYERS}
        answers = []
        for n, it in enumerate(items):
            imgs = [Image.open(p).convert("RGB") for p in it["frames"]]
            msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs]
                     + [{"type": "text", "text": it["question"] + GENERATION_SUFFIX}]}]
            enc = processor.apply_chat_template(msgs, enable_thinking=False, add_generation_prompt=True,
                                                tokenize=True, return_dict=True,
                                                return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states
                mask = (enc["input_ids"][0] == image_token)
                assert int(mask.sum()) > 0, "no image tokens found"
                for l in LAYERS:
                    h = hs[l][0].float()
                    feats[f"img_L{l}"].append(h[mask].mean(0).cpu().numpy().astype(np.float16))
                    feats[f"last_L{l}"].append(h[-1].cpu().numpy().astype(np.float16))
                del out, hs
                gen = model.generate(**enc, max_new_tokens=16, do_sample=False)
            answers.append(processor.decode(gen[0][enc["input_ids"].shape[1]:],
                                            skip_special_tokens=True).strip())
            if n % 40 == 0:
                print(f"[repr] {model_key} {clip} {n}/{len(items)} answer={answers[-1]!r} "
                      f"metric={it['metric']:.2f} px={it['px']:.1f}", flush=True)
        path = OUT / model_key / f"{clip.replace('@', '__')}.npz"
        if limit:
            path = path.with_name(path.stem + f"_smoke{limit}.npz")
        atomic_savez(path, **{k: np.stack(v) for k, v in feats.items()},
                     keys=np.array([i["key"] for i in items]),
                     category=np.array([i["category"] for i in items]),
                     window0=np.array([i["window0"] for i in items]),
                     metric=np.array([i["metric"] for i in items]),
                     px=np.array([i["px"] for i in items]),
                     answer=np.array(answers))
        print(f"[repr] {model_key} {clip}: saved {path.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--clips", nargs="*", default=CLIPS)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    extract(a.model, a.clips, a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
