#!/usr/bin/env python3
"""
本地 QLoRA SFT: Qwen3.5-4B (多模态) + CourtSI-1M/通用 QA, 单卡 8GB VRAM。

论文协议是 Qwen3-VL-8B 全参微调 (8x24GB+ZeRO3), 本机不可行, 等效降级:
  - 4-bit NF4 基座 + LoRA (仅语言模型 q/k/v/o/gate/up/down), 视觉塔冻结
  - batch 1 x grad_accum 16, bf16 计算, gradient checkpointing
  - 图片 max_pixels 限制 (默认 448*448 当量) 压序列长度
  - 从 CourtSI-1M 分层抽样 (按 image 所在 sport), 只保留本地图片存在的样本

用法:
  python train/train_qlora.py --smoke            # 32 条样本 8 步, 验证显存/流程
  python train/train_qlora.py --num_samples 20000 --output_dir models/courtsi-qwen3.5-4b-lora
  python train/train_qlora.py --qa_json <target_train.json> --img_root <images> \
      --init_adapter models/courtsi-qwen3.5-4b-lora-final \
      --budget_slice prompt_matched \
      --num_samples 0 --output_dir models/target-sft-v3
"""
import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# 底座位置随机器而异; 环境变量优先, 默认值是开发机布局 (换机器请设 QWEN35_4B)。
DEFAULT_MODEL = os.environ.get("QWEN35_4B") or (
    r"E:\models\hf\hub\models--Qwen--Qwen3.5-4B"
    r"\snapshots\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
DATA_JSON = os.path.join(ROOT, "data", "CourtSI-1M", "CourtSI-1M.json")
IMG_ROOT = os.path.join(ROOT, "data")  # 样本路径形如 CourtSI-1M/images/badminton/...
DEFAULT_OUTPUT = os.path.join(ROOT, "models", "courtsi-qwen3.5-4b-lora")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_existing_files(folder, names):
    records = {}
    if not folder:
        return records
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            records[name] = {"bytes": os.path.getsize(path), "sha256": file_sha256(path)}
    return records


def validate_init_adapter(adapter_path, target_family):
    """Validate a CPU-only adapter receipt before a formal continuation run.

    This deliberately checks more than directory existence: both config and
    weights must be present, and the adapter's recorded base model must resolve
    to the same registered model family as ``--model_path``.  It never loads
    tensors or creates a CUDA context.
    """
    if not adapter_path or not os.path.isdir(adapter_path):
        raise ValueError(f"init adapter directory missing: {adapter_path}")
    config_path = os.path.join(adapter_path, "adapter_config.json")
    weight_candidates = [
        os.path.join(adapter_path, "adapter_model.safetensors"),
        os.path.join(adapter_path, "adapter_model.bin"),
    ]
    if not os.path.isfile(config_path):
        raise ValueError(f"init adapter config missing: {config_path}")
    weights = next((path for path in weight_candidates if os.path.isfile(path)), None)
    if weights is None:
        raise ValueError(f"init adapter weights missing under: {adapter_path}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    recorded_base = config.get("base_model_name_or_path")
    if not recorded_base:
        raise ValueError("init adapter does not record base_model_name_or_path")
    from train.model_family import inspect_local_model
    try:
        adapter_family, _ = inspect_local_model(recorded_base)
    except Exception as exc:
        raise ValueError(
            f"cannot verify init adapter base model {recorded_base!r}: {exc}"
        ) from exc
    if adapter_family.name != target_family.name:
        raise ValueError(
            f"init adapter family mismatch: {adapter_family.name} != {target_family.name}"
        )
    return {
        "adapter_config.json": {
            "bytes": os.path.getsize(config_path),
            "sha256": file_sha256(config_path),
        },
        os.path.basename(weights): {
            "bytes": os.path.getsize(weights),
            "sha256": file_sha256(weights),
        },
        "recorded_base_model": os.path.abspath(recorded_base),
        "model_family": adapter_family.name,
    }


def sample_sequence_sha256(samples):
    """Hash the exact ordered supervision sequence without embedding local pixels."""
    canonical = []
    for sample in samples:
        entry = {
            "image_ref": (sample.get("image_id") or sample.get("image_path")
                          or sample.get("image") or sample.get("images")),
            "question": sample.get("question", sample.get("prompt")),
            "answer": sample.get("answer", sample.get("response")),
            "category": sample.get("category"),
            "messages": sample.get("messages"),
            "resolved_image": sample.get("_img_path"),
        }
        # 只有真正多帧时才加这个键 —— 单图样本的 canonical 保持逐字不变,
        # 论文 1 记录过的 selected_qa_sequence_sha256 才不会因为本次改动而变。
        _ps = sample.get("_img_paths") or []
        if len(_ps) > 1:
            entry["resolved_images"] = _ps
        canonical.append(entry)
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_training_image(image_ref, img_root):
    """解析训练图片，避免仅按 basename 猜图导致同名帧串样本。"""
    if not image_ref:
        return None
    ref = os.fspath(image_ref).replace("/", os.sep)
    candidates = [ref] if os.path.isabs(ref) else []
    candidates.extend([
        os.path.join(img_root, ref),
        os.path.join(img_root, "images", ref),
    ])
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def normalize_training_sample(item, img_root):
    """把 CourtSI messages/images 或 benchmark QA schema 归一成训练样本。

    QA schema 至少需要 image_id、question、answer；额外字段（category 等）保留，
    便于之后审计训练池组成。函数不改写输入 dict。
    """
    out = dict(item)
    messages = item.get("messages")
    images = item.get("images") or []
    if not isinstance(images, list):
        raise ValueError("images must be a list")
    # 多帧 (CourtDyn: image_ids 是 4 帧一组)。单图路径保持逐字不变 —— image_refs 只有一个元素时,
    # 下游行为与改动前完全相同, 论文 1 的单图训练不受影响。
    image_refs = list(images)
    if not image_refs and isinstance(item.get("image_ids"), list) and item["image_ids"]:
        image_refs = list(item["image_ids"])
    if not image_refs:
        single = item.get("image_id") or item.get("image_path") or item.get("image")
        image_refs = [single] if single else []
    image_ref = image_refs[0] if image_refs else None

    if not messages:
        question = item.get("question", item.get("prompt"))
        answer = item.get("answer", item.get("response"))
        if question is None or answer is None:
            raise ValueError("QA 样本缺少 question/answer（或 prompt/response）")
        messages = [
            {"role": "user", "content": "<image>" + str(question)},
            {"role": "assistant", "content": str(answer)},
        ]
        out["images"] = [image_ref] if image_ref else []
        out["_schema"] = "qa"
    else:
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("messages 必须至少含 user 与 assistant 两条消息")
        if not all(isinstance(message, dict) for message in messages):
            raise ValueError("messages entries must be objects")
        roles = [message.get("role") for message in messages]
        if "user" not in roles or roles[-1] != "assistant":
            raise ValueError("messages must contain a user turn and end with assistant")
        for message in messages:
            content = message.get("content")
            if not isinstance(content, (str, list)):
                raise ValueError("message content must be text or structured parts")
            if isinstance(content, list) and not all(
                    isinstance(part, dict) and part.get("type") in {"text", "image"}
                    for part in content):
                raise ValueError("structured message parts must be text/image objects")
        out["_schema"] = "messages"

    paths = [resolve_training_image(r, img_root) for r in image_refs]
    if not paths or any(p is None for p in paths):
        return None                       # 任一帧缺失就整条丢弃, 不能只训到一半的运动
    out["messages"] = messages
    out["_img_path"] = paths[0]           # 保持旧字段, 审计哈希与单图路径都还认它
    out["_img_paths"] = paths
    out["_stratum"] = (str(item.get("category")) if item.get("category") else
                       _sport_from_ref(str(image_ref or "")))
    return out


def _sport_from_ref(image_ref):
    parts = image_ref.replace("\\", "/").split("/")
    try:
        idx = parts.index("images")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def _stratified_pick(samples, n, seed):
    """按 category/sport 做确定性比例抽样；n<=0 表示全量。"""
    rng = random.Random(seed)
    if n <= 0 or n >= len(samples):
        picked = list(samples)
        rng.shuffle(picked)
        return picked
    groups = {}
    for item in samples:
        groups.setdefault(item["_stratum"], []).append(item)
    quotas = {}
    fractions = []
    total = len(samples)
    for name, items in groups.items():
        exact = n * len(items) / total
        quotas[name] = min(len(items), int(exact))
        fractions.append((exact - int(exact), name))
    remaining = n - sum(quotas.values())
    for _, name in sorted(fractions, key=lambda x: (-x[0], x[1])):
        if remaining <= 0:
            break
        if quotas[name] < len(groups[name]):
            quotas[name] += 1
            remaining -= 1
    # 极端情况下（大量已满小组）继续从尚有容量的小组补足。
    while remaining:
        progressed = False
        for name in sorted(groups):
            if quotas[name] < len(groups[name]):
                quotas[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    picked = []
    for name, items in groups.items():
        picked.extend(rng.sample(items, quotas[name]))
    rng.shuffle(picked)
    return picked


def load_samples(num_samples, seed=42, qa_json=None, img_root=None,
                 allow_drop_invalid=False):
    """读取并校验训练样本；支持 CourtSI 与通用 image/question/answer JSON。"""
    source = os.path.abspath(qa_json or DATA_JSON)
    root = os.path.abspath(img_root or (IMG_ROOT if qa_json is None
                                        else os.path.dirname(source)))
    print(f"[qlora] 读取 {source} (img_root={root}) ...", flush=True)
    with open(source, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("训练 JSON 顶层必须是 list")
    print(f"[qlora] 总样本 {len(data)}", flush=True)

    missing = 0
    invalid = 0
    normalized = []
    for it in data:
        try:
            sample = normalize_training_sample(it, root)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if sample is None:
            missing += 1
            continue
        normalized.append(sample)
    if qa_json is not None and not allow_drop_invalid and (missing or invalid):
        raise ValueError(
            f"target QA must match the frozen pool exactly; missing={missing}, "
            f"invalid={invalid}. Use --allow_drop_invalid only for an explicitly "
            "registered non-comparable diagnostic."
        )
    if not normalized:
        raise ValueError(f"没有可用训练样本（missing={missing}, invalid={invalid}）")
    by_stratum = {}
    for sample in normalized:
        by_stratum[sample["_stratum"]] = by_stratum.get(sample["_stratum"], 0) + 1
    print(f"[qlora] 图片缺失/无图={missing}; 非法={invalid}; 分布: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_stratum.items())), flush=True)

    picked = _stratified_pick(normalized, num_samples, seed)
    print(f"[qlora] 抽样 {len(picked)} 条 (seed={seed})", flush=True)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--num_samples", type=int, default=20000,
                    help="抽样数；0=使用全部可用样本")
    ap.add_argument("--qa_json", default=None,
                    help="target-SFT/general QA: list[image_id,question,answer,category]")
    ap.add_argument("--img_root", default=None,
                    help="--qa_json 中 image_id 的图片根目录")
    ap.add_argument("--init_adapter", default=None,
                    help="从已有可训练 LoRA 开始（公平 target-SFT 应指向 source-SFT）")
    ap.add_argument("--allow_base_init", action="store_true",
                    help="诊断专用：允许 target QA 从 base 新建 LoRA；正式公平基线禁用")
    ap.add_argument(
        "--budget_slice",
        choices=(
            "source_sft",
            "prompt_matched",
            "optimizer_update_matched",
            "applied_completion_label_token_matched",
            "diagnostic",
        ),
        default=None,
        help="冻结 target-SFT 公平性切片；正式 target QA 必须显式给出",
    )
    ap.add_argument("--soccer_synloc_training", action="store_true",
                    help="启用 SynLoc calibration-only provenance 硬门；GSR QA 会被拒绝")
    ap.add_argument("--validate_only", action="store_true",
                    help="仅在 CPU 上校验/抽样训练 JSON 与图片路径，不加载模型")
    ap.add_argument("--allow_drop_invalid", action="store_true",
                    help="允许丢坏样本（仅诊断；正式 target-SFT 禁止）")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--num_train_epochs", type=float, default=1.0,
                    help="--max_steps<0 时使用的 epoch 数；prompt-matched 默认 1")
    ap.add_argument("--seed", type=int, default=42,
                    help="数据顺序与 Trainer 随机种子")
    ap.add_argument("--lr", type=float, default=1e-4)          # LoRA 常用 1e-4 (全参才 5e-6)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_pixels", type=int, default=448 * 448)
    ap.add_argument("--vision_lora", action="store_true",
                    help="LoRA 同时挂到视觉塔 (blocks attn.qkv/proj + mlp.linear_fc1/2 + merger); 默认视觉塔冻结")
    ap.add_argument("--gpu_throttle", type=float, default=0.0,
                    help="GPU 占空比上限 (0-1, 如 0.8 = 每步后睡步时的 25%%, 给桌面留渲染窗口; 0=不限)")
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="32 样本 8 步冒烟测试")
    args = ap.parse_args()
    if args.smoke:
        args.num_samples, args.max_steps = 32, 8
    if args.qa_json and args.budget_slice is None:
        ap.error("正式/诊断 target QA 必须显式指定 --budget_slice")
    if not args.qa_json and args.budget_slice not in (None, "source_sft", "diagnostic"):
        ap.error("非 target QA 不能声明 target-SFT budget slice")
    if args.allow_base_init and args.budget_slice != "diagnostic":
        ap.error("--allow_base_init 只能与 --budget_slice diagnostic 一起使用")
    if args.qa_json and not args.allow_base_init and args.budget_slice == "diagnostic":
        ap.error("正式 source-adapter target-SFT 不能标为 diagnostic")

    samples = load_samples(args.num_samples, seed=args.seed,
                           qa_json=args.qa_json, img_root=args.img_root,
                           allow_drop_invalid=args.allow_drop_invalid)
    if args.soccer_synloc_training:
        if not args.qa_json:
            ap.error("--soccer_synloc_training requires --qa_json")
        from engine.soccer.audit import assert_training_eligible_qa
        assert_training_eligible_qa(samples)
    # AutoConfig only: freeze family-specific preprocessing/LoRA policy before
    # model tensors or a CUDA context are created.
    from train.model_family import (assert_processor_policy, inspect_local_model,
                                    processor_policy)
    model_family, _ = inspect_local_model(args.model_path)
    if args.vision_lora and model_family.name != "qwen_vl":
        ap.error("--vision_lora is currently frozen only for the Qwen visual tower")
    if args.qa_json and os.path.abspath(args.output_dir) == os.path.abspath(DEFAULT_OUTPUT):
        ap.error("target-SFT 必须显式指定独立 --output_dir，防止覆盖 source-SFT")
    if (args.init_adapter and
            os.path.abspath(args.init_adapter) == os.path.abspath(args.output_dir)):
        ap.error("--output_dir 不能与 --init_adapter 相同，防止原地覆盖源适配器")
    if args.qa_json and not args.init_adapter and not args.allow_base_init:
        ap.error(
            "正式 target-SFT 必须给 --init_adapter 从 source-SFT 开始；"
            "仅诊断时可显式传 --allow_base_init"
        )
    adapter_receipt = (
        validate_init_adapter(args.init_adapter, model_family)
        if args.init_adapter else {}
    )
    if args.validate_only:
        init_label = "verified" if adapter_receipt else (
            "diagnostic-base-init" if args.allow_base_init else "source-SFT run"
        )
        print(f"[qlora] validate-only OK（family={model_family.name}；"
              f"init={init_label}；未加载模型权重，未访问 GPU）", flush=True)
        return

    source_path = os.path.abspath(args.qa_json or DATA_JSON)
    selected_image_root = os.path.abspath(
        args.img_root or (IMG_ROOT if args.qa_json is None else os.path.dirname(source_path))
    )
    receipt = {
        "schema_version": "sft-training-protocol-v1",
        "qa_source": source_path,
        "qa_source_sha256": file_sha256(source_path),
        "image_root": selected_image_root,
        "selected_samples": len(samples),
        "ordered_sample_sequence_sha256": sample_sequence_sha256(samples),
        "allow_drop_invalid": bool(args.allow_drop_invalid),
        "model_path": os.path.abspath(args.model_path),
        "model_config": _hash_existing_files(args.model_path, ["config.json"]),
        "model_family": model_family.name,
        "processor_policy": processor_policy(model_family, args.max_pixels),
        "init_adapter": os.path.abspath(args.init_adapter) if args.init_adapter else None,
        "init_adapter_files": _hash_existing_files(
            args.init_adapter, ["adapter_config.json", "adapter_model.safetensors"]),
        "init_adapter_validation": adapter_receipt,
        "seed": args.seed,
        "num_samples_request": args.num_samples,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.lr,
        "gradient_accumulation_steps": args.grad_accum,
        "max_pixels": args.max_pixels,
        "vision_lora": bool(args.vision_lora),
        "allow_base_init": bool(args.allow_base_init),
        "budget_slice": args.budget_slice or "source_sft",
        "initialization_seed_applied_before_model_load": True,
        "soccer_synloc_training_gate": bool(args.soccer_synloc_training),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    receipt_path = os.path.join(args.output_dir, "training_protocol.json")
    if os.path.isfile(receipt_path):
        with open(receipt_path, encoding="utf-8") as handle:
            previous_receipt = json.load(handle)
        if previous_receipt != receipt:
            raise ValueError(
                "existing training_protocol.json differs from this run; use a new "
                "output directory instead of mixing experiments"
            )
    else:
        with open(receipt_path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, ensure_ascii=False)
        os.replace(receipt_path + ".tmp", receipt_path)

    import time
    import torch

    # 8/26-27 三次 nvlddmkm/UVM 蓝屏: VRAM 压到 94% 后驱动走 sysmem 换页路径崩溃。
    # 616.56 驱动已无 sysmem fallback 开关, 改在分配器层设上限: 超限抛干净 OOM,
    # 不给 WDDM 换页机会。VRAM_CAP_FRACTION 可调, 设 0 关闭。
    _vram_cap = float(os.environ.get("VRAM_CAP_FRACTION", "0.93"))
    if _vram_cap > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_vram_cap, 0)

    from PIL import Image
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)

    # Trainer 会在 train() 前再次设 seed，但新 LoRA 参数在那之前就初始化。
    # 必须在模型/adapter 构造前固定全部 RNG，才能让 source-SFT seed 真正可复现。
    set_seed(args.seed)

    class GPUThrottleCallback(TrainerCallback):
        """占空比限 GPU: 每个 micro-batch 后 sleep dt*(1-f)/f, 平均占用 ~f。

        在 substep 粒度 (梯度累积的每个小步, ~2-4s) 插空, 桌面合成器每隔
        一小步就能拿到 GPU, 比整个 optimizer step (~30s) 后再睡流畅得多。
        """

        def __init__(self, frac):
            self.frac = max(0.05, min(frac, 1.0))
            self._mark = None

        def _pause(self):
            torch.cuda.synchronize()
            now = time.time()
            if self._mark is not None:
                dt = now - self._mark
                time.sleep(dt * (1.0 - self.frac) / self.frac)
            self._mark = time.time()

        def on_substep_end(self, args, state, control, **kw):
            self._pause()

        def on_step_end(self, args, state, control, **kw):
            self._pause()

    print(f"[qlora] 加载 {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=model_family.trust_remote_code,
        **processor_policy(model_family, args.max_pixels))
    assert_processor_policy(model_family, processor)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=model_family.trust_remote_code,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    model.config.use_cache = False

    if model_family.name == "smolvlm2":
        # Trainer bf16 autocast 下 SmolVLM 视觉塔的 post_layernorm 输出 fp32, 连接器里的
        # bnb Linear4bit 又把输出转回输入 dtype, 于是 inputs_merger 里 image_hidden_states
        # (fp32) 与 inputs_embeds (bf16) 不匹配 → "Index put requires the source and
        # destination dtypes match" (T23 冒烟 2026-09-03)。无 autocast 的 run_bench /
        # train_grpo / smoke_second_vlm 不受影响。只对 smolvlm2 挂钩, Qwen 路径逐字不变。
        embed_dtype = model.get_input_embeddings().weight.dtype
        n_hook = 0
        for mod in model.modules():
            if type(mod).__name__ == "SmolVLMConnector":
                mod.register_forward_hook(
                    lambda m, i, out, _d=embed_dtype: out.to(_d))
                n_hook += 1
        if n_hook != 1:
            raise RuntimeError(f"expected exactly one SmolVLMConnector, found {n_hook}")
        print(f"[qlora] smolvlm2: connector 输出统一转为 {embed_dtype}", flush=True)

    if args.init_adapter:
        print(f"[qlora] 从可训练适配器继续: {args.init_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    elif args.vision_lora:
        # 语言侧 + 视觉塔 (24 blocks 的 qkv/proj/linear_fc1/2) + merger 投影
        lora = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=(r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                            r"|.*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
                            r"|.*visual\.merger\.linear_fc[12]"))
    else:
        lora = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=list(model_family.language_lora_targets),
            # 只挂在语言模型上, 视觉塔/投影层冻结
            exclude_modules=model_family.vision_exclude_pattern)
    if not args.init_adapter:
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    tok = processor.tokenizer

    class CourtDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            it = samples[idx]
            msgs = it["messages"]
            # 多帧: 按 image_ids 的顺序全部挂上 (CourtDyn 是 4 帧)。
            # 单图样本 _img_paths 只有一个元素, 走下面同一段代码, 行为与改动前逐字相同。
            imgs = [Image.open(p).convert("RGB")
                    for p in (it.get("_img_paths") or [it["_img_path"]])]
            # content 里的 <image> 占位换成结构化 content
            conv = []
            image_attached = False
            for m in msgs:
                role, content = m["role"], m.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part.get("type") == "image":
                            parts.append({"type": "image", "image": imgs[0]})
                            image_attached = True
                        elif part.get("type") == "text":
                            text = str(part.get("text", "")).replace("<image>", "")
                            parts.append({"type": "text", "text": text})
                    if role == "user" and not image_attached:
                        for j, im in enumerate(imgs):
                            parts.insert(j, {"type": "image", "image": im})
                        image_attached = True
                else:
                    text = str(content)
                    parts = [{"type": "text", "text": text.replace("<image>", "")}]
                    if role == "user" and ("<image>" in text or not image_attached):
                        for j, im in enumerate(imgs):
                            parts.insert(j, {"type": "image", "image": im})
                        image_attached = True
                conv.append({"role": role, "content": parts})
            kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
            try:
                full = processor.apply_chat_template(conv, enable_thinking=False, **kw)
                prompt = processor.apply_chat_template(
                    conv[:-1], add_generation_prompt=True, enable_thinking=False, **kw)
            except TypeError:
                full = processor.apply_chat_template(conv, **kw)
                prompt = processor.apply_chat_template(conv[:-1], add_generation_prompt=True, **kw)
            ids = full["input_ids"][0]
            labels = ids.clone()
            labels[: prompt["input_ids"].shape[1]] = -100   # 只训 assistant 回答
            # 只给 token 级张量去 batch 维; pixel_values/image_grid_thw 本身无 batch 维
            out = {"input_ids": ids,
                   "attention_mask": full["attention_mask"][0],
                   "labels": labels}
            for k, v in full.items():
                if k not in ("input_ids", "attention_mask"):
                    out[k] = v
            return out

    def collate_fn(batch):  # batch=1, 仅给文本张量补 batch 维
        assert len(batch) == 1
        out = {}
        for k, v in batch[0].items():
            if k in ("input_ids", "attention_mask", "labels"):
                out[k] = v.unsqueeze(0)
            else:
                out[k] = v  # pixel_values / image_grid_thw 已含正确维度
        return out

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if args.smoke else 10,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="paged_adamw_8bit",
        seed=args.seed,
        data_seed=args.seed,
    )
    callbacks = [GPUThrottleCallback(args.gpu_throttle)] if args.gpu_throttle else []
    trainer = Trainer(model=model, args=targs,
                      train_dataset=CourtDataset(), data_collator=collate_fn,
                      callbacks=callbacks)
    trainer.train(resume_from_checkpoint=bool(
        os.path.isdir(args.output_dir)
        and any(d.startswith("checkpoint-") for d in os.listdir(args.output_dir))))
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    final_files = _hash_existing_files(
        args.output_dir,
        ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"],
    )
    if "adapter_config.json" not in final_files or not any(
        name in final_files for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError("final target-SFT adapter is incomplete; no completion receipt written")
    completion = {
        "schema_version": "sft-training-completion-v1",
        "completed": True,
        "training_protocol_sha256": file_sha256(receipt_path),
        "seed": args.seed,
        "global_step": int(trainer.state.global_step),
        "selected_samples": len(samples),
        "budget_slice": args.budget_slice or "source_sft",
        "final_adapter_files": final_files,
    }
    completion_path = os.path.join(args.output_dir, "training_completion.json")
    with open(completion_path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2, ensure_ascii=False)
    os.replace(completion_path + ".tmp", completion_path)
    print(f"[qlora] done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
