#!/usr/bin/env python3
"""
CourtSI-Bench 端到端评测入口(在 GPU 机器上运行)。

流程:
  1. 解压 data/CourtSI-Bench/images.zip -> data/CourtSI-Bench/images/ (若尚未解压)
  2. 读取 data/CourtSI-Bench/qa_bench.json
  3. 用 Qwen3-VL (基础模型或微调后的 CourtSI-Qwen3-VL-8B) 推理得到 vlm_answer
  4. 调 evaluate.py 的 evaluate() 计算 exact / T-MRA 指标

本机无 GPU, 仅做 dry-run(--dry_run) 校验流程与格式。

用法:
  python run_bench.py --model_path <hf_or_local> --bench_dir data/CourtSI-Bench --output_folder results/bench
  python run_bench.py --dry_run        # 不加载模型, 仅校验数据/解压流程
"""
import argparse
import hashlib
import json
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import evaluate  # noqa: E402
from prompt_reference import (REFERENCE_PRESETS, compose_reference_prompt,
                              load_reference_text)  # noqa: E402


GENERATION_SUFFIX = "\nGive ONLY the final answer in the required format. Do not explain."


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_ref(value):
    if value is None:
        return None
    return os.path.abspath(value) if os.path.exists(value) else str(value)


def _qa_sequence_sha256(rows):
    payload = [{
        "image_id": row.get("image_id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "category": row.get("category"),
    } for row in rows]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _config_digest(config):
    raw = json.dumps(config, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_or_validate_run_config(path, config):
    """Prevent resume across a changed model/input/intervention configuration.

    Historical queues wrote a smaller config.  They may be upgraded only when
    every key they did record still matches; all newly written configs require
    exact equality.  This preserves watchdog recovery for the active legacy
    queue while making new experiments fail closed.
    """

    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("schema_version") == config["schema_version"]:
            if existing != config:
                # One narrow compatibility bridge for queues that were already
                # running when metric_version was added.  Only an absent field
                # whose requested value reproduces the historical metric may be
                # added; floor_v2 or any intervention/config change still fails.
                added = set(config) - set(existing)
                shared_match = all(
                    config.get(key) == value
                    for key, value in existing.items()
                    if key != "config_sha256"
                )
                safe_addition = (
                    added == {"metric_version"}
                    and config.get("metric_version") == "legacy_v1"
                    and shared_match
                )
                if safe_addition:
                    print("[bench] adding legacy_v1 metric receipt to active run",
                          flush=True)
                else:
                    raise ValueError(
                        "output_folder run_config differs; use a new output directory "
                        "instead of resuming across model/data/intervention settings"
                    )
        else:
            mismatched = {
                key: (value, config.get(key)) for key, value in existing.items()
                if key in config and value != config.get(key)
            }
            unknown = sorted(set(existing) - set(config))
            if mismatched or unknown:
                raise ValueError(
                    f"legacy run_config is incompatible: mismatched={mismatched}, "
                    f"unknown={unknown}"
                )
            print("[bench] legacy run_config material fields match; upgrading receipt",
                  flush=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def unzip_images(bench_dir):
    """解压 images.zip (可能含 z01/z02 分卷) 到 bench_dir/images/。返回图片根目录。"""
    img_root = os.path.join(bench_dir, "images")
    if os.path.isdir(img_root) and any(os.scandir(img_root)):
        print(f"[bench] images 已存在, 跳过解压: {img_root}")
        return img_root
    zip_path = os.path.join(bench_dir, "images.zip")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"找不到 {zip_path}, 请先下载 CourtSI-Bench")
    print(f"[bench] 解压 {zip_path} -> {img_root} ...")
    os.makedirs(img_root, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(img_root)
    print(f"[bench] 解压完成")
    return img_root


def resolve_image_path(image_id, img_root):
    """image_id 形如 'badminton/match10/frame/000/0048.jpg'。

    zip 内有时带顶层 images/ 前缀, 解压后变成 images/images/...,
    故同时尝试带/不带前缀的两条路径。正式评测禁止只按 basename 猜图：
    不同比赛/相机常有同名帧，静默 fallback 会把缺图伪装成有效样本。
    """
    if not isinstance(image_id, str) or not image_id.strip():
        return None
    normalized = image_id.replace("\\", "/")
    if os.path.isabs(image_id) or any(part == ".." for part in normalized.split("/")):
        return None
    root = os.path.abspath(img_root)
    cands = [os.path.abspath(os.path.join(root, image_id)),
             os.path.abspath(os.path.join(root, "images", image_id))]
    for p in cands:
        try:
            inside_root = os.path.commonpath([root, p]) == root
        except ValueError:
            inside_root = False
        if inside_root and os.path.isfile(p):
            return p
    return None


def build_cross_image_map(rows, img_root, seed=42):
    """Deterministic image-level derangement for the cross-image intervention."""
    image_ids = list(dict.fromkeys(row["image_id"] for row in rows))
    if len(image_ids) < 2:
        raise ValueError("cross-image intervention needs at least two unique images")
    import random
    shuffled = list(image_ids)
    random.Random(seed).shuffle(shuffled)
    rotated = shuffled[1:] + shuffled[:1]
    mapping = dict(zip(shuffled, rotated))
    if any(source == target for source, target in mapping.items()):
        raise AssertionError("cross-image mapping is not a derangement")
    missing = [target for target in mapping.values()
               if resolve_image_path(target, img_root) is None]
    if missing:
        raise ValueError(f"cross-image mapping contains missing images: {missing[:3]}")
    return mapping


def load_model(model_path, load_4bit=False, adapter=None, max_pixels=0,
               blank_images=False, max_new_tokens=128):
    """加载冻结 registry 中的 Qwen-VL/SmolVLM2 推理管线。

    adapter: LoRA 适配器目录 (train_qlora.py 产物), 挂到 4-bit 基座上做微调后评测。
    max_pixels: >0 时限制图片像素预算; 评微调模型要和训练时 (448²) 一致, 否则
                分辨率变化会和微调效果混在一起, 无法归因。
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    repo_root = os.path.dirname(HERE)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from train.model_family import (assert_processor_policy, inspect_local_model,
                                    processor_policy)
    model_family, _ = inspect_local_model(model_path)

    # 见 train_qlora.py: 蓝屏防线, 超过 VRAM 上限抛干净 OOM 而非驱动 UVM 崩溃
    _vram_cap = float(os.environ.get("VRAM_CAP_FRACTION", "0.93"))
    if _vram_cap > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_vram_cap, 0)

    print(f"[bench] 加载模型: {model_path} (4bit={load_4bit}, adapter={adapter}, "
          f"max_pixels={max_pixels or 'default'})")
    proc_kwargs = dict(trust_remote_code=model_family.trust_remote_code)
    if model_family.image_budget_kind != "max_pixels" or max_pixels:
        proc_kwargs.update(processor_policy(model_family, max_pixels or 200704))
    processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
    assert_processor_policy(model_family, processor)
    kwargs = dict(dtype=torch.bfloat16, device_map="auto",
                  trust_remote_code=model_family.trust_remote_code)
    if load_4bit:
        from transformers import BitsAndBytesConfig
        # CPU 卸载: 仅当显式设置 GPU_MAX_MEM 时启用, 默认关闭, 不影响任何既有队列。
        #
        # ⚠ 2026-09-08 实测: 对 Qwen3.5-9B (qwen_vl 族, trust_remote_code) **走不通**, 三步都试过:
        #   1) 不设本变量           -> bitsandbytes 直接拒绝: "Some modules are dispatched on the CPU"
        #   2) 设本变量 + double_quant -> accelerate 挂钩时 offset.item() 在 meta 张量上崩
        #   3) 再关掉 double_quant     -> 模型能加载, 但前向全崩 "Cannot copy out of meta tensor", 4/4 无输出
        # 根因是 accelerate 的 offload hook 没接管远程代码里的模块, 参数留在 meta 上没被换入。
        # 即使修通, 卸载 3-4 GB 到 CPU 以 fp32 跑, 预计 10-50x 减速, 单格数小时, 不划算。
        # 保留这段是为了记录"试过且为什么不行", 不是可用路径 —— 别再照着排队列。
        _gpu_max = os.environ.get("GPU_MAX_MEM", "").strip()
        # 卸载时必须关掉 double-quant: accelerate 挂执行设备钩子时会对 Linear4bit 调 state_dict(),
        # 而嵌套量化常数此时还在 meta 设备上, bitsandbytes 会在 offset.item() 处炸
        # ("Tensor.item() cannot be called on meta tensors")。这是库的兼容问题, 不是配置写错。
        # 代价: 该臂的量化方案与其余格子不同, 若用于论文必须写明。
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=not _gpu_max,
            llm_int8_enable_fp32_cpu_offload=bool(_gpu_max))
        if _gpu_max:
            _cpu_max = os.environ.get("CPU_MAX_MEM", "16GiB").strip()
            kwargs["max_memory"] = {0: _gpu_max, "cpu": _cpu_max}
            print(f"[bench] CPU 卸载已启用: GPU<={_gpu_max}, CPU<={_cpu_max} (会显著变慢)")
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    if model_family.name == "smolvlm2":
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
        if n_lora == 0:
            raise RuntimeError(f"适配器 {adapter} 没挂上任何 LoRA 权重")
        print(f"[bench] LoRA 已挂载: {n_lora} 个 lora_* 张量")
    model.eval()

    def infer(image_path, question):
        # image_path 可以是单张路径 (历史行为, 逐字节不变) 或多帧路径列表
        # (CourtDyn 动力学题: 同一条消息里按时序放 N 张图)。
        paths = [image_path] if isinstance(image_path, str) else list(image_path)
        imgs = [Image.open(p).convert("RGB") for p in paths]
        if blank_images:
            # 反 reward-hacking 对照 (arXiv:2604.15149): 换成同尺寸灰板。若模型
            # 得分不塌, 说明 RL 学到的是引擎问题的答案先验而非视觉感知。
            imgs = [Image.new("RGB", im.size, (127, 127, 127)) for im in imgs]
        msgs = [{
            "role": "user",
            "content": [{"type": "image", "image": im} for im in imgs]
            + [
                {"type": "text", "text": question + GENERATION_SUFFIX},
            ],
        }]
        tmpl_kwargs = dict(add_generation_prompt=True, tokenize=True,
                           return_dict=True, return_tensors="pt")
        try:
            enc = processor.apply_chat_template(
                msgs, enable_thinking=False, **tmpl_kwargs).to(model.device)
        except Exception:
            try:
                enc = processor.apply_chat_template(msgs, **tmpl_kwargs).to(model.device)
            except Exception:
                rendered = processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False
                )
                enc = processor(text=rendered, images=imgs, return_tensors="pt").to(
                    model.device
                )
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False
            )
        gen = out[0][enc["input_ids"].shape[1]:]
        return processor.decode(gen, skip_special_tokens=True).strip()

    return infer


_MULTIFRAME_LEAD = re.compile(
    r"^The \d+ frames are consecutive samples from one basketball clip spanning "
    r"([\d.]+) seconds, in chronological order\.")


def rewrite_single_frame_prompt(question):
    """single_v2: 题面从"N 帧按时序"改成"这是片段里的一帧", 其余逐字不动。

    T18 的 single 臂只裁了图, 题面仍告诉模型有 4 帧 —— 模型被告知 4 帧却只收到
    1 帧 (plan §〇 A3), 分数不可解读。记录里 `question` 保持原文 (它是打分器的
    对齐键), 实际送进模型的文本另存 `inference_question`。非多帧题面原样返回。
    """
    return _MULTIFRAME_LEAD.sub(
        r"This is a single frame from a basketball clip spanning \1 seconds.",
        question, count=1)


def apply_frame_mode(image_ids, mode, item):
    """CourtDyn 帧条件对照 (roadmap: 静帧 vs 完整视频 vs 打乱帧)。

    full    原样按时序;
    single  只留第一帧 —— 动力学信息被抽走, 分数应当塌回答案先验;
    shuffle 打乱顺序 —— 内容不变、时序破坏, 分数落差量化时序敏感性。

    打乱用 (seed=42, 题目身份) 派生的确定性排列: 同一条题在任何机器、任何
    续跑上得到同一个乱序, 且不同题的乱序互不相同 (避免固定排列被学成常量)。
    保证结果 != 原序 (帧数 >= 2 时), 否则 "打乱" 臂里会混入真·时序样本。
    """
    if mode in ("single", "single_v2"):
        return image_ids[:1]
    if mode == "static4":
        # T20 G1: 首帧重复 N 次 —— token 预算与 full 完全相同, 运动信息为零。
        # 这是 speed/path/faster 这些帧序不变家族唯一干净的时序对照 (plan §〇 A2),
        # 也把 "4 图输入拖垮 4bit 小模型" 的预算混淆从 full−single 里剥出来 (A3)。
        return [image_ids[0]] * len(image_ids)
    if mode != "shuffle" or len(image_ids) < 2:
        return list(image_ids)
    import random
    seed = int(hashlib.sha256(
        ("42|" + "|".join(image_ids)).encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    order = list(image_ids)
    for _ in range(20):
        rng.shuffle(order)
        if order != list(image_ids):
            return order
    return list(reversed(image_ids))


def extract_answer(raw):
    """取最后一个非空行作为最终答案 (模型偶尔仍带推理), 去掉 'Answer:' 前缀。"""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    ans = lines[-1]
    ans = __import__("re").sub(r"(?i)^(final\s+)?answer\s*[:：]\s*", "", ans)
    return ans.strip().strip("*").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="data/CourtSI-Qwen3-VL-8B",
                    help="Qwen3-VL 模型路径(基础或微调后)")
    ap.add_argument("--bench_dir", default="data/CourtSI-Bench")
    ap.add_argument("--output_folder", default="results/bench")
    ap.add_argument("--dry_run", action="store_true",
                    help="不加载模型, 仅校验解压+数据格式")
    ap.add_argument("--load_4bit", action="store_true",
                    help="bitsandbytes NF4 量化加载 (8GB 显存机器用)")
    ap.add_argument("--limit", type=int, default=0,
                    help="随机抽样 N 条评测 (0=全量), seed=42 可复现")
    ap.add_argument("--resume", action="store_true",
                    help="从 output_folder/predictions_partial.json 断点续跑")
    ap.add_argument("--adapter", default=None,
                    help="LoRA 适配器目录 (评微调模型时给, --model_path 仍指基座)")
    ap.add_argument("--max_pixels", type=int, default=0,
                    help="图片像素预算上限; 评 QLoRA 适配器请用训练时的 200704 (448²)")
    ap.add_argument("--qa_json", default=None,
                    help="直接指定 QA 文件 (如篮球 qa_real.json), 跳过 bench_dir 布局")
    ap.add_argument("--img_root", default=None,
                    help="图片根目录 (image_id 相对它解析), 跳过解压逻辑")
    ap.add_argument("--blank_images", action="store_true",
                    help="反 reward-hacking 对照: 图片换同尺寸灰板, 测答案先验")
    ap.add_argument("--cross_images", action="store_true",
                    help="反 reward-hacking 对照: 问题不变，换成确定性错配的另一张真实图片")
    ap.add_argument("--allow_missing_images", action="store_true",
                    help="仅诊断：允许缺图并以空预测计分；正式冻结评测禁用")
    ap.add_argument("--reference_prompt", default="none",
                    choices=sorted(REFERENCE_PRESETS),
                    help="零训练 metric-reference prompt 基线（仅公开球场尺寸）")
    ap.add_argument("--reference_text_file", default=None,
                    help="自定义 reference 文本；与 --reference_prompt 非 none 互斥")
    ap.add_argument("--frame_mode", default="full",
                    choices=["full", "single", "single_v2", "static4", "shuffle"],
                    help="多帧题 (QA 带 image_ids) 的帧条件对照: full=按时序全部; "
                         "single=只给第一帧 (⚠ 题面仍说 N 帧, 有提示/输入不匹配, "
                         "保留只为复现 T18/T19); single_v2=只给第一帧且题面改成"
                         "单帧措辞; static4=首帧重复 N 次 (预算=full, 运动=0); "
                         "shuffle=打乱顺序 (时序敏感性对照, seed=42)。"
                         "对单帧 QA 无影响。")
    ap.add_argument("--metric_version", default="legacy_v1",
                    choices=["legacy_v1", "floor_v2"],
                    help="旧队列用 legacy_v1；PR 冻结协议的新结果用 floor_v2")
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="生成 token 上限；默认保持历史评测的 128")
    args = ap.parse_args()
    if args.blank_images and args.cross_images:
        ap.error("--blank_images 与 --cross_images 互斥")
    if args.max_new_tokens <= 0:
        ap.error("--max_new_tokens 必须为正整数")

    custom_reference = None
    if args.reference_text_file:
        if args.reference_prompt != "none":
            ap.error("--reference_text_file 与非 none 的 --reference_prompt 不能同时使用")
        with open(args.reference_text_file, encoding="utf-8") as f:
            custom_reference = f.read()
    reference_tag, reference_text = load_reference_text(
        args.reference_prompt, custom_reference)

    bench_dir = args.bench_dir
    qa_path = args.qa_json or os.path.join(bench_dir, "qa_bench.json")
    if not os.path.isfile(qa_path):
        raise FileNotFoundError(f"找不到 {qa_path}")

    img_root = args.img_root or unzip_images(bench_dir)
    qa = json.load(open(qa_path, encoding="utf-8"))
    print(f"[bench] 读取 {len(qa)} 条 QA")

    if args.limit and args.limit < len(qa):
        import random
        qa = random.Random(42).sample(qa, args.limit)
        print(f"[bench] 抽样 {len(qa)} 条 (seed=42)")

    # 多帧题 (CourtDyn 动力学) 带 image_ids; 其余仍走单张 image_id
    multiframe_n = sum(1 for it in qa if it.get("image_ids"))
    if multiframe_n:
        print(f"[bench] 多帧题 {multiframe_n}/{len(qa)} 条, frame_mode={args.frame_mode}")
        if args.cross_images:
            ap.error("--cross_images 尚未定义多帧语义, 不能与 image_ids 数据集同用")

    # 校验图片存在性 (多帧题要求整组帧齐全, 缺一帧即算缺失)
    missing = 0
    for it in qa:
        ids = it.get("image_ids") or [it["image_id"]]
        if any(resolve_image_path(x, img_root) is None for x in ids):
            missing += 1
    print(f"[bench] 图片缺失: {missing}/{len(qa)}")
    if missing and not args.allow_missing_images:
        raise ValueError(
            "frozen evaluation has missing images; refusing to score a partial pool"
        )
    cross_image_map = (
        build_cross_image_map(qa, img_root, seed=42) if args.cross_images else None
    )

    if args.dry_run:
        if reference_text and qa:
            print(f"[bench] reference={reference_tag}; prompt preview:\n"
                  f"{compose_reference_prompt(qa[0]['question'], reference_text)}")
        print("[bench] DRY RUN: 跳过模型推理")
        # 自洽校验: 以 gt 当 pred
        for it in qa:
            it["vlm_answer"] = it["answer"]
        summary, _ = evaluate(
            qa,
            allow_ground_truth_self_check=True,
            metric_version=args.metric_version,
        )
        print(f"[bench] 自洽 exact={summary['exact_accuracy']:.4f} "
              f"(应为 1.0); T-MRA={summary['tmra_overall']}")
        return

    os.makedirs(args.output_folder, exist_ok=True)
    run_config = {
        "schema_version": "run-bench-config-v2",
        "qa_json": os.path.abspath(qa_path),
        "qa_json_sha256": _sha256_file(qa_path),
        "selected_qa_sequence_sha256": _qa_sequence_sha256(qa),
        "selected_qa_count": len(qa),
        "img_root": os.path.abspath(img_root),
        "model_path": _canonical_ref(args.model_path),
        "adapter": _canonical_ref(args.adapter),
        "load_4bit": bool(args.load_4bit),
        "max_pixels": int(args.max_pixels),
        "limit": int(args.limit),
        "limit_seed": 42 if args.limit else None,
        "blank_images": bool(args.blank_images),
        "reference_tag": reference_tag,
        "reference_text": reference_text,
        "generation": {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": False,
            "suffix": GENERATION_SUFFIX,
        },
        "metric_version": args.metric_version,
    }
    # 帧条件只在真的用到多帧 (或显式选了非默认模式) 时写进配置 —— 历史单帧
    # 实验的 run_config 保持逐字节不变, _write_or_validate_run_config 的
    # fail-closed 语义不被这次扩展削弱。
    if multiframe_n or args.frame_mode != "full":
        run_config["frame_mode"] = args.frame_mode
        run_config["n_multiframe_items"] = multiframe_n
        run_config["frame_shuffle_seed"] = 42 if args.frame_mode == "shuffle" else None
    if cross_image_map is not None:
        run_config["cross_images"] = True
        run_config["cross_image_seed"] = 42
        run_config["cross_image_mapping_sha256"] = _config_digest(cross_image_map)
    # Qwen historical receipts stay byte-compatible with active queues.  The
    # second-family arm records its non-Qwen preprocessing policy explicitly.
    try:
        repo_root = os.path.dirname(HERE)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from train.model_family import inspect_local_model, processor_policy
        family, _ = inspect_local_model(args.model_path)
        if family.name != "qwen_vl":
            run_config["model_family"] = family.name
            run_config["processor_policy"] = processor_policy(
                family, args.max_pixels or 200704
            )
    except (FileNotFoundError, OSError, ValueError):
        # Non-local legacy model identifiers are validated when load_model runs.
        pass
    run_config["config_sha256"] = _config_digest(run_config)
    _write_or_validate_run_config(
        os.path.join(args.output_folder, "run_config.json"), run_config
    )
    run_config_sha256 = run_config["config_sha256"]
    ckpt_path = os.path.join(args.output_folder, "predictions_partial.json")
    done = {}
    if args.resume and os.path.isfile(ckpt_path):
        prev = json.load(open(ckpt_path, encoding="utf-8"))
        done = {(d["image_id"], d["question"], d.get("reference_tag", "none")): d
                for d in prev if d.get("vlm_answer")}
        print(f"[bench] 断点续跑: 复用 {len(done)} 条已完成预测", flush=True)

    infer = load_model(args.model_path, load_4bit=args.load_4bit,
                       adapter=args.adapter, max_pixels=args.max_pixels,
                       blank_images=args.blank_images,
                       max_new_tokens=args.max_new_tokens)
    import time
    t0 = time.time()
    skipped = 0
    for i, it in enumerate(qa):
        inference_image_id = (
            cross_image_map[it["image_id"]] if cross_image_map is not None
            else it["image_id"]
        )
        it["image_intervention"] = (
            "cross_image" if cross_image_map is not None
            else "blank_image" if args.blank_images else "raw"
        )
        if cross_image_map is not None:
            it["cross_image_id"] = inference_image_id
        key = (it["image_id"], it["question"], reference_tag)
        if key in done:
            it["vlm_raw"] = done[key].get("vlm_raw", "")
            it["vlm_answer"] = done[key]["vlm_answer"]
            it["reference_tag"] = reference_tag
            it["run_config_sha256"] = run_config_sha256
            skipped += 1
            continue
        if it.get("image_ids"):
            frame_ids = apply_frame_mode(it["image_ids"], args.frame_mode, it)
            it["frame_mode"] = args.frame_mode
            it["inference_image_ids"] = frame_ids
            paths = [resolve_image_path(x, img_root) for x in frame_ids]
            ip = paths if all(p is not None for p in paths) else None
        else:
            ip = resolve_image_path(inference_image_id, img_root)
        if ip is None:
            it["vlm_answer"] = ""
            continue
        try:
            inference_question = compose_reference_prompt(it["question"], reference_text)
            if it.get("image_ids") and args.frame_mode == "single_v2":
                inference_question = rewrite_single_frame_prompt(inference_question)
                it["inference_question"] = inference_question
            raw = infer(ip, inference_question)
            it["vlm_raw"] = raw
            it["vlm_answer"] = extract_answer(raw)
            it["reference_tag"] = reference_tag
            it["run_config_sha256"] = run_config_sha256
            if reference_text:
                it["inference_question"] = inference_question
        except Exception as e:  # 单条失败不中断
            print(f"[bench] 推理失败 {it['image_id']}: {e}", flush=True)
            it["vlm_answer"] = ""
        if (i + 1) % 25 == 0 or i + 1 == len(qa):
            rate = max(i + 1 - skipped, 1) / (time.time() - t0)
            print(f"[bench] {i+1}/{len(qa)}  ({rate:.2f} it/s, "
                  f"剩余约 {(len(qa)-i-1)/rate/60:.0f} min)", flush=True)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(qa[:i+1], f, ensure_ascii=False)

    summary, records = evaluate(qa, metric_version=args.metric_version)
    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[bench] exact={summary['exact_accuracy']:.4f}  T-MRA={summary['tmra_overall']}")


if __name__ == "__main__":
    main()
