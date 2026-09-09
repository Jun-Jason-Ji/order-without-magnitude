#!/usr/bin/env python3
"""
LLM 答案抽取 (论文 "--parsed" 协议): 用本地文本模型 Qwen3-4B-Instruct-2507
从 vlm_raw 里抽取规范格式的最终答案, 再重跑 evaluate 出分。

只对"不合格式"的记录跑 LLM (合格式的答案 LLM 只能原样返回, 白耗 GPU 还引入风险);
用 --all 可强制全量过 LLM。

用法:
  python eval/llm_extract.py --pred results/bench_qwen3.5-4b_base/predictions.json \
      --output_folder results/bench_qwen3.5-4b_base_parsed --load_4bit
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import evaluate  # noqa: E402

# 抽取器底座的位置随机器而异; 环境变量优先 (换机器请设 QWEN3_4B_INSTRUCT)。
DEFAULT_MODEL = os.environ.get("QWEN3_4B_INSTRUCT") or (
    r"E:\models\hf\hub\models--Qwen--Qwen3-4B-Instruct-2507"
    r"\snapshots\cdbee75f17c01a7cc42f958dc650907174af0554")

FORMAT_SPEC = {
    "relational": "a single option letter: A, B, C, or D",
    "counting": "a single option letter (A/B/C/D) if the question lists options, otherwise a single integer",
    "distance": "a single number in meters (digits only, e.g. 3.25)",
    "localization": "three numbers as a tuple (x, y, z), e.g. (1.20, 0.35, 0.90)",
    # CourtDyn 动力学三组 (见 engine/dynamics_qa.py)。base 模型在这些题上会写
    # 一整段推导, 128 token 常在结论前被截断 —— 与 v3 的 parse_v3_base 同因,
    # 走同一条抽取协议。
    "speed": "a single number in meters per second (digits only, e.g. 2.4)",
    "path": "a single number in meters (digits only, e.g. 3.2)",
    "timing": "a single number in seconds (digits only, e.g. 1.8)",
}
NUMERIC_UNIT = {"speed": r"m/s|mps", "path": r"m", "timing": r"s|sec|seconds"}


def conforms(rec):
    """答案已是规范格式则无需 LLM。与统计脚本保持一致。"""
    a = (rec.get("vlm_answer") or "").strip()
    g = rec["group"]
    if g in ("relational", "counting"):
        gt = str(rec["answer"]).strip()
        if re.fullmatch(r"[A-Da-d]", a):
            return True
        if re.fullmatch(r"\d+", a) and re.fullmatch(r"\d+", gt):
            return True
        return False
    if g == "distance":
        return re.fullmatch(r"[-+]?\d*\.?\d+\s*m?", a) is not None
    # CourtDyn 三组: 与 distance 同形, 各自允许自己的单位后缀。distance 的
    # 正则原样保留 —— 冻结协议不因这次扩展改判任何一条历史记录。
    if g in NUMERIC_UNIT:
        return re.fullmatch(rf"[-+]?\d*\.?\d+\s*(?:{NUMERIC_UNIT[g]})?",
                            a) is not None
    if g == "localization":
        return len(re.findall(r"[-+]?\d*\.?\d+", a)) == 3
    return True


def load_llm(model_path, load_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 见 train_qlora.py: 蓝屏防线, 超过 VRAM 上限抛干净 OOM 而非驱动 UVM 崩溃
    _vram_cap = float(os.environ.get("VRAM_CAP_FRACTION", "0.93"))
    if _vram_cap > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_vram_cap, 0)

    print(f"[parse] 加载抽取模型: {model_path} (4bit={load_4bit})", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    kwargs = dict(dtype=torch.bfloat16, device_map="auto")
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()

    def parse(question, raw, group):
        prompt = (
            "Extract the final answer from a vision model's response to a question.\n"
            f"The required answer format is: {FORMAT_SPEC.get(group, 'the shortest canonical answer')}.\n\n"
            f"Question: {question}\n\n"
            f"Model response: {raw}\n\n"
            "Output ONLY the extracted answer in the required format, nothing else. "
            "If the response contains no answer, output NONE.")
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        return ans

    return parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--all", action="store_true", help="全量过 LLM (默认只处理不合格式的)")
    ap.add_argument("--metric_version", default="legacy_v1",
                    choices=["legacy_v1", "floor_v2"],
                    help="新 PR 比较显式使用 floor_v2；默认仅兼容历史队列")
    args = ap.parse_args()

    recs = json.load(open(args.pred, encoding="utf-8"))
    todo = [r for r in recs if r.get("vlm_raw") and (args.all or not conforms(r))]
    print(f"[parse] {len(todo)}/{len(recs)} 条需要 LLM 抽取", flush=True)

    if todo:
        parse = load_llm(args.model_path, load_4bit=args.load_4bit)
        changed = 0
        for i, r in enumerate(todo):
            old = r.get("vlm_answer", "")
            ans = parse(r["question"], r["vlm_raw"], r["group"])
            if ans and ans.upper() != "NONE" and ans != old:
                r["vlm_answer_rule"] = old
                r["vlm_answer"] = ans
                changed += 1
            if (i + 1) % 25 == 0:
                print(f"[parse] {i+1}/{len(todo)} (改写 {changed})", flush=True)
        print(f"[parse] 完成: 改写 {changed}/{len(todo)}", flush=True)

    summary, records = evaluate(recs, metric_version=args.metric_version)
    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[parse] ===== parsed summary =====")
    print(f"  exact acc : {summary['exact_accuracy']:.4f}")
    for grp, d in summary["per_group"].items():
        t = f" tmra={d['tmra']:.4f}" if d["tmra"] is not None else ""
        extra = f" acc@30cm={d['acc_at_30cm']:.4f}" if "acc_at_30cm" in d else ""
        print(f"    {grp:14s} n={d['n']:5d} acc={d['exact_accuracy']:.4f}{t}{extra}")


if __name__ == "__main__":
    main()
