#!/usr/bin/env python
"""Check an exported CourtDyn bundle from an unrelated, clean working directory.

No source data, real QA, model weights or network access are used. This verifies
entry points, imports, a synthetic trajectory build and evaluator behavior;
it is deliberately not a claim to reproduce any reported model result.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    checks = []
    with tempfile.TemporaryDirectory(prefix="courtdyn-smoke-") as scratch:
        work = Path(scratch)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)

        def run(name, command):
            result = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", *command],
                                    cwd=work, env=env, text=True, encoding="utf-8",
                                    capture_output=True, timeout=90)
            checks.append({"name": name, "returncode": result.returncode,
                           "output": (result.stdout + result.stderr)[-1600:]})
            if result.returncode:
                raise RuntimeError(f"{name}: {result.stderr}")

        for relative in ["tools/build_courtdyn_qa.py", "tools/build_courtdyn_zoom2.py",
                         "tools/build_courtdyn_declut.py", "tools/build_courtdyn_ruler.py",
                         "tools/build_courtdyn_noprior.py", "tools/build_courtdyn_hm3.py",
                         "tools/build_courtdyn_native_sft.py",
                         "tools/calibrate_court_homography.py", "eval/evaluate.py",
                         "eval/run_bench.py", "eval/llm_extract.py",
                         "eval/paired_relational_scorer.py", "train/train_qlora.py"]:
            run(relative + " --help", [str(bundle / relative), "--help"])
        # New candidates carry these entry points; historical audit bundles do not.
        for relative in ["tools/audit_paper2_evidence.py", "tools/rebuild_paper2_statistics.py",
                         "tools/prepare_courtdyn_revision_experiments.py",
                         "tools/run_courtdyn_revision_controls.py",
                         "paper2/audit/quick_quality_20260912/statistics/analyze_existing_controls.py"]:
            if (bundle / relative).is_file():
                run(relative + " --help", [str(bundle / relative), "--help"])
        imports = (
            "import pathlib,sys; r=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0,str(r)); "
            "import engine.court_homography as h; import tools.summarize_courtdyn_t28 as t; "
            "import tools.rescore_courtdyn_homography as s; "
            "assert pathlib.Path(h.__file__).is_relative_to(r); "
            "assert pathlib.Path(t.__file__).is_relative_to(r); "
            "assert pathlib.Path(s.__file__).is_relative_to(r); print('bundle imports resolved')"
        )
        run("core analysis imports", ["-c", imports, str(bundle)])
        seq = work / "synthetic"
        (seq / "gt").mkdir(parents=True)
        (seq / "seqinfo.ini").write_text(
            "[Sequence]\nname=synthetic\nframerate=30\nimwidth=1920\nimheight=1080\nseqlength=180\n",
            encoding="utf-8")
        rows = []
        for frame in range(1, 181):
            rows.append(f"{frame},1,{100 + 2 * frame},300,40,120,1,-1,-1,-1")
            rows.append(f"{frame},2,{400 + frame},400,40,120,1,-1,-1,-1")
            rows.append(f"{frame},3,1200,450,12,12,1,-1,-1,-1")
        (seq / "gt" / "gt.txt").write_text("\n".join(rows), encoding="utf-8")
        run("synthetic trajectory generation", [str(bundle / "tools/build_courtdyn_qa.py"),
             "--seq_dir", str(seq), "--max_per_family", "6", "--dry_run"])
        fixture = [
            {"category": "dynamics_speed_player", "answer": "2.0", "vlm_answer": "2.0"},
            {"category": "dynamics_speed_player", "answer": "2.0"},
        ]
        pred = work / "synthetic_predictions.json"
        pred.write_text(json.dumps(fixture), encoding="utf-8")
        run("synthetic evaluator", [str(bundle / "eval/evaluate.py"), "--file_path", str(pred),
             "--output_folder", str(work / "scored"), "--metric_version", "floor_v2"])
        summary = json.loads((work / "scored/summary.json").read_text(encoding="utf-8"))
        assert summary["n_total"] == 2 and summary["prediction_coverage"] == 0.5, summary
        assert summary["tmra_overall"] == 0.5, summary
        checks.append({"name": "missing-prediction penalization", "summary": summary})
    report = {"status": "PASS", "n_checks": len(checks), "python": sys.version,
              "scope": "clean cwd; isolated interpreter paths; installed CPU packages reused; synthetic data only",
              "not_validated": ["fresh dependency installation", "real video calibration", "GPU inference/training",
                                "all withheld pool hashes", "online v1.0.0 archive"], "checks": checks}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
