#!/usr/bin/env python
"""Rebuild aggregate paper2 controls from independently authorized local inputs.

Uses code beside this script and data under --workspace. No weights, inference,
network calls, manuscript changes, images, or per-item exports are performed.
Public sanitized predictions alone are insufficient: the original question-key
alignment and licensed source inputs must first be reconstructed and verified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='New aggregate output directory')
    ap.add_argument('--paired', action='store_true', help='Also rebuild 48 paired/bootstrap cells')
    ap.add_argument('--bootstrap-reps', type=int, default=2000)
    ap.add_argument('--check-reference', type=Path, help='Compare native-control numerical fields')
    args = ap.parse_args()
    root, out = args.workspace.resolve(), args.out.resolve()
    if out.exists():
        ap.error('Refusing an existing output directory; frozen results are never overwritten.')
    if args.bootstrap_reps < 2:
        ap.error('--bootstrap-reps must be at least 2')
    import numpy as np
    from tools import audit_paper2_evidence as audit
    from tools import summarize_courtdyn_t28 as t28
    from tools import courtdyn_pixel_reader as pr
    from engine.court_homography import CourtPlane
    from types import SimpleNamespace

    # All code imports remain in this bundle; only explicit data roots are rebound.
    for module in (audit, t28, pr):
        if not Path(module.__file__).resolve().is_relative_to(CODE_ROOT):
            raise RuntimeError('A dependency was imported outside the exported code root')
    cd = root / 'results/courtdyn'
    audit.ROOT, audit.CD = root, cd
    t28.ROOT, t28.CD = str(root), str(cd)
    pr.ROOT, pr.CD = str(root), str(cd)
    t28.CourtPlane = SimpleNamespace(load=lambda seq: CourtPlane.load(seq, h_dir=str(cd/'homography')))
    cells, sources = [], []
    for seq in t28.TOPS:
        pool_path = cd / f'seq_{seq}/qa_dyn_v1.json'
        pool = json.loads(pool_path.read_text(encoding='utf8'))
        ctx = t28.seq_ctx(seq)
        full_dir = cd / f't33_{seq}_main_full_cdnative_parsed'
        static_dir = cd / f't34_{seq}_main_static4_cdnative_parsed'
        full, static = audit.numeric(t28.preds(str(full_dir))), audit.numeric(t28.preds(str(static_dir)))
        v1 = {t28.ikey(r): float(r['answer']) for r in pool if r['category'] in dict(t28.FAMS)}
        for _, family in t28.FAMS:
            keys = sorted(k for k in full.keys() & static.keys() & ctx['v3'].keys()
                          if ctx['fam_of'].get(k) == family)
            if not keys:
                raise RuntimeError(f'Missing aligned native predictions for {seq}/{family}')
            f, s, gt = ([full[k] for k in keys], [static[k] for k in keys], [ctx['v3'][k] for k in keys])
            rf, rs = audit.rho(f, gt), audit.rho(s, gt)
            vf, vs = audit.rho(f, [v1[k] for k in keys]), audit.rho(s, [v1[k] for k in keys])
            cells.append({'seq': seq, 'family': family,
                          'n_pool': sum(ctx['fam_of'].get(k) == family for k in v1),
                          'n_full_valid': sum(ctx['fam_of'].get(k) == family for k in full),
                          'n_static4_valid': sum(ctx['fam_of'].get(k) == family for k in static),
                          'n_common': len(keys),
                          'v3': {'full_rho': rf, 'static4_rho': rs, 'delta_full_minus_static4': rf-rs,
                                 'full_tmra': t28.tmra(list(zip(f, gt)), family),
                                 'static4_tmra': t28.tmra(list(zip(s, gt)), family),
                                 'test_constant_tmra': ctx['const'][family, 'v3']},
                          'historical_v1': {'full_rho': vf, 'static4_rho': vs,
                                            'delta_full_minus_static4': vf-vs},
                          'n_distinct': {'full': len(set(f)), 'static4': len(set(s))},
                          'event_overlap': seq == 'Q1_top_0-30'})
        for p in (pool_path, cd/f'homography/H_{seq}.json',
                  full_dir/'predictions.json', static_dir/'predictions.json'):
            sources.append({'path': p.relative_to(root).as_posix(),
                            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        mapping=json.loads((cd/'teamtrack_probe_hits.json').read_text(encoding='utf8'))
        family, split, _=next(row for row in mapping if row[2] == seq)
        source_dir=root/'data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot'/family/split/seq
        for p in (source_dir/'gt/gt.txt',source_dir/'seqinfo.ini'):
            sources.append({'path':p.relative_to(root).as_posix(),
                            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    comparisons = 0
    if args.check_reference:
        reference = json.loads(args.check_reference.read_text(encoding='utf8'))['cells']
        for cell in cells:
            expected = next(c for c in reference if (c['seq'], c['family']) == (cell['seq'], cell['family']))
            for key in ('n_pool','n_full_valid','n_static4_valid','n_common','event_overlap'):
                assert cell[key] == expected[key], (cell['seq'], cell['family'], key)
                comparisons += 1
            for group in ('v3', 'historical_v1', 'n_distinct'):
                for key, value in cell[group].items():
                    assert np.isclose(value, expected[group][key], rtol=0, atol=1e-10), (cell['seq'], cell['family'], group, key)
                    comparisons += 1
    result = {'schema': 'paper2-native-controls-rebuild-v1',
              'scope': 'Author-input CPU rebuild; aggregate outputs only; no model inference.',
              'source_model_training_items': 1400,
              'limitation': 'Q1 shares a training event. The 1120-item corrected pool has not retrained this checkpoint.',
              'reference_numeric_checks': comparisons, 'cells': cells, 'sources': sources}
    paired = audit.stats(args.bootstrap_reps) if args.paired else None
    out.mkdir(parents=True, exist_ok=False)
    (out/'native_controls.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    if paired is not None:
        (out/'paired_statistics.json').write_text(json.dumps(paired, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'native_cells': len(cells), 'reference_numeric_checks': comparisons,
                      'paired_cells': len(paired['cells']) if paired else 0,
                      'bootstrap_replicates': args.bootstrap_reps if paired else 0,
                      'out': str(out), 'scope': 'No fresh environment/GPU/public archive certification'}))


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit('Missing authorized local input: '+str(exc.filename)+
                         '. Reconstruct licensed QA/H/MOT inputs and aligned original-format predictions; '
                         'the public code-only candidate does not contain them.') from None
