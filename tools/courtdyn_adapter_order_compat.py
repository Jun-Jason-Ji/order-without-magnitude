"""Explicit CPU-only compatibility for one frozen CourtDyn adapter-order gate.

The frozen analyzer and all inputs remain untouched. Only the two reviewed
target_modules arrays are sorted in temporary deep copies during the original
compare_models call. This tool cannot complete pipeline state or run a model.
Default invocation verifies evidence; --analyze writes a NEW analysis directory.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / 'paper2/audit/revision_execution_20260912'
EXECUTION = ROOT / 'results/courtdyn/revision_execution_20260912'
FROZEN = ROOT / 'tools/analyze_courtdyn_revision_results.py'
FROZEN_SHA = '37a2d98eefea7043fcb467aa5a223fced2c4474bc323e5f9bf5a2e561da1f6ea'
PLAN_SHA = 'b927a3b3a61a7c9b0eb8e6ffd37e8615c444b3d467394fca3a666a45bd7eb430'
FROZEN_PLAN = QA / 'frozen_execution_plan.json'
LABELS = ('eventholdout_v1_s42', 'eventholdout_v3_s42')
MODULES = ('down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj')
CONFIGS = {
    LABELS[0]: ('models/courtdyn-revision-v1-eventholdout-s42/adapter_config.json',
                'c2e8c2c4a5f1a83c6ff395d33f7055eaffc8279aef64deb1d5236afca87f0bd2'),
    LABELS[1]: ('models/courtdyn-revision-v3-eventholdout-s42/adapter_config.json',
                'c89eef577c0d5ce9db2f7f7d7084611f23b28d2ee34b01531234cbf46c7f27b4'),
}
PEFT_FILES = {
    'tuners/lora/config.py': ('bd3048b5d2ce364a98720ff48fed638c5819110ce53e3264a530cb5a9803dc31',
                            [915, 917], 'LoraConfig converts a target_modules list to a set.'),
    'config.py': ('9e1d9dd0979dc55112c4c39e0e9cd729bb444bd270697c620f52bf5d83b16ad7',
                  [149, 153], 'save_pretrained converts sets to lists without sorting.'),
    'tuners/tuners_utils.py': ('2f8f343737348014b71d775dda15a06dbe34df4190ed20afba8ad58ba471a5cf',
                             [1927, 1933], 'Target matching uses membership or any matching suffix.'),
}
SCHEMA = 'courtdyn-adapter-order-compatibility-v1'
RULE = 'unique-seven-target-modules-same-set-only'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def checked_hash(path, expected):
    require(Path(path).is_file() and not Path(path).is_symlink(), f'Missing/symlink evidence: {path}')
    require(sha(path) == expected, f'Compatibility evidence hash mismatch: {path}')


def canonical_pair(one, two):
    """Pure rule check, with no coercion of strings, duplicates, or other fields."""
    require(type(one) is dict and type(two) is dict, 'Adapter configurations must be dictionaries')
    arrays = []
    for config in (one, two):
        value = config.get('target_modules')
        require(type(value) is list and value and all(type(x) is str and x for x in value),
                'target_modules must be a nonempty string list, never a regex/string')
        require(len(value) == len(set(value)), 'Duplicate target_modules are not accepted')
        require(tuple(sorted(value)) == MODULES, 'target_modules differ from the seven reviewed members')
        arrays.append(value)
    require(set(arrays[0]) == set(arrays[1]), 'target_modules sets differ')
    require(canonical({k: v for k, v in one.items() if k != 'target_modules'}) ==
            canonical({k: v for k, v in two.items() if k != 'target_modules'}),
            'Adapter configurations differ outside target_modules ordering')
    copies = deepcopy(one), deepcopy(two)
    for config in copies:
        config['target_modules'] = list(MODULES)
    return copies


def incident_evidence():
    """Read and hash the exact reviewed incident; importing PEFT is unnecessary."""
    checked_hash(FROZEN, FROZEN_SHA)
    checked_hash(FROZEN_PLAN, PLAN_SHA)
    plan = read(FROZEN_PLAN)
    for relative, expected in plan['source_sha256'].items():
        checked_hash(ROOT / relative, expected)
    training = {s['label']: s for s in plan['stages'] if s['kind'] == 'training'}
    require(set(training) == set(LABELS), 'Compatibility is limited to the two planned training labels')
    configs, records = {}, {}
    for label, (relative, expected) in CONFIGS.items():
        path = ROOT / relative
        require(path.resolve() == (Path(training[label]['output']) / 'adapter_config.json').resolve(),
                'Reviewed adapter path differs from frozen training plan')
        checked_hash(path, expected)
        configs[label] = read(path)
        records[label] = dict(path=relative, sha256=expected,
                              original_target_modules=configs[label]['target_modules'])
    canonical_pair(*(configs[label] for label in LABELS))
    peft_root = Path(plan['python']).parent / 'Lib/site-packages/peft'
    peft = {}
    for relative, (expected, lines, observation) in PEFT_FILES.items():
        checked_hash(peft_root / relative, expected)
        peft[relative] = dict(sha256=expected, lines=lines, observation=observation)
    return dict(rule=RULE, labels=list(LABELS), execution_plan_sha256=PLAN_SHA,
                frozen_analyzer_sha256=FROZEN_SHA, compatibility_tool_sha256=sha(__file__),
                adapter_configs=records, canonical_target_modules=list(MODULES),
                peft_version='0.20.0', peft_source_sha256=peft,
                scope='Temporary configuration comparison only; raw inputs, recorded configuration order, '
                      'source hashes, statistics, missing-data decisions and all other gates are unchanged.')


def load_frozen_analyzer():
    checked_hash(FROZEN, FROZEN_SHA)
    spec = importlib.util.spec_from_file_location('courtdyn_frozen_with_explicit_order_compat', FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compatible_class(module, evidence):
    """Keep the original method and its exceptions; no new scoring implementation."""
    class CompatibleAuditor(module.Auditor):
        def __init__(self, inputs, plan=None):
            require(plan is not None, 'Adapter compatibility requires the frozen execution plan')
            checked_hash(plan, PLAN_SHA)
            super().__init__(inputs, plan)
            self.compatibility_events = []
            module.compatibility_auditors.append(self)

        def compare_models(self, left, right):
            if (left, right) != LABELS:
                return super().compare_models(left, right)
            require(all(label in self.training_records for label in LABELS),
                    'Both verified training records are required for compatibility')
            records = [self.training_records[label] for label in LABELS]
            originals = [record['actual_adapter_config'] for record in records]
            for label, config in zip(LABELS, originals):
                relative, expected = CONFIGS[label]
                path = ROOT / relative
                checked_hash(path, expected)
                require(self.hashes.get(str(path.resolve())) == expected,
                        'Original analyzer has not verified the reviewed adapter configuration')
                require(canonical(config) == canonical(read(path)),
                        'In-memory adapter configuration differs from the raw reviewed file')
            copies = canonical_pair(*originals)
            event = dict(labels=list(LABELS), rule=RULE,
                         original_target_modules=[deepcopy(x['target_modules']) for x in originals],
                         canonical_target_modules=list(MODULES), original_method_returned=False,
                         original_objects_restored=False)
            try:
                for record, config in zip(records, copies):
                    record['actual_adapter_config'] = config
                result = super().compare_models(left, right)
                event['original_method_returned'] = True
                return result
            finally:
                for record, config in zip(records, originals):
                    record['actual_adapter_config'] = config
                event['original_objects_restored'] = all(
                    record['actual_adapter_config'] is config for record, config in zip(records, originals))
                self.compatibility_events.append(event)

    CompatibleAuditor.__name__ = 'CompatibleAuditor'
    return CompatibleAuditor


def load_compatible_analyzer():
    """Return a new isolated frozen module with the explicit compatible Auditor.

    Consumers must also verify compatibility.json, not merely import this class.
    No previously imported module, source file, or model configuration is patched.
    """
    evidence = incident_evidence()
    module = load_frozen_analyzer()
    module.compatibility_evidence = evidence
    module.compatibility_auditors = []
    module.Auditor = _compatible_class(module, evidence)
    return module


def original_failure(path):
    report = read(path)
    require(report.get('schema') == 'courtdyn-revision-results-analysis-v1' and
            report.get('status') == 'FAIL' and report.get('analysis_script_sha256') == FROZEN_SHA and
            report.get('errors') == ['ValueError: P2 actual adapter configurations differ'],
            'Original failure must document precisely the frozen adapter-order gate')
    return dict(path=str(Path(path).resolve()), sha256=sha(path), status='FAIL', errors=report['errors'])


def verify_receipt(analysis_path):
    """Verify the sidecar provenance. Full CPU score replay remains mandatory."""
    analysis_path = Path(analysis_path)
    receipt = read(analysis_path.parent / 'compatibility.json')
    require(receipt.get('schema') == SCHEMA, 'Unknown compatibility receipt schema')
    require(canonical(receipt.get('evidence')) == canonical(incident_evidence()),
            'Compatibility receipt is stale or has unreviewed evidence')
    checked_hash(analysis_path, receipt.get('analysis_sha256'))
    checked_hash(analysis_path.parent / 'summary.md', receipt.get('summary_sha256'))
    failure = receipt.get('original_failure', {})
    require(canonical(original_failure(failure.get('path', ''))) == canonical(failure),
            'Original failure record changed')
    analysis = read(analysis_path)
    require(receipt.get('status') == analysis.get('status') and
            analysis.get('analysis_script_sha256') == FROZEN_SHA,
            'Compatibility receipt and frozen analysis identity/status differ')
    require(receipt.get('execution_plan_sha256') == PLAN_SHA and receipt.get('state_mutated') is False,
            'Compatibility receipt cannot authorize a pipeline state change')
    events = receipt.get('comparison_events')
    require(type(events) is list and len(events) == 1, 'Expected exactly one reviewed model comparison')
    expected = dict(labels=list(LABELS), rule=RULE,
                    original_target_modules=[receipt['evidence']['adapter_configs'][x]['original_target_modules']
                                             for x in LABELS],
                    canonical_target_modules=list(MODULES), original_method_returned=True,
                    original_objects_restored=True)
    if analysis.get('status') == 'PASS':
        require(canonical(events[0]) == canonical(expected), 'PASS lacks a restored, successful original comparison')
        require(not analysis.get('errors') and not analysis.get('missing'), 'PASS analysis is inconsistent')
    return receipt


def analyze(args):
    checked_hash(args.plan, PLAN_SHA)
    plan = read(args.plan)
    require(Path(sys.executable).resolve() == Path(plan['python']).resolve(),
            'Use the frozen research Python for exact CPU analysis')
    require(args.plan.resolve() == (EXECUTION / 'plan.json').resolve(), 'Use the actual execution plan')
    require(args.inputs.resolve() == (ROOT / 'results/courtdyn/revision_controls_20260912').resolve(),
            'Use the actual frozen prepared inputs')
    state = read(EXECUTION / 'state.json')
    expected_ids = {s['id'] for s in plan['stages']}
    require(state.get('status') == 'FAILED' and state.get('plan_sha256') == PLAN_SHA and len(expected_ids) == 11 and
            set(state.get('stages', {})) == expected_ids and not state.get('active_child_pid'),
            'Cannot reanalyze before the original pipeline terminates FAILED and the child exits')
    require(all(s.get('status') == 'COMPLETE' and s.get('returncode') == 0
                for s in state['stages'].values()), 'All 11 actual stages must be complete')
    require(args.output_dir.resolve().is_relative_to(EXECUTION.resolve()),
            'Keep the separate compatible analysis inside the actual execution directory')
    require(not args.output_dir.exists(), 'Analysis output must be new; never overwrite earlier FAIL/PASS')
    failure = original_failure(args.original_failure)
    failed = read(args.original_failure)
    require(not failed.get('missing') and len(failed.get('runs', [])) == 9,
            'Use the final complete nine-run failure, not an interim analysis')
    require(Path(failure['path']).parent.parent == EXECUTION.resolve(),
            'Original failure must be in the actual pipeline execution directory')
    state_sha = sha(EXECUTION / 'state.json')
    module = load_compatible_analyzer()
    argv = [str(FROZEN), '--inputs', str(args.inputs), '--plan', str(args.plan),
            '--output-dir', str(args.output_dir), '--compare-models', *LABELS]
    for stage in plan['stages']:
        if stage['kind'] == 'evaluation':
            argv.extend(['--run', f'{stage["label"]}={stage["output"]}'])
    saved_argv = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = saved_argv
    checked_hash(args.original_failure, failure['sha256'])
    checked_hash(EXECUTION / 'state.json', state_sha)
    require(canonical(incident_evidence()) == canonical(module.compatibility_evidence),
            'Compatibility source evidence changed during analysis')
    report = read(args.output_dir / 'analysis.json')
    receipt = dict(schema=SCHEMA, generated_utc=datetime.now(timezone.utc).isoformat(),
                   status=report['status'], execution_plan_sha256=PLAN_SHA,
                   evidence=module.compatibility_evidence, original_failure=failure,
                   analysis_sha256=sha(args.output_dir / 'analysis.json'),
                   summary_sha256=sha(args.output_dir / 'summary.md'),
                   state_sha256_at_analysis=state_sha, state_mutated=False,
                   comparison_events=[event for a in module.compatibility_auditors
                                      for event in a.compatibility_events])
    with (args.output_dir / 'compatibility.json').open('x', encoding='utf8') as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analyze', action='store_true', help='Run only the frozen CPU analysis into a new directory')
    ap.add_argument('--plan', type=Path, default=EXECUTION / 'plan.json')
    ap.add_argument('--inputs', type=Path, default=ROOT / 'results/courtdyn/revision_controls_20260912')
    ap.add_argument('--original-failure', type=Path)
    ap.add_argument('--output-dir', type=Path)
    args = ap.parse_args()
    if args.analyze and (not args.original_failure or not args.output_dir):
        ap.error('--analyze requires --original-failure and a new --output-dir')
    if not args.analyze and (args.output_dir or args.original_failure):
        ap.error('Output/failure arguments require explicit --analyze')
    try:
        if args.analyze:
            return analyze(args)
        print(json.dumps(dict(status='EVIDENCE_VERIFIED_NOT_ANALYZED', evidence=incident_evidence()), indent=2))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
