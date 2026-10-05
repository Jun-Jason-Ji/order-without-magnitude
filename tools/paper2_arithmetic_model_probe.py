"""One-shot text-only arithmetic diagnostic; default preparation never queries GPU.

Run base/v1/v3 as separate, serial CLI processes with fresh output directories.
Only --run can load a model. No training, external inference, retry or scheduler.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, localcontext
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import paper2_local_model_smoke as smoke

SOURCE = ROOT/'results/soccer/arithmetic_dev_20260921_v2'
PREPARED = ROOT/'results/soccer/arithmetic_model_probe_dev100_20260921'
MODEL_ANCHOR = ROOT/'results/soccer/paper2_gpu_queue_20260921_v2/queue.json'
SOURCE_MANIFEST_SHA256 = 'cfc9e6ce9b2f4f44aa2293171546341219435a30c14eb408264ecd8c70dc6bed'
MODEL_ANCHOR_SHA256 = '033b878dfa5fe142c482da29159f9540edb5c206d3e948bd6a26876791d84053'
ROLE = 'synthetic_text_arithmetic_development_only'
COUNT = 100
SUFFIX = '\nGive ONLY the final answer in the required format. Do not explain.'
TASK_UNITS = {'length_conversion': ('m', 'cm'), 'speed_conversion': ('m/s', 'cm/s'),
              'distance_from_speed_time': ('m', 'cm'), 'speed_from_distance_time': ('m/s', 'cm/s'),
              'known_value_restatement': ('m', 'cm')}
CODE = ('tools/paper2_arithmetic_model_probe.py', 'tools/paper2_local_model_smoke.py',
        'tools/paper2_gpu_idle.py', 'tools/queue_guard.py', 'a100/eval_controls.py',
        'a100/config.py', 'a100/model_families.py', 'eval/run_bench.py', 'train/model_family.py')
PARAMETERS = {'precision': 'nf4', 'nf4_double_quant': True, 'max_pixels': 200704,
              'max_new_tokens': 128, 'do_sample': False, 'enable_thinking_requested': False,
              'image_count': 0, 'generation_suffix': SUFFIX,
              'template_fallback_policy': 'unchanged frozen eval.run_bench loader; branch is not exposed'}


class ProbeError(ValueError):
    pass


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def current_code_hashes():
    return {name: smoke.sha256_file(ROOT/name) for name in CODE}


def validate_source(source):
    """Hash private bytes but never parse their answers during preparation."""
    source = Path(source).resolve()
    if smoke.sha256_file(source/'manifest.json') != SOURCE_MANIFEST_SHA256:
        raise ProbeError('Only the existing frozen arithmetic development v2 dataset is allowed')
    manifest = smoke.read_json(source/'manifest.json')
    expected_files = {'public/questions.jsonl', 'private/references.jsonl', 'protocol.json', 'summary.json'}
    if set(manifest['files']) != expected_files:
        raise ProbeError('Unexpected source file contract')
    for name, digest in manifest['files'].items():
        if smoke.sha256_file(smoke.contained_file(source, name)) != digest:
            raise ProbeError('Frozen arithmetic input changed: '+name)
    return manifest


def select_questions(rows):
    """Selection metadata is public upstream, but never passed to inference."""
    if len(rows) != 500:
        raise ProbeError('Expected all 500 original development questions')
    groups, ids, prompts = defaultdict(list), set(), set()
    for row in rows:
        if set(row) != {'question_id', 'question', 'task', 'unit', 'canonical_answer_decade'}:
            raise ProbeError('Unknown public question fields')
        qid, question = row['question_id'], row['question']
        if not isinstance(qid, str) or not qid or qid in ids or not isinstance(question, str) or not question or question in prompts:
            raise ProbeError('Question IDs and prompts must be unique nonempty strings')
        task, unit, decade = row['task'], row['unit'], row['canonical_answer_decade']
        if task not in TASK_UNITS or unit not in TASK_UNITS[task] or type(decade) is not int or decade not in range(-2, 3):
            raise ProbeError('Unexpected task/unit/decade')
        ids.add(qid); prompts.add(question)
        groups[(task, decade, unit)].append(row)
    expected = {(t, d, u) for t, units in TASK_UNITS.items() for d in range(-2, 3) for u in units}
    if set(groups) != expected or any(len(group) != 10 for group in groups.values()):
        raise ProbeError('Expected 50 cells with 10 candidates each')
    selected = sorted([r for group in groups.values() for r in sorted(group, key=lambda x: x['question_id'])[:2]], key=lambda x: x['question_id'])
    public = [{'question_id': r['question_id'], 'question': r['question']} for r in selected]
    metadata = [{k: r[k] for k in ('question_id', 'task', 'unit', 'canonical_answer_decade')} for r in selected]
    return public, metadata


def prepare(source, output):
    manifest = validate_source(source)
    public, metadata = select_questions(read_jsonl(Path(source)/'public/questions.jsonl'))
    (output/'public').mkdir()
    write_jsonl(output/'public/questions.jsonl', public)
    smoke.write_new_json(output/'selection_metadata.json', metadata)
    protocol = {'schema': 'paper2-arithmetic-model-probe-v1', 'role': ROLE, 'source': str(Path(source).resolve()),
                'source_manifest_sha256': SOURCE_MANIFEST_SHA256, 'source_sha256': manifest['files'],
                'selection': 'first two lexicographic question IDs per task x canonical-answer decade x requested unit; final order by question ID',
                'selected_count': COUNT, 'selected_cells': 50, 'questions_per_cell': 2,
                'model_payload': 'question string only through infer([], question); no metadata or references',
                'parameters': PARAMETERS, 'code_sha256': current_code_hashes(),
                'model_hash_anchor': str(MODEL_ANCHOR), 'model_hash_anchor_sha256': MODEL_ANCHOR_SHA256,
                'private_answers_parsed': False, 'gpu_queried': False,
                'score_policy': {'strict_format': 'same numeric-only grammar/finite-float range as original smoke, parsed into Decimal',
                                 'exact_accuracy_denominator': 'all selected questions including parse failures',
                                 'mae': 'conditional on strict parse, 80-significant-digit Decimal arithmetic, requested units; never pool incompatible units',
                                 'no_training_or_external_inference': True}}
    smoke.write_new_json(output/'protocol.json', protocol)
    smoke.write_new_json(output/'manifest.json', {'role': ROLE, 'count': COUNT,
        'files': {name: smoke.sha256_file(output/name) for name in ('public/questions.jsonl', 'selection_metadata.json', 'protocol.json')}})
    return {'prepared': str(output), 'selected_questions': len(public), 'cells': 50,
            'public_sha256': smoke.sha256_file(output/'public/questions.jsonl')}


def validate_prepared(prepared):
    prepared = Path(prepared).resolve()
    manifest = smoke.read_json(prepared/'manifest.json')
    if manifest.get('role') != ROLE or manifest.get('count') != COUNT or set(manifest['files']) != {'public/questions.jsonl', 'selection_metadata.json', 'protocol.json'}:
        raise ProbeError('Invalid prepared selection contract')
    for name, digest in manifest['files'].items():
        if smoke.sha256_file(smoke.contained_file(prepared, name)) != digest:
            raise ProbeError('Prepared selection changed: '+name)
    protocol = smoke.read_json(prepared/'protocol.json')
    if (protocol['role'] != ROLE or protocol['selected_count'] != COUNT or protocol['parameters'] != PARAMETERS
            or protocol['code_sha256'] != current_code_hashes()
            or protocol['source_manifest_sha256'] != SOURCE_MANIFEST_SHA256
            or protocol['model_hash_anchor_sha256'] != MODEL_ANCHOR_SHA256
            or Path(protocol['model_hash_anchor']).resolve() != MODEL_ANCHOR.resolve()):
        raise ProbeError('Frozen protocol/source-code mismatch')
    source = Path(protocol['source'])
    original = validate_source(source)
    if original['files'] != protocol['source_sha256']:
        raise ProbeError('Original source bindings changed')
    expected_public, expected_metadata = select_questions(read_jsonl(source/'public/questions.jsonl'))
    public = read_jsonl(prepared/'public/questions.jsonl')
    metadata = smoke.read_json(prepared/'selection_metadata.json')
    if public != expected_public or metadata != expected_metadata:
        raise ProbeError('Prepared questions do not reproduce the frozen deterministic selection')
    return public, metadata, protocol


def verify_model_hashes(model_info, adapter_info):
    """Match all helper-declared base/adapter files to the prior frozen queue."""
    if smoke.sha256_file(MODEL_ANCHOR) != MODEL_ANCHOR_SHA256:
        raise ProbeError('Base/adapter hash anchor changed')
    anchor = smoke.read_json(MODEL_ANCHOR)
    if Path(anchor['model']).resolve() != Path(model_info['model_path']).resolve():
        raise ProbeError('Model differs from the previously frozen local snapshot')
    expected = {str(Path(path).resolve()): h for path, h in anchor['input_sha256'].items()}
    for info, key in ((model_info, 'model_path'), (adapter_info, 'adapter_path')):
        if info is None:
            continue
        for entry in info['files']:
            path = smoke.contained_file(Path(info[key]), entry['path'])
            got = smoke.sha256_file(path)
            if expected.get(str(path.resolve())) != got:
                raise ProbeError('Frozen model or adapter hash mismatch: '+str(path))
            entry['sha256'] = got
    return {'anchor_sha256': MODEL_ANCHOR_SHA256, 'base_files_checked': len(model_info['files']),
            'adapter_files_checked': len(adapter_info['files']) if adapter_info else 0}


def generate(public, infer, output, receipt):
    with (output/'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for item in public:
            if set(item) != {'question_id', 'question'}:
                raise ProbeError('Inference input must contain only an ID and question')
            start = time.perf_counter()
            raw = infer([], item['question'])
            if not isinstance(raw, str):
                raise ProbeError('Inference must return its unmodified raw string')
            row = {'question_id': item['question_id'], 'raw': raw, 'elapsed_s': time.perf_counter()-start,
                   'private_reference_parsed': False, 'image_count': 0}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            receipt['predictions_written'] += 1
            if receipt['predictions_written'] % 20 == 0:
                print(json.dumps({'event': 'prediction_progress', 'predictions_written': receipt['predictions_written'], 'expected': COUNT}), flush=True)
    if receipt['predictions_written'] != COUNT:
        raise ProbeError('Partial generation cannot be scored as a complete probe')
    seal = {'predictions_written': COUNT, 'predictions_sha256': smoke.sha256_file(output/'predictions.jsonl'),
            'sealed': smoke.utc_now(), 'private_reference_parsed': False}
    smoke.write_new_json(output/'prediction_seal.json', seal)
    receipt['predictions_sha256'] = seal['predictions_sha256']
    return seal


def strict_decimal(raw):
    if smoke.strict_number(raw) is None:
        return None
    try:
        value = Decimal(raw.strip())
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def group_summary(rows, allow_mae):
    parsed = [r for r in rows if r['prediction_decimal'] is not None]
    values = Counter(Decimal(r['prediction_decimal']) for r in parsed)
    refs = {Decimal(r['answer_decimal']) for r in rows}
    exact = sum(r['exact_correct'] for r in rows)
    n, count = len(rows), len(parsed)
    result = {'n': n, 'strict_parsed': count, 'parse_failures': n-count, 'parse_rate': count/n if n else None,
              'exact_correct': exact, 'exact_accuracy': exact/n if n else None,
              'conditional_exact_accuracy_parsed': exact/count if count else None,
              'zero_predictions': sum(v == 0 for v in (Decimal(r['prediction_decimal']) for r in parsed)),
              'negative_predictions': sum(v < 0 for v in (Decimal(r['prediction_decimal']) for r in parsed)),
              'distinct_parsed_predictions': len(values), 'reference_distinct_values': len(refs),
              'largest_prediction_repeat_count': max(values.values(), default=0),
              'all_predictions_constant': n >= 2 and count == n and len(values) == 1,
              'mae_decimal_conditional': None, 'mae_unit': None,
              'mae_not_pooled_reason': None if allow_mae else 'mixed dimensions or requested units'}
    if allow_mae and parsed:
        with localcontext() as ctx:
            ctx.prec = 80
            errors = [Decimal(r['absolute_error_decimal']) for r in parsed]
            result['mae_decimal_conditional'] = str(sum(errors, Decimal(0))/Decimal(count))
            result['max_absolute_error_decimal'] = str(max(errors))
            result['mae_unit'] = rows[0]['unit']
    result['constant_on_variable_reference'] = result['all_predictions_constant'] and len(refs) > 1
    return result


def score_after_seal(public, metadata, protocol, output, seal):
    pred_path = output/'predictions.jsonl'
    if smoke.sha256_file(pred_path) != seal['predictions_sha256'] or seal['predictions_written'] != COUNT:
        raise ProbeError('Prediction seal changed or is incomplete')
    predictions = read_jsonl(pred_path)
    if len(predictions) != COUNT or [r['question_id'] for r in predictions] != [r['question_id'] for r in public]:
        raise ProbeError('Prediction IDs/order are incomplete')
    # The first answer/reference JSON parse occurs only after all raw outputs are sealed.
    reference_path = Path(protocol['source'])/'private/references.jsonl'
    raw_refs = reference_path.read_bytes()
    import hashlib
    if hashlib.sha256(raw_refs).hexdigest() != protocol['source_sha256']['private/references.jsonl']:
        raise ProbeError('Private reference bytes changed before scoring')
    references = [json.loads(line) for line in raw_refs.decode('utf-8').splitlines() if line.strip()]
    indexed = {r['question_id']: r for r in references}
    if len(indexed) != len(references) or len(references) != 500:
        raise ProbeError('Private reference IDs are not the original unique 500')
    scored = []
    with localcontext() as ctx:
        ctx.prec = 80
        for pred, meta in zip(predictions, metadata):
            ref = indexed[meta['question_id']]
            if ref['task'] != meta['task'] or ref['unit'] != meta['unit']:
                raise ProbeError('Reference task/unit mismatch')
            answer = Decimal(ref['answer_decimal'])
            if not answer.is_finite():
                raise ProbeError('Reference must be a finite exact Decimal')
            value = strict_decimal(pred['raw'])
            scored.append({**meta, 'prediction_decimal': str(value) if value is not None else None,
                           'answer_decimal': str(answer), 'exact_correct': value == answer if value is not None else False,
                           'absolute_error_decimal': str(abs(value-answer)) if value is not None else None,
                           'strict_parse_failure': value is None})
    grouped = {}
    group_fields = {'per_task': ('task',), 'per_unit': ('unit',), 'per_decade': ('canonical_answer_decade',),
                    'per_task_unit': ('task', 'unit'), 'per_task_unit_decade': ('task', 'unit', 'canonical_answer_decade')}
    for name, fields in group_fields.items():
        cells = defaultdict(list)
        for row in scored:
            cells['__'.join(str(row[k]) for k in fields)].append(row)
        grouped[name] = {key: group_summary(rows, len({r['unit'] for r in rows}) == 1)
                         for key, rows in sorted(cells.items())}
    result = {'role': ROLE, 'complete': True, 'predictions_written': COUNT,
              'overall': group_summary(scored, False), **grouped,
              'constant_full_cells': sum(s['all_predictions_constant'] for s in grouped['per_task_unit_decade'].values()),
              'constant_on_variable_reference_cells': sum(s['constant_on_variable_reference'] for s in grouped['per_task_unit_decade'].values()),
              'predictions_sha256': seal['predictions_sha256'], 'private_references_sha256': protocol['source_sha256']['private/references.jsonl'],
              'all_raw_outputs_sealed_before_reference_parse': True,
              'decimal_mae_precision': 80, 'exact_accuracy_uses_decimal_equality_without_tolerance': True,
              'scope': '100 synthetic text questions for arithmetic/formatting diagnosis; no visual, real-test, training or generalization claim'}
    smoke.write_new_json(output/'scored_questions.json', scored)
    smoke.write_new_json(output/'summary.json', result)
    return result


def execute_run(prepared, adapter, output, receipt, *, gate_fn=None, lock_fn=None, loader=None, model_path=None):
    gate_fn = smoke.check_strict_resources if gate_fn is None else gate_fn
    lock_fn = smoke.gpu_locks if lock_fn is None else lock_fn
    loader = smoke.load_model if loader is None else loader
    model_path = smoke.MODEL if model_path is None else Path(model_path)
    public, metadata, protocol = validate_prepared(prepared)
    receipt['prepared_manifest_sha256'] = smoke.sha256_file(Path(prepared)/'manifest.json')
    receipt['model'] = smoke.validate_model_environment(model_path)
    receipt['adapter'] = smoke.validate_adapter(adapter, model_path)
    adapter = Path(receipt['adapter']['adapter_path']) if receipt['adapter'] else None
    with lock_fn() as locks:
        receipt['locks'] = locks
        receipt['resource_gate'] = gate_fn()
        if receipt['resource_gate'].get('passed') is not True:
            raise smoke.SmokeError('Strict idle gate blocked before model import')
        receipt['frozen_model_verification'] = verify_model_hashes(receipt['model'], receipt['adapter'])
        if receipt['adapter'] != smoke.validate_adapter(adapter, model_path):
            raise ProbeError('Adapter changed during base-file hashing')
        receipt['final_resource_gate'] = gate_fn()
        if receipt['final_resource_gate'].get('passed') is not True:
            raise smoke.SmokeError('Final strict idle gate blocked before model import')
        previous = {key: os.environ.get(key) for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
        try:
            for key in previous:
                os.environ[key] = '1'
            receipt['offline_environment'] = {key: os.environ[key] for key in previous}
            receipt['model_load_attempted'] = True
            smoke.write_new_json(output/'model_loading.json', receipt)
            start = time.perf_counter()
            infer, loader_info = loader(model_path, adapter=adapter) if adapter else loader(model_path)
            receipt['model_loaded'] = True
            receipt['load_elapsed_s'] = time.perf_counter()-start
            receipt['loader'] = loader_info
            if (loader_info.get('precision') != 'nf4' or loader_info.get('max_new_tokens') != 128
                    or loader_info.get('do_sample') is not False or loader_info.get('generation_suffix') != SUFFIX
                    or loader_info.get('loader') != 'eval.run_bench.load_model'
                    or loader_info.get('processor_policy', {}).get('max_pixels') != 200704):
                raise ProbeError('Unexpected effective loader/generation policy')
            smoke.write_new_json(output/'protocol.json', {**protocol, 'prepared': str(Path(prepared).resolve()),
                'adapter': str(adapter) if adapter else None, 'effective_loader': loader_info,
                'effective_generation_suffix': loader_info['generation_suffix'], 'model_input': 'infer([], question string only)'})
            write_jsonl(output/'prompts.jsonl', public)
            seal = generate(public, infer, output, receipt)
            result = score_after_seal(public, metadata, protocol, output, seal)
            receipt['exact_correct'] = result['overall']['exact_correct']
            if 'torch' in sys.modules:
                torch = sys.modules['torch']
                if torch.cuda.is_initialized():
                    receipt['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    receipt['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--prepared', type=Path, default=PREPARED)
    parser.add_argument('--adapter', type=smoke.adapter_argument, default=None)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        destination = args.output.resolve()
        if destination.is_relative_to(args.source.resolve()) or (args.run and destination.is_relative_to(args.prepared.resolve())):
            raise ProbeError('Output must stay outside frozen source/selection directories')
        output = smoke.new_output(destination)
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'output_rejected', 'error': str(exc)}), file=sys.stderr)
        return 2
    receipt = {'schema': 'paper2-arithmetic-model-probe-receipt-v1', 'role': ROLE, 'mode': 'run' if args.run else 'prepare',
               'started': smoke.utc_now(), 'model_load_attempted': False, 'model_loaded': False,
               'predictions_written': 0, 'strict_idle': True, 'python': sys.executable,
               'adapter_requested': str(args.adapter) if args.adapter else None, 'parameters': PARAMETERS,
               'code_sha256': current_code_hashes()}
    receipt['package_versions'] = {}
    for package in ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate'):
        try:
            receipt['package_versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            receipt['package_versions'][package] = None
    smoke.write_new_json(output/'started.json', receipt)
    start, code = time.perf_counter(), 0
    try:
        if args.run:
            execute_run(args.prepared, args.adapter, output, receipt)
        else:
            if args.adapter is not None:
                raise ProbeError('--adapter is only meaningful with --run')
            receipt['selection'] = prepare(args.source, output)
        receipt['status'] = 'passed'
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        receipt['status'] = 'blocked' if isinstance(exc, smoke.SmokeError) and not receipt['model_load_attempted'] else 'failed'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        code = 3 if receipt['status'] == 'blocked' else 1
    finally:
        path = output/'predictions.jsonl'
        if path.exists():
            receipt['predictions_written'] = len(path.read_text(encoding='utf-8').splitlines())
            receipt['predictions_sha256'] = smoke.sha256_file(path)
        receipt.update(finished=smoke.utc_now(), elapsed_s=time.perf_counter()-start, torch_imported='torch' in sys.modules)
        smoke.write_new_json(output/'receipt.json', receipt)
        # A run's artifact manifest additionally binds its final receipt and raw outputs.
        if args.run:
            smoke.write_new_json(output/'manifest.json', {'files': {p.name: smoke.sha256_file(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'status': receipt['status'], 'receipt': str(output/'receipt.json'),
                      'model_load_attempted': receipt['model_load_attempted'], 'model_loaded': receipt['model_loaded'],
                      'predictions_written': receipt['predictions_written']}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
