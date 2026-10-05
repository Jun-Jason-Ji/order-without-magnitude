"""Create 500 stratified text-only development probes without running a model."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random


CONFIG = {
    'schema': 'paper2-arithmetic-development-v2', 'seed': 20260921,
    'role': 'synthetic_development_only', 'questions': 500,
    'tasks': ['length_conversion', 'speed_conversion', 'distance_from_speed_time',
              'speed_from_distance_time', 'known_value_restatement'],
    'canonical_answer_decades': [-2, -1, 0, 1, 2],
    'cases_per_task_decade': 20, 'target_units_balanced': ['m', 'cm'],
    'scope': 'text-only arithmetic/formatting; no visual or cross-game inference',
    'existing_12_probe_questions_modified': False,
    'unique_prompt_required': True,
}


def number(value):
    return format(value, 'f').rstrip('0').rstrip('.') if '.' in format(value, 'f') else format(value, 'f')


def generate():
    rng = random.Random(CONFIG['seed'])
    questions, references = [], []
    for task in CONFIG['tasks']:
        for exponent in CONFIG['canonical_answer_decades']:
            mantissas = rng.sample(range(100, 1000), CONFIG['cases_per_task_decade'])
            for index, mantissa in enumerate(mantissas):
                x = Decimal(mantissa) / Decimal(100) * Decimal(10) ** exponent
                seconds = Decimal(rng.choice([2, 4, 5]))
                is_speed = task in ('speed_conversion', 'speed_from_distance_time')
                target = 'm' if index % 2 == 0 else 'cm'
                source = 'cm' if target == 'm' else 'm'
                factor = Decimal(1 if target == 'm' else 100)
                source_factor = Decimal(1 if source == 'm' else 100)
                unit = target + ('/s' if is_speed else '')
                if task == 'length_conversion':
                    question = f'Convert {number(x * source_factor)} {source} to {target}.'
                elif task == 'speed_conversion':
                    question = f'Convert {number(x * source_factor)} {source}/s to {target}/s.'
                elif task == 'distance_from_speed_time':
                    question = (f'An object travels at a constant {number(x / seconds)} m/s for '
                                f'{number(seconds)} seconds. What distance does it travel in {target}?')
                elif task == 'speed_from_distance_time':
                    question = (f'An object travels a total path length of {number(x * seconds)} m in '
                                f'{number(seconds)} seconds. What is its average speed in {target}/s?')
                else:
                    question = f'The measured distance is exactly {number(x * factor)} {target}. Restate its numeric value in {target}.'
                qid = f'arith_dev_{len(questions):04d}'
                questions.append({'question_id': qid, 'task': task, 'unit': unit,
                                  'canonical_answer_decade': exponent,
                                  'question': question + ' Output only the exact number, without units or explanation.'})
                references.append({'question_id': qid, 'answer_decimal': number(x * factor),
                                   'canonical_answer_decimal': number(x),
                                   'canonical_unit': 'm/s' if is_speed else 'm',
                                   'unit': unit, 'task': task, 'role': CONFIG['role']})
    assert len(questions) == 500 and len({q['question_id'] for q in questions}) == 500
    assert len({q['question'] for q in questions}) == 500
    assert Counter(q['task'] for q in questions) == {task: 100 for task in CONFIG['tasks']}
    assert all('answer' not in q for q in questions)
    assert all(Decimal(row['answer_decimal']) > 0 for row in references)
    return questions, references


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing probe dataset')
    questions, references = generate()
    (args.output / 'public').mkdir(parents=True)
    (args.output / 'private').mkdir()
    for relative, rows in [('public/questions.jsonl', questions), ('private/references.jsonl', references)]:
        with (args.output / relative).open('x', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
    (args.output / 'protocol.json').write_text(json.dumps(CONFIG, indent=2) + '\n', encoding='utf-8')
    summary = {'role': CONFIG['role'], 'generated_questions': len(questions),
               'task_counts': dict(Counter(q['task'] for q in questions)),
               'target_unit_counts': dict(Counter(q['unit'] for q in questions)),
               'model_responses': 0, 'model_inference_performed': False,
               'development_set_not_final_evaluation': True,
               'limitations': 'Generated arithmetic labels are exact Decimal calculations; no evidence of model arithmetic ability yet.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    manifest = {'code_sha256': sha256(Path(__file__)),
                'files': {p.relative_to(args.output).as_posix(): sha256(p)
                          for p in sorted(args.output.rglob('*')) if p.is_file()}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
