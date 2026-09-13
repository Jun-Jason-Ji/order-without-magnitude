"""CPU-only, read-only analysis of newly run CourtDyn revision controls.

Never loads a model or substitutes legacy predictions. Incomplete runs produce
PARTIAL reports; hash/cohort/config inconsistencies produce FAIL reports.
Outputs contain aggregates, not trajectory IDs or per-item predictions.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARMS = [f'{u}_{h}' for u in ('m', 'cm', 'px') for h in ('height', 'noheight')]
FAMILIES = ('dynamics_speed_player', 'dynamics_path_player')
NUMBER = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
PRIOR = 'Assume a typical player on this court is 1.93 m tall. '
CONFIG_KEYS = ('protocol', 'input_manifest_sha256', 'model_config_sha256',
               'max_pixels', 'max_new_tokens', 'load_4bit', 'decoding', 'strict_parse',
               'code_sha256', 'runtime', 'generation_suffix', 'vram_cap_fraction')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def parse(raw):
    require(isinstance(raw, str), 'raw_answer must be a string')
    text = raw.strip()
    return float(text) if NUMBER.fullmatch(text) and math.isfinite(float(text)) else None


def score(prediction, target, tolerance, floor):
    """Exact protocol strict inequality; 0--100 scale, invalid parse = zero."""
    if prediction is None:
        return 0.0
    relative = (abs(prediction-target)-tolerance)/max(abs(target), floor)
    return 10.0*sum(relative < 1-(0.5+0.05*i) for i in range(10))


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*q
    lower = math.floor(index)
    return values[lower] + (values[math.ceil(index)]-values[lower])*(index-lower)


def distribution(values):
    require(all(finite(v) for v in values), 'Nonfinite value in aggregate')
    return dict(n=len(values), median=quantile(values, .5),
                q1_q3=[quantile(values, .25), quantile(values, .75)] if values else None,
                min_max=[min(values), max(values)] if values else None)


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0]*len(values)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start+end-1)/2+1
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def spearman(x, y):
    require(len(x) == len(y), 'Spearman cohorts differ')
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    a, b = ranks(x), ranks(y)
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((u-ma)*(v-mb) for u, v in zip(a, b))
    denominator = math.sqrt(sum((u-ma)**2 for u in a)*sum((v-mb)**2 for v in b))
    return max(-1.0, min(1.0, numerator/denominator))


def item_key(item):
    meta = item.get('meta', item)
    return item['category'], meta['track'], tuple(meta['window'])


def trajectory_key(item):
    return item_key(item)[1:]


def exact_meta(actual, expected):
    if finite(expected):
        return finite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def constant_baselines(items):
    """Label-only diagnostics on the full prepared cohort, including oracle."""
    targets = [r['meta']['reference_value'] for r in items]
    if not targets:
        return None
    def evaluate(value):
        return sum(score(value, r['meta']['reference_value'], r['meta']['score_tolerance'],
                         r['meta']['score_floor']) for r in items)/len(items)
    # Each threshold defines an open interval of constants that earn 10 points.
    # A maximum exists in an interval between adjacent endpoints. Sweep avoids
    # an arbitrary grid and handles per-item floors/tolerances exactly.
    events = defaultdict(int)
    for item in items:
        m = item['meta']
        y, t, epsilon = (m[k] for k in ('reference_value', 'score_tolerance', 'score_floor'))
        for i in range(10):
            radius = t+(1-(.5+.05*i))*max(abs(y), epsilon)
            events[y-radius] += 1
            events[y+radius] -= 1
    points = sorted(events)
    active, best, candidate = 0, -1, statistics.median(targets)
    for left, right in zip(points, points[1:]):
        active += events[left]
        midpoint = left+(right-left)/2
        if left < midpoint < right and active > best:
            best, candidate = active, midpoint
    constants = {'mean':statistics.mean(targets), 'median':statistics.median(targets),
                 'test_label_oracle':candidate}
    return dict(n=len(items), scope='Full prepared family; test-label diagnostics, not deployable fitted baselines.',
                values={name:dict(prediction=value, tmra=evaluate(value))
                        for name, value in constants.items()})


def family_statistics(items, observations):
    valid = [(r, observations[item_key(r)]['prediction']) for r in items
             if item_key(r) in observations and observations[item_key(r)]['prediction'] is not None]
    observed = [r for r in items if item_key(r) in observations]
    scores = [score(pred, r['meta']['reference_value'], r['meta']['score_tolerance'],
                    r['meta']['score_floor']) for r, pred in valid]
    total = sum(scores)
    return dict(n_expected=len(items), n_observed=len(observed), n_parsed=len(valid),
                n_invalid_parse=len(observed)-len(valid), n_missing=len(items)-len(observed),
                coverage=len(observed)/len(items) if items else None,
                parse_rate_observed=len(valid)/len(observed) if observed else None,
                tmra_all_observed=total/len(observed) if observed else None,
                tmra_all_items=total/len(items) if len(observed) == len(items) and items else None,
                tmra_parsed=total/len(valid) if valid else None,
                tmra_missing_as_zero_lower_bound=total/len(items) if items else None,
                spearman=spearman([p for _, p in valid], [r['meta']['reference_value'] for r, _ in valid]),
                prediction=distribution([p for _, p in valid]),
                reference=distribution([r['meta']['reference_value'] for r in items]),
                constant_diagnostics=constant_baselines(items))


def paired_statistics(a, b, family, comparison, allow_image_change=False):
    """Numerator b / denominator a, always on explicit common keys."""
    ai = {k:r for k,r in a['items'].items() if k[0] == family}
    bi = {k:r for k,r in b['items'].items() if k[0] == family}
    require(ai.keys() == bi.keys(), f'{comparison}: prepared family cohorts differ')
    for key in ai:
        require(allow_image_change or ai[key]['image_ids'] == bi[key]['image_ids'],
                f'{comparison}: image identities differ')
    observed = sorted(ai.keys() & a['observations'].keys() & b['observations'].keys(), key=repr)
    parsed = [k for k in observed if a['observations'][k]['prediction'] is not None
              and b['observations'][k]['prediction'] is not None]
    positive = [k for k in parsed if a['observations'][k]['prediction'] > 0]
    reference_positive = [k for k in ai if ai[k]['meta']['reference_value'] > 0]
    ratios = [b['observations'][k]['prediction']/a['observations'][k]['prediction'] for k in positive]
    expected = [bi[k]['meta']['reference_value']/ai[k]['meta']['reference_value'] for k in positive
                if ai[k]['meta']['reference_value'] > 0]
    normalized = [b['observations'][k]['prediction']/a['observations'][k]['prediction']/
                  (bi[k]['meta']['reference_value']/ai[k]['meta']['reference_value']) for k in positive
                  if ai[k]['meta']['reference_value'] > 0 and bi[k]['meta']['reference_value'] > 0]
    def paired_score(cell, keys):
        return sum(score(cell['observations'][k]['prediction'],
                         cell['items'][k]['meta']['reference_value'],
                         cell['items'][k]['meta']['score_tolerance'],
                         cell['items'][k]['meta']['score_floor']) for k in keys)/len(keys) if keys else None
    before, after = paired_score(a, observed), paired_score(b, observed)
    before_parsed, after_parsed = paired_score(a, parsed), paired_score(b, parsed)
    rho_a = spearman([a['observations'][k]['prediction'] for k in parsed],
                     [ai[k]['meta']['reference_value'] for k in parsed])
    rho_b = spearman([b['observations'][k]['prediction'] for k in parsed],
                     [bi[k]['meta']['reference_value'] for k in parsed])
    return dict(comparison=comparison, family=family, numerator=b['arm'], denominator=a['arm'],
                n_expected=len(ai), n_common_observed=len(observed), n_common_parsed=len(parsed),
                n_positive_prediction_denominator=len(positive),
                n_nonpositive_prediction_denominator=len(parsed)-len(positive),
                n_negative_numerators=sum(b['observations'][k]['prediction'] < 0 for k in positive),
                ratio=distribution(ratios), expected_reference_ratio_on_ratio_cohort=distribution(expected),
                ratio_divided_by_itemwise_reference_ratio=distribution(normalized),
                expected_reference_ratio_full_cohort=distribution([
                    bi[k]['meta']['reference_value']/ai[k]['meta']['reference_value'] for k in reference_positive]),
                paired_all_observed_tmra=[before, after],
                paired_all_observed_delta_tmra=None if before is None else after-before,
                paired_common_parsed_tmra=[before_parsed, after_parsed],
                paired_common_parsed_delta_tmra=None if before_parsed is None else after_parsed-before_parsed,
                paired_common_parsed_spearman=[rho_a, rho_b],
                delta_spearman=None if rho_a is None or rho_b is None else rho_b-rho_a)


def speed_path_consistency(cell):
    by_family = {f:{trajectory_key(r):r for r in cell['items'].values() if r['category'] == f}
                 for f in FAMILIES}
    speed, path = (by_family[f] for f in FAMILIES)
    common = speed.keys() & path.keys()
    observed, parsed, positive, ratios, deviations = 0, 0, 0, [], []
    duration_values, correct, nominal_correct, changed, reference_error = [], 0, 0, 0, 0.0
    for key in common:
        s, d = speed[key], path[key]
        require(s['image_ids'] == d['image_ids'], 'Speed/path images differ on paired trajectory')
        require(s['meta']['fps'] == d['meta']['fps'], 'Speed/path FPS differs')
        dt = (key[1][1]-key[1][0])/s['meta']['fps']
        require(dt > 0, 'Nonpositive duration')
        duration_values.append(dt)
        reference_error = max(reference_error, abs(d['meta']['reference_value']-dt*s['meta']['reference_value']))
        sk, dk = item_key(s), item_key(d)
        if sk not in cell['observations'] or dk not in cell['observations']:
            continue
        observed += 1
        sp, dp = (cell['observations'][k]['prediction'] for k in (sk,dk))
        if sp is None or dp is None:
            continue
        parsed += 1
        error = abs(dp-dt*sp)
        c = error <= .05*(1+dt)+1e-12
        nominal = abs(dp-2.0*sp) <= .15+1e-12
        correct += c
        nominal_correct += nominal
        changed += c != nominal
        deviations.append(error)
        if sp > 0:
            positive += 1
            ratios.append(dp/(dt*sp))
    require(reference_error < 1e-7, 'Exact references fail path = duration * speed')
    return dict(n_expected_pairs=len(common), n_observed_pairs=observed, n_parsed_pairs=parsed,
                n_positive_speed_denominator=positive, S=distribution(ratios),
                n_rounding_consistent=correct, rounding_count_denominator=parsed,
                n_rounding_consistent_nominal_2s=nominal_correct,
                n_classification_changes_nominal_2s=changed,
                absolute_consistency_error=distribution(deviations), duration_seconds=distribution(duration_values),
                maximum_exact_reference_identity_error=reference_error,
                pairing='Same sequence, player track and window; image identifiers and FPS verified.',
                definition='S=path/(actual_duration*speed) for positive speed; C=abs(path-duration*speed)<=0.05*(1+duration).',
                scope='C permits one-decimal rounding, not accuracy; counts use all common parsed pairs, S only positive speed. Constants may be consistent.')


def aligned_consistency(cells):
    """Report S/C on a shared cross-arm cohort, separate from per-arm coverage."""
    if not cells:
        return None
    expected_sets, observed_sets, parsed_sets, positive_sets = [], [], [], []
    for cell in cells.values():
        lookup = {f:{trajectory_key(r):item_key(r) for r in cell['items'].values() if r['category']==f}
                  for f in FAMILIES}
        expected = set(lookup[FAMILIES[0]]) & set(lookup[FAMILIES[1]])
        observed = {k for k in expected if all(lookup[f][k] in cell['observations'] for f in FAMILIES)}
        parsed = {k for k in observed if all(cell['observations'][lookup[f][k]]['prediction'] is not None
                                            for f in FAMILIES)}
        positive = {k for k in parsed if cell['observations'][lookup[FAMILIES[0]][k]]['prediction'] > 0}
        expected_sets.append(expected);observed_sets.append(observed)
        parsed_sets.append(parsed);positive_sets.append(positive)
    require(all(s == expected_sets[0] for s in expected_sets), 'Cross-arm speed/path prepared cohorts differ')
    common_observed, common_parsed, common_positive = (set.intersection(*sets) for sets in
                                                      (observed_sets,parsed_sets,positive_sets))
    def summarize_on(cohort):
        return {arm:speed_path_consistency(dict(arm=arm,
                    items={k:r for k,r in cell['items'].items() if trajectory_key(r) in cohort},
                    observations=cell['observations'])) for arm,cell in cells.items()}
    return dict(arms=list(cells),n_expected=len(expected_sets[0]),n_common_observed=len(common_observed),
                n_common_parsed=len(common_parsed),n_common_positive_speed=len(common_positive),
                on_common_parsed=summarize_on(common_parsed),
                on_common_positive_speed=summarize_on(common_positive),
                scope='Shared cohorts across available declared arms only; empty/missing arms do not become evidence.')


class Auditor:
    def __init__(self, inputs, plan=None):
        self.inputs = inputs.resolve()
        self.hashes, self.errors, self.missing, self.warnings = {}, [], [], []
        self.manifest = self.read(self.inputs/'manifest.json')
        self.manifest_hash = digest(self.inputs/'manifest.json')
        self.pools, self.cells = {}, {}
        self.runs = []
        self.plan = self.read(plan) if plan else None
        self.training_records = {}
        if self.plan:
            if 'stages' in self.plan:
                self.plan['runs'] = [dict(r,expected_arms=r['arms']) for r in self.plan['stages']
                                     if r['kind']=='evaluation']
            if self.plan.get('input_manifest_sha256'):
                require(self.plan['input_manifest_sha256'] == self.manifest_hash, 'Execution plan manifest hash differs')
            for relative, expected in self.plan.get('source_sha256',{}).items():
                self.check_hash(ROOT/relative,expected)
            identities = [(r['label'],r['seq'],r.get('frame_mode','full')) for r in self.plan['runs']]
            require(len(identities) == len(set(identities)), 'Duplicate planned run identity')
            for row in self.plan['runs']:
                require(set(row['expected_arms']) <= set(ARMS) and
                        len(row['expected_arms']) == len(set(row['expected_arms'])), 'Invalid planned arms')

    def planned_run(self, label, seq, mode):
        if self.plan:
            return next((r for r in self.plan['runs'] if
                         (r['label'],r['seq'],r.get('frame_mode','full')) == (label,seq,mode)), None)
        return None

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        self.hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode('utf8'))

    def check_hash(self, path, expected):
        require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected), f'Invalid SHA for {path}')
        actual = digest(path)
        self.hashes[str(Path(path).resolve())] = actual
        require(actual == expected, f'Hash mismatch: {path}')

    def load_inputs(self):
        for relative, expected in self.manifest.get('source_sha256', {}).items():
            path = ROOT/Path(relative.replace('\\','/'))
            if not path.is_file():
                self.missing.append(f'Provenance source missing: {path}')
            else:
                self.check_hash(path, expected)
        for cell in self.manifest['cells']:
            path = self.inputs/Path(cell['file'].replace('\\','/'))
            self.check_hash(path, cell['sha256'])
            rows = self.read(path)
            require(len(rows) == cell['items'], 'Prepared cell size differs from manifest')
            keyed = {item_key(r):r for r in rows}
            require(len(keyed) == len(rows), 'Duplicate prepared category/track/window')
            for row in rows:
                m = row['meta']
                require(m['seq'] == cell['seq'] and m['arm'] == cell['arm'], 'Prepared cell metadata differ')
                require(row['category'] in FAMILIES and len(row['image_ids']) == 4, 'Unexpected task or frame count')
                require(all(finite(m[k]) and m[k] > 0 for k in ('fps','score_tolerance','score_floor')),
                        'Invalid FPS/tolerance/floor')
                require(finite(m['reference_value']) and m['reference_value'] >= 0, 'Invalid exact reference')
                require(score(float(row['answer']),m['reference_value'],m['score_tolerance'],m['score_floor']) == 100,
                        'Rounded prepared answer fails exact-reference score')
            self.pools[(cell['seq'],cell['arm'])] = rows
            self.cells[(cell['seq'],cell['arm'])] = cell
        # Check the intended factorial manipulation, coordinate prose and scaled scoring.
        for seq in self.manifest['summary']['test_sequences']:
            base = {item_key(r):r for r in self.pools[(seq,'m_height')]}
            for arm in ARMS:
                keyed = {item_key(r):r for r in self.pools[(seq,arm)]}
                require(base.keys() == keyed.keys(), f'{seq}/{arm}: different prepared cohort')
                for key, row in keyed.items():
                    b = base[key]
                    require(row['image_ids'] == b['image_ids'], 'Unit/height inputs have different frames')
                    q = row['question']
                    normalized = q.replace(PRIOR,'').rsplit(' in ',1)[0]
                    expected = b['question'].replace(PRIOR,'').rsplit(' in ',1)[0]
                    require(normalized == expected, 'Unexpected change outside unit/height wording')
                    require((PRIOR in q) == arm.endswith('_height'), 'Height manipulation differs from arm')
                    if arm.startswith('cm_'):
                        for field in ('reference_value','score_tolerance','score_floor'):
                            require(math.isclose(row['meta'][field],100*b['meta'][field],rel_tol=1e-12),
                                    'Centimetre reference/tolerance/floor did not scale by 100')
                    if arm.startswith('m_'):
                        for field in ('reference_value','score_tolerance','score_floor'):
                            require(row['meta'][field] == b['meta'][field], 'Height control changed metre scoring')
                    unit_text = {'m':'meters','cm':'centimeters','px':'pixels'}[row['meta']['unit']]
                    suffix = ' per second' if row['category']==FAMILIES[0] else ''
                    require(q.endswith(f'in {unit_text}{suffix}? Output only the number, one decimal place.'),
                            'Requested output unit differs from metadata')
                    require('Each supplied frame is 960 pixels wide and 540 pixels high.' in q and
                            'Pixel answers refer to this original canvas before any resizing.' in q,
                            'Explicit pixel coordinate declaration is absent')
            # Px scoring uses one common declared rendered-plane scale for T/epsilon,
            # while the expected px/m reference ratio is deliberately itemwise.
            for suffix in ('height','noheight'):
                rows = self.pools[(seq,'px_'+suffix)]
                tolerance_scales, floor_scales = [], []
                for row in rows:
                    m, b = row['meta'], base[item_key(row)]['meta']
                    tolerance_scales.append(m['score_tolerance']/b['score_tolerance'])
                    floor_scales.append(m['score_floor']/b['score_floor'])
                require(max(tolerance_scales)-min(tolerance_scales) < 1e-10 and
                        max(abs(a-b) for a,b in zip(tolerance_scales,floor_scales)) < 1e-10,
                        'Pixel tolerance/floor convention differs across family')
            height_pixels = {item_key(r):r for r in self.pools[(seq,'px_height')]}
            for r in self.pools[(seq,'px_noheight')]:
                for field in ('reference_value','score_tolerance','score_floor'):
                    require(r['meta'][field] == height_pixels[item_key(r)]['meta'][field],
                            'Removing height changed pixel reference/scoring')

    def load_run(self, label, directory):
        directory = directory.resolve()
        if not (directory/'run_config.json').is_file():
            self.missing.append(f'Run config missing: {label}={directory}')
            return
        config = self.read(directory/'run_config.json')
        require(config['input_manifest_sha256'] == self.manifest_hash, 'Run used another prepared manifest')
        require(config['protocol'] == self.manifest['summary']['protocol'], 'Run protocol differs')
        for key, value in dict(max_pixels=200704,max_new_tokens=128,load_4bit=True,decoding='greedy',strict_parse=True).items():
            require(config.get(key) == value, f'Unexpected inference config {key}')
        seq = config['seq']
        mode = config.get('frame_mode','full')
        require(mode in ('full','static4'), 'Unknown frame mode')
        if 'frame_mode' not in config:
            self.warnings.append(f'{label}/{seq}: legacy revision runner omitted frame_mode; interpreted as full.')
        require(len(set(config['arms'])) == len(config['arms']), 'Duplicate configured arms')
        require(set(config['arms']) <= set(ARMS), 'Unexpected configured arm')
        signature = (label, seq, mode)
        require(not any(r['signature'] == signature for r in self.runs), 'Duplicate label/sequence/frame-mode run')
        planned = self.planned_run(label,seq,mode)
        if self.plan and planned is None:
            self.warnings.append(f'Unplanned additional run is reported separately: {label}/{seq}/{mode}')
        expected_arms = (planned['expected_arms'] if planned else
                         config.get('expected_arms', ARMS if mode == 'full' else ['m_height']))
        require(set(expected_arms) <= set(ARMS), 'Unexpected expected_arms')
        if planned and 'expected_arms' in config:
            require(set(config['expected_arms']) == set(expected_arms), 'Run and analysis plan expected arms differ')
        if planned:
            require(not (set(config['arms'])-set(expected_arms)), 'Run includes unplanned arms in a planned directory')
            if planned.get('output'):
                require(Path(planned['output']).resolve() == directory, 'Run directory differs from execution plan')
            if planned.get('adapter'):
                require(Path(config['adapter']).resolve() == Path(planned['adapter']).resolve(),
                        'Run adapter differs from execution plan')
            if self.plan.get('model_path'):
                require(Path(config['model_path']).resolve() == Path(self.plan['model_path']).resolve(),
                        'Run backbone path differs from execution plan')
        for arm in set(expected_arms)-set(config['arms']):
            self.missing.append(f'Unscheduled expected arm: {label}/{seq}/{mode}/{arm}')
        if planned and planned.get('arithmetic') and not config.get('arithmetic'):
            self.missing.append(f'Planned arithmetic was not configured: {label}/{seq}/{mode}')
        if config.get('input_cells'):
            require({r['arm'] for r in config['input_cells']} == set(config['arms']),
                    'run_config input_cells does not match arms')
            for recorded in config['input_cells']:
                source = self.cells[(seq,recorded['arm'])]
                require(recorded['sha256'] == source['sha256'] and
                        recorded['file'].replace('\\','/') == source['file'].replace('\\','/'),
                        'Run input cell hash/path differs from manifest')
        else:
            self.warnings.append(f'{label}/{seq}: per-cell input hashes not recorded in run config.')
        for relative,expected in config.get('code_sha256',{}).items():
            self.check_hash(ROOT/relative,expected)
        for folder, filename, hash_key in [
            ('adapter','adapter_model.safetensors','adapter_sha256'),
            ('adapter','adapter_config.json','adapter_config_sha256'),
            ('model_path','config.json','model_config_sha256'),
        ]:
            if config.get(hash_key):
                path = Path(config[folder])/filename
                if path.is_file():
                    self.check_hash(path, config[hash_key])
                else:
                    self.missing.append(f'Recorded model provenance file absent: {path}')
            else:
                self.warnings.append(f'{label}/{seq}: {hash_key} not recorded.')
        if not config.get('adapter_sha256') or not config.get('model_config_sha256'):
            self.missing.append(f'{label}/{seq}: model identity hashes missing')
        if label == 'native_original' and self.plan and self.plan.get('original_adapter_sha256'):
            require(config.get('adapter_sha256') == self.plan['original_adapter_sha256'],
                    'Original native adapter hash differs from plan')
        if config.get('training_protocol_sha256'):
            protocol_path = Path(config['adapter'])/'training_protocol.json'
            self.check_hash(protocol_path,config['training_protocol_sha256'])
            require(config.get('training_protocol') == self.read(protocol_path),
                    'Embedded training protocol differs from adapter receipt')
        run = dict(signature=signature,label=label,directory=str(directory),seq=seq,frame_mode=mode,
                   config=config,cells={},arithmetic=None)
        for arm in config['arms']:
            require((seq,arm) in self.pools, 'Configured cell absent from manifest')
            rows = self.pools[(seq,arm)]
            items = {item_key(r):r for r in rows}
            observations = {}
            path = directory/(arm+'.jsonl')
            if path.is_file():
                raw = path.read_bytes()
                self.hashes[str(path)] = hashlib.sha256(raw).hexdigest()
                lines = raw.decode('utf8').splitlines()
                for line_index, line in enumerate(lines):
                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        if line_index == len(lines)-1 and not raw.endswith(b'\n'):
                            self.missing.append(f'Incomplete final JSONL row: {path}')
                            continue
                        raise ValueError(f'Invalid JSONL at {path}, line {line_index+1}')
                    index = result.get('index')
                    require(type(index) is int and 0 <= index < len(rows), 'Invalid output index')
                    source = rows[index]
                    require(result['category'] == source['category'], 'Output category differs from prepared index')
                    for field, expected in source['meta'].items():
                        require(exact_meta(result.get(field),expected), f'Output metadata differs: {path}/{index}/{field}')
                    key = item_key(source)
                    require(key not in observations, 'Duplicate observed category/track/window')
                    pred = parse(result['raw_answer'])
                    require(exact_meta(result.get('prediction'),pred), 'Stored prediction differs from strict raw parsing')
                    observations[key] = result
            if len(observations) != len(rows):
                self.missing.append(f'{label}/{seq}/{mode}/{arm}: {len(observations)}/{len(rows)} rows available')
            run['cells'][arm] = dict(arm=arm,items=items,observations=observations)
        if config.get('arithmetic'):
            stimuli = self.read(self.inputs/'arithmetic.json')
            self.check_hash(self.inputs/'arithmetic.json',config['arithmetic_sha256'])
            path = directory/'arithmetic_results.json'
            answers = self.read(path) if path.is_file() else []
            require(len(answers) <= len(stimuli), 'Too many arithmetic results')
            for observed, stimulus in zip(answers,stimuli):
                require(observed['question'] == stimulus['question'] and
                        observed['target'] == float(stimulus['answer']), 'Arithmetic cohort/reference differs')
                pred = parse(observed['raw_answer'])
                require(exact_meta(observed.get('prediction'),pred), 'Arithmetic parse differs')
                correct = pred is not None and abs(pred-float(stimulus['answer'])) < .05
                require(observed['correct'] == correct, 'Arithmetic correctness differs')
            if len(answers) != len(stimuli):
                self.missing.append(f'{label}/{seq}: arithmetic {len(answers)}/{len(stimuli)}')
            run['arithmetic'] = dict(n_expected=len(stimuli),n_observed=len(answers),
                                     n_parsed=sum(r['prediction'] is not None for r in answers),
                                     n_correct=sum(r['correct'] for r in answers),
                                     all_items_accuracy=sum(r['correct'] for r in answers)/len(stimuli)
                                         if len(answers) == len(stimuli) else None,
                                     scope='Text-only arithmetic; not visual unit conversion evidence.')
        self.runs.append(run)

    def check_plan_complete(self):
        if not self.plan:
            self.warnings.append('No execution plan supplied: completeness covers declared directories/configured expected_arms only.')
            return
        found = {r['signature'] for r in self.runs}
        for row in self.plan['runs']:
            identity = (row['label'],row['seq'],row.get('frame_mode','full'))
            if identity not in found:
                self.missing.append(f'Planned run missing: {identity}')
        for stage in self.plan.get('stages',[]):
            if stage['kind'] != 'training':
                continue
            folder=Path(stage['output'])
            protocol_path,completion_path=folder/'training_protocol.json',folder/'training_completion.json'
            self.check_hash(Path(stage['training_pool']),stage['training_pool_sha256'])
            if not protocol_path.is_file() or not completion_path.is_file():
                self.missing.append(f'Completed training receipts missing: {stage["label"]}')
                continue
            protocol, completion = self.read(protocol_path), self.read(completion_path)
            adapter_config = self.read(folder/'adapter_config.json')
            require(adapter_config.get('r')==16 and adapter_config.get('lora_alpha')==32 and
                    math.isclose(adapter_config.get('lora_dropout',-1),.05,rel_tol=0,abs_tol=1e-12),
                    'Actual new adapter LoRA rank/alpha/dropout differ from fixed P2 design')
            command = stage['command']
            require('--lora_r' in command and command[command.index('--lora_r')+1]=='16',
                    'Planned training command does not request rank 16')
            require(completion['completed'] is True and completion['selected_samples']==stage['training_items'] and
                    completion['global_step']==stage['expected_steps'] and completion['seed']==42,
                    'Training completion differs from execution plan')
            require(completion['training_protocol_sha256']==digest(protocol_path),
                    'Training completion protocol hash differs')
            require(protocol['qa_source_sha256']==stage['training_pool_sha256'] and
                    protocol['selected_samples']==stage['training_items'] and protocol['seed']==42 and
                    protocol['init_adapter'] is None and protocol['allow_drop_invalid'] is False and
                    protocol['initialization_seed_applied_before_model_load'] is True,
                    'Training protocol does not verify planned base initialization/seed/full pool')
            for name,row in completion['final_adapter_files'].items():
                self.check_hash(folder/name,row['sha256'])
            matching_runs=[r for r in self.runs if r['label']==stage['label']]
            for run in matching_runs:
                require(run['config'].get('training_protocol_sha256')==digest(protocol_path),
                        'Evaluation is not linked to its planned training protocol')
                require(run['config']['adapter_sha256']==completion['final_adapter_files']['adapter_model.safetensors']['sha256'],
                        'Evaluation adapter is not the completed trained checkpoint')
            self.training_records[stage['label']]=dict(training_convention=stage['training_convention'],
                protocol=protocol,completion=completion,protocol_sha256=digest(protocol_path),
                completion_sha256=digest(completion_path),actual_adapter_config=adapter_config,
                training_command=command,verified=True)

    def training_pool_audit(self):
        """Read the two prepared pools; this is provenance, not model evidence."""
        paths = [self.inputs/'training'/f'event_holdout_{v}.json' for v in ('v1','v3')]
        if not all(p.is_file() for p in paths):
            self.missing.append('Prepared v1/v3 training pools unavailable for paired provenance check')
            return None
        a,b = (self.read(p) for p in paths)
        require(len(a) == len(b) == self.manifest['summary']['training_items_per_convention'],
                'Paired training pool sizes differ')
        changed = 0
        sources = set()
        for left,right in zip(a,b):
            for field in ('category','question','image_ids'):
                require(left[field] == right[field], f'Training v1/v3 {field} differs')
            for field in ('source_seq','track','window'):
                require(left['meta'].get(field) == right['meta'].get(field), f'Training identity field differs: {field}')
            changed += left['answer'] != right['answer']
            sources.add(left['meta']['source_seq'])
        event_path = ROOT/'results/courtdyn/courtdyn_native_sft_train_event_holdout.manifest.json'
        event_manifest = self.read(event_path)
        groups = event_manifest['event_groups']
        held_out = self.manifest['summary']['test_sequences']
        overlap = {groups[s] for s in sources} & {groups[s] for s in held_out}
        require(not overlap and set(event_manifest['train_seqs']) == sources,
                'Prepared training pool overlaps a held-out event group')
        return dict(n_each=len(a),n_rounded_answers_changed=changed,source_sequences=sorted(sources),
                    identical_input_order_questions_and_frames=True,
                    pool_sha256={v:digest(p) for v,p in zip(('v1','v3'),paths)},
                    event_group_manifest_sha256=digest(event_path),overlapping_event_groups=[],
                    scope='Prepared-pool comparison only. It does not prove a checkpoint was trained on these files or verify its seed.')

    def summarize_run(self, run):
        cells, pairs = {}, []
        for arm, cell in run['cells'].items():
            cells[arm] = dict(families={f:family_statistics([r for r in cell['items'].values() if r['category']==f],
                                                           cell['observations']) for f in FAMILIES},
                             speed_path_consistency=speed_path_consistency(cell))
        if run['frame_mode'] == 'full':
            comparisons = [(f'm_{h}',f'{u}_{h}',f'{u}/m with {h}') for h in ('height','noheight') for u in ('cm','px')]
            comparisons += [(f'{u}_height',f'{u}_noheight',f'noheight/height in {u}') for u in ('m','cm','px')]
            for a, b, name in comparisons:
                if a in run['cells'] and b in run['cells']:
                    pairs.extend(paired_statistics(run['cells'][a],run['cells'][b],f,name) for f in FAMILIES)
        return dict(label=run['label'],seq=run['seq'],frame_mode=run['frame_mode'],
                    directory=run['directory'],configuration=run['config'],cells=cells,
                    within_run_pairing=pairs,cross_arm_speed_path_consistency=aligned_consistency(run['cells']),
                    arithmetic=run['arithmetic'],
                    training_event_overlap=(run['seq']=='Q1_top_0-30' if run['label']=='native_original'
                                            else False if run['label'] in self.training_records else None),
                    event_scope='Within the same game; event isolation is distinct from cross-game generalization.')

    def compare_models(self, left, right):
        comparisons = []
        training_verified = False
        if left in self.training_records and right in self.training_records:
            one,two = self.training_records[left],self.training_records[right]
            require([one['training_convention'],two['training_convention']]==['v1','v3'],
                    'Model comparison order must be planned v1, then planned v3')
            ignore = {'qa_source','qa_source_sha256','ordered_sample_sequence_sha256'}
            require({k:v for k,v in one['protocol'].items() if k not in ignore} ==
                    {k:v for k,v in two['protocol'].items() if k not in ignore},
                    'P2 training protocols differ beyond label source / ordered supervision hash')
            def normalized_command(command):
                result = command[:]
                for flag in ('--qa_json','--output_dir'):
                    require(flag in result, f'Training command lacks {flag}')
                    result[result.index(flag)+1]='[LABEL_DEPENDENT_PATH]'
                return result
            require(normalized_command(one['training_command']) == normalized_command(two['training_command']),
                    'P2 planned training commands differ beyond data/output paths')
            require(one['actual_adapter_config']==two['actual_adapter_config'],
                    'P2 actual adapter configurations differ')
            training_verified = True
        a = {(r['seq'],r['frame_mode']):r for r in self.runs if r['label']==left}
        b = {(r['seq'],r['frame_mode']):r for r in self.runs if r['label']==right}
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                self.missing.append(f'Model comparison unavailable: {left}/{right}/{key}')
                continue
            ar, br = a[key], b[key]
            for field in CONFIG_KEYS:
                require(ar['config'].get(field) == br['config'].get(field),
                        f'P2 comparison inference config mismatch: {field}')
            require(ar['config'].get('adapter_sha256') and br['config'].get('adapter_sha256') and
                    ar['config']['adapter_sha256'] != br['config']['adapter_sha256'],
                    'P2 comparison requires two distinct recorded adapter hashes')
            arm_results = []
            for arm in sorted(ar['cells'].keys() | br['cells'].keys()):
                if arm not in ar['cells'] or arm not in br['cells']:
                    self.missing.append(f'Paired models missing arm {key}/{arm}')
                    continue
                ac, bc = ar['cells'][arm], br['cells'][arm]
                for item in ac['items']:
                    require(ac['items'][item] == bc['items'][item], 'P2 models evaluated on different stimuli/references')
                for f in FAMILIES:
                    row = paired_statistics(ac,bc,f,f'{right}/{left}')
                    row['arm'] = arm
                    row['numerator_model'], row['denominator_model'] = right, left
                    arm_results.append(row)
            comparisons.append(dict(models=[left,right],seq=key[0],frame_mode=key[1],paired_results=arm_results,
                paired_training_protocol_verified=training_verified,
                scope='Same newly prepared inputs and common observed/parsed cohorts; no legacy full results used. Training identity is certified only when the paired receipt flag is true.'))
        if not a or not b:
            self.missing.append(f'Model comparison needs both labels: {left}, {right}')
        return comparisons

    def temporal_controls(self):
        results = []
        for run in self.runs:
            if run['frame_mode'] != 'static4':
                continue
            full = next((r for r in self.runs if r['label']==run['label'] and r['seq']==run['seq']
                         and r['frame_mode']=='full'), None)
            if full is None:
                self.missing.append(f'Static4 has no matching new full run: {run["label"]}/{run["seq"]}')
                continue
            for field in CONFIG_KEYS+('adapter_sha256',):
                require(full['config'].get(field) == run['config'].get(field), 'Static4/full inference/model config differs')
            for arm in run['cells']:
                if arm not in full['cells']:
                    self.missing.append(f'Static4 arm lacks full control: {arm}')
                    continue
                for family in FAMILIES:
                    row = paired_statistics(full['cells'][arm],run['cells'][arm],family,'static4/new full')
                    row.update(label=run['label'],seq=run['seq'])
                    results.append(row)
        return results


class Tests(unittest.TestCase):
    def test_scale_and_strict_scoring(self):
        self.assertEqual(score(1.8,1,.3,1),0)  # relative .5: strict endpoint
        self.assertEqual(score(None,1,.3,1),0)
        for y in (.2,1,4):
            for p in (0,.7,1,1.5,10):
                self.assertEqual(score(p,y,.3,1),score(100*p,100*y,30,100))

    def test_rank_ties(self):
        self.assertAlmostEqual(spearman([1,2,2,4],[4,3,3,1]),-1)
        self.assertIsNone(spearman([1,1,1],[1,2,3]))
        self.assertEqual(distribution([1,2,3,4])['q1_q3'],[1.75,3.25])

    def test_parse(self):
        self.assertEqual(parse(' 1.2e2 '),120)
        for raw in ('1 meter','NaN','inf','1\n2',''):
            self.assertIsNone(parse(raw))

    def test_partial_and_constant(self):
        items=[]
        for i,y in enumerate((1.,2.,3.)):
            items.append(dict(category=FAMILIES[0],meta=dict(track=i,window=[0,60],
                reference_value=y,score_tolerance=.3,score_floor=1.)))
        obs={item_key(items[0]):dict(prediction=1.),item_key(items[1]):dict(prediction=None)}
        result=family_statistics(items,obs)
        self.assertIsNone(result['tmra_all_items'])
        self.assertEqual(result['tmra_all_observed'],50)
        self.assertEqual(result['n_missing'],1)
        constants=result['constant_diagnostics']['values']
        self.assertGreaterEqual(constants['test_label_oracle']['tmra'],constants['median']['tmra'])

    def test_pair_denominators(self):
        def cell(arm,preds,scale):
            items={}
            obs={}
            for i,p in enumerate(preds):
                r=dict(category=FAMILIES[0],image_ids=['a']*4,meta=dict(track=i,window=[0,60],
                       reference_value=(i+1)*scale,score_tolerance=.3*scale,score_floor=scale))
                key=item_key(r);items[key]=r;obs[key]=dict(prediction=p)
            return dict(arm=arm,items=items,observations=obs)
        result=paired_statistics(cell('m',[1,0,None],1),cell('cm',[100,3,4],100),FAMILIES[0],'cm/m')
        self.assertEqual(result['n_common_parsed'],2)
        self.assertEqual(result['n_positive_prediction_denominator'],1)
        self.assertEqual(result['ratio']['median'],100)
        self.assertEqual(result['expected_reference_ratio_on_ratio_cohort']['median'],100)
        self.assertEqual(result['ratio_divided_by_itemwise_reference_ratio']['median'],1)

    def fixture(self, folder):
        """Tiny synthetic runner-format fixture, never used as model evidence."""
        inputs=folder/'inputs';out=folder/'run';model=folder/'model';adapter=folder/'adapter'
        for path in (inputs,out,model,adapter):
            path.mkdir()
        manifest=dict(summary=dict(protocol='fixture'),cells=[])
        (inputs/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
        (model/'config.json').write_text('{}',encoding='utf8')
        (adapter/'adapter_config.json').write_text('{}',encoding='utf8')
        (adapter/'adapter_model.safetensors').write_bytes(b'synthetic fixture; not a model')
        items=[]
        for i in range(2):
            items.append(dict(category=FAMILIES[0],image_ids=['a','b','c','d'],answer='1.0',
                              question='synthetic',meta=dict(seq='fixture',track=i,window=[0,60],
                              fps=30.,arm='m_height',unit='m',reference_value=1.,
                              reference_convention='fixture',score_tolerance=.3,score_floor=1.)))
        config=dict(protocol='fixture',seq='fixture',arms=['m_height'],expected_arms=['m_height'],frame_mode='full',
                    input_manifest_sha256=digest(inputs/'manifest.json'),arithmetic=False,model_path=str(model),
                    adapter=str(adapter),adapter_sha256=digest(adapter/'adapter_model.safetensors'),
                    adapter_config_sha256=digest(adapter/'adapter_config.json'),model_config_sha256=digest(model/'config.json'),
                    max_pixels=200704,max_new_tokens=128,load_4bit=True,decoding='greedy',strict_parse=True)
        (out/'run_config.json').write_text(json.dumps(config),encoding='utf8')
        auditor=Auditor(inputs);auditor.pools[('fixture','m_height')]=items
        auditor.cells[('fixture','m_height')]=dict(sha256='0'*64,file='unused')
        row=dict(index=0,category=items[0]['category'],**items[0]['meta'],prediction=1.,raw_answer='1.0')
        return auditor,out,row

    def test_output_partial_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            auditor,out,row=self.fixture(Path(tmp))
            (out/'m_height.jsonl').write_text(json.dumps(row)+'\n',encoding='utf8')
            auditor.load_run('fixture',out)
            self.assertTrue(any('1/2' in missing for missing in auditor.missing))
            second=Auditor(auditor.inputs);second.pools=auditor.pools;second.cells=auditor.cells
            (out/'m_height.jsonl').write_text((json.dumps(row)+'\n')*2,encoding='utf8')
            with self.assertRaisesRegex(ValueError,'Duplicate observed'):
                second.load_run('fixture',out)

    def test_output_metadata_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            auditor,out,row=self.fixture(Path(tmp))
            row['reference_value']=99.
            (out/'m_height.jsonl').write_text(json.dumps(row)+'\n',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'Output metadata differs'):
                auditor.load_run('fixture',out)
            config=json.loads((out/'run_config.json').read_text(encoding='utf8'))
            config['input_manifest_sha256']='0'*64
            (out/'run_config.json').write_text(json.dumps(config),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'another prepared manifest'):
                auditor.load_run('fixture',out)

    def test_shared_speed_path_cohort(self):
        def cell(arm, invalid=False):
            items,observations={},{}
            for track in (1,2):
                for family in FAMILIES:
                    value=1. if family==FAMILIES[0] else 2.
                    row=dict(category=family,image_ids=['a','b','c','d'],
                             meta=dict(track=track,window=[0,60],fps=30.,reference_value=value))
                    key=item_key(row);items[key]=row
                    observations[key]=dict(prediction=None if invalid and track==2 else value)
            return dict(arm=arm,items=items,observations=observations)
        aligned=aligned_consistency({'a':cell('a'),'b':cell('b',True)})
        self.assertEqual(aligned['n_expected'],2)
        self.assertEqual(aligned['n_common_parsed'],1)
        self.assertEqual(aligned['on_common_positive_speed']['a']['S']['median'],1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,default=ROOT/'results/courtdyn/revision_controls_20260912')
    ap.add_argument('--run',action='append',default=[],metavar='LABEL=DIRECTORY',
                    help='Repeat; one label may have distinct sequence/frame-mode directories. Missing directories are reported.')
    ap.add_argument('--compare-models',nargs=2,action='append',default=[],metavar=('V1_LABEL','V3_LABEL'))
    ap.add_argument('--plan',type=Path,help='Expected label/sequence/frame-mode/arms/arithmetic JSON; missing planned runs remain PARTIAL.')
    ap.add_argument('--output-dir',type=Path,help='Must be new; analysis never overwrites an earlier report.')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if result.wasSuccessful() else 1
    if not args.output_dir:
        ap.error('--output-dir is required except with --self-test')
    if args.output_dir.exists():
        ap.error('Use a new --output-dir; existing analysis artifacts are not overwritten.')
    auditor=None
    report=dict(schema='courtdyn-revision-results-analysis-v1',generated=datetime.now().isoformat(timespec='seconds'),
                analysis_script_sha256=digest(__file__),status='FAIL',errors=[],missing=[],warnings=[],runs=[],
                limitations=['CPU reanalysis of newly recorded inference only; no training or inference is performed.',
                             'Windows/players are dependent. Q1/Q2 are two events from the same game, not cross-game replication.',
                             'Physical units measure floor motion; pixels measure projected image motion. Explicit canvas prose does not equate the quantities.',
                             'Original native Q1 has training-event overlap. Event-holdout labels must be supported by training provenance.',
                             'No confidence intervals or claims about internal mechanisms are inferred from descriptive statistics.'])
    try:
        auditor=Auditor(args.inputs,args.plan)
        auditor.load_inputs()
        if not args.run:
            auditor.missing.append('No model-run directories were supplied.')
        for declaration in args.run:
            require('=' in declaration,'--run requires LABEL=DIRECTORY')
            label,directory=declaration.split('=',1)
            require(label and re.fullmatch(r'[A-Za-z0-9_.-]+',label),'Use a nonempty alphanumeric model label')
            auditor.load_run(label,Path(directory))
        auditor.check_plan_complete()
        report['runs']=[auditor.summarize_run(r) for r in auditor.runs]
        report['model_comparisons']=[r for a,b in args.compare_models for r in auditor.compare_models(a,b)]
        report['temporal_controls']=auditor.temporal_controls()
        report['training_pool_audit']=auditor.training_pool_audit()
        report['training_receipts']=auditor.training_records
        report['input_manifest_sha256']=auditor.manifest_hash
        report['execution_coverage']=dict(
            n_expected_evaluation_runs=len(auditor.plan['runs']) if auditor.plan else None,
            n_observed_evaluation_runs=len(auditor.runs),
            n_expected_visual_items=auditor.plan.get('visual_items') if auditor.plan else None,
            n_observed_visual_items=sum(len(c['observations']) for r in auditor.runs for c in r['cells'].values()),
            n_expected_arithmetic_items=auditor.plan.get('arithmetic_items') if auditor.plan else None,
            n_observed_arithmetic_items=sum(r['arithmetic']['n_observed'] for r in auditor.runs if r['arithmetic']))
        report['status']='PARTIAL' if auditor.missing else 'PASS'
    except (ValueError,KeyError,OSError,TypeError,OverflowError) as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    if auditor:
        report['source_sha256']=auditor.hashes
        report['missing']=auditor.missing
        report['warnings']=auditor.warnings
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    def number(value):
        return 'NA' if value is None else f'{value:.4f}'
    table=['| Model | Sequence / frames | Arm / family | Expected / observed / parsed | All-item T-MRA | Parsed T-MRA | Spearman |',
           '|---|---|---|---:|---:|---:|---:|']
    for run in report['runs']:
        for arm, cell in run['cells'].items():
            for family, row in cell['families'].items():
                table.append(f'| {run["label"]} | {run["seq"]} / {run["frame_mode"]} | {arm} / {family.removeprefix("dynamics_").removesuffix("_player")} | '
                             f'{row["n_expected"]} / {row["n_observed"]} / {row["n_parsed"]} | '
                             f'{number(row["tmra_all_items"])} | {number(row["tmra_parsed"])} | {number(row["spearman"])} |')
    (args.output_dir/'summary.md').write_text(
        '# CourtDyn revision results\n\n'+report['status']+'\n\n'+
        f'Analyzed run directories: {len(report["runs"])}. No legacy outputs were substituted.\n\n'+
        '\n'.join('- '+line for line in report['errors']+report['missing']+report['warnings'])+
        '\n\n'+'\n'.join(table)+
        '\n\nAll-item T-MRA is NA until the full prepared family is observed. Invalid parsed answers score zero; unobserved answers remain missing. '
        'All ratios, paired-cohort scores, constant diagnostics, S/C cohorts, arithmetic and training provenance are in analysis.json.\n\n'+
        '\n'.join('- '+line for line in report['limitations'])+'\n',encoding='utf8')
    print(json.dumps(dict(status=report['status'],runs=len(report['runs']),errors=report['errors'],
                         missing_count=len(report['missing']),output=str(args.output_dir)),ensure_ascii=False))
    return {'PASS':0,'FAIL':1,'PARTIAL':2}[report['status']]


if __name__=='__main__':
    sys.exit(main())
