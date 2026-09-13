"""Prepare, but do not run, the controlled follow-up experiments for paper2.

Outputs are local derived data: do not redistribute the generated question pools.
Original pools, frozen predictions and trained checkpoints are never overwritten.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from engine import dynamics_qa as DQ
from engine.court_homography import CourtPlane, H_DIR, recompute_answer
from tools.build_courtdyn_ruler import tt_split, px_answer, seq_dir
from tools.build_courtdyn_native_sft import TRAIN_SEQS, HELD_OUT, EVENT_GROUPS

SEQS = ['Q2_top_480-510', 'Q1_top_0-30']
SPEED, PATH = 'dynamics_speed_player', 'dynamics_path_player'
PRIOR = 'Assume a typical player on this court is 1.93 m tall. '
CONTEXT = ('The 4 frames are consecutive samples from one basketball clip spanning '
           '2.0 seconds, in chronological order. {prior}'
           'Each supplied frame is 960 pixels wide and 540 pixels high. '
           'Pixel answers refer to this original canvas before any resizing. '
           'Physical units describe floor-plane motion; pixels describe projected '
           'image-plane motion. ')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require_input(path, parser):
    path = Path(path)
    if not path.is_file():
        parser.error(f'Required local input is missing: {path}. '
                     'Restore the authorized source data or prepared input pool before continuing.')
    return path


def read_input_json(path, parser):
    path = require_input(path, parser)
    try:
        return json.loads(path.read_text(encoding='utf8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        parser.error(f'Cannot read required local JSON input {path}: {error}')


def check_image(path, parser):
    path = require_input(path, parser)
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (960, 540):
                parser.error(f'Expected a 960x540 rendered frame: {path}; got {image.size}.')
    except OSError as error:
        parser.error(f'Cannot decode required local image {path}: {error}')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

def key(row):
    m = row['meta']
    return row['category'], m['track'], tuple(m['window'])

def scalar_score(prediction, target, tolerance, floor):
    relative = (abs(prediction-target)-tolerance)/max(abs(target), floor)
    return 10.0*sum(relative < 1-(0.5+0.05*i) for i in range(10))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=ROOT/'results/courtdyn/revision_controls_20260912')
    args = ap.parse_args()
    out = args.out_dir.resolve()
    inputs, cells = {}, []
    source = ROOT/'results/courtdyn/courtdyn_native_sft_train_event_holdout.json'
    train_v1 = read_input_json(source, ap)
    assert len(train_v1) == 1120
    source_paths = {source}
    for seq in set(SEQS + TRAIN_SEQS):
        split = Path(tt_split(seq))
        source_paths.update([Path(H_DIR)/f'H_{seq}.json', split/'gt/gt.txt', split/'seqinfo.ini'])
    for seq in SEQS:
        sd = Path(seq_dir(seq))
        source_paths.update(sd/name for name in (
            'qa_dyn_v1.json', 'qa_dyn_v1_ruler.json', 'qa_dyn_v1_pxunit.json',
            'qa_dyn_v1_pxunit.manifest.json'))
    for path in sorted(source_paths):
        require_input(path, ap)
        try:
            inputs[str(path.relative_to(ROOT))] = sha(path)
        except OSError as error:
            ap.error(f'Cannot read required local input {path}: {error}')
    training_images = {ROOT/'data/courtdyn'/image_id for row in train_v1 for image_id in row['image_ids']}
    for path in sorted(training_images):
        check_image(path, ap)
    for seq in SEQS:
        sd = Path(seq_dir(seq))
        paths = [sd/'qa_dyn_v1.json', sd/'qa_dyn_v1_ruler.json', sd/'qa_dyn_v1_pxunit.json']
        original, ruler, pixels = [read_input_json(p, ap) for p in paths]
        original = {key(r):r for r in original if r['category'] in (SPEED, PATH)}
        ruler = {key(r):r for r in ruler}
        pixels = {key(r):r for r in pixels}
        assert original.keys() == ruler.keys() == pixels.keys() and len(original) == 280
        tracks, info = DQ.load_tracks(tt_split(seq)), DQ.load_seqinfo(tt_split(seq))
        plane, fps = CourtPlane.load(seq), float(info['fps'])
        k = float(read_input_json(sd/'qa_dyn_v1_pxunit.manifest.json', ap)['k_px_per_m']['median'])
        img_root = ROOT/f'data/courtdyn/frames_{seq}'
        for image_id in {v for r in original.values() for v in r['image_ids']}:
            check_image(img_root/image_id, ap)
        per_arm = {}
        for unit in ('m', 'cm', 'px'):
            for prior in (True, False):
                arm = unit + ('_height' if prior else '_noheight')
                rows = []
                for item_key, old in original.items():
                    cat = old['category']
                    r = recompute_answer(plane, old, tracks, fps)
                    assert r is not None and r[0] == ruler[item_key]['answer']
                    world = r[1]['speed_mps' if cat == SPEED else 'path_m']
                    px = px_answer(old, tracks, 960/float(info['width']), fps)
                    assert f'{px:.1f}' == pixels[item_key]['answer']
                    factor = {'m':1.0, 'cm':100.0, 'px':k}[unit]
                    target = {'m':world, 'cm':world*100, 'px':px}[unit]
                    unit_text = {'m':'meters', 'cm':'centimeters', 'px':'pixels'}[unit]
                    if cat == SPEED:
                        question = f'What is the average speed of the player marked with the red box, in {unit_text} per second? '
                    else:
                        question = f'What is the path length of the player marked with the red box over the whole clip, in {unit_text}? '
                    question = CONTEXT.format(prior=PRIOR if prior else '') + question + 'Output only the number, one decimal place.'
                    t, epsilon = ((0.30, 1.0) if cat == SPEED else (0.50, 2.0))
                    row = copy.deepcopy(old)
                    row.update(question=question, answer=f'{target:.1f}')
                    row['meta'] = dict(seq=seq, track=old['meta']['track'], window=old['meta']['window'], fps=fps,
                                       arm=arm, unit=unit, score_tolerance=t*factor, score_floor=epsilon*factor,
                                       reference_value=target, reference_convention='rendered pixels' if unit=='px' else 'v3 court homography')
                    rows.append(row)
                    assert scalar_score(float(row['answer']), target, t*factor, epsilon*factor) == 100
                    # The scoring convention must be invariant to a change of physical unit.
                    for multiplier in (0.0, 0.7, 1.0, 1.5):
                        assert scalar_score(multiplier*world, world, t, epsilon) == scalar_score(multiplier*world*100, world*100, t*100, epsilon*100)
                per_arm[arm] = rows
                path = out/seq/f'{arm}.json'
                write(path, rows)
                cells.append(dict(seq=seq, arm=arm, items=len(rows), file=str(path.relative_to(out)),
                                  sha256=sha(path), image_root=str(img_root), event_overlap=(seq==SEQS[1])))
        for i in range(280):
            for suffix in ('height', 'noheight'):
                # Only the requested unit (last occurrence) changes; coordinate prose is identical.
                rows = [per_arm[u+'_'+suffix][i] for u in ('m','cm','px')]
                normalize = lambda r: r['question'].rsplit(' in ',1)[0]+' in [UNIT]? Output only the number, one decimal place.'
                assert len({normalize(r) for r in rows}) == 1
            for u in ('m','cm','px'):
                assert per_arm[u+'_height'][i]['question'].replace(PRIOR,'') == per_arm[u+'_noheight'][i]['question']
    # Matched, event-isolated training pools: identical inputs/order, v1 or v3 labels.
    training = {}
    for seq in TRAIN_SEQS:
        training[seq]=(CourtPlane.load(seq), DQ.load_tracks(tt_split(seq)), float(DQ.load_seqinfo(tt_split(seq))['fps']))
    train_v3 = copy.deepcopy(train_v1)
    for row in train_v3:
        seq=row['meta']['source_seq']
        assert seq in TRAIN_SEQS and EVENT_GROUPS[seq] not in {EVENT_GROUPS[s] for s in HELD_OUT}
        plane, tracks, fps=training[seq]
        answer=recompute_answer(plane,row,tracks,fps)
        assert answer is not None
        for field, value in answer[1].items():
            if field in row['meta']:
                row['meta']['v1_'+field] = row['meta'][field]
            row['meta'][field] = value
        row['answer']=answer[0]
        row['meta']['label_convention']='v3 court homography'
    for variant,rows in [('v1',train_v1),('v3',train_v3)]:
        write(out/'training'/f'event_holdout_{variant}.json',rows)
    # Publicly shareable arithmetic stimuli; these are inputs, not model observations.
    arithmetic=[]
    for amount in (0.2,0.5,1.0,2.5,4.0,8.0):
        for unit,scale in [('centimeters',100.0),('millimeters',1000.0)]:
            arithmetic.append(dict(question=f'A distance is {amount:.1f} meters. One meter equals {scale:.0f} {unit}. Express that distance in {unit}. Output only the number, one decimal place.', answer=f'{amount*scale:.1f}'))
    write(out/'arithmetic.json',arithmetic)
    summary=dict(status='PREPARED_NOT_RUN', protocol='courtdyn-revision-controls-v1',
                 test_sequences=SEQS, visual_cells=len(cells), visual_items=sum(c['items'] for c in cells),
                 arithmetic_items=len(arithmetic), training_items_per_convention=1120,
                 training_conventions=['v1','v3'], training_images_checked=len(training_images), canvas=[960,540],
                 manipulations=['unit m/cm/px','height cue retained/removed'],
                 checks=['item/frame identity','prompt unit-only comparison','height-only comparison',
                         'actual canvas dimensions','frozen rounded references','scaled T/epsilon','event-disjoint training',
                         'decoded training images','homography/track/seqinfo source hashes','v3 metadata references'],
                 limitations=['No new inference or training has run.','Q1 remains event-overlapped for the frozen original model.',
                              'New event-isolated pools still use the same game.','Generated data require original data permissions.'])
    write(out/'protocol_summary.json',summary)
    write(out/'manifest.json',dict(summary=summary,source_sha256=inputs,cells=cells,arithmetic_file='arithmetic.json'))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    main()
