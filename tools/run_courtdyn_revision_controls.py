"""Validate or explicitly run the follow-up unit controls with the frozen model.

Default is CPU validation only. --run requires an idle CUDA device and does not
start a service, wait in a background queue, or modify any existing result.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_courtdyn_revision_experiments import read_input_json, require_input, scalar_score, sha

NUMBER=re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

def parse(raw):
    raw=raw.strip()
    return float(raw) if NUMBER.fullmatch(raw) and math.isfinite(float(raw)) else None

def score_rows(rows):
    result={}
    for category in sorted({r['category'] for r in rows}):
        group=[r for r in rows if r['category']==category]
        valid=[r for r in group if r['prediction'] is not None]
        scores=[scalar_score(r['prediction'],r['reference_value'],r['score_tolerance'],r['score_floor']) for r in valid]
        result[category]=dict(n=len(group),parsed=len(valid),parse_rate=len(valid)/len(group),
                              tmra_all_items=sum(scores)/len(group),
                              tmra_parsed=sum(scores)/len(valid) if valid else None)
    return result


def validate_saved_rows(path, items):
    """A resume may append only after an intact prefix of these exact inputs."""
    if not path.exists():
        return []
    content=path.read_text(encoding='utf8')
    if content and not content.endswith('\n'):
        raise ValueError(f'Resume requires newline-terminated intact JSONL records: {path}')
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if len(rows) > len(items):
        raise ValueError(f'Too many saved predictions: {path}')
    for index, row in enumerate(rows):
        expected = dict(index=index, category=items[index]['category'], **items[index]['meta'])
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Resume input identity differs at {path}:{index+1}')
        if row.get('prediction') != parse(row['raw_answer']):
            raise ValueError(f'Saved parsing differs at {path}:{index+1}')
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,default=ROOT/'results/courtdyn/revision_controls_20260912')
    ap.add_argument('--seq',default='Q2_top_480-510')
    ap.add_argument('--arms',nargs='+',default=['m_height','cm_height','px_height'])
    ap.add_argument('--arithmetic',action='store_true')
    ap.add_argument('--frame-mode',choices=['full','static4'],default='full')
    ap.add_argument('--resume',action='store_true',help='Append only to a verified identical incomplete run.')
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--model-path',type=Path)
    ap.add_argument('--adapter',type=Path,default=ROOT/'models/courtdyn-native-sft')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if len(args.arms) != len(set(args.arms)):
        ap.error('--arms must contain each arm only once; duplicate arms would reuse an output filename.')
    manifest=read_input_json(args.inputs/'manifest.json', ap)
    cells=[]
    for arm in args.arms:
        cell=next((c for c in manifest['cells'] if c['seq']==args.seq and c['arm']==arm), None)
        if cell is None:
            ap.error(f'The prepared manifest has no cell for sequence {args.seq} and arm {arm}.')
        path=args.inputs/cell['file']
        rows=read_input_json(path, ap)
        if sha(path)!=cell['sha256']:
            ap.error(f'Prepared input hash does not match its manifest: {path}.')
        image_root=Path(cell['image_root'])
        for image_id in {i for row in rows for i in row['image_ids']}:
            require_input(image_root/image_id, ap)
        cells.append((cell,rows))
    # Exercise parsing and scale-aware scoring without inventing model observations.
    assert parse('1.5')==1.5 and parse('1.5 meters') is None and parse('nan') is None
    for _,rows in cells:
        for row in rows:
            m=row['meta']
            assert scalar_score(float(row['answer']),m['reference_value'],m['score_tolerance'],m['score_floor'])==100
    arithmetic=read_input_json(args.inputs/'arithmetic.json', ap) if args.arithmetic else []
    config=dict(protocol=manifest['summary']['protocol'],seq=args.seq,arms=args.arms,
                frame_mode=args.frame_mode,
                input_cells=[dict(arm=c['arm'],file=c['file'],sha256=c['sha256']) for c,_ in cells],
                input_manifest_sha256=sha(args.inputs/'manifest.json'),
                arithmetic=bool(args.arithmetic),arithmetic_sha256=sha(args.inputs/'arithmetic.json') if arithmetic else None,
                model_path=str(args.model_path.resolve()) if args.model_path else None,
                adapter=str(args.adapter.resolve()),max_pixels=200704,max_new_tokens=128,
                load_4bit=True,decoding='greedy',strict_parse=True,
                score='per-item scale-aware T-MRA; invalid parses receive zero in all-item score')
    if not args.run:
        print(json.dumps(dict(status='VALIDATED_NOT_RUN',visual_items=sum(len(rows) for _,rows in cells),
                              arithmetic_items=len(arithmetic),configuration=config),ensure_ascii=False))
        return
    if args.model_path is None or args.output is None:
        ap.error('--run requires --model-path and --output')
    if args.output.exists() and not args.resume:
        ap.error('Use a new output directory; this runner never overwrites an existing experiment.')
    if args.resume and not (args.output/'run_config.json').is_file():
        ap.error('--resume requires an existing run_config.json; no unverified partial run may be reused.')
    if os.environ.get('GPU_MAX_MEM') or os.environ.get('CPU_MAX_MEM'):
        ap.error('This matched protocol does not permit CPU offloading overrides.')
    adapter_file=require_input(args.adapter/'adapter_model.safetensors', ap)
    require_input(args.adapter/'adapter_config.json', ap)
    require_input(args.model_path/'config.json', ap)
    config['adapter_sha256']=sha(adapter_file)
    config['adapter_config_sha256']=sha(args.adapter/'adapter_config.json')
    config['model_config_sha256']=sha(args.model_path/'config.json')
    config['code_sha256']={p:sha(ROOT/p) for p in ['tools/run_courtdyn_revision_controls.py','eval/run_bench.py']}
    config['runtime']={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes','accelerate']}
    config['generation_suffix']='\nGive ONLY the final answer in the required format. Do not explain.'
    config['vram_cap_fraction']=float(os.environ.get('VRAM_CAP_FRACTION','0.93'))
    protocol_path=args.adapter/'training_protocol.json'
    config['training_protocol_sha256']=sha(protocol_path) if protocol_path.exists() else None
    config['training_protocol']=read_input_json(protocol_path,ap) if protocol_path.exists() else None
    saved={}
    if args.resume:
        if read_input_json(args.output/'run_config.json',ap)!=config:
            ap.error('Resume configuration or source hashes changed; preserve the run and use a new output directory.')
        for cell,items in cells:
            saved[cell['arm']]=validate_saved_rows(args.output/(cell['arm']+'.jsonl'),items)
        if (args.output/'summary.json').exists():
            ap.error('This run already has a completion summary; verify it without rerunning inference.')
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if gpu.stdout.strip() or not memory.stdout.strip().isdigit() or int(memory.stdout.strip())>=500:
        raise RuntimeError('GPU is not idle; existing work was left untouched.')
    from eval.run_bench import GENERATION_SUFFIX, load_model
    assert GENERATION_SUFFIX==config['generation_suffix']
    if not args.resume:
        args.output.mkdir(parents=True)
        with (args.output/'run_config.json').open('x',encoding='utf8') as handle:
            handle.write(json.dumps(config,indent=2)+'\n')
    infer=load_model(str(args.model_path),load_4bit=True,adapter=str(args.adapter),max_pixels=200704,max_new_tokens=128)
    summaries={}
    for cell,items in cells:
        rows=saved.get(cell['arm'],[])
        result_path=args.output/(cell['arm']+'.jsonl')
        with result_path.open('a' if args.resume and result_path.exists() else 'x',encoding='utf8') as handle:
            for index,item in enumerate(items[len(rows):],start=len(rows)):
                paths=[str(Path(cell['image_root'])/i) for i in item['image_ids']]
                if args.frame_mode=='static4':
                    paths=[paths[0]]*len(paths)
                raw=infer(paths,item['question'])
                row=dict(index=index,category=item['category'],**item['meta'],prediction=parse(raw),raw_answer=raw)
                rows.append(row)
                handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush()
                if (index+1)%20==0:print(datetime.now(timezone.utc).isoformat(),cell['arm'],index+1,'/',len(items),flush=True)
        summaries[cell['arm']]=score_rows(rows)
    if arithmetic:
        arithmetic_path=args.output/'arithmetic_results.json'
        if args.resume and arithmetic_path.exists():
            observations=read_input_json(arithmetic_path,ap)
            assert len(observations)==len(arithmetic)
            for row,item in zip(observations,arithmetic):
                assert row['question']==item['question'] and row['target']==float(item['answer'])
                assert row['prediction']==parse(row['raw_answer'])
                assert row['correct']==(row['prediction'] is not None and abs(row['prediction']-row['target'])<0.05)
        else:
            observations=[]
            for item in arithmetic:
                raw=infer([],item['question'])
                pred=parse(raw)
                observations.append(dict(question=item['question'],target=float(item['answer']),raw_answer=raw,
                                         prediction=pred,correct=(pred is not None and abs(pred-float(item['answer']))<0.05)))
            with arithmetic_path.open('x',encoding='utf8') as handle:
                handle.write(json.dumps(observations,indent=2)+'\n')
        summaries['arithmetic']=dict(n=len(observations),correct=sum(r['correct'] for r in observations))
    with (args.output/'summary.json').open('x',encoding='utf8') as handle:
        handle.write(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',output=str(args.output),summary=summaries)))

if __name__=='__main__':
    main()
