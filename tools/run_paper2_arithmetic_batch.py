"""Finite serial arithmetic probe batch; no polling, scheduler, resume or retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.paper2_local_model_smoke import ALLOWED_ADAPTERS, sha256_file, write_new_json

PYTHON=Path(sys.executable)   # the interpreter running this script (was a hard-coded user path)
JOBS=(('base',None),('v1_s42',ALLOWED_ADAPTERS[0]),('v3_s42',ALLOWED_ADAPTERS[1]))
CODE=('tools/run_paper2_arithmetic_batch.py','tools/paper2_arithmetic_model_probe.py',
      'tools/paper2_local_model_smoke.py','tools/paper2_gpu_idle.py','tools/queue_guard.py',
      'a100/eval_controls.py','a100/model_families.py','eval/run_bench.py')


def utc(): return datetime.now(timezone.utc).isoformat()


def save_state(path,state):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)


def successful_attempt(returncode,attempt,expected_ids,prepared_sha,expected_adapter):
    receipt_path=attempt/'receipt.json'
    if not receipt_path.is_file(): return False,{'reason':'missing_receipt','exit_code':returncode}
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    passed=(returncode==0 and receipt.get('status')=='passed' and receipt.get('model_loaded') is True
            and receipt.get('predictions_written')==100 and (attempt/'summary.json').is_file())
    if not passed: return False,receipt
    try:
        summary=json.loads((attempt/'summary.json').read_text(encoding='utf-8'))
        path=attempt/'predictions.jsonl'; raw=path.read_text(encoding='utf-8')
        predictions=[json.loads(line) for line in raw.splitlines()]
        ids=[row['question_id'] for row in predictions]
        adapter=receipt.get('adapter')
        observed_adapter=adapter.get('adapter_path') if isinstance(adapter,dict) else None
        passed=(raw.endswith('\n') and len(ids)==100 and len(set(ids))==100 and set(ids)==set(expected_ids)
                and all(isinstance(p['raw'],str) and p['image_count']==0 and p['private_reference_parsed'] is False for p in predictions)
                and summary['complete'] is True and summary['predictions_written']==100 and summary['overall']['n']==100
                and summary['role']=='synthetic_text_arithmetic_development_only'
                and receipt['prepared_manifest_sha256']==prepared_sha and observed_adapter==expected_adapter
                and receipt['predictions_sha256']==sha256_file(path)==summary['predictions_sha256'])
    except (OSError,ValueError,KeyError,TypeError):
        passed=False
    return passed,receipt


def stop_owned_child(child):
    """Stop only the Popen object created by this finite batch after an error."""
    result={'pid':child.pid,'termination_requested':False,'kill_required':False}
    if child.poll() is None:
        result['termination_requested']=True; child.terminate()
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            result['kill_required']=True; child.kill(); child.wait(timeout=10)
    result['exit_code']=child.poll()
    return result


def run(prepared,output):
    if output.exists() or output.resolve().is_relative_to(prepared.resolve()): raise ValueError('Fresh batch output outside prepared input required')
    if not PYTHON.is_file() or not prepared.is_dir(): raise ValueError('Fixed interpreter and prepared inputs required')
    output.mkdir(parents=True)
    code_hashes={name:sha256_file(ROOT/name) for name in CODE}
    input_hashes={p.relative_to(prepared).as_posix():sha256_file(p) for p in sorted(prepared.rglob('*')) if p.is_file()}
    questions=[json.loads(line) for line in (prepared/'public/questions.jsonl').read_text(encoding='utf-8').splitlines()]
    expected_ids=[q['question_id'] for q in questions]
    if len(expected_ids)!=100 or len(set(expected_ids))!=100: raise ValueError('Exactly100 unique prepared questions required')
    prepared_sha=sha256_file(prepared/'manifest.json')
    protocol={'role':'synthetic_text_arithmetic_development_only','created_utc':utc(),'python':str(PYTHON),
              'jobs':[{'id':name,'adapter':str(adapter) if adapter else None} for name,adapter in JOBS],
              'code_sha256':code_hashes,'prepared_sha256':input_hashes,
              'no_retry':True,'no_resume':True,'strict_idle_gate_in_each_child':True,'automatic_startup_configured':False}
    write_new_json(output/'protocol.json',protocol)
    state={'status':'running','started_utc':utc(),'jobs':[{'id':name,'status':'pending'} for name,_ in JOBS]}
    save_state(output/'state.json',state)
    child=None; index=None
    try:
        for index,(name,adapter) in enumerate(JOBS):
            for relative,h in code_hashes.items():
                if sha256_file(ROOT/relative)!=h: raise ValueError('Source changed during finite batch: '+relative)
            current={p.relative_to(prepared).as_posix():sha256_file(p) for p in sorted(prepared.rglob('*')) if p.is_file()}
            if current!=input_hashes: raise ValueError('Prepared input changed during batch')
            attempt=output/name
            command=[str(PYTHON),'-m','tools.paper2_arithmetic_model_probe','--run','--prepared',str(prepared.resolve()),
                     '--adapter',str(adapter) if adapter else 'base','--output',str(attempt.resolve())]
            state['jobs'][index].update(status='running',started_utc=utc(),output=str(attempt.resolve()))
            save_state(output/'state.json',state)
            print(json.dumps({'event':'starting','job':name,'output':str(attempt)}),flush=True)
            env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            started=time.perf_counter()
            with (output/(name+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                state['jobs'][index]['pid']=child.pid; save_state(output/'state.json',state)
                code=child.wait()
            passed,receipt=successful_attempt(code,attempt,expected_ids,prepared_sha,str(adapter.resolve()) if adapter else None)
            state['jobs'][index].update(status='passed' if passed else 'failed',finished_utc=utc(),exit_code=code,
                                       wall_seconds=time.perf_counter()-started,model_loaded=receipt.get('model_loaded'),
                                       predictions_written=receipt.get('predictions_written',0))
            if (attempt/'receipt.json').is_file(): state['jobs'][index]['receipt_sha256']=sha256_file(attempt/'receipt.json')
            save_state(output/'state.json',state)
            print(json.dumps({'event':'completed' if passed else 'failed','job':name,'predictions':receipt.get('predictions_written',0)}),flush=True)
            if not passed: raise RuntimeError(f'{name} did not complete; no retry or next model started')
        state.update(status='complete',finished_utc=utc()); save_state(output/'state.json',state)
        write_new_json(output/'summary.json',{'role':protocol['role'],'arms':{
            name:json.loads((output/name/'summary.json').read_text(encoding='utf-8')) for name,_ in JOBS}})
        print(json.dumps({'status':'complete','summary':str(output/'summary.json')}),flush=True)
        return 0
    except BaseException as exc:
        if child is not None:
            try: state['owned_child_cleanup']=stop_owned_child(child)
            except Exception as cleanup_error:
                state['owned_child_cleanup']={'pid':child.pid,'error':str(cleanup_error),'exit_code':child.poll()}
        if index is not None and state['jobs'][index]['status']=='running':
            state['jobs'][index].update(status='failed',finished_utc=utc(),reason='batch_interrupted_or_validation_failed')
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_utc=utc())
        save_state(output/'state.json',state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.exit(run(args.prepared,args.output))
