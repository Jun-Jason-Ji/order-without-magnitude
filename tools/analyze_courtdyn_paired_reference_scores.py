"""CPU-only paired v1/v3 rescoring of the two new P2 adapters.

Primary results use exact references; one-decimal references are a separately
labelled sensitivity analysis. MAE uses strictly parsed items, with N and units.
Never substitutes original native predictions,
loads a model, modifies frozen files, or overwrites an analysis directory.
"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine import dynamics_qa as DQ
from engine.court_homography import CourtPlane,recompute_answer

FROZEN_ANALYZER=ROOT/'tools/analyze_courtdyn_revision_results.py'
FROZEN_SHA='37a2d98eefea7043fcb467aa5a223fced2c4474bc323e5f9bf5a2e561da1f6ea'
MODELS=('eventholdout_v1_s42','eventholdout_v3_s42')
SEQUENCES=('Q1_top_0-30','Q2_top_480-510')
MODES=('full','static4')
FAMILIES={'dynamics_speed_player':('speed','speed_mps',.30,1.),
          'dynamics_path_player':('path','path_m',.50,2.)}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_math(compatibility_analysis=None):
    require(sha(FROZEN_ANALYZER)==FROZEN_SHA,'Frozen analyzer changed; review its new semantics before importing.')
    if compatibility_analysis is not None:
        from tools import courtdyn_adapter_order_compat as compat
        from tools import finalize_ivc_revision_execution as final
        plan,state,actual,_=final.verify_completed_state(final.EXECUTION)
        require(Path(sys.executable).resolve()==Path(plan['python']).resolve(),
                'Use the frozen research Python for exact compatible rescoring')
        require(state['status']=='FAILED' and Path(compatibility_analysis).resolve()==actual.resolve(),
                'Explicit compatibility analysis must be the certified correction of the preserved FAILED pipeline')
        receipt=compat.verify_receipt(actual)
        module=compat.load_compatible_analyzer()
        module.analysis_correction=dict(rule=receipt['evidence']['rule'],
            completion_receipt_sha256=sha(final.EXECUTION/'analysis_correction_completion.json'),
            compatibility_receipt_sha256=sha(actual.parent/'compatibility.json'),
            compatibility_tool_sha256=receipt['evidence']['compatibility_tool_sha256'],
            original_pipeline_state_sha256=sha(final.EXECUTION/'state.json'),
            original_failure_analysis_sha256=receipt['original_failure']['sha256'],
            corrected_analysis_sha256=sha(actual))
        return module
    spec=importlib.util.spec_from_file_location('courtdyn_reference_frozen_math',FROZEN_ANALYZER)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M=load_math()


def identity(row):
    return row['category'],row['meta']['track'],tuple(row['meta']['window'])


def family_score(keys,observations,reference,tolerance,floor,mae_unit):
    """Missing outputs remain missing; observed invalid parses score zero."""
    observed=[k for k in keys if k in observations]
    parsed=[k for k in observed if observations[k] is not None]
    scores=[M.score(observations[k],reference[k],tolerance,floor) for k in parsed]
    errors=[abs(observations[k]-reference[k]) for k in parsed]
    total=sum(scores)
    return dict(n_expected=len(keys),n_observed=len(observed),n_parsed=len(parsed),
                n_invalid_parse=len(observed)-len(parsed),n_missing=len(keys)-len(observed),
                coverage=len(observed)/len(keys) if keys else None,
                parse_rate_observed=len(parsed)/len(observed) if observed else None,
                tmra_all_items=total/len(keys) if keys and len(observed)==len(keys) else None,
                tmra_all_observed=total/len(observed) if observed else None,
                tmra_parsed=total/len(parsed) if parsed else None,
                mae_parsed=statistics.mean(errors) if errors else None,
                mae_n=len(parsed),mae_unit=mae_unit,
                spearman=M.spearman([observations[k] for k in parsed],[reference[k] for k in parsed]),
                reference_median=M.distribution([reference[k] for k in keys])['median'],
                score_tolerance=tolerance,score_floor=floor)


def paired_score(keys,observations_a,observations_b,reference,tolerance,floor):
    observed=[k for k in keys if k in observations_a and k in observations_b]
    parsed=[k for k in observed if observations_a[k] is not None and observations_b[k] is not None]
    def score_on(observations,cohort):
        return sum(M.score(observations[k],reference[k],tolerance,floor) for k in cohort)/len(cohort) if cohort else None
    a,b=score_on(observations_a,observed),score_on(observations_b,observed)
    ap,bp=score_on(observations_a,parsed),score_on(observations_b,parsed)
    ar=M.spearman([observations_a[k] for k in parsed],[reference[k] for k in parsed])
    br=M.spearman([observations_b[k] for k in parsed],[reference[k] for k in parsed])
    return dict(n_expected=len(keys),n_common_observed=len(observed),n_common_parsed=len(parsed),
                all_common_observed_tmra=[a,b],delta_all_common_observed_tmra=None if a is None else b-a,
                common_parsed_tmra=[ap,bp],delta_common_parsed_tmra=None if ap is None else bp-ap,
                common_parsed_spearman=[ar,br],delta_common_parsed_spearman=None if ar is None or br is None else br-ar)


def paired_reference_delta(keys,observations,v1,v3,tolerance,floor):
    """Same predictions and same cohort; delta is reference v3 minus v1."""
    observed=[k for k in keys if k in observations]
    parsed=[k for k in observed if observations[k] is not None]
    def mean(reference,cohort):
        return sum(M.score(observations[k],reference[k],tolerance,floor) for k in cohort)/len(cohort) if cohort else None
    a,b=mean(v1,observed),mean(v3,observed)
    ap,bp=mean(v1,parsed),mean(v3,parsed)
    ar=M.spearman([observations[k] for k in parsed],[v1[k] for k in parsed])
    br=M.spearman([observations[k] for k in parsed],[v3[k] for k in parsed])
    return dict(n_expected=len(keys),n_observed=len(observed),n_parsed=len(parsed),
                delta_all_items_tmra=b-a if a is not None and len(observed)==len(keys) else None,
                delta_all_observed_tmra=None if a is None else b-a,
                delta_parsed_tmra=None if ap is None else bp-ap,
                delta_spearman=None if ar is None or br is None else br-ar,
                direction='reference v3 minus reference v1, with predictions held identical')


class References:
    def __init__(self,auditor):
        self.auditor=auditor
        self.items={};self.values={};self.validation=[]

    def build(self):
        for seq in SEQUENCES:
            original=self.auditor.read(ROOT/'results/courtdyn'/('seq_'+seq)/'qa_dyn_v1.json')
            original=[r for r in original if r['category'] in FAMILIES]
            old={identity(r):r for r in original}
            prepared=self.auditor.pools[(seq,'m_height')]
            new={identity(r):r for r in prepared}
            require(len(old)==len(original)==len(new)==280 and old.keys()==new.keys(),
                    'Original v1 and prepared m_height cohorts differ')
            self.items[seq]=new
            seqdir=ROOT/'data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot/basketball_top/train'/seq
            tracks=DQ.load_tracks(str(seqdir));info=DQ.load_seqinfo(str(seqdir));plane=CourtPlane.load(seq)
            for path in (seqdir/'gt/gt.txt',seqdir/'seqinfo.ini',ROOT/'results/courtdyn/homography'/f'H_{seq}.json'):
                self.auditor.hashes[str(path)]=sha(path)
            values={name:{} for name in ('exact_v1','exact_v3','rounded_v1','rounded_v3')}
            for category,(family,field,tolerance,floor) in FAMILIES.items():
                count=0;v1_error=0.;v3_error=0.;rounded_matches=0
                for key,row in new.items():
                    if key[0]!=category:
                        continue
                    previous=old[key];m=previous['meta'];f0,f1=m['window']
                    require(previous['image_ids']==row['image_ids'] and row['meta']['fps']==m['fps']==info['fps'],
                            'v1/v3 presented image identity or fps differs')
                    require(row['meta']['unit']=='m' and row['meta']['score_tolerance']==tolerance and
                            row['meta']['score_floor']==floor,'m_height scoring convention differs')
                    require(field in m and M.finite(m[field]) and m[field]>=0,'Exact v1 meta reference unavailable')
                    exact_v1=float(m[field])
                    require(f'{exact_v1:.1f}'==previous['answer'],'Exact v1 does not serialize to original answer')
                    check_v1=(DQ.mean_speed(tracks[m['track']],f0,f1,info['fps']) if family=='speed'
                              else DQ.path_m(tracks[m['track']],f0,f1))
                    require(check_v1 is not None,'v1 cannot be recomputed from original MOT')
                    computed_v3=recompute_answer(plane,previous,tracks,info['fps'])
                    require(computed_v3 is not None,'v3 cannot be recomputed from original MOT')
                    exact_v3=float(row['meta']['reference_value'])
                    e1,e3=abs(exact_v1-check_v1),abs(exact_v3-computed_v3[1][field])
                    require(e1<1e-12 and e3<1e-12,'Exact reference differs from its independent geometric recomputation')
                    require(computed_v3[0]==row['answer'] and f'{exact_v3:.1f}'==row['answer'],
                            'Prepared rounded v3 differs from geometric reference')
                    values['exact_v1'][key]=exact_v1;values['exact_v3'][key]=exact_v3
                    values['rounded_v1'][key]=float(previous['answer']);values['rounded_v3'][key]=float(row['answer'])
                    count+=1;rounded_matches+=1;v1_error=max(v1_error,e1);v3_error=max(v3_error,e3)
                require(count==140,'Unexpected family size')
                self.validation.append(dict(seq=seq,family=family,n=count,
                    all_presented_images_and_fps_identical=True,v1_meta_rounds_to_original_answer=rounded_matches,
                    maximum_v1_meta_recompute_error=v1_error,maximum_v3_prepared_recompute_error=v3_error,
                    exact_v1_basis='original meta.'+field,rounded_fallback_used=False))
            self.values[seq]=values


class PublicPaths:
    def __init__(self,backbone=None):
        self.backbone=Path(backbone).resolve() if backbone else None

    def path(self,value):
        path=Path(value).resolve()
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            if self.backbone:
                try:
                    return '$BACKBONE/'+path.relative_to(self.backbone).as_posix()
                except ValueError:
                    pass
        return '$EXTERNAL/'+path.name

    def text(self,value):
        result=str(value)
        roots=[(str(ROOT),'$REPOSITORY')]
        if self.backbone:
            roots.append((str(self.backbone),'$BACKBONE'))
        for source,alias in sorted(roots,key=lambda p:len(p[0]),reverse=True):
            for spelling in (source,source.replace('\\','/')):
                result=result.replace(spelling,alias)
        return re.sub(r'[A-Za-z]:[\\/][^\s,;"<>]*','[external path]',result)


def public_provenance(auditor,paths):
    sources={paths.path(path):digest for path,digest in auditor.hashes.items()}
    require(len(sources)==len(auditor.hashes),'Two input files collapse to one public provenance alias')
    trainings={}
    protocol_fields=('schema_version','selected_samples','qa_source_sha256','ordered_sample_sequence_sha256',
                     'allow_drop_invalid','model_family','processor_policy','seed','num_samples_request',
                     'max_steps','num_train_epochs','learning_rate','gradient_accumulation_steps',
                     'max_pixels','vision_lora','allow_base_init','budget_slice',
                     'initialization_seed_applied_before_model_load')
    for label,row in auditor.training_records.items():
        protocol=row['protocol'];completion=row['completion'];adapter=row['actual_adapter_config']
        trainings[label]=dict(verified=row['verified'],training_convention=row['training_convention'],
            protocol_sha256=row['protocol_sha256'],completion_sha256=row['completion_sha256'],
            protocol={k:protocol[k] for k in protocol_fields if k in protocol},
            completion={k:completion[k] for k in ('completed','seed','global_step','selected_samples','final_adapter_files')},
            actual_lora_config={k:adapter[k] for k in ('r','lora_alpha','lora_dropout','target_modules')})
    runs=[]
    for run in auditor.runs:
        cfg=run['config']
        runs.append(dict(label=run['label'],seq=run['seq'],frame_mode=run['frame_mode'],
                         directory=paths.path(run['directory']),
                         run_config_sha256=auditor.hashes[str(Path(run['directory'])/'run_config.json')],
                         adapter_sha256=cfg['adapter_sha256'],adapter_config_sha256=cfg.get('adapter_config_sha256'),
                         model_config_sha256=cfg['model_config_sha256'],
                         training_protocol_sha256=cfg.get('training_protocol_sha256'),
                         max_pixels=cfg['max_pixels'],max_new_tokens=cfg['max_new_tokens'],load_4bit=cfg['load_4bit'],
                         decoding=cfg['decoding'],strict_parse=cfg['strict_parse'],runtime=cfg.get('runtime')))
    return dict(source_sha256=sources,training_receipts=trainings,run_identity=runs)


def observations_for(run):
    if run is None or 'm_height' not in run['cells']:
        return {}
    return {key:row['prediction'] for key,row in run['cells']['m_height']['observations'].items()}


def verify_completion_summary(run,summary):
    expected_names=set(run['cells']) | ({'arithmetic'} if run['config'].get('arithmetic') else set())
    require(set(summary)==expected_names,'Evaluation completion summary has different configured arms')
    for arm,cell in run['cells'].items():
        require(len(cell['observations'])==len(cell['items']),'Completion summary exists for an incomplete visual arm')
        require(set(summary[arm])==set(FAMILIES),'Completion summary family set differs')
        for category in FAMILIES:
            rows=[r for r in cell['observations'].values() if r['category']==category]
            valid=[r for r in rows if r['prediction'] is not None]
            total=sum(M.score(r['prediction'],r['reference_value'],r['score_tolerance'],r['score_floor']) for r in valid)
            expected=dict(n=len(rows),parsed=len(valid),parse_rate=len(valid)/len(rows),
                          tmra_all_items=total/len(rows),tmra_parsed=total/len(valid) if valid else None)
            require(set(summary[arm][category])==set(expected),'Completion summary score schema differs')
            for field,value in expected.items():
                actual=summary[arm][category][field]
                require(actual==value if value is None else
                        M.finite(actual) and math.isclose(actual,value,rel_tol=0,abs_tol=1e-10),
                        'Completion summary score differs: '+field)
    if run['config'].get('arithmetic'):
        arithmetic=run['arithmetic']
        require(arithmetic and arithmetic['n_observed']==arithmetic['n_expected'],
                'Completion summary exists with incomplete arithmetic')
        require(summary['arithmetic']==dict(n=arithmetic['n_observed'],correct=arithmetic['n_correct']),
                'Arithmetic completion summary differs')


def rescore(auditor,references):
    lookup={(r['label'],r['seq'],r['frame_mode']):r for r in auditor.runs}
    layers={'primary_exact':('exact_v1','exact_v3'),'rounded_historical_sensitivity':('rounded_v1','rounded_v3')}
    result={}
    for layer,bases in layers.items():
        cells=[];deltas=[];model_pairs=[];temporal_pairs=[]
        for seq in SEQUENCES:
            for category,(family,field,tolerance,floor) in FAMILIES.items():
                keys=sorted((k for k in references.items[seq] if k[0]==category),key=repr)
                values=references.values[seq]
                for label in MODELS:
                    for mode in MODES:
                        observations=observations_for(lookup.get((label,seq,mode)))
                        for basis in bases:
                            row=family_score(keys,observations,values[basis],tolerance,floor,
                                             'm/s' if family=='speed' else 'm')
                            row.update(model=label,seq=seq,frame_mode=mode,family=family,reference_basis=basis)
                            cells.append(row)
                        delta=paired_reference_delta(keys,observations,values[bases[0]],values[bases[1]],tolerance,floor)
                        delta.update(model=label,seq=seq,frame_mode=mode,family=family)
                        deltas.append(delta)
                    full=observations_for(lookup.get((label,seq,'full')))
                    static=observations_for(lookup.get((label,seq,'static4')))
                    for basis in bases:
                        row=paired_score(keys,full,static,values[basis],tolerance,floor)
                        row.update(model=label,seq=seq,family=family,reference_basis=basis,
                                   direction='static4 minus matching new full')
                        temporal_pairs.append(row)
                for mode in MODES:
                    a=observations_for(lookup.get((MODELS[0],seq,mode)))
                    b=observations_for(lookup.get((MODELS[1],seq,mode)))
                    for basis in bases:
                        row=paired_score(keys,a,b,values[basis],tolerance,floor)
                        row.update(models=list(MODELS),seq=seq,frame_mode=mode,family=family,reference_basis=basis,
                                   direction='v3-trained adapter minus v1-trained adapter')
                        model_pairs.append(row)
        require(len(cells)==32,'Unexpected reference-score grid')
        result[layer]=dict(cells=cells,within_prediction_reference_deltas=deltas,
                           paired_models=model_pairs,paired_temporal_controls=temporal_pairs)
    return result


class Tests(unittest.TestCase):
    def test_missing_is_not_parse_failure(self):
        keys=[1,2,3];pred={1:1.,2:None};ref={1:1.,2:2.,3:3.}
        d=family_score(keys,pred,ref,.3,1,'m/s')
        self.assertEqual(d['n_missing'],1);self.assertEqual(d['n_invalid_parse'],1)
        self.assertIsNone(d['tmra_all_items']);self.assertEqual(d['tmra_all_observed'],50.)
        pred[3]=None
        self.assertAlmostEqual(family_score(keys,pred,ref,.3,1,'m/s')['tmra_all_items'],100/3)

    def test_all_invalid_complete_is_zero(self):
        d=family_score([1,2,3],{1:None,2:None,3:None},{1:1.,2:2.,3:3.},.3,1,'m/s')
        self.assertEqual(d['tmra_all_items'],0);self.assertEqual(d['n_missing'],0)
        self.assertIsNone(d['tmra_parsed']);self.assertIsNone(d['spearman'])
        self.assertIsNone(d['mae_parsed']);self.assertEqual(d['mae_n'],0)

    def test_mae_uses_same_parsed_cohort_without_zero_imputation(self):
        d=family_score([1,2,3,4],{1:2.,2:None,3:-1.},{1:1.,2:5.,3:3.,4:6.},.5,2,'m')
        self.assertEqual(d['mae_n'],d['n_parsed'])
        self.assertEqual(d['mae_n'],2)
        self.assertEqual(d['mae_parsed'],2.5)  # abs(2-1), abs(-1-3)
        self.assertEqual(d['mae_unit'],'m')
        self.assertEqual(d['n_invalid_parse'],1);self.assertEqual(d['n_missing'],1)
        self.assertIsNone(d['tmra_all_items'])

    def test_mae_distinguishes_tolerance_and_reference_rounding(self):
        d=family_score([1],{1:1.2},{1:1.},.3,1,'m/s')
        self.assertEqual(d['tmra_all_items'],100.)
        self.assertAlmostEqual(d['mae_parsed'],.2)
        exact=family_score([1],{1:1.},{1:1.04},.3,1,'m/s')
        rounded=family_score([1],{1:1.},{1:1.},.3,1,'m/s')
        self.assertAlmostEqual(exact['mae_parsed'],.04)
        self.assertEqual(rounded['mae_parsed'],0.)

    def test_reference_delta_holds_predictions_fixed(self):
        d=paired_reference_delta([1,2,3],{1:1.,2:2.,3:3.},
                                {1:1.,2:2.,3:3.},{1:10.,2:20.,3:30.},.3,1)
        self.assertEqual(d['delta_all_items_tmra'],-100)
        self.assertAlmostEqual(d['delta_spearman'],0)

    def test_paired_missing_and_invalid_denominators(self):
        d=paired_score([1,2,3],{1:1.,2:None},{1:1.,2:2.,3:3.},{1:1.,2:2.,3:3.},.3,1)
        self.assertEqual(d['n_common_observed'],2);self.assertEqual(d['n_common_parsed'],1)
        self.assertEqual(d['delta_all_common_observed_tmra'],50)
        self.assertEqual(d['delta_common_parsed_tmra'],0)

    def test_strict_parser(self):
        self.assertEqual(M.parse('1.0'),1)
        self.assertIsNone(M.parse('1.0 m'));self.assertIsNone(M.parse('NaN'))

    def test_public_paths(self):
        p=PublicPaths(Path('E:/models/example'))
        self.assertEqual(p.path(ROOT/'tools/a.py'),'tools/a.py')
        self.assertNotRegex(p.text(str(ROOT/'data/test.json')),r'[A-Za-z]:[\\/]')
        self.assertNotRegex(p.text('unrecognized Z:/private/file.json'),r'[A-Za-z]:[\\/]')

    def test_evaluation_completion_scores(self):
        observations={};summary={'m_height':{}}
        for index,(category,(_,_,tolerance,floor)) in enumerate(FAMILIES.items()):
            observations[index]=dict(category=category,prediction=1.,reference_value=1.,
                                     score_tolerance=tolerance,score_floor=floor)
            summary['m_height'][category]=dict(n=1,parsed=1,parse_rate=1.,tmra_all_items=100.,tmra_parsed=100.)
        run=dict(config=dict(arithmetic=False),cells={'m_height':dict(items={0:{},1:{}},observations=observations)})
        verify_completion_summary(run,summary)
        summary['m_height']['dynamics_speed_player']['tmra_all_items']=99.
        with self.assertRaisesRegex(ValueError,'Completion summary score differs'):
            verify_completion_summary(run,summary)


def main():
    global M
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,default=ROOT/'results/courtdyn/revision_controls_20260912')
    ap.add_argument('--plan',type=Path,default=ROOT/'results/courtdyn/revision_execution_20260912/plan.json')
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--restricted-provenance',action='store_true',help='Also write absolute local provenance separately; never share that file.')
    ap.add_argument('--compatibility-analysis',type=Path,
                    help='Explicit certified full analysis.json with the audited target_modules-order sidecar')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if r.wasSuccessful() else 1
    if not args.output_dir:
        ap.error('--output-dir is required')
    if args.output_dir.exists():
        ap.error('Use a new output directory; prior analysis is never overwritten.')
    auditor=None;paths=PublicPaths()
    report=dict(schema='courtdyn-paired-reference-scores-v1',status='FAIL',
                generated=datetime.now().isoformat(timespec='seconds'),script_sha256=sha(__file__),
                frozen_analyzer_sha256=FROZEN_SHA,errors=[],missing=[],warnings=[],
                primary_basis='Exact v1 from original QA meta and exact v3 from prepared inputs; both independently recomputed.',
                sensitivity_basis='Original stored one-decimal v1 and stored one-decimal v3; separate historical serialization sensitivity.',
                score='0–100 T-MRA with original strict inequalities. Metre units: speed T=.30/floor=1; path T=.50/floor=2.',
                missing_policy='Observed invalid parses score zero in all-item score. Unobserved items remain missing; all-item score is null until complete.',
                mae_policy='Mean absolute error on the same strictly parsed cohort as parsed T-MRA/Spearman; invalid and missing outputs are excluded, never imputed zero. mae_n and mae_unit are explicit; speed m/s, path m.',
                limitations=['CPU-only rescoring; no new model training or inference.',
                    'Same predictions are scored against two label conventions. Changing reference is not an independent replication.',
                    'Windows and players are dependent. Q1/Q2 belong to the same game, not cross-game validation.',
                    'Behavioral/reference differences cannot identify an internal visual measurement mechanism.',
                    'Full/static4 and v1/v3 model comparisons use only matching new P2 outputs, never original native full.',
                    'Public aggregate has no per-item identifiers or local absolute paths. Optional restricted provenance must remain local.'])
    try:
        if args.compatibility_analysis:
            M=load_math(args.compatibility_analysis)
            report['analysis_correction']=M.analysis_correction
        auditor=M.Auditor(args.inputs,args.plan)
        paths=PublicPaths(auditor.plan.get('model_path'))
        selected=[r for r in auditor.plan['runs'] if r['label'] in MODELS]
        training=[s for s in auditor.plan.get('stages',[]) if s['kind']=='training']
        require(len(training)==2 and {s['label'] for s in training}==set(MODELS) and
                {s['label']:s['training_convention'] for s in training}==dict(zip(MODELS,('v1','v3'))),
                'Execution plan must include both required P2 training stages and their label conventions')
        expected={(label,seq,mode) for label in MODELS for seq in SEQUENCES for mode in MODES}
        require({(r['label'],r['seq'],r['frame_mode']) for r in selected}==expected and len(selected)==8,
                'Execution plan must contain exactly the eight required P2 evaluation stages')
        auditor.plan['runs']=selected
        auditor.load_inputs()
        references=References(auditor);references.build()
        for entry in selected:
            require('m_height' in entry['expected_arms'],'Required m_height arm absent from plan')
            auditor.load_run(entry['label'],Path(entry['output']))
        for run in auditor.runs:
            summary_path=Path(run['directory'])/'summary.json'
            if not summary_path.is_file():
                auditor.missing.append(f'Evaluation completion summary missing: {run["label"]}/{run["seq"]}/{run["frame_mode"]}')
            else:
                verify_completion_summary(run,auditor.read(summary_path))
        auditor.check_plan_complete()
        # Reuse strict paired-model and temporal configuration gates, not their
        # original score summaries. These also verify training receipts.
        auditor.compare_models(*MODELS)
        auditor.temporal_controls()
        require(set(auditor.training_records)<=set(MODELS),'Unplanned training label in P2 provenance')
        report['reference_mapping']=references.validation
        report['training_pool_audit']=auditor.training_pool_audit()
        report.update(rescore(auditor,references))
        report['coverage']=dict(expected_evaluation_runs=8,observed_evaluation_runs=len(auditor.runs),
            expected_primary_score_cells=32,expected_rounded_sensitivity_cells=32,
            expected_m_height_predictions=2240,
            observed_m_height_predictions=sum(len(r['cells'].get('m_height',{}).get('observations',{})) for r in auditor.runs))
        report['status']='PARTIAL' if auditor.missing else 'PASS'
    except (ValueError,KeyError,OSError,TypeError,OverflowError) as error:
        report['errors'].append(paths.text(f'{type(error).__name__}: {error}'))
    if auditor:
        report['missing']=[paths.text(m) for m in auditor.missing]
        report['warnings']=[paths.text(m) for m in auditor.warnings]
        report.update(public_provenance(auditor,paths))
    public_text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    require(re.search(r'[A-Za-z]:[\\/]',public_text) is None,'Absolute local path escaped public sanitization')
    require('image_ids' not in public_text and 'window_index' not in public_text,
            'Per-item identifiers escaped public report schema')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(public_text,encoding='utf8')
    if args.restricted_provenance and auditor:
        restricted=dict(restriction='LOCAL ONLY: absolute file paths; exclude from public package.',
                        source_sha256=auditor.hashes,training_records=auditor.training_records)
        (args.output_dir/'restricted_local_provenance.json').write_text(json.dumps(restricted,indent=2)+'\n',encoding='utf8')
    lines=['# P2 同预测双参考重评分','',report['status'],'',
           '主表采用 exact v1/v3；一位小数双参考敏感性独立列在 JSON。未完成的题目不补零成完整结果。MAE 使用同一严格解析成功集合，单列 N 与单位。',
           '', '| 模型 | 序列/帧条件 | Family | 参考 | 观察/解析/应有N | all-item T-MRA | parsed T-MRA | Spearman | MAE（单位；N） |',
           '|---|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(v):
        return 'NA' if v is None else f'{v:.4f}'
    for r in report.get('primary_exact',{}).get('cells',[]):
        lines.append(f'| {r["model"]} | {r["seq"]}/{r["frame_mode"]} | {r["family"]} | {r["reference_basis"]} | '
                     f'{r["n_observed"]}/{r["n_parsed"]}/{r["n_expected"]} | {fmt(r["tmra_all_items"])} | '
                     f'{fmt(r["tmra_parsed"])} | {fmt(r["spearman"])} | {fmt(r["mae_parsed"])} ({r["mae_unit"]}; {r["mae_n"]}) |')
    lines+=['','所有分数使用同一预测分别与两参考比较；解析失败在完整 all-item 分数中记零。'
            'MAE 不将无效解析或缺失输出当作零误差；MAE 的样本数与 parsed T-MRA/Spearman 相同。'
            '逐项配对差、共同观察/共同可解析队列、full/static4及跨模型差另见 JSON。'
            '这些仍是同一比赛内的依赖数据，不能据此识别内部视觉机制。']
    if report['errors'] or report['missing']:
        lines+=['','错误或待完成项：']+['- '+m for m in report['errors']+report['missing']]
    (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status=report['status'],errors=report['errors'],missing_count=len(report['missing']),
                          files=['analysis.json','summary.md']),ensure_ascii=False))
    return {'PASS':0,'FAIL':1,'PARTIAL':2}[report['status']]


if __name__=='__main__':
    sys.exit(main())
