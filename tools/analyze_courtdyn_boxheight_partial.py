"""Post-hoc CPU-only box-height partial-rank diagnostic of frozen native results.

Writes aggregate.json and findings.md suitable for review, plus an explicitly
restricted local item mapping. No model loading, inference, or frozen-file edits.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import unittest

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine import dynamics_qa as DQ
from engine.court_homography import CourtPlane,recompute_answer

SEQUENCES=('Q1_top_0-30','Q2_top_480-510')
FAMILIES={'dynamics_speed_player':'speed','dynamics_path_player':'path'}
IMAGE_NAME=re.compile(r'f(\d+)_r(-?\d+)\.jpg')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def key(row):
    m=row['meta']
    return row['category'],m['window_index'],m['track']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tie_ranks(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    result=np.empty(len(values),dtype=float)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:
            j+=1
        result[order[i:j]]=(i+j-1)/2+1
        i=j
    return result


def pearson(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if len(x)<3:
        return None
    x,y=x-x.mean(),y-y.mean()
    denominator=np.linalg.norm(x)*np.linalg.norm(y)
    if denominator<1e-15:
        return None
    return float(np.clip(np.dot(x,y)/denominator,-1,1))


def partial_rank(prediction,reference,height):
    require(len(prediction)==len(reference)==len(height),'Rank cohorts differ')
    require(all(math.isfinite(v) for a in (prediction,reference,height) for v in a),'Nonfinite rank input')
    if len(height)<3:
        return dict(n=len(height),plain_rho=None,prediction_height_rho=None,reference_height_rho=None,
                    partial_rho=None,undefined_reason='fewer than three complete items')
    pr,gr,hr=(tie_ranks(v) for v in (prediction,reference,height))
    design=np.column_stack((np.ones(len(hr)),hr))
    beta_p=np.linalg.lstsq(design,pr,rcond=None)[0]
    beta_g=np.linalg.lstsq(design,gr,rcond=None)[0]
    residual_p,residual_g=pr-design@beta_p,gr-design@beta_g
    # Exact rank collinearity can leave floating-point residuals; do not turn
    # numerical roundoff into a spurious correlation.
    zero_p=np.linalg.norm(residual_p)<=1e-10*max(np.linalg.norm(pr-pr.mean()),1)
    zero_g=np.linalg.norm(residual_g)<=1e-10*max(np.linalg.norm(gr-gr.mean()),1)
    value=None if zero_p or zero_g else pearson(residual_p,residual_g)
    plain,ph,gh=pearson(pr,gr),pearson(pr,hr),pearson(gr,hr)
    formula=None
    if ph is not None and gh is not None and (1-ph*ph)*(1-gh*gh)>1e-12:
        formula=(plain-ph*gh)/math.sqrt((1-ph*ph)*(1-gh*gh))
        require(value is not None and abs(value-formula)<1e-9,'OLS and one-control correlation identity disagree')
    return dict(n=len(height),plain_rho=plain,prediction_height_rho=ph,reference_height_rho=gh,
                partial_rho=value,closed_form_check=formula,
                undefined_reason='zero residual rank variance' if zero_p or zero_g else None,
                covariate_distinct_values=len(set(height)),
                covariate_constant=len(set(height))==1,
                design_matrix_rank=int(np.linalg.matrix_rank(design)),
                residual_sum_squares_prediction=float(np.dot(residual_p,residual_p)),
                residual_sum_squares_reference=float(np.dot(residual_g,residual_g)))


class MappingUnavailable(Exception):
    pass


def presented_heights(item,boxes,scale):
    """Use exactly the four named presented frames; no nearest-frame fallback."""
    m=item['meta'];frames=m['frames'];names=item['image_ids'];track=m['track']
    if len(frames)!=4 or len(names)!=4 or frames!=sorted(frames) or len(set(frames))!=4:
        raise MappingUnavailable('not four ordered distinct frame identifiers')
    if frames[0]!=m['window'][0] or frames[-1]!=m['window'][1]:
        raise MappingUnavailable('presented frame endpoints do not match window')
    heights=[]
    for frame,name in zip(frames,names):
        match=IMAGE_NAME.fullmatch(name)
        if not match or int(match[1])!=frame or int(match[2])!=track:
            raise MappingUnavailable('image filename and frame/track metadata differ')
        if (track,frame) not in boxes:
            raise MappingUnavailable('exact MOT frame/track row missing')
        box=boxes[(track,frame)]
        if not all(math.isfinite(v) for v in box) or box[3]<=0:
            raise MappingUnavailable('nonfinite/nonpositive MOT box height')
        heights.append(box[3]*scale)
    return heights


class Audit:
    def __init__(self):
        self.sources={}
        self.private_items=[]
        self.image_inventory={}
        self.failures=[]
        self.expected=self.read(ROOT/'paper2/audit/independence_native_controls_20260912.json')
        for p in ('engine/dynamics_qa.py','engine/court_homography.py','tools/build_courtdyn_qa.py'):
            self.record(ROOT/p)

    def record(self,path):
        path=Path(path)
        digest=sha(path)
        self.sources[path.relative_to(ROOT).as_posix()]=digest
        return digest

    def read(self,path):
        self.record(path)
        return json.loads(Path(path).read_text(encoding='utf8'))

    def verify_frozen_sources(self,seq):
        files=next(r['files'] for r in self.expected['sources'] if r['seq']==seq)
        for row in files:
            # First-frame results are not inputs to this full-frame diagnostic.
            if '/t34_' not in row['path']:
                require(self.record(ROOT/row['path'])==row['sha256'],'Frozen core input changed: '+row['path'])

    def mot_boxes(self,seqdir):
        boxes={}
        path=seqdir/'gt/gt.txt';self.record(path)
        for line in path.read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            p=line.split(',');frame,track=int(p[0]),int(p[1])
            require((track,frame) not in boxes,'Duplicate MOT frame/track row')
            boxes[(track,frame)]=tuple(float(v) for v in p[2:6])
        return boxes

    def check_image(self,path):
        path=Path(path)
        name=path.relative_to(ROOT).as_posix()
        if name not in self.image_inventory:
            if not path.is_file():
                raise MappingUnavailable('presented image file missing')
            try:
                with Image.open(path) as image:
                    if image.size!=(960,540):
                        raise MappingUnavailable('presented image is not 960x540')
                    image.verify()
            except OSError:
                raise MappingUnavailable('presented image cannot be decoded')
            self.image_inventory[name]=sha(path)

    def sequence(self,seq):
        self.verify_frozen_sources(seq)
        pooldir=ROOT/'results/courtdyn'/('seq_'+seq)
        rawdir=ROOT/'results/courtdyn'/f't33_{seq}_main_full_cdnative'
        parsedpath=Path(str(rawdir)+'_parsed')/'predictions.json'
        pool=self.read(pooldir/'qa_dyn_v1.json')
        ruler=self.read(pooldir/'qa_dyn_v1_ruler.json')
        source_manifest=self.read(pooldir/'qa_dyn_v1.manifest.json')
        config=self.read(rawdir/'run_config.json')
        parsed=self.read(parsedpath)
        raw=self.read(rawdir/'predictions.json')
        require(config['qa_json_sha256']==sha(pooldir/'qa_dyn_v1.json'),'Run QA hash differs')
        require(config['frame_mode']=='full' and config['blank_images'] is False and
                config['reference_tag']=='none','Original run is not the expected full-image condition')
        require(Path(config['adapter']).resolve()==(ROOT/'models/courtdyn-native-sft').resolve(),
                'Wrong adapter in frozen run config')
        def indexed(rows):
            rows=[r for r in rows if r['category'] in FAMILIES]
            result={key(r):r for r in rows}
            require(len(result)==len(rows),'Duplicate category/window-index/track item')
            return result
        items,v3,observed,raw_observed=map(indexed,(pool,ruler,parsed,raw))
        require(items.keys()==v3.keys(),'Original and v3 reference cohorts differ')
        require(set(observed)<=set(items) and set(raw_observed)<=set(items),'Prediction contains an unexpected item')
        require(len(items)==280,'Unexpected core prepared item count')
        for k,p in observed.items():
            require(k in raw_observed,'Parsed item absent from raw output')
            for field in ('question','answer','meta','image_ids','image_id','vlm_raw','vlm_answer','run_config_sha256'):
                require(p.get(field)==raw_observed[k].get(field),'Raw/parsed identity differs: '+field)
            for field in ('question','answer','meta','image_ids','image_id'):
                require(p[field]==items[k][field],'Prediction does not match original prepared item: '+field)
            require(p['run_config_sha256']==config['config_sha256'] and p['image_intervention']=='raw',
                    'Prediction run identity/intervention differs')
        seqdir=ROOT/'data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot/basketball_top/train'/seq
        self.record(seqdir/'seqinfo.ini')
        info=DQ.load_seqinfo(str(seqdir))
        require(info['name']==seq,'Sequence name differs')
        require(source_manifest['source']['gt_sha256']==sha(seqdir/'gt/gt.txt'),'Rendered-pool MOT hash differs')
        require(source_manifest['params']['render_width']==960,'Unexpected original render width')
        require(source_manifest['source']['seqinfo']==info,'Source frame metadata differ from original manifest')
        scale=960/info['width']
        require(math.isclose(info['height']*scale,540),'Source-to-canvas scale is not isotropic')
        boxes=self.mot_boxes(seqdir)
        tracks=DQ.load_tracks(str(seqdir))
        plane=CourtPlane.load(seq)
        good=defaultdict(list);failed=defaultdict(Counter);pool_counts=Counter(r['category'] for r in items.values())
        for k,item in items.items():
            category=item['category'];m=item['meta']
            row=dict(seq=seq,category=category,window_index=m['window_index'],track=m['track'],
                     window=m['window'],frames=m['frames'],image_ids=item['image_ids'])
            try:
                require(m['seq']==seq and math.isclose(m['fps'],info['fps']),'Item sequence/FPS differs')
                require(v3[k]['image_ids']==item['image_ids'] and v3[k]['meta']['frames']==m['frames'] and
                        v3[k]['meta']['window']==m['window'],'V3 reference frame mapping differs')
                if k not in observed:
                    raise MappingUnavailable('prediction missing')
                try:
                    prediction=float(observed[k]['vlm_answer'])
                    if not math.isfinite(prediction):
                        raise ValueError()
                except (TypeError,ValueError):
                    raise MappingUnavailable('prediction not finite/parsed')
                heights=presented_heights(item,boxes,scale)
                for name in item['image_ids']:
                    self.check_image(Path(config['img_root'])/name)
                recomputed=recompute_answer(plane,item,tracks,info['fps'])
                if recomputed is None:
                    raise MappingUnavailable('v3 reference cannot be recomputed')
                require(recomputed[0]==v3[k]['answer'],'Recomputed rounded v3 differs from frozen reference')
                gt=float(recomputed[0])
                exact=recomputed[1]['speed_mps' if FAMILIES[category]=='speed' else 'path_m']
                height=statistics.median(heights)
                row.update(status='MATCHED',prediction=prediction,v3_rounded=gt,v3_exact=exact,
                           four_frame_heights_rendered=heights,median_height_rendered=height,
                           raw_to_rendered_scale=scale)
                good[category].append(row)
            except MappingUnavailable as error:
                row.update(status='UNAVAILABLE',reason=str(error))
                failed[category][str(error)]+=1
            self.private_items.append(row)
        cells=[]
        for category,family in FAMILIES.items():
            rows=good[category]
            stats=partial_rank([r['prediction'] for r in rows],[r['v3_rounded'] for r in rows],
                               [r['median_height_rendered'] for r in rows])
            expected=next(r for r in self.expected['cells'] if r['seq']==seq and r['family']==family)
            match=None
            if len(rows)==expected['n_common']:
                match=math.isclose(stats['plain_rho'],expected['v3']['full_rho'],rel_tol=0,abs_tol=1e-12)
                require(match,'Plain rho does not reproduce frozen manuscript core value')
            # Sensitivity to reference serialization, clearly secondary.
            exact_stats=partial_rank([r['prediction'] for r in rows],[r['v3_exact'] for r in rows],
                                     [r['median_height_rendered'] for r in rows])
            cells.append(dict(seq=seq,family=family,n_pool=pool_counts[category],n_prediction_rows=sum(k[0]==category for k in observed),
                              n_complete=len(rows),coverage=len(rows)/pool_counts[category],
                              failure_counts=dict(failed[category]),rank_diagnostic=stats,
                              exact_reference_sensitivity=exact_stats,
                              frozen_core_plain_rho_reproduced=match,event_overlap=(seq=='Q1_top_0-30'),
                              median_height_rendered_range=[min(r['median_height_rendered'] for r in rows),
                                                            max(r['median_height_rendered'] for r in rows)] if rows else None))
        return cells


class Tests(unittest.TestCase):
    def test_average_ties_and_formula(self):
        self.assertEqual(tie_ranks([4,1,1,3]).tolist(),[4,1.5,1.5,3])
        d=partial_rank([1,2,2,5,4,6],[2,1,3,4,6,5],[2,2,1,3,4,3])
        self.assertAlmostEqual(d['partial_rho'],d['closed_form_check'],places=12)

    def test_collinear_and_constant_height(self):
        self.assertIsNone(partial_rank([1,2,3,4],[2,1,4,3],[1,2,3,4])['partial_rho'])
        d=partial_rank([1,3,2,4],[1,2,4,3],[9,9,9,9])
        self.assertAlmostEqual(d['partial_rho'],d['plain_rho'],places=12)
        self.assertTrue(d['covariate_constant'])

    def test_height_scale_rank_invariance(self):
        p,g,h=[1,3,2,5,4],[5,1,3,2,4],[2,4,3,1,3]
        self.assertAlmostEqual(partial_rank(p,g,h)['partial_rho'],
                               partial_rank(p,g,[v*.25 for v in h])['partial_rho'],places=12)

    def test_exact_frame_mapping(self):
        frames=[10,30,50,70]
        item=dict(meta=dict(track=7,frames=frames,window=[10,70]),
                  image_ids=[f'f{f:04d}_r7.jpg' for f in frames])
        boxes={(7,f):(0,0,10,h) for f,h in zip(frames,[40,80,120,160])}
        boxes[(7,11)]=(0,0,10,999)
        hs=presented_heights(item,boxes,.25)
        self.assertEqual(hs,[10,20,30,40])
        self.assertEqual(statistics.median(hs),25)
        del boxes[(7,10)]
        with self.assertRaises(MappingUnavailable):
            presented_heights(item,boxes,.25)  # Must not borrow frame 11.

    def test_frame_track_mismatch(self):
        item=dict(meta=dict(track=7,frames=[10,30,50,70],window=[10,70]),
                  image_ids=['f0010_r8.jpg','f0030_r7.jpg','f0050_r7.jpg','f0070_r7.jpg'])
        with self.assertRaises(MappingUnavailable):
            presented_heights(item,{},.25)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'paper2/audit/revision_execution_20260912/boxheight_partial')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if result.wasSuccessful() else 1
    if args.output_dir.exists():
        ap.error('Use a new output directory; existing audit outputs are never overwritten.')
    report=dict(schema='courtdyn-boxheight-partial-rank-v1',status='FAIL',cells=[],errors=[],
                generated=datetime.now().isoformat(timespec='seconds'),script_sha256=sha(__file__),
                model='Frozen original CourtDyn-native, v1 training, 1400 samples; T33 full four-frame predictions.',
                reference='Primary: one-decimal v3 court-homography reference, exactly matching frozen manuscript rho. Secondary exact-reference sensitivity is separate.',
                covariate='Median raw MOT bbox height at exactly the four presented frames, multiplied by 960 / original sequence width. Not the all-window v1 ruler height.',
                method='Average tie ranks for prediction, reference and covariate. Separate OLS of prediction/reference ranks on intercept + height rank. Pearson correlation of residuals.',
                limitations=['Post-hoc descriptive adjustment, not a causal identification strategy.',
                             'Adjusts only one scalar rank covariate; does not control all scene/player/motion confounders or identify a visual mechanism.',
                             'Items share windows/players; these are not independent observations or independent games. No significance test or confidence interval is claimed.',
                             'Q1 contains a training event for this original model. Q2 is from the same game.',
                             'Current supplied-image files and source mapping are checked; pixel tensors processed during historic inference were not archived.',
                             'Only aggregate.json, findings.md and the script are candidates for sharing. restricted_local_items.json contains trajectory IDs/image names and must remain local.'])
    audit=None
    try:
        audit=Audit()
        for seq in SEQUENCES:
            report['cells'].extend(audit.sequence(seq))
        report['status']='PASS' if all(c['coverage']==1 for c in report['cells']) else 'PARTIAL'
    except (ValueError,KeyError,OSError,TypeError) as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    args.output_dir.mkdir(parents=True)
    if audit:
        report['source_sha256']=audit.sources
        report['unique_presented_images_checked']=len(audit.image_inventory)
        inventory=json.dumps(sorted(audit.image_inventory.items()),separators=(',',':')).encode('utf8')
        report['presented_image_inventory_sha256']=hashlib.sha256(inventory).hexdigest()
        private=dict(restriction='LOCAL RESTRICTED: do not publish trajectory IDs, per-item mapping or image filenames.',
                     items=audit.private_items,image_sha256=audit.image_inventory)
        (args.output_dir/'restricted_local_items.json').write_text(json.dumps(private,indent=2)+'\n',encoding='utf8')
    (args.output_dir/'aggregate.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    lines=['# 原始 native/v3 四格框高偏相关诊断','',report['status'],'',
           '这是使用已有预测的后验 CPU 描述性诊断；没有新增训练或推理。主参考保留现稿一位小数 v3 口径，exact v3 敏感性另列于 JSON。',
           '', '| 序列 | Family | N / 题池 | 原 Spearman | pred–height | gt–height | 调整框高后的相关 |',
           '|---|---|---:|---:|---:|---:|---:|']
    def fmt(v):
        return '未定义' if v is None else f'{v:.6f}'
    for cell in report['cells']:
        d=cell['rank_diagnostic']
        lines.append(f'| {cell["seq"]} | {cell["family"]} | {cell["n_complete"]}/{cell["n_pool"]} | '
                     f'{fmt(d["plain_rho"])} | {fmt(d["prediction_height_rho"])} | {fmt(d["reference_height_rho"])} | {fmt(d["partial_rho"])} |')
    lines+=['','每题严格使用 meta.frames 对应的四张 image_ids，名称中的 frame/track 与 MOT 原框逐一匹配，检查实际图像为960×540；不借用邻帧，也不使用整个窗口的框高中位数。该常数缩放不影响秩。',
            '', '对预测、参考和四帧框高中位数分别取平均并列秩；预测秩和参考秩各以“截距 + 框高秩”做 OLS，再求残差 Pearson。数学夹具及单控制变量闭式公式用于复核。失败/覆盖数量、原始来源 SHA 和 exact-reference 敏感性在 aggregate.json。',
            '', '即使调整后的相关仍较高，也只能限制“这一单一框高秩变量足以解释全部相关”的解释；不能说明已控制所有混杂、证明依赖真实运动、证明内部测量机制或建立跨比赛泛化。Q1 对原模型含训练事件重叠。没有进行显著性或独立样本推断。',
            '', '逐项映射和图像清单仅保存在 restricted_local_items.json，本地受限，禁止纳入公开包。可公开候选限本脚本、aggregate.json 和本说明。']
    if report['errors']:
        lines+=['','错误：']+['- '+e for e in report['errors']]
    (args.output_dir/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status=report['status'],cells=report['cells'],errors=report['errors'],
                         images=report.get('unique_presented_images_checked'),output=str(args.output_dir)),ensure_ascii=False))
    return {'PASS':0,'FAIL':1,'PARTIAL':2}[report['status']]


if __name__=='__main__':
    sys.exit(main())
