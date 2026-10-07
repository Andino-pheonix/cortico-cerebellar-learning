"""Independently recompute selections, held-out summaries and probe replays."""
import json,csv,sys,argparse,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
if not (ROOT/'distributed_simulation.py').exists():
    raise RuntimeError('Keep this file in experiments/distributed/tuning.')
sys.path.insert(0,str(ROOT))
from distributed_simulation import DistributedModel,PROBES
from model_core import rotation

def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def objective(s):return .25*(s['late_abs_early_deg']/5+s['late_endpoint_mm']+s['adapt_abs_early_deg']/30+s['adapt_endpoint_mm']/15.3)
def model_from(path,seed):
    m=DistributedModel(seed)
    with np.load(path) as c:
        for target,key in [('recurrent','recurrent'),('input','input'),('bias','bias'),('expansion','expansion'),('expansion_bias','expansion_bias'),('hidden_projection','hidden_projection')]:setattr(m,target,c[key].copy())
        m.initialize_adaptation(c['base_readout']);m.cortical_delta=c['cortical_delta'].copy();m.cerebellar=c['cb_weights'].copy()
    return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'results');args=ap.parse_args();d=args.data
    protocol=json.loads((d/'protocol.json').read_text(encoding='utf-8'));selection=json.loads((d/'frozen_selection.json').read_text(encoding='utf-8'))
    train=read(d/'search_per_seed.csv');agg=read(d/'search_aggregate.csv');rows=read(d/'test_trials.csv');probes=read(d/'test_probes.csv');ss=read(d/'test_per_seed.csv');saved=json.loads((d/'test_summary.json').read_text(encoding='utf-8'))
    checks={};maxdiff=0.
    assert not set(protocol['train_seeds'])&set(protocol['test_seeds'])
    checks['train_test_seeds_disjoint']=True
    for r in train:
        s={k:float(r[k]) for k in ['late_abs_early_deg','late_endpoint_mm','adapt_abs_early_deg','adapt_endpoint_mm']}
        maxdiff=max(maxdiff,abs(objective(s)-float(r['objective'])))
        expected=r['all_finite']=='True' and float(r['clip_events'])==0 and float(r['max_adapt_abs_early_deg'])<=60 and float(r['max_adapt_endpoint_mm'])<=50
        assert expected==(r['stable']=='True')
    for kind in ('all','joint','cortex','cerebellum'):
        candidates=[r for r in agg if r['stable']=='True' and (kind=='all' or (kind=='joint' and float(r['eta_c'])>0 and float(r['eta_cb'])>0) or(kind=='cortex' and float(r['eta_c'])>0 and float(r['eta_cb'])==0) or(kind=='cerebellum' and float(r['eta_c'])==0 and float(r['eta_cb'])>0))]
        best=min(candidates,key=lambda r:(float(r['objective']),float(r['eta_c'])+float(r['eta_cb'])))
        assert float(best['eta_c'])==selection[kind]['eta_c'] and float(best['eta_cb'])==selection[kind]['eta_cb']
        selectedrows=[r for r in train if float(r['eta_c'])==float(best['eta_c']) and float(r['eta_cb'])==float(best['eta_cb'])]
        maxdiff=max(maxdiff,abs(np.mean([float(r['objective']) for r in selectedrows])-selection[kind]['objective']))
    checks['selection_recomputed_from_train_only']=True
    conditions=list(saved);seed_list=protocol['test_seeds'];assert len(rows)==len(seed_list)*len(conditions)*120 and len(probes)==len(seed_list)*len(conditions)*10
    original=dict(feedback=(0.,0.),original_joint=(.4,.4))
    expected_pairs={**original,**{f'tuned_{k}':(selection[k]['eta_c'],selection[k]['eta_cb']) for k in ('cortex','cerebellum','joint')}}
    for seed in seed_list:
        hashes=[]
        for condition in conditions:
            subset=[r for r in rows if r['condition']==condition and int(r['seed'])==seed];assert [int(r['trial']) for r in subset]==list(range(1,121))
            assert all((float(r['eta_c']),float(r['eta_cb']))==expected_pairs[condition] for r in subset)
            late=subset[90:100];adapt=subset[20:100]
            recalculated=dict(late_abs_early_deg=np.mean([abs(float(r['early_error_deg'])) for r in late]),late_signed_early_deg=np.mean([float(r['early_error_deg']) for r in late]),late_endpoint_mm=np.mean([float(r['endpoint_error_mm']) for r in late]),adapt_abs_early_deg=np.mean([abs(float(r['early_error_deg'])) for r in adapt]),adapt_endpoint_mm=np.mean([float(r['endpoint_error_mm']) for r in adapt]))
            recalculated['objective']=objective(recalculated)
            own=next(r for r in ss if r['condition']==condition and int(r['seed'])==seed)
            for key,value in recalculated.items():maxdiff=max(maxdiff,abs(float(own[key])-value))
            assert all(np.isfinite(float(r[k])) for r in subset for k in ('early_error_deg','endpoint_error_mm','path_tracking_rmse_mm','velocity_tracking_rmse_mm_s'))
            cp=d/f'checkpoint_{condition}_seed{seed}.npz'
            with np.load(cp) as c:
                assert all(np.all(np.isfinite(c[k])) for k in c.files)
                hashes.append(hashlib.sha256(b''.join(c[k].tobytes() for k in ['recurrent','input','bias','base_readout','expansion','expansion_bias','hidden_projection'])).hexdigest())
            m=model_from(cp,seed);weightbefore=(m.cortical_delta.copy(),m.cerebellar.copy())
            for name,opts in PROBES:
                metric,detail=m.trial(**opts,noise_seed=seed+3000)
                recorded=next(r for r in probes if r['condition']==condition and int(r['seed'])==seed and r['test']==name)
                for key in ('early_error_deg','endpoint_error_mm'):maxdiff=max(maxdiff,abs(metric[key]-float(recorded[key])))
                assert np.array_equal(weightbefore[0],m.cortical_delta) and np.array_equal(weightbefore[1],m.cerebellar)
            metric,a=m.trial(theta=0);_,b=m.trial(theta=30)
            maxdiff=max(maxdiff,float(np.max(np.abs(a['hidden'][:12]-b['hidden'][:12]))),float(np.max(np.abs(a['commands'][:12]-b['commands'][:12]))))
            predicted=np.vstack([np.zeros(2),np.cumsum(b['commands']*.01,axis=0)])@rotation(30).T
            maxdiff=max(maxdiff,float(np.max(np.abs(predicted-b['display']))))
        assert len(set(hashes))==1
    checks['all_test_conditions_use_frozen_rates']=True;checks['same_pretrained_maps_per_test_seed']=True
    checks['all_250_probes_replayed_and_weights_frozen']=True;checks['all_metrics_and_checkpoints_finite']=True
    checks['no_rotation_leakage_before_delayed_feedback']=True;checks['only_cortical_output_drives_body']=True
    for label in conditions:
        for key in ['late_abs_early_deg','late_endpoint_mm','objective']:
            values=[float(r[key]) for r in ss if r['condition']==label];v=saved[label][key]
            maxdiff=max(maxdiff,abs(np.mean(values)-v['mean']),abs(np.std(values,ddof=1)-v['sd']))
    paired=[];changes=[]
    for seed in seed_list:
        j=next(r for r in ss if r['condition']=='tuned_joint' and int(r['seed'])==seed)
        for other in ('tuned_cerebellum','tuned_cortex','original_joint'):
            o=next(r for r in ss if r['condition']==other and int(r['seed'])==seed)
            paired.append(dict(seed=seed,comparison=f'joint_minus_{other}',**{k:float(j[k])-float(o[k]) for k in ['late_abs_early_deg','late_endpoint_mm','objective','adapt_abs_early_deg','adapt_endpoint_mm']}))
        intact=next(r for r in probes if r['condition']=='tuned_joint' and int(r['seed'])==seed and r['test']=='intact')
        for test in ('cb_block','cortex_reset','both_reset'):
            r=next(r for r in probes if r['condition']=='tuned_joint' and int(r['seed'])==seed and r['test']==test)
            changes.append(dict(seed=seed,test=test,abs_early_increase_deg=abs(float(r['early_error_deg']))-abs(float(intact['early_error_deg'])),endpoint_increase_mm=float(r['endpoint_error_mm'])-float(intact['endpoint_error_mm'])))
    write(d/'test_paired_differences.csv',paired);write(d/'probe_paired_changes.csv',changes)
    pp=[r for r in agg if r['stable']=='True'];pareto=[]
    fields=['late_abs_early_deg','late_endpoint_mm','adapt_abs_early_deg','adapt_endpoint_mm'];points=np.array([[float(r[k]) for k in fields] for r in pp])
    for i,r in enumerate(pp):
        dominated=np.any(np.all(points<=points[i]+1e-12,axis=1)&np.any(points<points[i]-1e-12,axis=1))
        if not dominated:pareto.append(r)
    write(d/'pareto_candidates.csv',pareto)
    checks['joint_improves_all5_test_seeds_vs_tuned_CB_for_late_errors_and_objective']=all(r[k]<0 for r in paired if r['comparison']=='joint_minus_tuned_cerebellum' for k in ('late_abs_early_deg','late_endpoint_mm','objective'))
    checks['each_module_removal_worsens_both_errors_all5_seeds']=all(r[k]>0 for r in changes for k in ('abs_early_increase_deg','endpoint_increase_mm'))
    checks['max_independent_recompute_difference']=maxdiff;checks['train_candidates']=len(agg);checks['train_trials']=len(train)*120;checks['heldout_trials']=len(rows);checks['heldout_probes']=len(probes);checks['pareto_candidates']=len(pareto)
    assert maxdiff<1e-8
    (d/'independent_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
