"""Two-rate search with fixed objective and held-out random networks.

Pure NumPy. The batched search has identical within-trial fixed weights to the
scalar reference; selected test conditions are replayed by that reference.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys
import argparse
import json
import csv
import time
from pathlib import Path
import numpy as np

MODEL_DIR=Path(__file__).resolve().parent.parent
if not (MODEL_DIR/'distributed_simulation.py').exists():
    raise RuntimeError('Keep this file in experiments/distributed/tuning.')
sys.path.insert(0,str(MODEL_DIR))
from distributed_simulation import DistributedModel, PROBES, NORMALIZATION_FLOOR, RETENTION
from model_core import reference, rotation, digest, HIDDEN, FEATURES, N, PREP, DT, DURATION, DISTANCE

TRAIN_SEEDS=[11,22,33,44,55]
TEST_SEEDS=[66,77,88,99,110]
RATES=[0.,.025,.05,.1,.2,.4,.8,1.2,1.6,2.4,3.2]
KEYS=['late_abs_early_deg','late_endpoint_mm','adapt_abs_early_deg','adapt_endpoint_mm']


def load_model(seed,out,retrain=False):
    """Reuse the exact existing pretraining where available; cache new seeds."""
    model=DistributedModel(seed)
    original=MODEL_DIR/'results'/f'checkpoint_feedback_seed{seed}.npz'
    cached=out/'pretrained'/f'seed{seed}.npz'
    path=original if original.exists() else cached
    if path.exists() and not retrain:
        with np.load(path) as cp:
            for target,source in [('recurrent','recurrent'),('input','input'),('bias','bias'),
                                  ('expansion','expansion'),('expansion_bias','expansion_bias'),
                                  ('hidden_projection','hidden_projection'),('readout','base_readout')]:
                setattr(model,target,cp[source].copy())
    else:
        model.train_cortex()
        cached.parent.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(cached,recurrent=model.recurrent,input=model.input,bias=model.bias,
            expansion=model.expansion,expansion_bias=model.expansion_bias,
            hidden_projection=model.hidden_projection,base_readout=model.readout)
    model.initialize_adaptation(model.readout.copy())
    return model


def batch_run(model,pairs):
    """One shared network, independent plastic weights for each coefficient pair."""
    pairs=np.asarray(pairs,dtype=float);p=len(pairs)
    c=np.zeros((p,2*HIDDEN+1,2));b=np.zeros((p,FEATURES,2))
    base=model.base_readout.copy();direction=np.array([1.,0.]);ref,vv=reference(direction)
    metrics=np.empty((120,p,4));clips=np.zeros(p,dtype=int);finite=np.ones(p,dtype=bool)
    for trial in range(120):
        R=rotation(30. if 20<=trial<100 else 0.)
        h=np.zeros((p,HIDDEN));x=np.zeros((p,2));y=np.zeros((N+1,p,2))
        qs=np.empty((N,p,2*HIDDEN+1));phis=np.empty((N,p,FEATURES))
        readout=base[None,:,:]+c
        for i in range(-PREP,N):
            go=i>=0;v=vv[i] if go else direction*.8;phase=max(i,0)/N
            context=np.column_stack([np.broadcast_to(direction,(p,2)),np.broadcast_to(v,(p,2)),
                                     np.full(p,phase),h@model.hidden_projection.T])
            phi=np.tanh(context@model.expansion.T+model.expansion_bias)/np.sqrt(FEATURES)*np.linalg.norm(v)
            raw=np.einsum('pj,pjk->pk',phi,b);cb=np.clip(raw,-2.,2.)
            clips+=np.any(raw!=cb,axis=1)
            e=ref[i-12]-y[i-12] if i>=12 else np.zeros((p,2))
            inp=np.column_stack([v+cb,e,np.broadcast_to(direction,(p,2)),np.full(p,phase),np.full(p,float(go))])
            new=.3*h+.7*np.tanh(h@model.recurrent.T+inp@model.input.T+model.bias)
            q=np.column_stack([new,h,np.ones(p)])
            u=np.einsum('pj,pjk->pk',q,readout)*go
            h=new
            finite&=np.all(np.isfinite(h),axis=1)&np.all(np.isfinite(u),axis=1)&np.all(np.isfinite(phi),axis=1)
            if not go:continue
            qs[i]=q;phis[i]=phi
            x+=DT/DURATION*u;y[i+1]=x@R.T
        early=y[10]
        metrics[trial,:,0]=np.rad2deg(np.arctan2(early[:,1],early[:,0]))
        metrics[trial,:,1]=np.linalg.norm(y[-1]-direction,axis=1)*100.
        metrics[trial,:,2]=np.sqrt(np.mean((ref[:,None,:]-y)**2,axis=(0,2)))*100.
        metrics[trial,:,3]=np.sqrt(np.mean((vv[:,None,:]-np.diff(y,axis=0)*DURATION/DT)**2,axis=(0,2)))*125.
        # The last available completed segment is 66 (inclusive): 67 total.
        error=vv[:67,None,:]-np.diff(y,axis=0)[:67]*DURATION/DT
        qc=qs[:67]/np.maximum(NORMALIZATION_FLOOR,np.sum(qs[:67]**2,axis=2))[:,:,None]
        pb=phis[:67]/np.maximum(NORMALIZATION_FLOOR,np.sum(phis[:67]**2,axis=2))[:,:,None]
        gc=np.einsum('npj,npk->pjk',qc,error);gb=np.einsum('npj,npk->pjk',pb,error)
        c=np.where((pairs[:,0]>0)[:,None,None],RETENTION*c+pairs[:,0,None,None]*gc/N,c)
        b=np.where((pairs[:,1]>0)[:,None,None],RETENTION*b+pairs[:,1,None,None]*gb/N,b)
        finite&=np.all(np.isfinite(c),axis=(1,2))&np.all(np.isfinite(b),axis=(1,2))
    finite&=np.all(np.isfinite(metrics),axis=(0,2))
    return metrics,clips,dict(finite=finite,final_c=c,final_b=b)


def summarize(metrics):
    late=metrics[90:100];adapt=metrics[20:100]
    return dict(late_abs_early_deg=float(np.mean(np.abs(late[:,0]))),
        late_signed_early_deg=float(np.mean(late[:,0])),late_endpoint_mm=float(np.mean(late[:,1])),
        adapt_abs_early_deg=float(np.mean(np.abs(adapt[:,0]))),adapt_endpoint_mm=float(np.mean(adapt[:,1])),
        max_adapt_abs_early_deg=float(np.max(np.abs(adapt[:,0]))),
        max_adapt_endpoint_mm=float(np.max(adapt[:,1])),
        max_baseline_abs_early_deg=float(np.max(np.abs(metrics[:20,0]))),
        max_baseline_endpoint_mm=float(np.max(metrics[:20,1])),
        max_washout_abs_early_deg=float(np.max(np.abs(metrics[100:,0]))),
        max_washout_endpoint_mm=float(np.max(metrics[100:,1])),
        late_path_rmse_mm=float(np.mean(late[:,2])),late_velocity_rmse_mm_s=float(np.mean(late[:,3])),
        first_washout_deg=float(metrics[100,0]))


def score(s):
    # Fixed BEFORE inspecting search results: late accuracy and learning history.
    return float(.25*(s['late_abs_early_deg']/5.+s['late_endpoint_mm']/1.+
                      s['adapt_abs_early_deg']/30.+s['adapt_endpoint_mm']/15.3))


def stable(s):
    return bool(s['all_finite'] and s['clip_events']==0 and s['max_adapt_abs_early_deg']<=60. and
                s['max_adapt_endpoint_mm']<=50. and all(np.isfinite(v) for v in s.values()))


def export_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def scan(pairs,stage,models,out):
    rows=[]
    for seed,model in models.items():
        start=time.perf_counter();metrics,clips,check=batch_run(model,pairs)
        for i,(ec,eb) in enumerate(pairs):
            s=summarize(metrics[:,i]);s['clip_events']=int(clips[i]);s['objective']=score(s);s['all_finite']=bool(check['finite'][i])
            good=stable(s)
            rows.append(dict(stage=stage,seed=seed,eta_c=float(ec),eta_cb=float(eb),**s,stable=good))
        print(json.dumps(dict(stage=stage,seed=seed,pairs=len(pairs),seconds=round(time.perf_counter()-start,2)),ensure_ascii=False),flush=True)
    export_csv(out/f'{stage}_per_seed.csv',rows)
    return rows


def aggregate(rows):
    pairs=sorted(set((r['eta_c'],r['eta_cb']) for r in rows));agg=[]
    for ec,eb in pairs:
        rr=[r for r in rows if r['eta_c']==ec and r['eta_cb']==eb]
        item=dict(eta_c=ec,eta_cb=eb,stable=all(r['stable'] for r in rr),n_seeds=len(rr))
        item['all_finite']=all(r['all_finite'] for r in rr)
        for key in list(summarize(np.zeros((120,4))))+['clip_events','objective']:
            values=[r[key] for r in rr];item[key]=float(np.mean(values));item[key+'_sd']=float(np.std(values,ddof=1))
        agg.append(item)
    return agg


def pick(agg,kind):
    eligible=[r for r in agg if r['stable'] and
        ((kind=='joint' and r['eta_c']>0 and r['eta_cb']>0) or
         (kind=='cortex' and r['eta_c']>0 and r['eta_cb']==0) or
         (kind=='cerebellum' and r['eta_c']==0 and r['eta_cb']>0) or kind=='all')]
    if not eligible:raise ValueError(f'No stable {kind} condition')
    return min(eligible,key=lambda r:(r['objective'],r['eta_c']+r['eta_cb']))


def scalar_run(model,pair,seed,label,save_to=None):
    model.initialize_adaptation(model.base_readout.copy());base=model.base_readout.copy()
    ec,eb=pair;rows=[];history=np.empty((120,4));learned=None;saved={};finite=True
    for t in range(120):
        theta=30. if 20<=t<100 else 0.
        m,detail=model.trial(theta=theta,learn_cortex=ec,learn_cb=eb,rate=1.)
        finite=finite and all(np.all(np.isfinite(v)) for v in detail.values())
        finite=finite and np.all(np.isfinite(model.cortical_delta)) and np.all(np.isfinite(model.cerebellar))
        history[t]=[m['early_error_deg'],m['endpoint_error_mm'],m['path_tracking_rmse_mm'],m['velocity_tracking_rmse_mm_s']]
        phase='baseline' if t<20 else 'adaptation' if t<100 else 'washout'
        rows.append(dict(seed=seed,condition=label,eta_c=ec,eta_cb=eb,trial=t+1,phase=phase,**m))
        if t==99:learned=(model.cortical_delta.copy(),model.cerebellar.copy());saved['trial100']=detail
    final_c=model.cortical_delta.copy();final_b=model.cerebellar.copy()
    model.cortical_delta,model.cerebellar=learned
    checkpoint=dict(recurrent=model.recurrent,input=model.input,bias=model.bias,base_readout=base,
        cortical_delta=model.cortical_delta,cb_weights=model.cerebellar,expansion=model.expansion,
        expansion_bias=model.expansion_bias,hidden_projection=model.hidden_projection)
    before=(digest(model.cortical_delta),digest(model.cerebellar));probes=[]
    for test,options in PROBES:
        m,detail=model.trial(**options,noise_seed=seed+3000)
        assert before==(digest(model.cortical_delta),digest(model.cerebellar))
        probes.append(dict(seed=seed,condition=label,eta_c=ec,eta_cb=eb,test=test,**m))
        saved[test]=detail
    if save_to:
        np.savez_compressed(save_to/f'checkpoint_{label}_seed{seed}.npz',**checkpoint)
        np.savez_compressed(save_to/f'activity_{label}_seed{seed}.npz',**{f'{k}__{f}':v for k,d in saved.items() for f,v in d.items()})
    return rows,probes,history,dict(final_c=final_c,final_b=final_b,finite=bool(finite))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results')
    ap.add_argument('--phase',choices=['search','test','all'],default='all')
    ap.add_argument('--retrain-pretraining',action='store_true',help='Fit all basic networks afresh; ignore pretrained caches.')
    args=ap.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(train_seeds=TRAIN_SEEDS,test_seeds=TEST_SEEDS,coarse_rates=RATES,
        objective='0.25*(late absolute early/5deg + late endpoint/1mm + adaptation mean absolute early/30deg + adaptation mean endpoint/15.3mm)',
        selection='minimum mean objective across training seeds, all seeds must pass stability; select global, positive joint, cortex-only and CB-only',
        stability='predefined numerical/behavior screening, not a dynamics proof: no CB clipping; adaptation max absolute early <=60deg and endpoint <=50mm; all120 trials states/weights/metrics finite',
        refinement='7x7 linear grids between neighboring coarse rates around up to 3 best joint candidates; 13-point solo refinements; candidates chosen using train only',
        task='unchanged 20 baseline + 80 rotation30deg +20 washout; dt10ms duration800ms delay120ms',
        fixed='retention .999, network, teaching and update rules unchanged; vary only eta_C, eta_CB',
        test_policy='freeze selected rates before new seeds; no post-test retuning; network-initialization holdout only, same task',
        weighting='equal four normalized terms; analyst-selected pragmatic priorities, no universal optimum',
        claim='best tested coefficients under this objective/domain, not global or biological optimum')
    protocol['amendment_before_holdout']='After initial training-only search found joint and CB optima at upper boundary3.2, added conditional two-level range extensions up to25.6; objective, thresholds and held-out seeds unchanged. No test results inspected.'
    if args.phase in ('search','all'):
        (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
        models={s:load_model(s,out,args.retrain_pretraining) for s in TRAIN_SEEDS}
        # Check vectorization against scalar reference over an entire 120-trial run.
        verify_pairs=[(0.,0.),(.4,.4),(0.,.8),(.1,1.2)]
        batch,batchclips,batchcheck=batch_run(models[11],verify_pairs);diffs=[];wdiffs=[]
        for i,pair in enumerate(verify_pairs):
            sr,_,scalar,check=scalar_run(models[11],pair,11,'verification')
            diffs.append(float(np.max(np.abs(batch[:,i]-scalar))))
            wdiffs.append(max(float(np.max(np.abs(batchcheck['final_c'][i]-check['final_c']))),float(np.max(np.abs(batchcheck['final_b'][i]-check['final_b'])))))
            assert batchclips[i]==sum(r['clip_events'] for r in sr)
        assert max(diffs)<1e-7,(diffs,'batched/scalar mismatch')
        assert max(wdiffs)<1e-7
        (out/'batch_validation.json').write_text(json.dumps(dict(max_abs_metric_difference=max(diffs),max_final_weight_difference=max(wdiffs),clip_counts_identical=True,per_pair=diffs,tolerance=1e-7),indent=2),encoding='utf-8')
        rates=RATES.copy();pairs=[(c,b) for c in rates for b in rates]
        coarse=scan(pairs,'coarse',models,out);ca=aggregate(coarse)
        expansions=[]
        while max(rates)<25.6 and any(max(pick(ca,k)['eta_c'],pick(ca,k)['eta_cb'])==max(rates) for k in ('all','joint','cortex','cerebellum')):
            hi=max(rates);rates=sorted(rates+[round(hi*1.5,8),round(hi*2,8)])
            new_pairs=[(c,b) for c in rates for b in rates if (c,b) not in set(pairs)]
            er=scan(new_pairs,f'extend{len(expansions)+1}',models,out)
            coarse+=er;pairs+=new_pairs;ca=aggregate(coarse)
            expansions.append(dict(max_rate=max(rates),added_pairs=len(new_pairs)))
        export_csv(out/'coarse_aggregate.csv',ca)
        known=set(pairs);fine=set()
        joints=sorted([r for r in ca if r['stable'] and r['eta_c']>0 and r['eta_cb']>0],key=lambda r:r['objective'])[:3]
        def neighbors(v):
            idx=rates.index(v);return rates[max(0,idx-1)],rates[min(len(rates)-1,idx+1)]
        for best in joints:
            cr=np.linspace(*neighbors(best['eta_c']),7);br=np.linspace(*neighbors(best['eta_cb']),7)
            fine.update((round(float(c),8),round(float(b),8)) for c in cr for b in br)
        for kind in ('cortex','cerebellum'):
            best=pick(ca,kind);key='eta_c' if kind=='cortex' else 'eta_cb'
            for v in np.linspace(*neighbors(best[key]),13):
                fine.add((round(float(v),8),0.) if kind=='cortex' else (0.,round(float(v),8)))
        fine=sorted(fine-known)
        fr=scan(fine,'fine',models,out);allrows=coarse+fr;agg=aggregate(allrows)
        export_csv(out/'search_per_seed.csv',allrows);export_csv(out/'search_aggregate.csv',agg)
        choices={kind:pick(agg,kind) for kind in ('all','joint','cortex','cerebellum')}
        choices['counts']=dict(coarse=len(pairs),fine=len(fine),total=len(agg),training_simulations=len(allrows),training_trials=len(allrows)*120)
        choices['coarse_rates_final']=rates;choices['range_expansions']=expansions
        choices['unconstrained_best']={k:min([r for r in agg if r['all_finite'] and r['clip_events']==0 and ((k=='cortex' and r['eta_cb']==0 and r['eta_c']>0) or(k=='cerebellum' and r['eta_c']==0 and r['eta_cb']>0) or(k=='joint' and r['eta_c']>0 and r['eta_cb']>0))],key=lambda r:r['objective']) for k in ('cortex','cerebellum','joint')}
        # Save the frozen selection BEFORE constructing any held-out networks.
        (out/'frozen_selection.json').write_text(json.dumps(choices,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(frozen_selection=choices),ensure_ascii=False),flush=True)
    if args.phase in ('test','all'):
        selected=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
        choices={'feedback':(0.,0.),'original_joint':(.4,.4),
            'tuned_cortex':(selected['cortex']['eta_c'],0.),
            'tuned_cerebellum':(0.,selected['cerebellum']['eta_cb']),
            'tuned_joint':(selected['joint']['eta_c'],selected['joint']['eta_cb'])}
        rows=[];probes=[];summaries=[]
        for seed in TEST_SEEDS:
            model=load_model(seed,out,args.retrain_pretraining);base_hash=digest(model.base_readout)
            for label,pair in choices.items():
                rr,pp,hist,check=scalar_run(model,pair,seed,label,out);rows+=rr;probes+=pp
                s=summarize(hist);s['clip_events']=int(sum(r['clip_events'] for r in rr));s['objective']=score(s);s['all_finite']=check['finite']
                summaries.append(dict(seed=seed,condition=label,eta_c=pair[0],eta_cb=pair[1],**s,stable=stable(s)))
                assert digest(model.base_readout)==base_hash
                print(json.dumps(dict(test_seed=seed,condition=label,summary=s),ensure_ascii=False),flush=True)
        export_csv(out/'test_trials.csv',rows);export_csv(out/'test_probes.csv',probes);export_csv(out/'test_per_seed.csv',summaries)
        aggregate_test={}
        for label,pair in choices.items():
            rr=[r for r in summaries if r['condition']==label]
            obj=dict(eta_c=pair[0],eta_cb=pair[1],stable=all(r['stable'] for r in rr))
            for key in list(summarize(np.zeros((120,4))))+['objective']:
                values=[r[key] for r in rr];obj[key]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),values=values)
            obj['probes']={}
            for test,_ in PROBES:
                obj['probes'][test]={}
                for key in ('early_error_deg','endpoint_error_mm'):
                    v=[r[key] for r in probes if r['condition']==label and r['test']==test]
                    obj['probes'][test][key]=dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),values=v)
            aggregate_test[label]=obj
        (out/'test_summary.json').write_text(json.dumps(aggregate_test,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(test_complete=True,trial_rows=len(rows),probe_rows=len(probes)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
