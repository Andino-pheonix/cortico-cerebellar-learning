"""Four-arm cortical/cerebellar readout plasticity experiment.

Same pretrained cortex and random feature maps per seed; equal total learning
coefficient split across modules in the joint arm. No anatomical identification.
"""
import argparse
import csv
import json
from pathlib import Path
from model_core import CoupledModel, rotation, reference, digest
from model_core import DT, DURATION, DISTANCE, SPEED_SCALE, N, PREP, HIDDEN, FEATURES
import numpy as np

ARMS = dict(feedback=(0.,0.), cortex=(1.,0.), cerebellum=(0.,1.), joint=(.5,.5))
RETENTION=.999
DEFAULT_RATE=.8
NORMALIZATION_FLOOR=.1


class DistributedModel(CoupledModel):
    def __init__(self, seed):
        super().__init__(seed)
        self.base_readout=None
        self.cortical_delta=np.zeros_like(self.readout)

    def initialize_adaptation(self, base):
        self.base_readout=base.copy()
        self.cortical_delta[:]=0
        self.cerebellar[:]=0
        self.readout=base.copy()

    def trial(self, theta=30., target_angle=0., delay=.12, connection=True,
              cortical_memory=True, feedback=True, learn_cortex=0., learn_cb=0.,
              rate=DEFAULT_RATE, sensory_noise=0., noise_seed=0):
        d=round(delay/DT)
        direction=rotation(target_angle)@np.array([1.,0.])
        ref,vv=reference(direction)
        R=rotation(theta)
        # Cortex reset means an intervention on the learned readout residual;
        # it leaves basic pretrained feedback control and recurrent activity intact.
        self.readout=self.base_readout+(self.cortical_delta if cortical_memory else 0.)
        hidden=np.zeros(HIDDEN)
        x,y,observed=[np.zeros(2)],[np.zeros(2)],[np.zeros(2)]
        rng=np.random.default_rng(noise_seed)
        phi_history,q_history=[],[]
        states,prep,cb_outputs,commands=[],[],[],[]
        grad_c=np.zeros_like(self.cortical_delta)
        grad_b=np.zeros_like(self.cerebellar)
        teaching_count,clipping=0,0
        for i in range(-PREP,N):
            go=float(i>=0)
            v=vv[i] if go else direction*.8
            phi=self.cb_features(hidden,direction,v,max(i,0)/N)
            cb=phi@self.cerebellar if connection else np.zeros(2)
            limited=np.clip(cb,-2.,2.)
            clipping+=int(np.any(limited!=cb));cb=limited
            e=ref[i-d]-observed[i-d] if feedback and i>=d else np.zeros(2)
            hidden,u,q=self.cortex_step(hidden,v+cb,e,direction,max(i,0)/N,go)
            if not go:
                prep.append(hidden.copy());continue
            phi_history.append(phi.copy());q_history.append(q.copy())
            states.append(hidden.copy());cb_outputs.append(cb.copy());commands.append(u.copy())
            x.append(x[-1]+DT/DURATION*u)
            y.append(R@x[-1])
            observed.append(y[-1]+rng.normal(0,sensory_noise/DISTANCE,2))
            j=i-d-1
            if j>=0 and (learn_cortex or learn_cb):
                measured=(observed[j+1]-observed[j])*DURATION/DT
                error=vv[j]-measured
                # Locally normalized feature-error correlation. No privileged R,
                # derivative through dynamics, or same-trial weight changes.
                q_old,phi_old=q_history[j],phi_history[j]
                grad_c+=np.outer(q_old,error)/max(NORMALIZATION_FLOOR,float(q_old@q_old))
                grad_b+=np.outer(phi_old,error)/max(NORMALIZATION_FLOOR,float(phi_old@phi_old))
                teaching_count+=1
        if learn_cortex:
            self.cortical_delta=RETENTION*self.cortical_delta+rate*learn_cortex*grad_c/N
        if learn_cb:
            self.cerebellar=RETENTION*self.cerebellar+rate*learn_cb*grad_b/N
        self.readout=self.base_readout+self.cortical_delta
        yy=np.array(y);early=yy[round(.10/DT)]
        cross=direction[0]*early[1]-direction[1]*early[0]
        metric=dict(early_error_deg=float(np.rad2deg(np.arctan2(cross,direction@early))),
                    endpoint_error_mm=float(np.linalg.norm(yy[-1]-direction)*DISTANCE*1000),
                    velocity_tracking_rmse_mm_s=float(np.sqrt(np.mean((vv-np.diff(yy,axis=0)*DURATION/DT)**2))*SPEED_SCALE*1000),
                    path_tracking_rmse_mm=float(np.sqrt(np.mean((ref-yy)**2))*DISTANCE*1000),
                    cortex_delta_norm=float(np.linalg.norm(self.cortical_delta)),
                    cb_weight_norm=float(np.linalg.norm(self.cerebellar)),
                    teaching_segments=teaching_count,clip_events=clipping)
        detail=dict(display=yy*DISTANCE*1000,hidden=np.array(states),preparation=np.array(prep),
                    cb=np.array(cb_outputs)*SPEED_SCALE*1000,commands=np.array(commands)*SPEED_SCALE*1000)
        return metric,detail


PROBES=[('intact',{}),('cb_block',dict(connection=False)),
        ('cortex_reset',dict(cortical_memory=False)),
        ('both_reset',dict(connection=False,cortical_memory=False)),
        ('feedback_off',dict(feedback=False)),
        ('delay60',dict(delay=.06)),('delay180',dict(delay=.18)),
        ('target_minus45',dict(target_angle=-45.)),('target_plus45',dict(target_angle=45.)),
        ('noise_1mm',dict(sensory_noise=.001))]


def run_arm(model,base,arm,rate):
    model.initialize_adaptation(base)
    fixed_hash=digest(np.r_[model.recurrent.ravel(),model.input.ravel(),model.bias.ravel()])
    lc,lb=ARMS[arm]
    rows,saved=[],{}
    for trial in range(1,121):
        phase='baseline' if trial<=20 else 'adaptation' if trial<=100 else 'washout'
        theta=30. if phase=='adaptation' else 0.
        before_c,before_b=model.cortical_delta.copy(),model.cerebellar.copy()
        metric,detail=model.trial(theta=theta,learn_cortex=lc,learn_cb=lb,rate=rate)
        assert lc or np.array_equal(before_c,model.cortical_delta)
        assert lb or np.array_equal(before_b,model.cerebellar)
        rows.append(dict(seed=model.seed,arm=arm,rate=rate,trial=trial,phase=phase,**metric))
        if trial in (20,21,100,101):saved[f'trial{trial}']=detail
        if trial==100:
            learned_c,learned_b=model.cortical_delta.copy(),model.cerebellar.copy()
    late=[r for r in rows if 91<=r['trial']<=100]
    summary=dict(late_abs_early_deg=float(np.mean([abs(r['early_error_deg']) for r in late])),
                 late_signed_early_deg=float(np.mean([r['early_error_deg'] for r in late])),
                 late_endpoint_mm=float(np.mean([r['endpoint_error_mm'] for r in late])),
                 late_velocity_rmse_mm_s=float(np.mean([r['velocity_tracking_rmse_mm_s'] for r in late])),
                 late_path_rmse_mm=float(np.mean([r['path_tracking_rmse_mm'] for r in late])),
                 first_washout_deg=rows[100]['early_error_deg'])
    model.cortical_delta=learned_c.copy();model.cerebellar=learned_b.copy()
    c_hash,b_hash=digest(learned_c),digest(learned_b)
    probe_rows=[]
    for name,options in PROBES:
        metric,detail=model.trial(**options,noise_seed=model.seed+3000)
        assert digest(model.cortical_delta)==c_hash and digest(model.cerebellar)==b_hash
        probe_rows.append(dict(seed=model.seed,arm=arm,rate=rate,test=name,**metric))
        saved[name]=detail
    assert fixed_hash==digest(np.r_[model.recurrent.ravel(),model.input.ravel(),model.bias.ravel()])
    first20=[r['endpoint_error_mm'] for r in rows[:20]]
    validation=dict(seed=model.seed,arm=arm,rate=rate,
                    only_enabled_readouts_updated=True,core_recurrent_input_bias_fixed=True,
                    all_probes_weights_frozen=True,max_baseline_endpoint_mm=max(first20),
                    all_metrics_finite=all(np.isfinite(r['endpoint_error_mm']) for r in rows),
                    cb_clip_events=sum(r['clip_events'] for r in rows),
                    cortical_parameters=2*(2*HIDDEN+1) if lc else 0,
                    cb_parameters=2*FEATURES if lb else 0)
    assert validation['all_metrics_finite']
    checkpoint=dict(recurrent=model.recurrent,input=model.input,bias=model.bias,base_readout=base,
                    cortical_delta=learned_c,cb_weights=learned_b,expansion=model.expansion,
                    expansion_bias=model.expansion_bias,hidden_projection=model.hidden_projection)
    return rows,probe_rows,summary,validation,saved,checkpoint


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def aggregate(summary_rows,probe_rows):
    result={}
    for arm in ARMS:
        result[arm]={}
        for metric in ('late_abs_early_deg','late_signed_early_deg','late_endpoint_mm','late_velocity_rmse_mm_s','late_path_rmse_mm','first_washout_deg'):
            values=[r[metric] for r in summary_rows if r['arm']==arm]
            result[arm][metric]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else 0.,values=values)
        result[arm]['probes']={}
        for test,_ in PROBES:
            result[arm]['probes'][test]={}
            for metric in ('early_error_deg','endpoint_error_mm'):
                values=[r[metric] for r in probe_rows if r['arm']==arm and r['test']==test]
                result[arm]['probes'][test][metric]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else 0.,values=values)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--seeds',default='11,22,33,44,55')
    parser.add_argument('--rate',type=float,default=DEFAULT_RATE)
    parser.add_argument('--sensitivity',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    seeds=[int(s) for s in args.seeds.split(',')]
    rows,probes,summaries,validations,sensitivity=[],[],[],[],[]
    for seed in seeds:
        model=DistributedModel(seed);rmse=model.train_cortex();base=model.readout.copy()
        initial_hash=digest(base)
        for arm in ARMS:
            rr,pp,ss,vv,saved,cp=run_arm(model,base,arm,args.rate)
            rows+=rr;probes+=pp;summaries.append(dict(seed=seed,arm=arm,**ss));validations.append(vv)
            assert digest(base)==initial_hash
            np.savez_compressed(args.out/f'checkpoint_{arm}_seed{seed}.npz',**cp)
            traces={f'{name}__{kind}':v for name,detail in saved.items() for kind,v in detail.items()}
            np.savez_compressed(args.out/f'activity_{arm}_seed{seed}.npz',**traces)
            print(json.dumps(dict(seed=seed,arm=arm,summary=ss,cb_clips=vv['cb_clip_events']),ensure_ascii=False),flush=True)
        if args.sensitivity:
            for rate in (.4,1.6):
                for arm in ('cortex','cerebellum','joint'):
                    _,_,ss,vv,_,_=run_arm(model,base,arm,rate)
                    sensitivity.append(dict(seed=seed,rate=rate,arm=arm,**ss,cb_clip_events=vv['cb_clip_events']))
    write_csv(args.out/'trial_metrics.csv',rows);write_csv(args.out/'probe_metrics.csv',probes)
    if sensitivity:
        central=[dict(seed=r['seed'],rate=args.rate,arm=r['arm'],**{k:v for k,v in r.items() if k not in ('seed','arm')},cb_clip_events=next(v['cb_clip_events'] for v in validations if v['seed']==r['seed'] and v['arm']==r['arm'])) for r in summaries if r['arm']!='feedback']
        write_csv(args.out/'rate_sensitivity.csv',central+sensitivity)
    config=dict(seeds=seeds,main_rate=args.rate,sensitivity_rates=[.4,args.rate,1.6] if args.sensitivity else [],
                arms=ARMS,dt_s=DT,duration_s=DURATION,distance_m=DISTANCE,delay_s=.12,
                retention=RETENTION,normalization_floor=NORMALIZATION_FLOOR,
                cortical_plasticity='readout residual only; recurrent/input connections fixed',
                cb_plasticity='output weights of fixed feature expansion',
                update='normalized feature x delayed velocity error; applied at trial end',
                matching='same pretrained model; same trial schedule, error and total rate; joint splits rate 1/2 each',
                caveat='plastic parameter counts differ: cortex 258, CB 96, joint 354. Equal total rate is not equal effective function change.',
                rotation_deg=30.,phases=[20,80,20],hidden_units=HIDDEN,cb_features=FEATURES)
    for name,data in [('summary.json',aggregate(summaries,probes)),('per_seed_summary.json',summaries),('validation.json',validations),('parameters.json',config)]:
        (args.out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
