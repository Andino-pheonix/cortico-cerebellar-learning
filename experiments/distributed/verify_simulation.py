"""Checkpoint replay and information-routing checks for all four arms."""
import json
import argparse
from pathlib import Path
import numpy as np
from distributed_simulation import DistributedModel,ARMS,DT,rotation,digest

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,default=ROOT/'results')
DATA=parser.parse_args().data
params=json.loads((DATA/'parameters.json').read_text(encoding='utf-8'))
checks=[];base_hashes={}
for seed in params['seeds']:
    for arm in ARMS:
        cp=np.load(DATA/f'checkpoint_{arm}_seed{seed}.npz')
        model=DistributedModel(seed)
        for name in ('recurrent','input','bias','expansion','expansion_bias','hidden_projection','base_readout','cortical_delta'):
            setattr(model,name,cp[name].copy())
        model.cerebellar=cp['cb_weights'].copy()
        base_hashes.setdefault(seed,set()).add(digest(model.base_readout))
        old_c,old_b=digest(model.cortical_delta),digest(model.cerebellar)
        z=np.load(DATA/f'activity_{arm}_seed{seed}.npz')
        metric,detail=model.trial()
        replay=max(float(np.max(np.abs(detail[k]-z['intact__'+k]))) for k in detail)
        _,unrotated=model.trial(theta=0.)
        no_future=(np.array_equal(detail['hidden'][:12],unrotated['hidden'][:12]) and
                   np.array_equal(detail['commands'][:12],unrotated['commands'][:12]))
        true_x=np.vstack([np.zeros(2),np.cumsum(detail['commands']/1000*DT,axis=0)])
        route_error=float(np.max(np.abs(true_x@rotation(30).T*1000-detail['display'])))
        _,blocked=model.trial(connection=False)
        _,reset=model.trial(cortical_memory=False)
        baseline,baseline_detail=model.trial(connection=False,cortical_memory=False)
        weights_fixed=digest(model.cortical_delta)==old_c and digest(model.cerebellar)==old_b
        old_c_delta=model.cortical_delta.copy();old_cb=model.cerebellar.copy()
        model.cortical_delta[:]=0;model.cerebellar[:]=0
        original,original_detail=model.trial()
        both_resets_restore=np.array_equal(baseline_detail['display'],original_detail['display'])
        assert replay<1e-10 and no_future and route_error<1e-10 and weights_fixed and both_resets_restore
        assert np.max(np.abs(blocked['cb']))==0
        if arm=='cortex':assert np.array_equal(detail['display'],blocked['display'])
        if arm=='cerebellum':assert np.array_equal(detail['display'],reset['display'])
        checks.append(dict(seed=seed,arm=arm,checkpoint_replay_difference=replay,
                           no_theta_or_future_observation_leak=no_future,
                           cortex_only_plant_route_error_mm=route_error,
                           probes_preserve_both_learned_weight_sets=weights_fixed,
                           double_reset_restores_pretrained_feedback=both_resets_restore))
assert all(len(h)==1 for h in base_hashes.values())
model=DistributedModel(params['seeds'][0]);model.train_cortex()
cp=np.load(DATA/f"checkpoint_feedback_seed{params['seeds'][0]}.npz")
repeat=float(np.max(np.abs(model.readout-cp['base_readout'])))
assert repeat<1e-7, 'Pretraining exceeds the numerical replay tolerance.'
rr=list(__import__('csv').DictReader((DATA/'trial_metrics.csv').open(encoding='utf-8-sig')))
assert len(rr)==len(params['seeds'])*4*120
validation=dict(shared_pretrained_cortex_per_seed=True,pretraining_repeatability_difference=repeat,
                trial_count=len(rr),checks=checks,
                notes='cortex_reset removes only learned readout residual; it is not M1 ablation. Equal rates do not match feature Jacobians or parameter counts.')
(DATA/'independent_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in validation.items() if k!='checks'},ensure_ascii=False,indent=2))
