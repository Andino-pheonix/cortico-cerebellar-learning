"""Portable workflow; reference results stay separate from regenerated outputs."""
import argparse,csv,json,subprocess,sys,platform
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/distributed'
TUNE=BASE/'tuning'

def run(*args):
    subprocess.run([sys.executable,'-B',*[str(a) for a in args]],cwd=ROOT,check=True)

def test():
    run('-m','unittest','discover','-s',ROOT/'tests','-v')

def compare_csv(expected,actual):
    with expected.open(encoding='utf-8-sig') as f:a=list(csv.DictReader(f))
    with actual.open(encoding='utf-8-sig') as f:b=list(csv.DictReader(f))
    if len(a)!=len(b):raise AssertionError(f'Row count differs: {expected.name}')
    delta=0.
    for aa,bb in zip(a,b):
        if set(aa)!=set(bb):raise AssertionError(f'Columns differ: {expected.name}')
        for key in aa:
            try:
                va,vb=float(aa[key]),float(bb[key])
            except ValueError:
                if aa[key]!=bb[key]:raise AssertionError(f'Text differs: {expected.name}, {key}')
                continue
            if not (__import__('math').isfinite(va) and __import__('math').isfinite(vb)):
                raise AssertionError('Non-finite data in comparison')
            delta=max(delta,abs(va-vb))
    if delta>1e-6:raise AssertionError(f'Numerical difference {delta} exceeds tolerance: {expected.name}')
    return dict(rows=len(a),max_numeric_difference=delta)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','verify','full'],default='verify')
    args=ap.parse_args();test()
    if args.mode=='smoke':return
    if args.mode=='verify':
        run(BASE/'verify_simulation.py')
        run(TUNE/'verify_tuning.py')
        return
    out=ROOT/'reproduced/distributed';data=out/'results';td=out/'tuning/results'
    run(BASE/'distributed_simulation.py','--sensitivity','--out',data)
    run(TUNE/'tune_learning.py','--phase','all','--retrain-pretraining','--out',td)
    run(BASE/'verify_simulation.py','--data',data)
    run(TUNE/'verify_tuning.py','--data',td)
    run(BASE/'render_figures.py','--data',data,'--out',out/'figures')
    run(TUNE/'render_tuning.py','--data',td,'--out',out/'tuning/figures')
    comparisons={}
    for file in ['trial_metrics.csv','probe_metrics.csv','rate_sensitivity.csv']:
        comparisons['distributed/'+file]=compare_csv(BASE/'results'/file,data/file)
    for file in ['search_per_seed.csv','search_aggregate.csv','test_trials.csv','test_probes.csv','test_per_seed.csv','test_paired_differences.csv','probe_paired_changes.csv']:
        comparisons['tuning/'+file]=compare_csv(TUNE/'results'/file,td/file)
    expected=json.loads((TUNE/'results/frozen_selection.json').read_text(encoding='utf-8'))
    actual=json.loads((td/'frozen_selection.json').read_text(encoding='utf-8'))
    for kind in ['all','joint','cortex','cerebellum']:
        for key in ['eta_c','eta_cb']:
            if expected[kind][key]!=actual[kind][key]:
                raise AssertionError(f'Selected coefficient changed: {kind} {key}')
    import numpy as np
    validation=dict(status='passed',reference_data_unchanged=True,tolerance=1e-6,
                    coefficients_match=True,python=sys.version.split()[0],numpy=np.__version__,
                    operating_system=platform.system(),csv_comparisons=comparisons)
    (ROOT/'reproduced/REPRODUCTION_CHECK.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(validation,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
