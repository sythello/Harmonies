"""Resumable multi-process audit; never changes training models or replay."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from variance_report import build_report

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--state-limit', type=int, default=0, help='Smoke tests only; 0 evaluates every state')
    args = parser.parse_args()
    directory = args.out.resolve(); directory.mkdir(parents=True, exist_ok=True)
    if min(args.games,args.workers) < 1 or args.repeats < 2 or args.state_limit < 0:
        parser.error('Invalid configuration')
    with (directory/'run.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: parser.error('Another audit process owns this directory')
        if (directory/'config.json').exists():
            config = json.loads((directory/'config.json').read_text())
            if args.model and str(args.model.resolve()) != config['modelSource']:
                parser.error('Model differs from saved run')
        else:
            if not args.model:parser.error('--model required for a new audit')
            model = args.model.resolve(); raw = model.read_bytes()
            config = dict(games=args.games,repeats=args.repeats,workers=args.workers,seed=args.seed,
                          stateLimit=args.state_limit,simulations=16,modelSource=str(model),
                          modelHash=hashlib.sha256(raw).hexdigest(),createdAt=time.time())
            atomic(directory/'config.json',config); (directory/'model.json').write_bytes(raw)
        assert hashlib.sha256((directory/'model.json').read_bytes()).hexdigest()==config['modelHash']
        if (directory/'verification.json').exists() and json.loads((directory/'state.json').read_text())['phase']=='complete':
            print('Already complete');return
        node=shutil.which('node')
        if not node:raise RuntimeError('Node not found')
        started=time.time();processes=[];logs=[]
        # Worker locks also survive an unexpectedly terminated coordinator.
        worker_locks=[]
        for worker in range(config['workers']):
            worker_lock=(directory/f'worker-{worker}.lock').open('a+')
            try:fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('An earlier worker still owns this run; do not start duplicates')
            worker_locks.append(worker_lock)
        try:
            for worker,worker_lock in enumerate(worker_locks):
                log=(directory/f'worker-{worker}.log').open('a');logs.append(log)
                command=[node,'--import','tsx','scripts/value-variance.ts','--out',str(directory),'--worker',str(worker)]
                processes.append(subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(worker_lock.fileno(),)))
            atomic(directory/'process.json',{'pid':os.getpid(),'workerPids':[p.pid for p in processes], 'startedAt':started})
            while True:
                statuses=[json.loads(p.read_text()) for p in sorted(directory.glob('worker-*.json'))]
                status={'phase':'running','baseGamesCompleted':sum(s['gamesDone'] for s in statuses),
                        'statesCompleted':sum(s['statesDone'] for s in statuses),
                        'continuationsCompleted':sum(s['statesDone'] for s in statuses)*config['repeats'],
                        'rolloutActions':sum(s['rolloutActions'] for s in statuses),'updatedAt':time.time(),
                        'freeDiskGiB':shutil.disk_usage(directory).free/2**30,'workers':statuses}
                atomic(directory/'state.json',status)
                failed=[p.returncode for p in processes if p.poll() not in (None,0)]
                if failed:raise RuntimeError(f'Worker failed: {failed}; see worker logs')
                if all(p.poll()==0 for p in processes):break
                time.sleep(5)
            atomic(directory/'state.json',{**status,'phase':'verifying'})
            with (directory/'verify.log').open('a') as log:
                subprocess.run([node,'--import','tsx','scripts/value-variance.ts','--out',str(directory),'--verify'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=build_report(directory)
            replay=json.loads((directory/'replay-verification.json').read_text())
            assert replay['passed'] and replay['states']==result['states'] and replay['actions']==result['continuationActions']
            verification={'passed':True,'replay':replay,'aggregation':'Independent NumPy means/sample variances and total variance/MSE identities verified',
                          'states':result['states'],'continuations':result['continuations'],'wallSeconds':time.time()-started,
                          'configHash':hashlib.sha256((directory/'config.json').read_bytes()).hexdigest()}
            atomic(directory/'verification.json',verification)
            atomic(directory/'state.json',{**status,'phase':'complete','statesCompleted':result['states'],'continuationsCompleted':result['continuations'],'wallSeconds':time.time()-started})
            print(json.dumps(verification),flush=True)
        except BaseException as error:
            for p in processes:
                if p.poll() is None:p.terminate()
            for p in processes:p.wait()
            atomic(directory/'state.json',{'phase':'failed','error':str(error),'updatedAt':time.time()})
            raise
        finally:
            for log in logs:log.close()
            for worker_lock in worker_locks:worker_lock.close()


if __name__=='__main__':run()
