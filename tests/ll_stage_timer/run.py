"""Shared target launcher for both LL kernel branches; no implicit device selection."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
VISIBILITY=('CUDA_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','PROBE_VISIBLE_DEVICES')


def environment(devices):
    env=dict(os.environ)
    paths={'TMPDIR':'.tmp','TMP':'.tmp','TEMP':'.tmp','XDG_CACHE_HOME':'.cache','PIP_CACHE_DIR':'.cache/pip',
           'CUDA_CACHE_PATH':'.cache/cuda','CCACHE_DIR':'.cache/ccache','TORCH_EXTENSIONS_DIR':'.cache/torch_extensions',
           'TRITON_CACHE_DIR':'.cache/triton','ASCEND_CACHE_PATH':'.cache/ascend','ASCEND_WORK_PATH':'.logs/ascend',
           'ASCEND_PROCESS_LOG_PATH':'.logs/ascend','SHMEM_LOG_PATH':'.logs/shmem'}
    for key,relative in paths.items():
        path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    # Keep JIT kernels and scratch in-project unless a shared root is provided.
    if 'DEEPEP_JIT_CACHE_DIR' not in env:
        path=ROOT/'.cache/jit';path.mkdir(parents=True,exist_ok=True);env['DEEPEP_JIT_CACHE_DIR']=str(path)
    env.update({key:devices for key in VISIBILITY});env['PYTHONDONTWRITEBYTECODE']='1'
    libraries=[str(ROOT/'third-party/shmem/install/shmem/lib')]
    if env.get('DEEPEP_RDMA_PROVIDER_PATH'):
        path=Path(env['DEEPEP_RDMA_PROVIDER_PATH'])
        path=path if path.is_absolute() else ROOT/path
        if not path.is_dir(): raise ValueError('Invalid explicit RDMA provider path')
        libraries.insert(0,str(path))
    env['LD_LIBRARY_PATH']=':'.join(libraries+[env.get('LD_LIBRARY_PATH','')])
    return env


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--backend',choices=('mte','rdma'),help='Optional override; default comes from this branch')
    parser.add_argument('--devices')
    parser.add_argument('--validation',action='store_true',help='A5 CPU --plan only; fixes all device variables to 0,1')
    args,remaining=parser.parse_known_args()
    descriptor=json.loads(Path(__file__).with_name('branch.json').read_text())
    if args.validation:
        if args.devices not in (None,'0,1') or '--plan' not in remaining:
            parser.error('A5 validation is CPU --plan only, with devices 0,1')
        args.devices='0,1'
    pieces=(args.devices or '').split(',')
    if any(not p.isdecimal() for p in pieces) or len({int(p) for p in pieces})!=len(pieces):
        parser.error('Supply distinct explicit --devices numbers')
    env=environment(args.devices)
    env['DEEPEP_LL_EXPECT_IMPLEMENTATION']=descriptor['implementation']
    for key,command in [('DEEPEP_LL_RUN_REVISION',['rev-parse','HEAD']),('DEEPEP_LL_RUN_DIRTY',['status','--porcelain'])]:
        result=subprocess.run(['git','-C',str(ROOT),*command],capture_output=True,text=True)
        env[key]=(result.stdout.strip() if key.endswith('REVISION') else str(bool(result.stdout.strip()))) if result.returncode==0 else 'unknown'
    test=ROOT/'tests/elastic/test_ep_low_latency.py';st=runpy.run_path(str(test))
    options=st['parse_args'](remaining)
    args.backend=args.backend or options.backend or ('rdma' if descriptor['rdma'] else 'mte')
    if args.backend=='rdma' and not descriptor['rdma']:
        parser.error('This branch is MTE-only; RDMA cannot be requested or silently substituted')
    env['DEEPEP_LL_BACKEND']=args.backend
    explicit_processes=any(value=='--num-processes' or value.startswith('--num-processes=') for value in remaining)
    if options.config:
        config=Path(options.config);config=config if config.is_absolute() else ROOT/config
        explicit_processes=explicit_processes or 'num_processes' in json.loads(config.read_text())
    if not explicit_processes:
        remaining += ['--num-processes',str(16 if args.validation else len(pieces))]
    remaining += ['--backend',args.backend]
    options=st['parse_args'](remaining)
    world,_,_=st['launch_config'](options,env);st['validate_args'](options,world)
    local=int(env.get('LOCAL_WORLD_SIZE',options.num_processes))
    if local!=16: parser.error('Comparison launcher requires 16 contiguous ranks per host')
    if not args.validation and len(pieces)!=local: parser.error('Device count must match local ranks')
    if args.backend=='rdma' and world>16 and '--plan' not in remaining:
        subprocess.run([sys.executable,str(ROOT/'tests/ll_rdma/preflight.py')],cwd=ROOT,env=env,check=True)
    return subprocess.call([sys.executable,str(test),*remaining],cwd=ROOT,env=env)


if __name__=='__main__':sys.exit(main())
