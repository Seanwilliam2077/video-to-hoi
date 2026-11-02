"""Task-local path adapter; immutable source bundles are mapped only in memory."""
import sys,os,runpy
from pathlib import Path
ROOT=Path('/data1/defu/zly/nvda/continuation-20261102-v1')
sys.path.insert(0,str(ROOT/'a6000-deps'))
sys.path.insert(0,str(ROOT/'src'))
sys.executable=str(ROOT/'.venv/bin/python')
import torch
load_original=torch.load
MAPPING=[('/mnt/zlydisk/zly/nvda/upstream-reproduction-20261029',str(ROOT)),('/mnt/zlydisk/zly/nvda','/data1/defu/zly/nvda')]
def map_paths(value):
    if isinstance(value,str):
        for old,new in MAPPING:value=value.replace(old,new)
        return value
    if isinstance(value,Path):return Path(map_paths(str(value)))
    if isinstance(value,dict):return {map_paths(k):map_paths(v) for k,v in value.items()}
    if isinstance(value,list):return [map_paths(v) for v in value]
    if isinstance(value,tuple):return tuple(map_paths(v) for v in value)
    return value
def load_mapped(*a,**kw):return map_paths(load_original(*a,**kw))
torch.load=load_mapped
args=sys.argv[1:]
if not args:raise SystemExit('Pass a task script, -c, -m, or -')
if args[0]=='-c':sys.argv=['-c',*args[2:]];exec(compile(args[1],'<string>','exec'),{'__name__':'__main__'})
elif args[0]=='-m':sys.argv=args[1:];runpy.run_module(args[1],run_name='__main__',alter_sys=True)
elif args[0]=='-':sys.argv=args;exec(compile(sys.stdin.read(),'<stdin>','exec'),{'__name__':'__main__'})
else:
    sys.argv=args;sys.path.insert(0,str(Path(args[0]).resolve().parent));runpy.run_path(args[0],run_name='__main__')
