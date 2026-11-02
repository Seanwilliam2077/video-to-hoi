from pathlib import Path
import hashlib,json,os
root=Path('/mnt/zlydisk/zly/nvda/upstream-reproduction-20261029')
skip={'.git','.venv','__pycache__','runtime-deps','runtime-tools','weights','artifacts'}
exts={'.py','.sh','.md','.txt','.json','.yaml','.yml','.toml','.cfg','.ini','.cpp','.cu','.c','.h','.hpp','.html','.css','.js','.svg'}
required=['experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth','runs/full-episode16-20261030/object-refinement-r2/episode_000016/foam_grass_block/refined.pth','runs/full-episode16-20261030/object-refinement-r2/episode_000016/foam_grass_block/object.glb','runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4']
files=[]
for parent,ds,fs in os.walk(root):
 ds[:]=[d for d in ds if d not in skip]
 for name in fs:
  p=Path(parent)/name
  if p.is_symlink() or p.suffix.lower() not in exts or p.stat().st_size>16*1024*1024:continue
  if any(t in p.relative_to(root).parts for t in ('track_2','track2','track_3','track3')):continue
  files.append(p)
files.extend(root/p for p in required)
rows=[]
for p in sorted(set(files)):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 rows.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps({'source_root':str(root),'files':rows,'required_binary_paths':required,'total_bytes':sum(x['bytes'] for x in rows)}))
