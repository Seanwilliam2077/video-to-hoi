from pathlib import Path
import json,hashlib,os,shutil
root=Path('/data1/defu/zly/nvda/continuation-20261102-v1');original=root/'source-original';manifest=json.loads((root/'source-manifest.json').read_text())
changes=[];verified=0
mapping=[('/mnt/zlydisk/zly/nvda/upstream-reproduction-20261029',str(root)),('/mnt/zlydisk/zly/nvda','/data1/defu/zly/nvda')]
for row in manifest['files']:
 p=original/row['path'];assert p.stat().st_size==row['bytes'],str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==row['sha256'],str(p);verified+=1
 dst=root/row['path'];dst.parent.mkdir(parents=True,exist_ok=True)
 if row['path'] in manifest['required_binary_paths']:
  if dst.exists() or dst.is_symlink():dst.unlink()
  dst.symlink_to(os.path.relpath(p,dst.parent));continue
 data=p.read_bytes()
 try:
  s=data.decode('utf-8');adapt=s
  for old,new in mapping:adapt=adapt.replace(old,new)
  data=adapt.encode('utf-8')
 except UnicodeDecodeError:pass
 tmp=dst.with_name(dst.name+'.a6000-tmp');tmp.write_bytes(data);tmp.chmod(p.stat().st_mode&0o777);os.replace(tmp,dst)
 if hashlib.sha256(data).hexdigest()!=row['sha256']:changes.append({'path':row['path'],'source_sha256':row['sha256'],'adapted_sha256':hashlib.sha256(data).hexdigest(),'kind':'absolute path substitution only'})
report={'status':'verified_and_materialized','source_files_verified':verified,'source_bytes':manifest['total_bytes'],'source_copy_preserved':str(original),'source_changes':False,'original_a6000_tree_changed':False,'path_mapping':mapping,'adapted_files':changes,'complete_paired_reference_sha256':'61decc2bae1b4c5db67c30c7aee521edf61e5c1f5a352e70eec1f03c57234f32'}
(root/'migration-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='adapted_files'}));print('Path-adapted files:',len(changes))
