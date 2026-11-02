"""Acquire exactly one approved matching checkpoint and pinned author source trees."""
from pathlib import Path
import urllib.request,tarfile,hashlib,json,subprocess,time,traceback,os
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');T=R/'mast3r-pilot';T.mkdir(exist_ok=True);(T/'downloads').mkdir(exist_ok=True)
def state(**kw):(T/'setup-state.json').write_text(json.dumps({'time':time.time(),**kw},indent=2))
def get(url,p):
 subprocess.run(['curl','--proxy','http://127.0.0.1:17897','--fail','--location','--retry','3','--retry-delay','2','--connect-timeout','20','--max-time','1800','--continue-at','-','--silent','--show-error',url,'-o',str(p)],check=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
try:
 sources=[('mast3r','naver/mast3r','f5209afc300cec36239a7ac992263f36847bbba0',T/'code'),('dust3r','naver/dust3r','3cc8c88c413bb9e34c41db0e0eef99c2ee010b12',T/'code/dust3r'),('croco','naver/croco','d7de0705845239092414480bd829228723bf20de',T/'code/dust3r/croco')]
 receipts=[]
 for name,repo,rev,dst in sources:
  state(status='fetching_author_code',component=name,revision=rev)
  archive=T/'downloads'/f'{name}-{rev}.tar.gz';get(f'https://codeload.github.com/{repo}/tar.gz/{rev}',archive);dst.mkdir(parents=True,exist_ok=True)
  with tarfile.open(archive) as tar:
   for m in tar:
    rel=Path(*Path(m.name).parts[1:])
    if not rel.parts or '..' in rel.parts or m.issym() or m.islnk():continue
    if any(t in rel.parts for t in ('assets','data','examples','checkpoints','.git')):continue
    target=dst/rel
    if m.isdir():target.mkdir(parents=True,exist_ok=True)
    elif m.isfile():
     target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(m).read())
  receipts.append({'repo':repo,'revision':rev,'archive_sha256':digest(archive),'source_only':True,'excluded_demo_assets':True})
 (T/'author-source-receipts.json').write_text(json.dumps(receipts,indent=2))
 model=T/'model';model.mkdir(exist_ok=True);base='https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/06e7259f34c3060f322df5cb0c7b9094f57e41fc/'
 state(status='fetching_model_metadata')
 for name in ['config.json','README.md']:
  get(base+name,model/name)
 state(status='fetching_single_checkpoint',expected_bytes=2754661648)
 p=model/'model.safetensors.partial';get(base+'model.safetensors',p)
 assert p.stat().st_size==2754661648,p.stat().st_size
 h=digest(p);assert h=='0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb',h
 p.rename(model/'model.safetensors')
 (T/'model-receipt.json').write_text(json.dumps({'repo':'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric','revision':'06e7259f34c3060f322df5cb0c7b9094f57e41fc','bytes':2754661648,'sha256':h,'other_models_downloaded':False},indent=2))
 state(status='ready_for_import_and_fixed_pilot',sources=receipts,model_sha256=h)
except Exception as e:
 state(status='setup_failed_no_gpu_run',error=repr(e));traceback.print_exc();raise
