"""Resume public official checkpoint in bounded ranges, with complete SHA verification."""
from pathlib import Path
import urllib.request,concurrent.futures,json,time,hashlib,os,traceback,threading
T=Path('/data1/defu/zly/nvda/continuation-20261102-v1/mast3r-pilot');D=T/'model/mirror-parts';D.mkdir(exist_ok=True)
SIZE=2754661648;STEP=8*1024*1024;URL='https://hf-mirror.com/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/06e7259f34c3060f322df5cb0c7b9094f57e41fc/model.safetensors?download=true'
start=time.time();lock=threading.Lock();completed={};failures=[]
def state(status):
 with lock:(T/'mirror-download-state.json').write_text(json.dumps({'status':status,'parts_completed':len(completed),'bytes_completed':sum(completed.values()),'expected_bytes':SIZE,'seconds':time.time()-start,'failures':failures[-4:]},indent=2))
def one(i):
 lo=i*STEP;hi=min(SIZE,lo+STEP)-1;n=hi-lo+1;p=D/f'{i:04}.part'
 if p.exists() and p.stat().st_size==n:
  with lock:completed[i]=n
  return i
 err=None
 for attempt in range(4):
  try:
   op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
   req=urllib.request.Request(URL+'&nvda_range='+str(lo)+'-'+str(hi),headers={'Range':f'bytes={lo}-{hi}','User-Agent':'nvda-fixed-checkpoint-transfer','Cache-Control':'no-cache'})
   with op.open(req,timeout=90) as r:
    cr=r.headers.get('Content-Range','')
    if not cr.startswith(f'bytes {lo}-'):raise RuntimeError(f'Range not honored for part {i}; response {r.status}, Content-Range {cr}')
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:
     remaining=n
     while remaining:
      b=r.read(min(1024*1024,remaining))
      if not b:raise EOFError(f'part {i} truncated')
      f.write(b);remaining-=len(b)
   assert tmp.stat().st_size==n;os.replace(tmp,p)
   with lock:completed[i]=n
   state('downloading');return i
  except Exception as e:err=repr(e)
 with lock:failures.append({'part':i,'error':err})
 state('downloading_with_failed_parts');return None
try:
 state('downloading')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:done=list(ex.map(one,range((SIZE+STEP-1)//STEP)))
 if any(x is None for x in done):raise RuntimeError('Some exact ranges failed; verified-size parts preserved for resume')
 state('assembling_and_hashing');target=T/'model/model.safetensors.assembled';h=hashlib.sha256()
 with target.open('wb') as f:
  for i in range(len(done)):
   with (D/f'{i:04}.part').open('rb') as src:
    for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b);f.write(b)
 assert target.stat().st_size==SIZE;assert h.hexdigest()=='0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb',h.hexdigest()
 os.replace(target,T/'model/model.safetensors')
 (T/'model-receipt.json').write_text(json.dumps({'revision':'06e7259f34c3060f322df5cb0c7b9094f57e41fc','sha256':h.hexdigest(),'bytes':SIZE,'source':'hf-mirror.com, exact pinned revision; final original SHA256 verified','other_models_downloaded':False},indent=2));state('verified_complete')
except Exception as e:state('failed_parts_preserved');traceback.print_exc();raise
