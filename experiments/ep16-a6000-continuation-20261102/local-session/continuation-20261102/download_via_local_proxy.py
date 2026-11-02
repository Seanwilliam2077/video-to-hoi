"""Download fixed official ranges via the local proxy and stage each part over SSH."""
from pathlib import Path
import urllib.request,concurrent.futures,json,time,threading,subprocess,hashlib
B=Path('/home/lonpyer/nvda/continuation-20261102');R='/data1/defu/zly/nvda/continuation-20261102-v1';D=R+'/mast3r-pilot/model/local-proxy-parts';SIZE=2754661648;STEP=16*1024*1024;start=time.time();lock=threading.Lock();done={};failed=[]
S=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','a6000dove']
subprocess.run(S+['mkdir -p '+D],check=True)
existing=json.loads(subprocess.check_output(S+['python3 -c '+__import__('shlex').quote('from pathlib import Path;import json;print(json.dumps({p.name:p.stat().st_size for p in Path('+repr(D)+').glob("*.part")}))')]))
def status(state):
 with lock:(B/'local-proxy-download-state.json').write_text(json.dumps({'status':state,'parts_completed':len(done),'bytes_completed':sum(done.values()),'expected_bytes':SIZE,'seconds':time.time()-start,'failed':failed[-8:]},indent=2))
def one(i):
 lo=i*STEP;hi=min(SIZE,lo+STEP)-1;n=hi-lo+1;name=f'{i:04}.part'
 if existing.get(name)==n:
  with lock:done[i]=n
  status('downloading');return True
 err=None
 for attempt in range(4):
  try:
   proxy=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}))
   url='https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/06e7259f34c3060f322df5cb0c7b9094f57e41fc/model.safetensors?download=true&nvda_range='+str(lo)+'-'+str(hi)
   req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}','User-Agent':'nvda-fixed-checkpoint-transfer','Cache-Control':'no-cache'})
   with proxy.open(req,timeout=90) as r:
    cr=r.headers.get('Content-Range','');assert cr.startswith(f'bytes {lo}-'),(i,r.status,cr)
    raw=r.read(n+1);assert len(raw)==n,(i,len(raw),n)
   dst=D+'/'+name
   subprocess.run(S+['cat > '+dst+'.tmp && mv '+dst+'.tmp '+dst],input=raw,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=90,check=True)
   with lock:done[i]=n
   status('downloading')
   if i%8==0:print('Staged',len(done),'parts',sum(done.values()),'bytes',flush=True)
   return True
  except Exception as e:err=repr(e)
 with lock:failed.append({'part':i,'error':err})
 status('downloading_with_failures');return False
status('downloading')
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:results=list(ex.map(one,range((SIZE+STEP-1)//STEP)))
if not all(results):status('incomplete_parts_preserved');raise SystemExit('Some ranges failed; complete parts retained')
status('assembling_on_a6000')
script="""from pathlib import Path
import hashlib,json,os
r=Path('/data1/defu/zly/nvda/continuation-20261102-v1/mast3r-pilot');parts=sorted((r/'model/local-proxy-parts').glob('*.part'));h=hashlib.sha256();target=r/'model/model.safetensors.assembled'
with target.open('wb') as out:
 for p in parts:
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);out.write(block)
assert target.stat().st_size==2754661648
assert h.hexdigest()=='0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb',h.hexdigest()
os.replace(target,r/'model/model.safetensors')
(r/'model-receipt.json').write_text(json.dumps({'revision':'06e7259f34c3060f322df5cb0c7b9094f57e41fc','bytes':2754661648,'sha256':h.hexdigest(),'network_route':'Local existing proxy 127.0.0.1:7897; staged over authenticated SSH','other_models_downloaded':False},indent=2));print('Fixed checkpoint SHA256 verified')
"""
subprocess.run(S+['python3 -'],input=script.encode(),check=True);status('verified_complete')
