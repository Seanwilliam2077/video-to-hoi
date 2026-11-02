from pathlib import Path
import subprocess,json,shlex,time
b=Path('/home/lonpyer/nvda/continuation-20261102');d=json.loads((b/'source-manifest.json').read_text())
snapshot='/data1/defu/zly/nvda/continuation-20261102-v1'
ssh_a=['ssh','-o','BatchMode=yes','a6000dove']
ssh_s=['ssh','-S','/tmp/nvda-1080ti-live.sock','-o','BatchMode=yes','zly@121.48.165.39']
prep=f'test ! -e {snapshot} && mkdir -p {snapshot} && cp -al /data1/defu/zly/nvda/upstream-reproduction-20261029/. {snapshot}/ && mkdir {snapshot}/source-original'
subprocess.run(ssh_a+[prep],check=True)
source_code="""import sys,json,tarfile
from pathlib import Path
root=Path('/mnt/zlydisk/zly/nvda/upstream-reproduction-20261029')
d=json.load(sys.stdin)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=3) as t:
 for x in d['files']:
  p=Path(x['path']); assert not p.is_absolute() and '..' not in p.parts
  t.add(root/p,arcname=str(p),recursive=False)
"""
with (b/'transfer-source.log').open('wb') as err1,(b/'transfer-destination.log').open('wb') as err2:
 dest=subprocess.Popen(ssh_a+[f'tar xzf - --no-same-owner -C {snapshot}/source-original'],stdin=subprocess.PIPE,stderr=err2)
 src=subprocess.Popen(ssh_s+['python3 -c '+shlex.quote(source_code)],stdin=subprocess.PIPE,stdout=dest.stdin,stderr=err1)
 dest.stdin.close();src.stdin.write(json.dumps(d).encode());src.stdin.close()
 (b/'transfer-state.json').write_text(json.dumps({'status':'transferring','source_pid':src.pid,'destination_pid':dest.pid,'snapshot':snapshot,'files':len(d['files']),'uncompressed_bytes':d['total_bytes'],'started':time.time()},indent=2))
 rc1=src.wait();rc2=dest.wait()
 if rc1 or rc2:raise RuntimeError(f'transfer failed source={rc1} destination={rc2}')
subprocess.run(ssh_a+[f'cat > {snapshot}/source-manifest.json'],input=json.dumps(d,indent=2).encode(),check=True)
(b/'transfer-state.json').write_text(json.dumps({'status':'transferred_pending_hash_verification','snapshot':snapshot,'files':len(d['files']),'bytes':d['total_bytes'],'completed':time.time()},indent=2))
print('Transferred',len(d['files']),'files to',snapshot)
