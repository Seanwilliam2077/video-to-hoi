import zipfile
from pathlib import Path
from prepare_assets import ROOT, download, digest
import json

def extract_code(archive, destination):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            rel = Path(*Path(name).parts[1:])
            if name.endswith('/') or not rel.parts:
                continue
            if rel.parts[0] in ('data', 'assets', 'examples', 'docs', '.github'):
                continue
            p = destination / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(name))

specs = [('dinov3','facebookresearch/dinov3','6876159a11b4df116f30f667f8c9888617df0751'),
         ('moge','microsoft/MoGe','925b8ed835a7a9cdb7578ba15c658a0afc969030'),
         ('utils3d','EasternJournalist/utils3d','3fab839f0be9931dac7c8488eb0e1600c236e183')]
receipts = []
for name, repo, rev in specs:
    p = download('https://gh-proxy.com/https://github.com/' + repo + '/archive/' + rev + '.zip',
                 ROOT / 'downloads' / (name + '-source.zip'))
    extract_code(p, ROOT / 'sources' / name)
    receipts.append({'name':name, 'revision':rev, 'sha256':digest(p)})
p = download('https://hf-mirror.com/Ruicheng/moge-2-vitl-normal/resolve/'
             'b135031bae30b5ac2ae141a0e68717795ce38340/model.pt', ROOT / 'assets' / 'moge2' / 'model.pt')
receipts.append({'name':'moge2-model', 'revision':'b135031bae30b5ac2ae141a0e68717795ce38340','sha256':digest(p)})
(ROOT / 'extra-assets-receipt.json').write_text(json.dumps(receipts,indent=2))
print('Extra assets ready', flush=True)
