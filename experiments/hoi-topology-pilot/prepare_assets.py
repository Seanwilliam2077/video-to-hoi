"""Acquire only approved Track 1 inputs and pinned model/source bytes via mirrors."""
import concurrent.futures
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SOURCE_COMMIT = 'b5c765a0d89d789985e186d396315e7590887b94'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected is None or digest(path) == expected):
        return path
    tmp = path.with_name(path.name + '.partial')
    print('Downloading', path.name, flush=True)
    with OPENER.open(url, timeout=120) as src, tmp.open('wb') as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
    if expected and digest(tmp) != expected:
        raise ValueError('SHA256 mismatch: ' + path.name)
    tmp.replace(path)
    print('Verified', path.name, path.stat().st_size, flush=True)
    return path

def main():
    manifest = json.loads((ROOT / 'hf-models-manifest.json').read_text())
    body = next(x for x in manifest['assets'] if x['id'] == 'sam3d-body')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, 'https://gh-proxy.com/' + x['url'],
                              ROOT / 'downloads' / x['name'], x['sha256']) for x in body['parts']]
        video = pool.submit(download,
            'https://hf-mirror.com/datasets/nvidia/video_to_data_challenge/resolve/'
            '5f68335f3acc802033d1e80728c1633197521de8/track_1/videos/chunk-000/'
            'observation.images.exo_camera/episode_000016.mp4', ROOT / 'data' / 'episode_000016.mp4')
        source = pool.submit(download, 'https://gh-proxy.com/https://github.com/'
            'facebookresearch/sam-3d-body/archive/' + SOURCE_COMMIT + '.zip',
            ROOT / 'downloads' / 'sam3d-body-source.zip')
        parts = [f.result() for f in futures]
        video.result(); source.result()
    archive = ROOT / 'downloads' / body['archive']['name']
    if not archive.exists():
        with archive.open('wb') as dst:
            for path in parts:
                with path.open('rb') as src:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    if digest(archive) != body['archive']['sha256']:
        raise ValueError('Reassembled model archive hash mismatch')
    with zipfile.ZipFile(archive) as z:
        z.extractall(ROOT / 'assets')
    source_root = ROOT / 'sources' / 'sam-3d-body'
    with zipfile.ZipFile(ROOT / 'downloads' / 'sam3d-body-source.zip') as z:
        for name in z.namelist():
            rel = Path(*Path(name).parts[1:])
            if name.endswith('/') or not rel.parts:
                continue
            if rel.parts[0] not in ('sam_3d_body', 'tools', 'LICENSE', 'README.md', 'INSTALL.md'):
                continue
            dst = source_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(z.read(name))
    receipt = {'dataset': 'Track 1 only', 'episode': 16, 'original_frame_ids': list(range(90, 210)),
               'video_sha256': digest(ROOT / 'data' / 'episode_000016.mp4'),
               'model_archive_sha256': digest(archive), 'model_revision': body['revision'],
               'source_commit': SOURCE_COMMIT, 'source_archive_sha256': digest(ROOT / 'downloads' / 'sam3d-body-source.zip'),
               'mirrors': ['hf-mirror.com', 'gh-proxy.com', 'pypi.tuna.tsinghua.edu.cn'],
               'source_filter': 'Code and licenses only; no example datasets extracted.'}
    (ROOT / 'assets-receipt.json').write_text(json.dumps(receipt, indent=2))
    print('Asset preparation complete', flush=True)

if __name__ == '__main__':
    main()
