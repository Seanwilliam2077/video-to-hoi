"""Independent transport fixtures only; no challenge media or model outputs."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
SPEC = importlib.util.spec_from_file_location('object_catalog', ROOT/'tools/build_object_catalog.py')
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def transport(tmp_path, monkeypatch):
    package = tmp_path/'package'
    package.mkdir()
    delivery = 'independent-object-fixture'
    base = 'https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-'+delivery
    archive = tmp_path/(delivery+'.zip')
    archive.write_bytes(b'Independent archive identity fixture, not a real ZIP')
    checks = tmp_path/'producer-checks.json'
    save(checks, dict(passed=True, videos_fully_decoded=True, archive_crc_and_hashes_verified=True,
                     episodes=30, original_frames=150))
    files = [archive, checks]
    episodes, records = [], []
    for ep in range(30):
        eid = f'episode_{ep:06d}'
        path = package/'episodes'/eid/'hoi.mp4'
        path.parent.mkdir(parents=True)
        path.write_bytes(f'Independent media identity fixture {ep}, not real video'.encode())
        files.append(path)
        episodes.append(dict(episode=ep, frames=5, object='test_cube', source_video_sha256='a'*64,
                             selected_result_sha256='b'*64, artifacts=dict(hoi_video=f'episodes/{eid}/hoi.mp4')))
        for stage in ('objects', 'motion'):
            records.append(dict(id=f'fixture-{ep:02d}-{stage}', episode_id=eid, stage=stage, iteration='fixture-v1',
                state='submitted', kind='real_track1', created_at='2026-11-02T12:00:00Z', contributor='Independent fixture',
                reviewer='', summary='Synthetic metadata only', delivery_url=base, source_commit='', upstream=[],
                coverage=dict(frames=5, objects_complete=True), blocker='Review pending', next_action='Inspect',
                artifacts=[dict(requirement_id=req, label='Fixture', url=base+'/'+archive.name,
                                sha256=catalog.sha256(archive), bytes=archive.stat().st_size)
                           for req in (('object_record', 'mesh') if stage == 'objects' else ('transforms',))],
                checks=dict(structural=dict(status='pass', evidence_url=base+'/'+checks.name),
                            quality=dict(status='not_run', evidence_url=''), export=dict(status='not_run', evidence_url=''))))
    manifest = package/'manifest.json'
    save(manifest, dict(schema='v2hoi.object-handoff.v1', provenance='track1', official_submission=False,
                        delivery_id=delivery, episodes=episodes))
    files.append(manifest)
    assets = []
    for path in files:
        name = path.name if path.name != 'hoi.mp4' else path.parent.name+'_test_cube.mp4'
        assets.append(dict(name=name, url=base+'/'+name, sha256=catalog.sha256(path), bytes=path.stat().st_size,
                           server_digest_verified=True, anonymous_download_verified=True))
    location = tmp_path/'deliveries/object'/delivery
    save(location/'checks/publication.json', dict(complete=True, published=True, assets=assets))
    save(location/'dashboard-submitted-records.json', dict(schema_version='hoi_progress.v1', records=records))
    destination = tmp_path/'docs/progress-board/delivery-view.js'
    destination.parent.mkdir(parents=True)
    destination.write_bytes((ROOT/'docs/progress-board/delivery-view.js').read_bytes())
    monkeypatch.setattr(catalog, 'load_episodes', lambda root: [dict(id=f'episode_{ep:06d}', frames=5) for ep in range(30)])
    monkeypatch.setattr(catalog, 'read_records', lambda root, eps: records)
    return tmp_path, package, archive, checks, location, records


def test_exact_bytes_and_records_build_without_decoding_or_accepting(transport):
    root, package, archive, checks, _, records = transport
    value = catalog.build(root, package, archive, checks)
    catalog.validate(value, root)
    assert len(value['entries']) == 60
    assert all(not row['quality_accepted'] and not row['official_export_accepted'] for row in value['entries'])
    # Another record may not silently inherit the same media.
    records[0]['summary'] = 'Changed immutable declaration'
    with pytest.raises(ValueError, match='exact immutable'):
        catalog.validate(value, root)


def test_modified_local_video_is_rejected_before_preview_publication(transport):
    root, package, archive, checks, _, _ = transport
    (package/'episodes/episode_000000/hoi.mp4').write_bytes(b'Modified independent transport fixture')
    with pytest.raises(ValueError, match='Local release bytes changed'):
        catalog.build(root, package, archive, checks)


def test_unverified_public_download_is_rejected(transport):
    root, package, archive, checks, location, _ = transport
    value = catalog.build(root, package, archive, checks)
    path = location/'checks/publication.json'
    receipt = json.loads(path.read_text())
    receipt['assets'][0]['anonymous_download_verified'] = False
    save(path, receipt)
    with pytest.raises(ValueError, match='public-download'):
        catalog.validate(value, root)
