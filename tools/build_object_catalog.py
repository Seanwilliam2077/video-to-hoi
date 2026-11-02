"""Bind M3 HOI previews to exact submitted records and verified release bytes.

Build reads only the explicit Track 1 handoff and hashes its local files.
--check is offline: validate committed declarations/publication evidence without
loading videos, model outputs or reference data. Neither path grants acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

try:
    from tools.progress_board import load_json, load_episodes, read_records
except ModuleNotFoundError:
    from progress_board import load_json, load_episodes, read_records

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate(catalog, root):
    script = "const fs=require('node:fs');const api=require(process.argv[1]);const errors=api.validateObjectCatalog(JSON.parse(fs.readFileSync(0,'utf8')));if(errors.length){process.stderr.write(errors.join('\\n'));process.exit(1)}"
    result = subprocess.run(['node', '-e', script, str(root/'docs/progress-board/delivery-view.js')],
                            input=json.dumps(catalog), text=True, capture_output=True)
    require(result.returncode == 0, 'Invalid Object catalog: '+result.stderr.strip())
    episodes = {row['id']: row for row in load_episodes(root)}
    records = {row['id']: row for row in read_records(root, list(episodes.values()))}
    for entry in catalog['entries']:
        record = entry['record']
        require(records.get(record['id']) == record, 'Preview differs from its exact immutable record')
        require(entry['frame_count'] == episodes[entry['episode_id']]['frames'], 'Incomplete inventory frame coverage')
        # The strict browser schema has already checked these safe release filenames.
        delivery = entry['archive']['url'].rsplit('/', 1)[1][:-4]
        publication = load_json(root/'deliveries/object'/delivery/'checks/publication.json')
        require(publication.get('complete') is True and publication.get('published') is True,
                'Missing completed publication evidence')
        published = {row['url']: row for row in publication['assets']}
        require(len(published) == len(publication['assets']), 'Duplicate published asset URL')
        for key in ('archive', 'manifest', 'checks', 'preview'):
            asset = entry[key]
            actual = published.get(asset['url'], {})
            require(actual.get('server_digest_verified') is True and actual.get('anonymous_download_verified') is True
                    and all(actual.get(field) == asset[field] for field in ('bytes', 'sha256')),
                    'Preview lacks exact public-download byte verification')


def build(root, package, archive, producer_checks):
    manifest = load_json(package/'manifest.json')
    require(manifest.get('schema') == 'v2hoi.object-handoff.v1' and manifest.get('provenance') == 'track1'
            and manifest.get('official_submission') is False, 'Require permitted Track 1 research delivery')
    rows = manifest['episodes']
    require(len(rows) == 30 and {row['episode'] for row in rows} == set(range(30)), 'Require exact all-30 delivery')
    delivery = manifest['delivery_id']
    require(archive.name == delivery+'.zip', 'Archive name differs from delivery ID')
    require(delivery and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-.' for c in delivery)
            and '/' not in delivery and '..' not in delivery, 'Unsafe delivery identity')
    publication = load_json(root/'deliveries/object'/delivery/'checks/publication.json')
    published = {row['name']: row for row in publication['assets']}
    checks = load_json(producer_checks)
    require(all(checks.get(k) is True for k in ('passed', 'videos_fully_decoded', 'archive_crc_and_hashes_verified'))
            and checks['episodes'] == 30 and checks['original_frames'] == sum(row['frames'] for row in rows),
            'Require complete portable-scene and full-video producer checks')

    def asset(name, path):
        remote = published[name]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == remote['bytes']
                and sha256(path) == remote['sha256'], 'Local release bytes changed: '+name)
        return {key: remote[key] for key in ('url', 'sha256', 'bytes')}

    common = dict(archive=asset(archive.name, archive), manifest=asset('manifest.json', package/'manifest.json'),
                  checks=asset('producer-checks.json', producer_checks))
    envelope = load_json(root/'deliveries/object'/delivery/'dashboard-submitted-records.json')
    records = envelope['records']
    require(envelope['schema_version'] == 'hoi_progress.v1' and len(records) == 60, 'Require 60 submitted M3 declarations')
    entries = []
    for row in rows:
        eid = f'episode_{row["episode"]:06d}'
        relative = row['artifacts']['hoi_video']
        require(relative == f'episodes/{eid}/hoi.mp4', 'Unexpected local video path')
        video = asset(f'{eid}_{row["object"]}.mp4', package/relative)
        matches = [record for record in records if record['episode_id'] == eid]
        require(len(matches) == 2 and {record['stage'] for record in matches} == {'objects', 'motion'},
                'Each episode needs one Objects and one Motion declaration')
        for record in matches:
            entries.append(dict(record=record, episode_id=eid, profile='temporal_object_hoi_review.v1',
                frame_count=row['frames'], object_name=row['object'], source_video_sha256=row['source_video_sha256'],
                selected_result_sha256=row['selected_result_sha256'], **common,
                preview=dict(**video, content_type='video/mp4', original_frame_count=row['frames']),
                quality_accepted=False, metric_scale_verified=False, official_export_accepted=False))
    return dict(schema_version='hoi_object_catalog.v1', entries=sorted(entries, key=lambda entry: entry['record']['id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--package', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--producer-checks', type=Path)
    args = parser.parse_args()
    output = args.root/'tracking/object-catalog.json'
    try:
        if args.check:
            require(not any((args.package, args.archive, args.producer_checks)), '--check accepts no build inputs')
            catalog = load_json(output)
        else:
            require(all((args.package, args.archive, args.producer_checks)), 'Building requires package, archive and producer checks')
            catalog = build(args.root, args.package, args.archive, args.producer_checks)
            if output.exists():
                previous = load_json(output)
                validate(previous, args.root)
                incoming = {row['record']['id']: row for row in catalog['entries']}
                for row in previous['entries']:
                    require(row['record']['id'] not in incoming or incoming[row['record']['id']] == row,
                            'Cannot rewrite an existing immutable preview entry')
                    incoming[row['record']['id']] = row
                catalog['entries'] = sorted(incoming.values(), key=lambda row: row['record']['id'])
        validate(catalog, args.root)
        if not args.check:
            temporary = output.with_suffix('.tmp')
            temporary.write_text(json.dumps(catalog, indent=2)+'\n', encoding='utf-8')
            temporary.replace(output)
        print(json.dumps(dict(entries=len(catalog['entries']), written=not args.check,
            scope='Preview declarations and prior publication evidence; no quality/native/export acceptance')))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('Object catalog failed: '+str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
