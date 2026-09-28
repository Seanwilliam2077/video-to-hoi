"""Copy one modern runtime profile with checksum checks; never install or run it."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Manifest path outside the selected root')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--profile', required=True, choices=('sam3', 'moge3'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = args.output.resolve()
    if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise SystemExit('Input and output directories must not overlap')
    manifest = bundle / 'profiles' / (args.profile + '.json')
    doc = json.loads(manifest.read_text(encoding='utf-8'))
    rows = doc['packages'] + doc.get('supplemental_files', [])
    allowed = {row['relative_path'] for row in rows}
    for directory in ('wheels', 'sources', 'source-companions', 'license-supplements'):
        folder = output / directory
        if folder.exists():
            foreign = [str(path.relative_to(output)) for path in folder.rglob('*')
                       if path.is_file() and path.relative_to(output).as_posix() not in allowed]
            if foreign:
                raise SystemExit('Output contains files outside this profile: ' + ', '.join(foreign))
    report_path = output / 'ASSEMBLY_REPORT.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        if previous.get('profile') != args.profile or previous.get('manifest_sha256') != digest(manifest):
            raise SystemExit('Output belongs to a different profile or inventory snapshot')
    copied = []
    missing = []
    for row in rows:
        source = safe(bundle, row['relative_path'])
        target = safe(output, row['relative_path'])
        if not source.is_file():
            missing.append({'name': row.get('name', row['relative_path']),
                            'path': row['relative_path'], 'url': row.get('url')})
            continue
        if source.stat().st_size != row['bytes'] or digest(source) != row['sha256']:
            raise SystemExit('Checksum mismatch: ' + str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
                raise SystemExit('Existing output differs; select an empty output directory')
        else:
            shutil.copyfile(source, target)
        copied.append(row['relative_path'])
    output.mkdir(parents=True, exist_ok=True)
    requirements = '\n'.join(row['name'] + '==' + row['version'] for row in doc['packages']) + '\n'
    (output / 'requirements.lock.txt').write_text(requirements, encoding='utf-8')
    shutil.copyfile(manifest, output / 'runtime-acquisition.json')
    report = {'profile': args.profile, 'manifest_sha256': digest(manifest),
              'files_copied_or_verified': len(copied), 'missing': missing,
              'installed_or_runtime_tested': False}
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if missing and not args.allow_missing:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
