"""Download exact runtime distributions; no installation or third-party execution."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener, Request

ALLOWED_HOSTS = {
    "files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org",
    "nvidia-kaolin.s3.us-east-2.amazonaws.com", "pypi.nvidia.com"
}

class AllowedRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.hostname not in ALLOWED_HOSTS:
            raise ValueError("Redirect outside approved distribution hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)

OPENER = build_opener(AllowedRedirects())

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('runtime-acquisition.json'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--names', nargs='+', help='Optional exact package names from the manifest')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    doc = json.loads(args.manifest.read_text(encoding='utf-8'))
    rows = doc['packages']
    if args.names:
        unknown = set(args.names) - {r['name'] for r in rows}
        if unknown:
            raise SystemExit('Unknown package names: ' + ', '.join(sorted(unknown)))
        rows = [r for r in rows if r['name'] in args.names]
    root = args.root.resolve()
    for row in rows:
        path = (root / row['relative_path']).resolve()
        if not path.is_relative_to(root):
            raise SystemExit('Unsafe manifest path')
        parsed = urlsplit(row['url'])
        if parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS:
            raise SystemExit('Unexpected download origin')
        valid = path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
        if valid:
            print(row['name'] + ': verified')
            continue
        if args.verify_only:
            raise SystemExit(row['name'] + ': missing or checksum mismatch')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.part')
        print(row['name'] + ': retrieving original upstream distribution', flush=True)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > row['bytes']:
            raise SystemExit('Partial file is larger than manifest entry')
        with partial.open('ab' if offset else 'wb') as output:
            while offset < row['bytes']:
                end = min(offset + 8 * 1024 * 1024, row['bytes']) - 1
                request = Request(row['url'], headers={'Range': f'bytes={offset}-{end}'})
                with OPENER.open(request, timeout=180) as response:
                    if response.status != 206:
                        raise SystemExit('Origin did not honor byte-range request')
                    block = response.read()
                    if len(block) != end - offset + 1:
                        raise SystemExit('Incomplete byte-range response')
                    output.write(block)
                offset = end + 1
        if partial.stat().st_size != row['bytes'] or digest(partial) != row['sha256']:
            raise SystemExit(row['name'] + ': upstream checksum mismatch; partial file retained')
        partial.replace(path)
        print(row['name'] + ': verified')

if __name__ == '__main__':
    main()
