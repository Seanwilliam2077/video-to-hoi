"""Assemble only the CPython 3.11 profile; copy and hash-check, never install."""
import argparse,hashlib,json,shutil
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def safe(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Path outside selected root')
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True,type=Path)
    p.add_argument('--companion',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--allow-missing',action='store_true',help='Prepare available files and report missing restricted distributions')
    a=p.parse_args();base=a.base.resolve();companion=a.companion.resolve();output=a.output.resolve()
    if any(output==root or output.is_relative_to(root) or root.is_relative_to(output) for root in (base,companion)):
        raise SystemExit('Output must not overlap either input directory')
    manifest=companion/'runtime-cp311-acquisition.json';doc=json.loads(manifest.read_text())
    if digest(base/'runtime-acquisition.json')!=doc['base_manifest_sha256']:
        raise SystemExit('Base acquisition manifest does not match the published base snapshot required by this companion')
    rows=[(base,r['base_relative_path'],r) for r in doc['reused_packages']]+[(companion,r['relative_path'],r) for r in doc['packages']]
    base_doc=json.loads((base/'runtime-acquisition.json').read_text())
    extras=base_doc.get('license_supplements',[])+base_doc.get('source_companions',[])
    for group in base_doc.get('source_inventory_extra',[]):
        if group.get('kind')=='source-files':extras+=group['files']
    if base_doc.get('eigen'):extras.append(base_doc['eigen'])
    for row in extras:rows.append((base,row['relative_path'],{**row,'name':row.get('name',row['relative_path']),'redistribution_verified':True}))
    allowed={relative for _,relative,_ in rows}
    for directory in ('wheels','sources'):
        folder=output/directory
        if folder.exists():
            foreign=[str(p.relative_to(output)) for p in folder.rglob('*') if p.is_file() and p.relative_to(output).as_posix() not in allowed]
            if foreign:raise SystemExit('Output contains files outside the selected profile: '+', '.join(foreign))
    missing=[];copied=[]
    for root,relative,row in rows:
        src=safe(root,relative);dst=safe(output,relative)
        if not src.is_file():missing.append({'name':row['name'],'path':relative,'redistribution_verified':row['redistribution_verified']});continue
        if src.stat().st_size!=row['bytes'] or digest(src)!=row['sha256']:raise SystemExit('Checksum mismatch: '+str(src))
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if digest(dst)!=row['sha256']:raise SystemExit('Existing output differs; use an empty output directory')
        else:shutil.copyfile(src,dst)
        copied.append(relative)
    report={'files_copied_or_verified':len(copied),'missing':missing,'runtime_tested':False}
    output.mkdir(parents=True,exist_ok=True)
    (output/'ASSEMBLY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if missing and not a.allow_missing:raise SystemExit(2)

if __name__=='__main__':main()
