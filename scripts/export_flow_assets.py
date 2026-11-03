"""Export existing preprocessed PAC-NeRF scenes separately from Git.

The archive restores model/pacnerf/... under a fresh clone. It includes all
scene files so optional masks, geometry and moving-point selections survive.
No model reconstruction or prompt generation is needed on the destination.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True, help='Original PhysFlow workspace containing model/pacnerf')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--scenes', nargs='+', default=['letter','droplet','torus','bird','playdoh','cat','cream','toothpaste','trophy'])
a = p.parse_args()
if a.output.exists() or Path(str(a.output)+'.sha256').exists():
    raise FileExistsError(a.output)
scene_paths = [a.source/'model/pacnerf'/s for s in a.scenes]
for path in scene_paths:
    if not (path/'images_generated').is_dir() or not (path/'cameras.json').is_file():
        raise FileNotFoundError(path)
a.output.parent.mkdir(parents=True, exist_ok=True)
manifest = {}
temporary = a.output.with_suffix(a.output.suffix+'.partial')
with tarfile.open(temporary, 'w:gz', compresslevel=1, dereference=True) as archive:
    for path in scene_paths:
        for file in sorted(path.rglob('*')):
            if file.is_file():
                relative = file.relative_to(a.source).as_posix()
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
                manifest[relative] = dict(bytes=file.stat().st_size, sha256=digest)
                archive.add(file, arcname=relative, recursive=False)
        print('PACKED',path.name,flush=True)
temporary.replace(a.output)
digest = hashlib.sha256()
with a.output.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024), b''):
        digest.update(chunk)
Path(str(a.output)+'.sha256').write_text(digest.hexdigest()+'  '+a.output.name+'\n')
Path(str(a.output)+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('DONE',a.output,'files',len(manifest),'bytes',a.output.stat().st_size,flush=True)
