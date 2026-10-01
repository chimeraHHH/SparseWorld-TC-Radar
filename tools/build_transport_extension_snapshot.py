"""Reconstruct immutable original scientific source plus additive committed overlays."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

BASE='2fc2feaf5edbf0e1bc839b6b34c694be7e85af83'
OVERLAYS=(
    'configs/sw-radar-transport-reliable-extend20.py',
    'configs/sw-radar-transport-reliable-extend20-smoke.py',
    'tools/transport_extension_spool.py','tools/transport_extension_hooks.py',
    'tools/train_transport_extension.py','tools/evaluate_transport_extension.py',
    'tools/check_transport_extension.py','tools/run_transport_extension.py',
    'tools/build_transport_extension_snapshot.py',
    'tests/test_transport_extension_resume.py','tests/test_transport_extension_spool.py',
    'docs/TRANSPORT_RELIABLE_EXTEND20_20261001.md')


def build(output):
    root=Path(__file__).resolve().parents[1]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    blob=subprocess.check_output(['git','archive',BASE],cwd=root)
    files={}
    with tarfile.open(fileobj=io.BytesIO(blob)) as source:
        for entry in source:
            if entry.isfile(): files[entry.name]=source.extractfile(entry).read()
    for name in OVERLAYS:
        # Read committed bytes, never a mutable worktree or uncommitted script.
        files[name]=subprocess.check_output(['git','show',f'{revision}:{name}'],cwd=root)
    manifest=dict(git_revision=revision,scientific_base_revision=BASE,overlays=list(OVERLAYS),
        sha256={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
    with tarfile.open(output,'w') as target:
        for name,data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o644
            target.addfile(member,io.BytesIO(data))
        data=json.dumps(manifest,indent=2).encode();member=tarfile.TarInfo('code_manifest.json');member.size=len(data)
        target.addfile(member,io.BytesIO(data))
    return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    m=build(args.out);print(json.dumps({k:v for k,v in m.items() if k!='sha256'},indent=2))
