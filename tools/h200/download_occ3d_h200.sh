#!/usr/bin/env bash
set -euo pipefail
ROOT=/storage/data/metaiot_data/huayiming/SparseWorld
TARGET="$ROOT/datasets/downloads/gts.tar.gz"
# Public transfer mirror; checksum is from its Hugging Face LFS manifest.
# Official Google Drive file is currently quota-limited. This does NOT claim
# a checksum comparison with the unavailable original archive.
curl -fsSL --retry 5 --connect-timeout 15 -C - 'https://hf-mirror.com/datasets/RoyYao233/occ3d-nuscenes-transfer/resolve/main/gts.tar.gz' -o "$TARGET.partial"
[[ $(stat -c %s "$TARGET.partial") == 2737816708 ]]
[[ $(sha256sum "$TARGET.partial" | cut -d ' ' -f1) == 0635d1383d8b99cce26a0344ec3ca3c4346f53907a06e82ea525acf7b1abba53 ]]
mv "$TARGET.partial" "$TARGET"
echo OCC3D_MIRROR_ARCHIVE_HASH_VERIFIED
# Validate archive member paths before extraction.
python3 - "$TARGET" <<'PY'
import sys,tarfile,pathlib
with tarfile.open(sys.argv[1]) as archive:
 for member in archive:
  p=pathlib.PurePosixPath(member.name)
  if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk():
   raise ValueError(member.name)
PY
tar -xzf "$TARGET" -C "$ROOT/datasets/occ3d"
echo OCC3D_EXTRACT_COMPLETE
