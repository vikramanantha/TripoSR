#!/usr/bin/env bash
set -uo pipefail
SRC=/home/markiv/sdfer/TripoSR/sdf_dataset
DST=/mnt/ws-frb/users/markiv/sdfer/TripoSR/precomputed
rm -f "$DST/.probe"
mkdir -p "$DST/samples" "$DST/mesh_cache"
cp -f "$SRC/metadata.json" "$DST/metadata.json"

# 16 parallel rsync streams keyed on the first hex char of each dir name.
# Many small files over NFS are metadata-bound, so parallelism matters far
# more than raw throughput here. -a preserves times/perms; no --delete.
copy_tree() {  # $1 = subdir
  cd "$SRC/$1" || return 1
  for c in 0 1 2 3 4 5 6 7 8 9 a b c d e f _; do
    (
      L=$(mktemp)
      ls -d ${c}* 2>/dev/null > "$L"
      if [ -s "$L" ]; then rsync -ar --files-from="$L" . "$DST/$1/"; fi
      rm -f "$L"
    ) &
  done
  wait
}
echo "[$(date +%T)] copying samples/ ..."; copy_tree samples
echo "[$(date +%T)] copying mesh_cache/ ..."; copy_tree mesh_cache
echo "[$(date +%T)] done."
echo "src samples: $(ls "$SRC/samples" | wc -l)   dst samples: $(ls "$DST/samples" | wc -l)"
echo "src mesh:    $(ls "$SRC/mesh_cache" | wc -l)   dst mesh:    $(ls "$DST/mesh_cache" | wc -l)"
