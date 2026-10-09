#!/usr/bin/env bash
# From the public WEAR raw videos, download only the competition's training participants (sbj_0-21). For the paper's analysis only (CC BY-NC-SA 4.0).
# sbj_22 and later overlap in numbering with the competition's test participants, so never download them.
# Skip files whose size matches the server's; resume interrupted downloads (up to 20 tries).
set -uo pipefail
BASE=https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/camera
OUT=${WEAR_DATA:-data}/wear_raw_video
mkdir -p "$OUT"
for i in $(seq 0 21); do
  f=sbj_${i}.mp4
  for try in $(seq 1 20); do
    remote=$(curl -sI "$BASE/$f" | grep -i '^content-length' | tr -dc 0-9)
    local_sz=$(stat -c %s "$OUT/$f" 2>/dev/null || echo 0)
    if [ -n "$remote" ] && [ "$remote" = "$local_sz" ]; then break; fi
    echo "$(date +%T) $f (try $try, $local_sz / $remote)"
    curl -sS -C - --retry 5 -o "$OUT/$f" "$BASE/$f" || sleep 30
  done
  echo "$(date +%T) ok $f"
done
echo "done"
