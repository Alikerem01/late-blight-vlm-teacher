#!/usr/bin/env bash
# VLM ağırlıklarını curl ile indirir: kesilirse kaldığı yerden sürer, her LFS dosyası
# Hugging Face'in bildirdiği SHA-256 ile doğrulanır, sürüm sabitlenir (resolve/<commit>).
# huggingface_hub indiricisi bu ağda takıldığı için (2026-09-25) yazıldı.
#
# Kullanım: ./model_indir.sh mlx-community/Qwen3-VL-4B-Instruct-4bit [başka depolar...]
# Hedef:    $HOME/vlm_modeller/<depo adı>/   (bitince .tamam = commit)
set -uo pipefail

HEDEF_KOK="${HEDEF_KOK:-$HOME/vlm_modeller}"
PY="${PY:-python3}"
zaman() { date +%H:%M:%S; }

for depo in "$@"; do
  ad="${depo#*/}"
  dizin="$HEDEF_KOK/$ad"
  mkdir -p "$dizin"
  if [ -f "$dizin/.tamam" ]; then echo "[$ad] zaten tamam"; continue; fi
  "$PY" - "$depo" > "$dizin/.liste.tsv" <<'EOF' || { echo "[$ad] dosya listesi alınamadı"; exit 66; }
import json, sys, urllib.request
depo = sys.argv[1]
d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{depo}?blobs=true", timeout=30))
print("#commit\t" + d["sha"])
for s in d["siblings"]:
    lfs = s.get("lfs") or {}
    print(f"{s['rfilename']}\t{s.get('size') or lfs.get('size') or 0}\t{lfs.get('sha256', '')}")
EOF
  commit="$(grep '^#commit' "$dizin/.liste.tsv" | cut -f2)"
  echo "[$ad] başladı $(zaman) · commit $commit"
  while IFS=$'\t' read -r dosya boyut sha; do
    case "$dosya" in "#"*) continue ;; esac
    hedef="$dizin/$dosya"
    mkdir -p "$(dirname "$hedef")"
    if [ -f "$hedef" ] && [ "$(stat -f %z "$hedef")" = "$boyut" ]; then continue; fi
    url="https://huggingface.co/$depo/resolve/$commit/$dosya"
    for deneme in $(seq 1 30); do
      mevcut=0; [ -f "$hedef.part" ] && mevcut="$(stat -f %z "$hedef.part")"
      [ "$mevcut" = "$boyut" ] && break
      curl -fsSL -C - --connect-timeout 20 --speed-time 60 --speed-limit 20480 -o "$hedef.part" "$url" && break
      echo "[$ad] $dosya: deneme $deneme kesildi ($(stat -f %z "$hedef.part" 2>/dev/null || echo 0)/$boyut bayt), sürdürülüyor" >&2
      sleep 5
    done
    if [ -n "$sha" ]; then
      h="$(shasum -a 256 "$hedef.part" | cut -d' ' -f1)"
      if [ "$h" != "$sha" ]; then echo "[$ad] SHA-256 UYUŞMADI: $dosya — dosya silindi"; rm -f "$hedef.part"; exit 65; fi
    fi
    mv "$hedef.part" "$hedef"
    echo "[$ad] $dosya ($((boyut / 1000000)) MB) tamam $(zaman)"
  done < "$dizin/.liste.tsv"
  echo "$commit" > "$dizin/.tamam"
  echo "[$ad] MODEL TAMAM $(zaman)"
done
