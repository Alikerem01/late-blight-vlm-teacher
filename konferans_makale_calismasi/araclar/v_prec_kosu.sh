#!/usr/bin/env bash
# Karar D6 (hakem R1-M7): Gemma 4 E2B'nin 8-bit ve bf16 MLX dönüşümlerini indirir ve
# kör kümede (60) + aynalı kör kümede (60) aynı istem, girdi ve yazılımla koşar.
# İndirme model_indir.sh ile (curl, devam, SHA-256, sabit commit). Koşular sırayla (bellek).
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
./model_indir.sh mlx-community/gemma-4-e2b-it-8bit mlx-community/gemma-4-e2b-it-bf16 || exit $?
for m in gemma4-e2b-8bit gemma4-e2b-bf16; do
  "$PY" vlm_kosu.py --model "$m" 2>&1 | grep -v "Fetching" || exit 1
  "$PY" vlm_kosu.py --model "$m" --aynala --etiket _fliph 2>&1 | grep -v "Fetching" || exit 1
done
echo "V-PREC KOŞULARI TAMAM $(date +%H:%M:%S)"
