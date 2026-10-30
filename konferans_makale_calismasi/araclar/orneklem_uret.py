"""Uzman değerlendirme örneklemi için sıralı rastgele liste üretir — SALT OKUMA.

Uygun görüntüler: 301 özgünden, araç testinde bizim düzenlediğimiz 3 özgün
çıkarılır (298). `--haric` ile sunucuda protokolden önce incelenmiş görüntüler
de çıkarılabilir (sunucu dışa aktarımından gelen, satır başına bir ad içeren
dosya). Liste tohum 42 ile karıştırılır; uzmanlar bu SIRAYLA çalışır. Rastgele
bir sıralamanın her ön kesiti de rastgele örneklemdir; son tarihte kaç görüntü
bitmişse o n raporlanır. Bir öğeyi listeden çıkarmak kalanların rastgeleliğini
bozmaz.

Çıktı: veri/uzman_orneklem_sirasi.csv (sira, goruntu, koleksiyon, sahne)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

VERI = Path(__file__).resolve().parent.parent / "veri"
TOHUM = 42
TEST_DUZENLENEN = {"a_0002.jpg", "b_0001.jpg", "b_0002.jpg"}  # Faz 3 araç testi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--haric", type=Path, help="sunucuda önceden incelenmiş görüntü adları (satır başına bir)")
    args = ap.parse_args()

    sahne = json.loads((VERI / "sahne_turu.json").read_text(encoding="utf-8"))["goruntuler"]
    haric = set(TEST_DUZENLENEN)
    if args.haric:
        haric |= {s.strip() for s in args.haric.read_text(encoding="utf-8").splitlines() if s.strip()}
    uygun = sorted(a for a in sahne if a not in haric)
    random.Random(TOHUM).shuffle(uygun)

    out = VERI / "uzman_orneklem_sirasi.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sira", "goruntu", "koleksiyon", "sahne"])
        for i, a in enumerate(uygun, 1):
            w.writerow([i, a, sahne[a]["koleksiyon"], sahne[a]["sahne"]])
    ilk = uygun[:150]
    print(f"uygun: {len(uygun)} (hariç: {len(haric)}) → {out.name}")
    print(f"ilk 150: a={sum(a.startswith('a_') for a in ilk)} b={sum(a.startswith('b_') for a in ilk)} | "
          f"detached={sum(sahne[a]['sahne'] == 'detached' for a in ilk)} in_situ={sum(sahne[a]['sahne'] == 'in_situ' for a in ilk)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
