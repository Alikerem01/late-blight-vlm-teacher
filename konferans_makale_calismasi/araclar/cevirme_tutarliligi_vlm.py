"""Kör kümede beş VLM'in çevirme tutarlılığı (V-FLIP) — etiketsiz, SALT OKUMA.

Her model için: 1280 px özgün girdideki kutular (A) ile aynı girdinin yatay aynasındaki
kutuların geri çevrilmişi (B: cx → 1 − cx) görüntü başına Macar algoritmasıyla
eşleştirilir. Simetrik F1 = 2·eşleşme / (|A| + |B|). Şans düzeyi: A ile BAŞKA bir
görüntünün B'si (20 permütasyon, tohum 42). "Geri çevirmeden" satırı, kutu yerleşiminin
içerikten mi çerçeve konumundan mı geldiğini gösterir (bkz. brif 5.5).

Çıktı: veri/kor_kume_cevirme_tutarliligi.json
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

KOK = Path(__file__).resolve().parent.parent
K = KOK / "veri" / "vlm_kosulari"
MODELLER = ["gemma4-e2b", "gemma4-e4b", "gemma4-12b", "qwen3vl-4b", "qwen3vl-8b"]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    u = a[2] * a[3] + b[2] * b[3] - iw * ih
    return iw * ih / u if u > 0 else 0.0


def eslesme(a: list, b: list, esik: float) -> tuple[int, list[float]]:
    if not a or not b:
        return 0, []
    m = np.array([[iou(x, y) for y in b] for x in a])
    ri, ci = linear_sum_assignment(-m)
    v = [float(m[i, j]) for i, j in zip(ri, ci)]
    return sum(x >= esik for x in v), v


def f1(ciftler: list[tuple[list, list]], esik: float = 0.5) -> float | None:
    tp = n = 0
    for a, b in ciftler:
        tp += eslesme(a, b, esik)[0]
        n += len(a) + len(b)
    return round(2 * tp / n, 3) if n else None


def yukle(ad: str) -> dict[str, list]:
    satirlar = (K / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["goruntu"]: [tuple(t["yolo"]) for t in r["tespitler"] if t.get("yolo") and t.get("sinif")]
            for r in map(json.loads, satirlar) if r["durum"] == "ok"}


def main() -> int:
    sonuc = {}
    for m in MODELLER:
        A, Bham = yukle(m), yukle(f"{m}_fliph")
        ortak = sorted(set(A) & set(Bham))
        B = {k: [(1 - cx, cy, w, h) for cx, cy, w, h in Bham[k]] for k in ortak}
        ciftler = [(A[k], B[k]) for k in ortak]
        ious = [v for a, b in ciftler for v in eslesme(a, b, 0.5)[1]]
        rng = random.Random(42)
        taban = []
        for _ in range(20):
            karisik = ortak[:]
            rng.shuffle(karisik)
            taban.append(f1([(A[i], B[j]) for i, j in zip(ortak, karisik) if i != j]))
        sonuc[m] = {
            "goruntu": len(ortak), "kutu_ozgun": sum(len(A[k]) for k in ortak), "kutu_aynali": sum(len(Bham[k]) for k in ortak),
            "F1_05": f1(ciftler, 0.5), "F1_03": f1(ciftler, 0.3),
            "geri_cevirmeden_F1_05": f1([(A[k], Bham[k]) for k in ortak], 0.5),
            "sans_F1_05_ort": round(statistics.mean(taban), 3), "sans_F1_05_aralik": [min(taban), max(taban)],
            "ayni_sayi_orani": round(sum(len(A[k]) == len(Bham[k]) for k in ortak) / len(ortak), 3),
            "ikisi_bos": sum(1 for k in ortak if not A[k] and not Bham[k]),
            "macar_medyan_iou": round(statistics.median(ious), 3) if ious else None,
        }
        s = sonuc[m]
        print(f"{m:11s} F1@0,5 {s['F1_05']} · @0,3 {s['F1_03']} · geri çevirmeden {s['geri_cevirmeden_F1_05']} · "
              f"şans {s['sans_F1_05_ort']} {s['sans_F1_05_aralik']} · aynı sayı %{100 * s['ayni_sayi_orani']:.0f} · medyan IoU {s['macar_medyan_iou']}")
    (KOK / "veri" / "kor_kume_cevirme_tutarliligi.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
