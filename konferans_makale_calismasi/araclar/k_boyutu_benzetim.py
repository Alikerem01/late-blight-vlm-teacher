"""Kör küme (K) boyutu için güç benzetimi — SALT OKUMA (hakem R1-m2: yeniden üretilebilirlik).

2026-09-25'te satır içi koşulan iki benzetimin aynı çağrı sırasıyla kaydı; K = 60 kararı buna
dayanıyor. Varsayımlar:
  - görüntü başına lezyon sayısı: ilk koşunun (v0) 301 özgündeki kutu sayısı dağılımından
    yeniden örnekleme, en az 1 (uzman her görüntüde en az bir lezyon çizer varsayımı)
  - iki model aynı görüntülerde; görüntü zorluğu ortak, logit ölçeğinde N(0, 1)
  - lezyon başına tespit olasılığı: logit(p_model) + zorluk
  - görüntü başına yanlış pozitif: Poisson(0,8), iki model için bağımsız
  - F1 = 2·TP / (referans + tahmin), görüntüler üzerinden toplanmış
  - güç: eşleştirilmiş farkın %95 aralığının sıfırı dışlama olasılığı, normal yaklaşımla
    Φ(ortalama/SS − 1,96)
Koşu 1 (p = 0,55 / 0,45) tek model F1'inin %95 aralık yarı genişliğini, koşu 2
(0,60 / 0,40 ve 0,65 / 0,35) güç değerlerini verir. Kesim tarihinde tamamlanan n farklıysa
`--n` ile yeniden hesaplanır (bu durumda tohum sırası özgün koşudan ayrılır).

Kullanım:  python k_boyutu_benzetim.py            # 2026-09-25 koşularının yeniden üretimi
           python k_boyutu_benzetim.py --n 47     # gerçek n için güç ve aralık
Çıktı: veri/k_boyutu_benzetim.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

V0 = Path(__file__).resolve().parents[2] / "VLM_auto_annotation"
CIKTI = Path(__file__).resolve().parent.parent / "veri" / "k_boyutu_benzetim.json"
TEKRAR = 4000
FP = 0.8


def lezyon_sayilari() -> np.ndarray:
    n = [sum(1 for d in json.loads(l)["detections"] if d.get("kept"))
         for s in ("kaynak_a_images", "kaynak_b_images")
         for l in open(V0 / f"{s}_autolabel_manifest.jsonl", encoding="utf-8")
         if '"augmentation_type": "original"' in l]
    return np.array([max(1, x) for x in n])


def f1(tp, gt, pr) -> float:
    return 2 * tp.sum() / (gt.sum() + pr.sum())


def lg(p: float) -> float:
    return math.log(p / (1 - p))


def tekrarla(rng, n_lez, N, pA, pB):
    fa, fb = [], []
    for _ in range(TEKRAR):
        gt = rng.choice(n_lez, N)
        zor = rng.normal(0, 1.0, N)
        qa = 1 / (1 + np.exp(-(lg(pA) + zor)))
        qb = 1 / (1 + np.exp(-(lg(pB) + zor)))
        tpa, tpb = rng.binomial(gt, qa), rng.binomial(gt, qb)
        pra, prb = tpa + rng.poisson(FP, N), tpb + rng.poisson(FP, N)
        fa.append(f1(tpa, gt, pra))
        fb.append(f1(tpb, gt, prb))
    return np.array(fa), np.array(fb)


def ozet(fa, fb) -> dict:
    fark = fa - fb
    m, s = fark.mean(), fark.std()
    return {"F1_A_ort": round(float(fa.mean()), 3), "F1_B_ort": round(float(fb.mean()), 3),
            "tek_model_GA95_yari": round(float(1.96 * fa.std()), 3), "fark_ort": round(float(m), 3),
            "guc": round(0.5 * (1 + math.erf((m / s - 1.96) / math.sqrt(2))), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, help="gerçek tamamlanan K boyutu")
    args = ap.parse_args()
    n_lez = lezyon_sayilari()
    sonuc: dict = {"varsayim": {"tekrar": TEKRAR, "fp_poisson": FP, "zorluk_ss": 1.0,
                                "lezyon_ort": round(float(n_lez.mean()), 3)}}
    if args.n:
        rng = np.random.default_rng(42)
        sonuc[f"n={args.n}"] = {f"{a}/{b}": ozet(*tekrarla(rng, n_lez, args.n, a, b))
                                for a, b in ((0.55, 0.45), (0.60, 0.40), (0.65, 0.35))}
    else:
        rng = np.random.default_rng(42)                 # koşu 1: aralık genişliği
        sonuc["kosu1_055_045"] = {N: ozet(*tekrarla(rng, n_lez, N, 0.55, 0.45)) for N in (30, 40, 50, 60, 80, 100, 150)}
        rng = np.random.default_rng(42)                 # koşu 2: güç
        k2 = {}
        for N in (30, 50, 60, 80, 100):
            k2[N] = {"060_040": ozet(*tekrarla(rng, n_lez, N, 0.60, 0.40)),
                     "065_035": ozet(*tekrarla(rng, n_lez, N, 0.65, 0.35))}
        sonuc["kosu2_guc"] = k2
    print(json.dumps(sonuc, ensure_ascii=False, indent=1))
    # --n ile ayrı dosya: makaledeki 25 Eylül değerlerinin kaynağı (k_boyutu_benzetim.json) ezilmesin
    hedef = CIKTI.with_name(f"k_boyutu_benzetim_n{args.n}.json") if args.n else CIKTI
    hedef.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {hedef}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
