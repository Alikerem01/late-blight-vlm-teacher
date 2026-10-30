"""Özgün görüntüleri sahne türüne ayırır ve çevirme tutarlılığını tür bazında ölçer — SALT OKUMA.

Sahne türü:
  * ``detached`` — koparılmış yaprak, düz zemin (kâğıt/beton) üzerinde
  * ``in_situ``  — bitki yerinde (tarla / tünel), toprak, el, yabani ot görünür

Yöntem: görüntü 256 px'e küçültülür; yeşil bitki pikseli = HSV ton 55–170°,
doygunluk > 0,20. Yeşil oranı histogramı iki tepelidir (vadi 0,40–0,45);
eşik 0,40. 301 görüntünün tamamı küçük resim sayfalarında gözle kontrol
edildi; eşiğin yanlış tarafına düşen iki yerinde bitki görüntüsü elle
düzeltildi (ELLE_DUZELTME).

Çıktı: veri/sahne_turu.json, veri/sahne_bazinda_tutarlilik.json
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2] / "VLM_auto_annotation"
SETS = ["kaynak_a_images", "kaynak_b_images"]
VERI = Path(__file__).resolve().parent.parent / "veri"
ESIK = 0.40
ELLE_DUZELTME = {"b_0014.jpg": "in_situ", "a_0022.jpg": "in_situ"}  # gözle kontrol, 2026-09-25


def yesil_orani(p: Path) -> float:
    with Image.open(p) as im:
        im = im.convert("HSV")
        im.thumbnail((256, 256))
        a = np.asarray(im).astype(float)
    ton, doy = a[..., 0] * 360 / 255, a[..., 1] / 255
    return float(((ton > 55) & (ton < 170) & (doy > 0.20)).mean())


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    u = a[2] * a[3] + b[2] * b[3] - iw * ih
    return iw * ih / u if u > 0 else 0.0


def eslesme(ref: list, pred: list, esik: float) -> tuple[int, list[float]]:
    if not ref or not pred:
        return 0, []
    m = np.array([[iou(r, p) for p in pred] for r in ref])
    ri, ci = linear_sum_assignment(-m)
    v = [float(m[i, j]) for i, j in zip(ri, ci)]
    return sum(x >= esik for x in v), v


def geri_cevir(b, aug: str):
    cx, cy, w, h = b
    return (1 - cx, cy, w, h) if aug == "fliph" else (cx, 1 - cy, w, h)


def main() -> int:
    kayit: dict[str, dict] = defaultdict(dict)
    for s in SETS:
        with open(ROOT / f"{s}_autolabel_manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                kayit[r["source_image"]]["_set"] = s
                kayit[r["source_image"]][r["augmentation_type"]] = [
                    tuple(d["yolo_box"]) for d in r["detections"] if d.get("kept")]

    tur: dict[str, dict] = {}
    for ad in sorted(kayit):
        g = yesil_orani(ROOT / kayit[ad]["_set"] / ad)
        t = ELLE_DUZELTME.get(ad, "detached" if g < ESIK else "in_situ")
        tur[ad] = {"koleksiyon": kayit[ad]["_set"], "yesil_orani": round(g, 3), "sahne": t,
                   "elle": ad in ELLE_DUZELTME}
    (VERI / "sahne_turu.json").write_text(json.dumps(
        {"yontem": __doc__.strip().splitlines()[0], "esik": ESIK, "elle_duzeltme": ELLE_DUZELTME,
         "sayim": {k: sum(1 for v in tur.values() if v["sahne"] == k) for k in ("detached", "in_situ")},
         "goruntuler": tur}, ensure_ascii=False, indent=1), encoding="utf-8")

    sonuc: dict = {}
    for sahne in ("detached", "in_situ", "tumu"):
        adlar = [a for a in kayit if sahne == "tumu" or tur[a]["sahne"] == sahne]
        blok: dict = {"goruntu": len(adlar)}
        ozgun = [kayit[a]["original"] for a in adlar]
        n_kutu = [len(k) for k in ozgun]
        alan = [b[2] * b[3] for k in ozgun for b in k]
        blok["ozgun_kutu"] = sum(n_kutu)
        blok["goruntu_basina_kutu"] = round(statistics.mean(n_kutu), 2)
        blok["bos_ozgun"] = sum(1 for n in n_kutu if n == 0)
        blok["kutu_alani_medyan"] = round(statistics.median(alan), 4) if alan else None
        for aug in ("fliph", "flipv"):
            tp = n1 = n2 = 0
            ious: list[float] = []
            for a in adlar:
                ref = kayit[a]["original"]
                pred = [geri_cevir(b, aug) for b in kayit[a][aug]]
                k, v = eslesme(ref, pred, 0.5)
                tp, n1, n2 = tp + k, n1 + len(ref), n2 + len(pred)
                ious += v
            blok[f"{aug}_F1@0.5"] = round(2 * tp / (n1 + n2), 3) if n1 + n2 else None
            blok[f"{aug}_medyan_iou"] = round(statistics.median(ious), 3) if ious else None
        sonuc[sahne] = blok
    (VERI / "sahne_bazinda_tutarlilik.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(sonuc, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
