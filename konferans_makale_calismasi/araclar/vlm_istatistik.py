"""VLM (Gemma 4 E2B) ilk etiketleme çıktısının istatistikleri — SALT OKUMA.

Kaynak: VLM_auto_annotation/*_autolabel_manifest.jsonl (ham model çıktısı, v0).
Hiçbir proje dosyasına yazmaz; sonuç scratchpad'e JSON olarak düşer.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2] / "VLM_auto_annotation"
SETS = ["kaynak_a_images", "kaynak_b_images"]
OUT = Path(__file__).resolve().parent.parent / "veri" / "vlm_istatistik.json"


def iou(a, b) -> float:
    """YOLO (cx,cy,w,h) normalize kutular arası IoU (anizotropik ölçeğe değişmez)."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def eslestir(ref: list, pred: list, esik: float) -> tuple[int, list[float]]:
    """Macar algoritmasıyla bire bir eşleştirme; eşik üstü eşleşme sayısı + IoU'lar."""
    if not ref or not pred:
        return 0, []
    m = np.array([[iou(r, p) for p in pred] for r in ref])
    ri, ci = linear_sum_assignment(-m)
    ious = [float(m[i, j]) for i, j in zip(ri, ci)]
    return sum(1 for v in ious if v >= esik), ious


def unflip(box, aug: str):
    cx, cy, w, h = box
    if aug == "fliph":
        return (1.0 - cx, cy, w, h)
    if aug == "flipv":
        return (cx, 1.0 - cy, w, h)
    return (cx, cy, w, h)


def yuzdelik(v: list[float], q: float) -> float:
    return float(np.percentile(v, q)) if v else float("nan")


def main() -> int:
    recs = []
    for s in SETS:
        with open(ROOT / f"{s}_autolabel_manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["_set"] = s
                recs.append(r)

    res: dict = {"kaynak": [str(ROOT / f"{s}_autolabel_manifest.jsonl") for s in SETS]}
    res["goruntu_sayisi"] = len(recs)
    res["set_bazinda"] = dict(Counter(r["_set"] for r in recs))
    res["augment_turu"] = dict(Counter(r["augmentation_type"] for r in recs))
    res["status"] = dict(Counter(r["status"] for r in recs))

    dets = [(r, d) for r in recs for d in r["detections"]]
    kept = [(r, d) for r, d in dets if d.get("kept")]
    res["kutu_ham"] = len(dets)
    res["kutu_tutulan"] = len(kept)
    res["sinif_adlari"] = dict(Counter(d.get("class_name") for _, d in dets))

    per_img = [sum(1 for d in r["detections"] if d.get("kept")) for r in recs]
    res["goruntu_basina_kutu"] = {
        "ortalama": round(statistics.mean(per_img), 3), "medyan": statistics.median(per_img),
        "max": max(per_img), "dagilim": dict(sorted(Counter(per_img).items())),
    }
    bos = [r["image_name"] for r, n in zip(recs, per_img) if n == 0]
    res["bos_goruntu"] = {"sayi": len(bos), "oran": round(len(bos) / len(recs), 4), "liste": bos}
    orj = [r for r in recs if r["augmentation_type"] == "original"]
    res["ozgun_goruntu_sayisi"] = len(orj)
    res["ozgun_bos"] = sum(1 for r in orj if not any(d.get("kept") for d in r["detections"]))
    res["ozgun_kutu"] = sum(sum(1 for d in r["detections"] if d.get("kept")) for r in orj)

    conf = [float(d["confidence"]) for _, d in kept]
    res["guven"] = {
        "min": min(conf), "max": max(conf), "ortalama": round(statistics.mean(conf), 4),
        "medyan": statistics.median(conf), "p5": yuzdelik(conf, 5), "p25": yuzdelik(conf, 25),
        "ayrik_deger_sayisi": len(set(conf)),
        "en_sik_10": Counter(round(c, 2) for c in conf).most_common(10),
        "oran_ge_0_9": round(sum(c >= 0.9 for c in conf) / len(conf), 4),
        "oran_ge_0_8": round(sum(c >= 0.8 for c in conf) / len(conf), 4),
        "oran_lt_0_5": round(sum(c < 0.5 for c in conf) / len(conf), 4),
    }

    koord = [int(round(v)) for _, d in kept for v in d["box_2d_raw"]]
    res["koordinat_nicemleme"] = {
        "deger_sayisi": len(koord),
        "oran_kat_100": round(sum(v % 100 == 0 for v in koord) / len(koord), 4),
        "oran_kat_50": round(sum(v % 50 == 0 for v in koord) / len(koord), 4),
        "oran_kat_10": round(sum(v % 10 == 0 for v in koord) / len(koord), 4),
        "ayrik_deger_sayisi": len(set(koord)),
    }

    alan = [d["yolo_box"][2] * d["yolo_box"][3] for _, d in kept]
    res["kutu_alani_goruntu_orani"] = {
        "medyan": round(statistics.median(alan), 4), "p10": round(yuzdelik(alan, 10), 4),
        "p90": round(yuzdelik(alan, 90), 4), "oran_gt_0_25": round(sum(a > 0.25 for a in alan) / len(alan), 4),
        "oran_gt_0_5": round(sum(a > 0.5 for a in alan) / len(alan), 4),
        "oran_lt_0_01": round(sum(a < 0.01 for a in alan) / len(alan), 4),
    }

    ms = [r["elapsed_ms"] for r in recs]
    res["sure_ms"] = {
        "ortalama": round(statistics.mean(ms), 1), "medyan": round(statistics.median(ms), 1),
        "p5": round(yuzdelik(ms, 5), 1), "p95": round(yuzdelik(ms, 95), 1),
        "toplam_dakika": round(sum(ms) / 60000, 1),
    }

    gerekce = [d.get("reasoning", "") for _, d in kept]
    tr = set("çğıöşüÇĞİÖŞÜ")
    res["gerekce"] = {
        "bos_olmayan": sum(1 for g in gerekce if g.strip()),
        "turkce_karakterli_oran": round(sum(any(c in tr for c in g) for g in gerekce) / len(gerekce), 4),
        "ornekler": gerekce[:3],
    }

    # --- Çevirme tutarlılığı (etiketsiz öz-tutarlılık ölçüsü) -----------------
    grup: dict[str, dict[str, list]] = defaultdict(dict)
    for r in recs:
        kutular = [tuple(d["yolo_box"]) for d in r["detections"] if d.get("kept")]
        grup[r["_set"] + "/" + r["source_image"]][r["augmentation_type"]] = kutular
    tutarlilik = {}
    for aug in ("fliph", "flipv"):
        for esik in (0.5, 0.3):
            tp = n_ref = n_pred = ayni_sayi = cift = ikisi_bos = 0
            eslesen_iou: list[float] = []
            for g in grup.values():
                if "original" not in g or aug not in g:
                    continue
                cift += 1
                ref, pred = g["original"], [unflip(b, aug) for b in g[aug]]
                n_ref += len(ref)
                n_pred += len(pred)
                ayni_sayi += len(ref) == len(pred)
                ikisi_bos += (not ref and not pred)
                k, ious = eslestir(ref, pred, esik)
                tp += k
                eslesen_iou.extend(ious)
            tutarlilik[f"{aug}@{esik}"] = {
                "goruntu_cifti": cift, "kutu_ozgun": n_ref, "kutu_cevrilmis": n_pred,
                "eslesen": tp, "F1_simetrik": round(2 * tp / (n_ref + n_pred), 4) if n_ref + n_pred else None,
                "ayni_kutu_sayisi_orani": round(ayni_sayi / cift, 4),
                "ikisi_de_bos": ikisi_bos,
                "macar_cifti_medyan_iou": round(statistics.median(eslesen_iou), 4) if eslesen_iou else None,
            }
    res["cevirme_tutarliligi"] = tutarlilik

    # --- v0 (JSONL) ile yereldeki güncel .txt karşılaştırması -----------------
    fark, ayni, sinif_idleri = [], 0, Counter()
    for r in recs:
        lbl = ROOT / f"{r['_set']}_labels" / (Path(r["image_name"]).stem + ".txt")
        v0 = sorted(tuple(round(x, 6) for x in d["yolo_box"]) for d in r["detections"] if d.get("kept"))
        yerel = []
        if lbl.exists():
            for satir in lbl.read_text().split("\n"):
                p = satir.split()
                if len(p) == 5:
                    sinif_idleri[p[0]] += 1
                    yerel.append(tuple(round(float(x), 6) for x in p[1:]))
        yerel.sort()
        if not lbl.exists():
            fark.append({"goruntu": r["image_name"], "durum": "label dosyası yok"})
        elif v0 == yerel:
            ayni += 1
        else:
            fark.append({"goruntu": r["image_name"], "v0_kutu": len(v0), "yerel_kutu": len(yerel)})
    res["v0_yerel_karsilastirma"] = {"ayni": ayni, "farkli": len(fark), "farklar": fark,
                                     "yerel_sinif_idleri": dict(sinif_idleri)}

    # --- Özgün görüntülerin çözünürlüğü ve EXIF çekim bilgisi -----------------
    boyut, tarih, cihaz = Counter(), [], Counter()
    for r in orj:
        p = ROOT / r["_set"] / r["image_name"]
        with Image.open(p) as im:
            boyut[f"{im.width}x{im.height}"] += 1
            ex = im.getexif()
            ifd = ex.get_ifd(0x8769) if ex else {}
            t = ifd.get(36867) or ex.get(306) if ex else None
            if t:
                tarih.append(str(t))
            if ex and (ex.get(271) or ex.get(272)):
                cihaz[f"{ex.get(271, '')} {ex.get(272, '')}".strip()] += 1
    res["ozgun_cozunurluk"] = dict(boyut.most_common())
    res["ozgun_exif"] = {"tarihli": len(tarih), "ilk": min(tarih) if tarih else None,
                         "son": max(tarih) if tarih else None,
                         "gunler": dict(Counter(t[:10] for t in tarih).most_common()),
                         "cihaz": dict(cihaz.most_common())}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    ozet = {k: v for k, v in res.items() if k not in ("bos_goruntu", "v0_yerel_karsilastirma")}
    ozet["bos_goruntu"] = {k: v for k, v in res["bos_goruntu"].items() if k != "liste"}
    ozet["v0_yerel_karsilastirma"] = res["v0_yerel_karsilastirma"]
    print(json.dumps(ozet, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
