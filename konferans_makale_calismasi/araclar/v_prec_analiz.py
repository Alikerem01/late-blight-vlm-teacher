"""Nicemleme kontrolü (V-PREC, karar D6, hakem R1-M7) — SALT OKUMA.

Gemma 4 E2B'nin 4-bit (öğretmen adaylarıyla aynı koşu), 8-bit ve bf16 MLX dönüşümleri kör
kümede (60 görüntü, 1280 px ortak girdi, aynı istem ve yazılım) karşılaştırılır:
  - çıktı özeti: kutu, görüntü başına kutu, boş görüntü, beyan edilen güven (aralık, ayrık değer),
    koordinat nicemlemesi (10 / 50 / 100'ün katı, düzgün beklentiye oranı), medyan kutu alanı,
    medyan süre
  - yatay çevirme tutarlılığı: F1 @ 0,3 / 0,5 / 0,75, bootstrap %95 GA, 1.000 permütasyonluk
    şans düzeyi, ters dönüşümsüz F1, ters dönüşüm kazancı (GA), eşleşen çiftlerin medyan IoU'su,
    küçük kutu imzası (alan katmanlı geri bulunma)
  - eşleştirilmiş bootstrap: 8-bit − 4-bit ve bf16 − 4-bit tutarlılık farkı
  - modeller arası uyum (F1 @ 0,5): üç duyarlık birbiriyle ve öbür dört adayla
Yöntem `etiketsiz_ek_analiz.py` ile birebir aynı (fonksiyonlar oradan alınır).

Çıktı: veri/v_prec_analiz.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etiketsiz_ek_analiz import (KOSU, N_BOOT, TOHUM, alan_katmanli, bootstrap_f1,  # noqa: E402
                                  f1_havuz, nicemleme, permutasyon_sansi, tp, eslesme_iou, yukle_k)

CIKTI = Path(__file__).resolve().parent.parent / "veri" / "v_prec_analiz.json"
DUYARLIK = {"4-bit": "gemma4-e2b", "8-bit": "gemma4-e2b-8bit", "bf16": "gemma4-e2b-bf16"}
OBUR = ["gemma4-e4b", "gemma4-12b", "qwen3vl-4b", "qwen3vl-8b"]


def ozet(ad: str) -> dict:
    kayit = [json.loads(l) for l in (KOSU / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = [r for r in kayit if r["durum"] == "ok"]
    tes = [t for r in ok for t in r["tespitler"] if t.get("yolo") and t.get("sinif")]
    guven = [t["guven"] for t in tes if t.get("guven") is not None]
    alan = [t["yolo"][2] * t["yolo"][3] for t in tes]
    return {
        "goruntu": len(kayit), "ok": len(ok), "parse_hatasi": sum(r["durum"] == "parse_hatasi" for r in kayit),
        "kutu": len(tes), "kutu_goruntu": round(len(tes) / len(ok), 2),
        "bos": sum(1 for r in ok if not [t for t in r["tespitler"] if t.get("yolo") and t.get("sinif")]),
        "guven_min": min(guven) if guven else None, "guven_max": max(guven) if guven else None,
        "guven_ayrik": len(set(guven)),
        "alan_medyan": round(statistics.median(alan), 4) if alan else None,
        "sure_medyan_s": round(statistics.median(r["sure_ms"] for r in ok) / 1000, 1),
    }


def cevirme(ad: str, rng: np.random.Generator) -> tuple[dict, dict]:
    A, ham = yukle_k(ad)
    Bham, _ = yukle_k(f"{ad}_fliph")
    anahtar = sorted(set(A) & set(Bham))
    B = {k: [(1 - cx, cy, w, h) for cx, cy, w, h in Bham[k]] for k in anahtar}
    ns = np.array([len(A[k]) + len(B[k]) for k in anahtar])
    t05 = np.array([tp(A[k], B[k], 0.5) for k in anahtar])
    t05_ham = np.array([tp(A[k], Bham[k], 0.5) for k in anahtar])
    idx = rng.integers(0, len(ns), size=(N_BOOT, len(ns)))
    kaz = 2 * (t05[idx].sum(1) - t05_ham[idx].sum(1)) / np.maximum(ns[idx].sum(1), 1)
    ious = [v for k in anahtar for _, _, v in eslesme_iou(A[k], B[k])]
    sans = permutasyon_sansi(A, B, anahtar)
    f05 = 2 * t05.sum() / ns.sum()
    sonuc = {
        "F1_03": round(f1_havuz([(A[k], B[k]) for k in anahtar], 0.3), 3),
        "F1_05": round(float(f05), 3), "F1_05_GA95": bootstrap_f1(t05, ns, rng),
        "F1_075": round(f1_havuz([(A[k], B[k]) for k in anahtar], 0.75), 3),
        "ters_donusumsuz_F1_05": round(float(2 * t05_ham.sum() / ns.sum()), 3),
        "ters_donusum_kazanci": round(float(f05 - 2 * t05_ham.sum() / ns.sum()), 3),
        "ters_donusum_kazanci_GA95": [round(float(x), 3) for x in np.percentile(kaz, [2.5, 97.5])],
        "sans_1000": sans, "macar_medyan_iou": round(statistics.median(ious), 3) if ious else None,
        "alan_katmanli_geri_bulunma": alan_katmanli(A, B, Bham, anahtar),
        "nicemleme": nicemleme(ham),
    }
    return sonuc, {"anahtar": anahtar, "tp": t05, "n": ns}


def uyum(a: str, b: str) -> float:
    A, _ = yukle_k(a)
    B, _ = yukle_k(b)
    ortak = sorted(set(A) & set(B))
    return round(f1_havuz([(A[k], B[k]) for k in ortak], 0.5), 3)


def main() -> int:
    rng = np.random.default_rng(TOHUM)
    sonuc: dict = {"yontem": __doc__.strip().splitlines()[0], "duyarlik": {}}
    gb = {}
    for etiket, ad in DUYARLIK.items():
        c, gb[etiket] = cevirme(ad, rng)
        sonuc["duyarlik"][etiket] = {"anahtar": ad, "ozet": ozet(ad), "cevirme": c}
        o = sonuc["duyarlik"][etiket]["ozet"]
        print(f"{etiket:6s} kutu {o['kutu']} ({o['kutu_goruntu']}/görüntü, boş {o['bos']}) · güven {o['guven_min']}–{o['guven_max']} "
              f"({o['guven_ayrik']}) · 50'nin katı {c['nicemleme']['kat50']['oran']} ({c['nicemleme']['kat50']['kat_fazla']}×) · "
              f"çevirme F1@.5 {c['F1_05']} {c['F1_05_GA95']} · ham {c['ters_donusumsuz_F1_05']} · kazanç {c['ters_donusum_kazanci']} "
              f"{c['ters_donusum_kazanci_GA95']} · şans {c['sans_1000']['ort']} · medyan IoU {c['macar_medyan_iou']} · süre {o['sure_medyan_s']} s")
    # eşleştirilmiş fark: yüksek duyarlık − 4-bit (aynı görüntüler)
    ortak = sorted(set.intersection(*(set(v["anahtar"]) for v in gb.values())))
    idx = np.random.default_rng(TOHUM).integers(0, len(ortak), size=(N_BOOT, len(ortak)))
    hiza = {e: {k: i for i, k in enumerate(v["anahtar"])} for e, v in gb.items()}
    def dizi(e):
        i = np.array([hiza[e][k] for k in ortak])
        return gb[e]["tp"][i], gb[e]["n"][i]
    t4, n4 = dizi("4-bit")
    sonuc["esli_fark_4bite_gore"] = {}
    for e in ("8-bit", "bf16"):
        te, ne = dizi(e)
        fark = 2 * te.sum() / ne.sum() - 2 * t4.sum() / n4.sum()
        dag = 2 * te[idx].sum(1) / np.maximum(ne[idx].sum(1), 1) - 2 * t4[idx].sum(1) / np.maximum(n4[idx].sum(1), 1)
        lo, hi = np.percentile(dag, [2.5, 97.5])
        sonuc["esli_fark_4bite_gore"][e] = {"fark": round(float(fark), 3), "GA95": [round(float(lo), 3), round(float(hi), 3)]}
        print(f"{e} − 4-bit çevirme tutarlılığı farkı {fark:.3f} [{lo:.3f}, {hi:.3f}]")
    # eşleştirilmiş fark: yüksek duyarlık − öbür dört aday (8-bit V-POOL kararıyla aday oldu, 2026-09-29)
    sonuc["esli_fark_obur_adaylara_gore"] = {}
    obur_gb = {}
    for m in OBUR:
        A, _ = yukle_k(m); Bh, _ = yukle_k(f"{m}_fliph")
        k = sorted(set(A) & set(Bh)); B = {x: [(1 - cx, cy, w, h) for cx, cy, w, h in Bh[x]] for x in k}
        obur_gb[m] = {"anahtar": k, "tp": np.array([tp(A[x], B[x], 0.5) for x in k]), "n": np.array([len(A[x]) + len(B[x]) for x in k])}
    for e in ("8-bit", "bf16"):
        te, ne = dizi(e)
        for m in OBUR:
            h = {x: i for i, x in enumerate(obur_gb[m]["anahtar"])}
            i = np.array([h[x] for x in ortak])
            to, no = obur_gb[m]["tp"][i], obur_gb[m]["n"][i]
            fark = 2 * te.sum() / ne.sum() - 2 * to.sum() / no.sum()
            dag = 2 * te[idx].sum(1) / np.maximum(ne[idx].sum(1), 1) - 2 * to[idx].sum(1) / np.maximum(no[idx].sum(1), 1)
            lo, hi = np.percentile(dag, [2.5, 97.5])
            sonuc["esli_fark_obur_adaylara_gore"][f"{e}-{m}"] = {"fark": round(float(fark), 3), "GA95": [round(float(lo), 3), round(float(hi), 3)],
                                                                   "sifiri_disliyor": bool(lo > 0 or hi < 0)}
            print(f"{e} − {m}: {fark:.3f} [{lo:.3f}, {hi:.3f}]")
    # modeller arası uyum
    u = {}
    adlar = list(DUYARLIK.values())
    for i, a in enumerate(adlar):
        for b in adlar[i + 1:]:
            u[f"{a}~{b}"] = uyum(a, b)
        for b in OBUR:
            u[f"{a}~{b}"] = uyum(a, b)
    sonuc["modeller_arasi_uyum_F1_05"] = u
    for k, v in u.items():
        print(f"  uyum {k}: {v}")
    CIKTI.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {CIKTI}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
