"""Hakem Tur 1 isteklerine karşılık etiketsiz ek analizler — SALT OKUMA.

Kör küme K (60 görüntü, 1280 px ortak girdi), beş aday:
  - yatay çevirme tutarlılığı: simetrik F1 @ IoU 0,3 / 0,5 / 0,75 (kutular üzerinden
    toplanmış: iki kopyada da kutusu olmayan görüntü paya da paydaya da katkı vermez);
    F1@0,5 için görüntü düzeyinde %95 bootstrap GA (10.000 yeniden örnekleme, tohum 42)
  - ters dönüşümsüz F1@0,5 ve "ters dönüşüm kazancı" (doğru ters dönüşümle − ham
    koordinatlarla), bootstrap GA: kendi aynasıyla örtüşen büyük/merkezî kutunun
    kurgu gereği aldığı tutarlılığı ayırır (R1-M2)
  - özgün kutunun aynalı kopyada geri bulunma oranı, kutu alanına göre katmanlı
    (< %2, %2–5, ≥ %5 görüntü alanı), ters dönüşümlü ve ters dönüşümsüz
  - şans düzeyi: 1.000 permütasyon (tohum 42), ortalama ve %2,5–%97,5 yüzdelikleri;
    gözlenen − şans ve gözlenen / şans (R1-m1, R1-m12)
  - adaylar arası eşleştirilmiş bootstrap: çevirme tutarlılığı farkı, 10 çift
  - sahne türüne göre (koparılmış yaprak / yerinde bitki) F1@0,5
  - koordinat nicemlemesi: 10 / 50 / 100'ün katı oranı ve düzgün tam sayı ızgarasındaki
    beklentiye oranı (R1-M8a)
Ağustos koşusu (Gemma 4 E2B v0, 301 özgün, yatay ve dikey): F1@0,5 için bootstrap GA,
1.000 permütasyonla şans düzeyi (4.2'deki 20 permütasyonluk değerin güncellemesi).

Girdi: veri/vlm_kosulari/<model>{,_fliph}.jsonl, veri/sahne_turu.json,
       VLM_auto_annotation/*_autolabel_manifest.jsonl (v0).
Çıktı: veri/etiketsiz_ek_analiz.json (yeni dosya; mevcut hiçbir dosyaya yazmaz).
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

KOK = Path(__file__).resolve().parent.parent
KOSU = KOK / "veri" / "vlm_kosulari"
V0_KOK = Path(__file__).resolve().parents[2] / "VLM_auto_annotation"
V0_KUMELER = ["kaynak_a_images", "kaynak_b_images"]
CIKTI = KOK / "veri" / "etiketsiz_ek_analiz.json"
MODELLER = ["gemma4-e2b", "gemma4-e4b", "gemma4-12b", "qwen3vl-4b", "qwen3vl-8b"]
TOHUM = 42
N_BOOT = 10_000
N_PERM = 1_000
ALAN_KATMAN = [(0.0, 0.02, "<2%"), (0.02, 0.05, "2-5%"), (0.05, 1.01, ">=5%")]
BEKLENEN = {10: 101 / 1001, 50: 21 / 1001, 100: 11 / 1001}  # 0–1000 tam sayı ızgarası


def iou(a, b) -> float:
    """YOLO (cx, cy, w, h) normalize kutular arası IoU."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    u = a[2] * a[3] + b[2] * b[3] - iw * ih
    return iw * ih / u if u > 0 else 0.0


def eslesme_iou(a: list, b: list) -> list[tuple[int, int, float]]:
    """Macar algoritmasıyla toplam IoU'yu en büyükleyen bire bir atama: (i, j, IoU)."""
    if not a or not b:
        return []
    m = np.array([[iou(x, y) for y in b] for x in a])
    ri, ci = linear_sum_assignment(-m)
    return [(int(i), int(j), float(m[i, j])) for i, j in zip(ri, ci)]


def tp(a: list, b: list, esik: float) -> int:
    return sum(v >= esik for _, _, v in eslesme_iou(a, b))


def f1_havuz(ciftler: list[tuple[list, list]], esik: float) -> float | None:
    t = sum(tp(a, b, esik) for a, b in ciftler)
    n = sum(len(a) + len(b) for a, b in ciftler)
    return 2 * t / n if n else None


def bootstrap_f1(tps: np.ndarray, ns: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(ns), size=(N_BOOT, len(ns)))
    t, n = tps[idx].sum(1), ns[idx].sum(1)
    f = np.where(n > 0, 2 * t / np.maximum(n, 1), np.nan)
    lo, hi = np.nanpercentile(f, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def yukle_k(ad: str) -> tuple[dict[str, list], dict[str, list]]:
    """K koşusu: görüntü → YOLO kutuları ve ham 0–1000 koordinatlar."""
    kutu, ham = {}, {}
    for r in map(json.loads, (KOSU / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines()):
        if r["durum"] != "ok":
            continue
        tes = [t for t in r["tespitler"] if t.get("yolo") and t.get("sinif")]
        kutu[r["goruntu"]] = [tuple(t["yolo"]) for t in tes]
        ham[r["goruntu"]] = [v for t in tes for v in t["ham_kutu"]]
    return kutu, ham


def permutasyon_sansi(A: dict, B: dict, anahtarlar: list[str], esik: float = 0.5) -> dict:
    rng = random.Random(TOHUM)
    degerler = []
    for _ in range(N_PERM):
        karisik = anahtarlar[:]
        rng.shuffle(karisik)
        degerler.append(f1_havuz([(A[i], B[j]) for i, j in zip(anahtarlar, karisik) if i != j], esik))
    d = np.array([x for x in degerler if x is not None])
    return {"ort": round(float(d.mean()), 3), "p2_5": round(float(np.percentile(d, 2.5)), 3),
            "p97_5": round(float(np.percentile(d, 97.5)), 3), "min": round(float(d.min()), 3),
            "max": round(float(d.max()), 3), "permutasyon": N_PERM}


def alan_katmanli(A: dict, B: dict, Bham: dict, anahtarlar: list[str]) -> dict:
    """Özgün kutunun aynalı kopyada IoU ≥ 0,5 ile geri bulunma oranı, alan katmanına göre."""
    sonuc = {}
    for alt, ust, ad in ALAN_KATMAN:
        say = bulundu = bulundu_ham = 0
        for k in anahtarlar:
            es = {i: v for i, _, v in eslesme_iou(A[k], B[k])}
            es_ham = {i: v for i, _, v in eslesme_iou(A[k], Bham[k])}
            for i, kt in enumerate(A[k]):
                if alt <= kt[2] * kt[3] < ust:
                    say += 1
                    bulundu += es.get(i, 0.0) >= 0.5
                    bulundu_ham += es_ham.get(i, 0.0) >= 0.5
        sonuc[ad] = {"ozgun_kutu": say,
                     "geri_bulunma_ters_donusumlu": round(bulundu / say, 3) if say else None,
                     "geri_bulunma_ters_donusumsuz": round(bulundu_ham / say, 3) if say else None}
    return sonuc


def nicemleme(ham: dict) -> dict:
    v = [int(round(x)) for vs in ham.values() for x in vs]
    out = {"deger_sayisi": len(v)}
    for kat, bek in BEKLENEN.items():
        oran = sum(x % kat == 0 for x in v) / len(v)
        out[f"kat{kat}"] = {"oran": round(oran, 3), "beklenen": round(bek, 4), "kat_fazla": round(oran / bek, 1)}
    return out


def kor_kume() -> dict:
    sahne = {k: v["sahne"] for k, v in json.loads((KOK / "veri" / "sahne_turu.json").read_text(encoding="utf-8"))["goruntuler"].items()}
    rng = np.random.default_rng(TOHUM)
    idx_ortak = None
    goruntu_basi: dict[str, dict] = {}
    sonuc: dict = {}
    for m in MODELLER:
        A, ham = yukle_k(m)
        Bham, _ = yukle_k(f"{m}_fliph")
        anahtar = sorted(set(A) & set(Bham))
        B = {k: [(1 - cx, cy, w, h) for cx, cy, w, h in Bham[k]] for k in anahtar}
        ns = np.array([len(A[k]) + len(B[k]) for k in anahtar])
        t05 = np.array([tp(A[k], B[k], 0.5) for k in anahtar])
        t05_ham = np.array([tp(A[k], Bham[k], 0.5) for k in anahtar])
        goruntu_basi[m] = {"anahtar": anahtar, "n": ns, "tp": t05}
        f05 = 2 * t05.sum() / ns.sum()
        f05_ham = 2 * t05_ham.sum() / ns.sum()
        idx = rng.integers(0, len(ns), size=(N_BOOT, len(ns)))
        kaz = 2 * (t05[idx].sum(1) - t05_ham[idx].sum(1)) / np.maximum(ns[idx].sum(1), 1)
        sans = permutasyon_sansi(A, B, anahtar)
        sahne_f1 = {}
        for s in ("detached", "in_situ"):
            alt = [k for k in anahtar if sahne[k] == s]
            f = f1_havuz([(A[k], B[k]) for k in alt], 0.5)
            sahne_f1[s] = {"goruntu": len(alt), "kutu_ozgun": sum(len(A[k]) for k in alt),
                           "F1_05": round(f, 3) if f is not None else None}
        sonuc[m] = {
            "goruntu": len(anahtar),
            "F1_03": round(f1_havuz([(A[k], B[k]) for k in anahtar], 0.3), 3),
            "F1_05": round(float(f05), 3),
            "F1_05_GA95": bootstrap_f1(t05, ns, rng),
            "F1_075": round(f1_havuz([(A[k], B[k]) for k in anahtar], 0.75), 3),
            "ters_donusumsuz_F1_05": round(float(f05_ham), 3),
            "ters_donusumsuz_F1_05_GA95": bootstrap_f1(t05_ham, ns, rng),
            "ters_donusum_kazanci": round(float(f05 - f05_ham), 3),
            "ters_donusum_kazanci_GA95": [round(float(x), 3) for x in np.percentile(kaz, [2.5, 97.5])],
            "sans_1000": sans,
            "gozlenen_eksi_sans": round(float(f05) - sans["ort"], 3),
            "gozlenen_bolu_sans": round(float(f05) / sans["ort"], 1) if sans["ort"] else None,
            "alan_katmanli_geri_bulunma": alan_katmanli(A, B, Bham, anahtar),
            "sahne": sahne_f1,
            "nicemleme": nicemleme(ham),
        }
        idx_ortak = anahtar if idx_ortak is None else idx_ortak
        print(f"{m:11s} F1@.5 {sonuc[m]['F1_05']} {sonuc[m]['F1_05_GA95']} · @.3 {sonuc[m]['F1_03']} · @.75 {sonuc[m]['F1_075']} · "
              f"ham {sonuc[m]['ters_donusumsuz_F1_05']} · kazanç {sonuc[m]['ters_donusum_kazanci']} {sonuc[m]['ters_donusum_kazanci_GA95']} · "
              f"şans {sans['ort']} [{sans['p2_5']}, {sans['p97_5']}]")

    # Eşleştirilmiş bootstrap: çevirme tutarlılığı farkı (satır − sütun), aynı görüntüler
    ortak = sorted(set.intersection(*(set(goruntu_basi[m]["anahtar"]) for m in MODELLER)))
    hiza = {m: {k: i for i, k in enumerate(goruntu_basi[m]["anahtar"])} for m in MODELLER}
    idx = np.random.default_rng(TOHUM).integers(0, len(ortak), size=(N_BOOT, len(ortak)))
    farklar = {}
    for a_i, a in enumerate(MODELLER):
        for b in MODELLER[a_i + 1:]:
            ia = np.array([hiza[a][k] for k in ortak])
            ib = np.array([hiza[b][k] for k in ortak])
            ta, na = goruntu_basi[a]["tp"][ia], goruntu_basi[a]["n"][ia]
            tb, nb = goruntu_basi[b]["tp"][ib], goruntu_basi[b]["n"][ib]
            fa = 2 * ta[idx].sum(1) / np.maximum(na[idx].sum(1), 1)
            fb = 2 * tb[idx].sum(1) / np.maximum(nb[idx].sum(1), 1)
            fark = 2 * ta.sum() / na.sum() - 2 * tb.sum() / nb.sum()
            lo, hi = np.percentile(fa - fb, [2.5, 97.5])
            farklar[f"{a}-{b}"] = {"fark": round(float(fark), 3), "GA95": [round(float(lo), 3), round(float(hi), 3)],
                                   "sifiri_disliyor": bool(lo > 0 or hi < 0)}
    return {"modeller": sonuc, "esli_tutarlilik_farki": farklar, "ortak_goruntu": len(ortak)}


def agustos() -> dict:
    """Gemma 4 E2B v0 (Ağustos) koşusu, 301 özgün: F1@0,5 bootstrap GA ve 1.000 permütasyonluk şans."""
    kayit = [json.loads(l) | {"_set": s} for s in V0_KUMELER
             for l in (V0_KOK / f"{s}_autolabel_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    g: dict[str, dict] = defaultdict(dict)
    for r in kayit:
        g[r["_set"] + "/" + r["source_image"]][r["augmentation_type"]] = [
            tuple(d["yolo_box"]) for d in r["detections"] if d.get("kept")]
    anahtar = sorted(k for k, v in g.items() if {"original", "fliph", "flipv"} <= v.keys())
    rng = np.random.default_rng(TOHUM)
    sonuc = {"goruntu": len(anahtar)}
    for aug, ters in (("fliph", lambda b: (1 - b[0], b[1], b[2], b[3])), ("flipv", lambda b: (b[0], 1 - b[1], b[2], b[3]))):
        A = {k: g[k]["original"] for k in anahtar}
        B = {k: [ters(b) for b in g[k][aug]] for k in anahtar}
        ns = np.array([len(A[k]) + len(B[k]) for k in anahtar])
        t05 = np.array([tp(A[k], B[k], 0.5) for k in anahtar])
        sonuc[aug] = {"F1_05": round(float(2 * t05.sum() / ns.sum()), 3), "F1_05_GA95": bootstrap_f1(t05, ns, rng),
                      "F1_075": round(f1_havuz([(A[k], B[k]) for k in anahtar], 0.75), 3),
                      "sans_1000": permutasyon_sansi(A, B, anahtar)}
        print(f"Ağustos {aug}: F1@.5 {sonuc[aug]['F1_05']} {sonuc[aug]['F1_05_GA95']} · @.75 {sonuc[aug]['F1_075']} · "
              f"şans {sonuc[aug]['sans_1000']['ort']} [{sonuc[aug]['sans_1000']['p2_5']}, {sonuc[aug]['sans_1000']['p97_5']}]")
    return sonuc


def main() -> int:
    sonuc = {"yontem": __doc__.strip().splitlines()[0], "tohum": TOHUM, "bootstrap": N_BOOT,
             "kor_kume": kor_kume(), "agustos_e2b_v0": agustos()}
    CIKTI.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {CIKTI}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
