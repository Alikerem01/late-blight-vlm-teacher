"""Uzman etiketleri ↔ VLM çıktıları — SALT OKUMA (brif bölüm 7; kayıt defteri R, V, IAA grupları).

K (1_kor_etiketleme, kör referans): altı öğretmen adayının (V-POOL, 2026-09-29) yanlısız
    karşılaştırması ve önceden yazılmış kurala göre öğretmen seçimi (V-COMP): K'de F1 @ 0,5
    nokta tahmini en yüksek aday; en iyiyle öbür beşinin eşleştirilmiş bootstrap farkı ve
    Holm düzeltmeli bootstrap p değerleri; ayrışmayanlar "not separable". bf16 kontrolü
    (aday değil), ilk koşu (v0) ve tam çözünürlük koşusu ek satır. Görüntü düzeyinde
    "lezyon var" kesinliği ve duyarlılığı (R2-M6), sahne türüne göre kırılım (yerinde bitki
    birincil, R2-M3), güven değerinin ayırt ediciliği (R-CONF-AUROC), çevirme tutarlılığı
    ile doğruluk sıralamasının ilişkisi (V-FLIP-RANK) ve görüntü başına ilişkisi (R-FLIP-CORR).
O (1b_ortak_okuma, D1): K'nin 20 görüntüsünün öbür uzmanca kör okuması → uzmanlar arası
    uyum (IAA): kutu F1 @ 0,3 / 0,5 / 0,75 ve bootstrap GA, görüntü düzeyinde uyum ve
    Cohen κ, aynı görüntülerde adayların F1'i (tavan karşılaştırması).
H (2_duzeltme_onay, HITL): Gemma 4 E2B'nin ilk koşu (v0) ön-etiketlerinin uzman kabul oranı
    (R-VLM-PRF), düzeltme dökümü (R-EDIT), dokunulmadan onaylanan görüntüler (R-UNTOUCHED).
Uzman başına kutu özeti (IAA) ve süre (R-EFFORT): aynı uzmanın ardışık iki kayıt ya da
    onayı arasındaki süre görüntü başına toplanır; öncülü olmayan ya da 10 dakikadan uzun
    boşlukla (mola) başlayan görüntü süresiz; uzman × küme ve sahne kırılımında medyan.

Eşleştirme: görüntü başına Macar algoritması (IoU'yu en büyükleyen bire bir atama);
IoU ≥ eşik → doğru pozitif. Mikro toplam; görüntü düzeyinde bootstrap (10.000, tohum 42;
aynı görüntü sayısında bütün modeller aynı indeksleri kullanır, farklar eşleştirilmiştir).
Yalnız uzmanın incelediği (image_reviews: approved / edited) görüntüler sayılır. Lezyon
kutusu = sunucunun tek sınıfı (Late_Blight, id 0); başka sınıftaki kutu sayılmaz, uyarılır.

Kullanım:
  python uzman_vlm_karsilastir.py --paket ~/Downloads/degerlendirme_<zaman>.tar.gz
  python uzman_vlm_karsilastir.py --klasor ../uzman_veriler_2
  (--cikti verilmezse sonuçlar veri/ altına yazılır)

--klasor (2026-10-11): uzmanlar etiketleri sunucu dışında, Drive'daki `uzman_veriler/` kopyasında çizip
klasör olarak geri gönderdi. Görüntü ↔ küme ↔ uzman eşlemesi `veri/uzman_veriler_manifest.csv`'den (uzman_1 =
birinci, uzman_2 = ikinci, aralıklar atamadaki gibi), lezyon sınıfı her `labels/classes.txt`'teki `Late_Blight`'ın
sırası. H'nin uzman etiketi `labels/`, VLM ön-etiketi yine `*_autolabel_manifest.jsonl`'den. Bu kaynakta
inceleme durumu (onay / düzenleme) ve işlem zaman damgası YOK: durum "kayit_yok", süre (R-EFFORT) boş kalır,
uydurulmaz. Dosya zaman damgaları aktarım anını gösterdiği için süreye çevrilmez.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, spearmanr

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "veri"
VLM_KLASOR = KOK.parent / "VLM_auto_annotation"
ADAYLAR = ["gemma4-e2b", "gemma4-e2b-8bit", "gemma4-e4b", "gemma4-12b", "qwen3vl-4b", "qwen3vl-8b"]
KONTROL = ["gemma4-e2b-bf16"]             # nicemleme kontrolü (D6), öğretmen adayı değil
MODELLER = ADAYLAR
ESIKLER = (0.3, 0.5, 0.75)
BOOT, TOHUM = 10_000, 42
ALFA = 0.05
MOLA_S = 600
SAHNELER = ("detached", "in_situ")
KOR_AD, HITL_AD, ORTAK_AD = "1_kor_etiketleme_images", "2_duzeltme_onay_images", "1b_ortak_okuma_images"
KLASOR = {"kor": KOR_AD, "ortak": ORTAK_AD, "hitl": HITL_AD}


# ------------------------------------------------------------------ yardımcılar
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    u = a[2] * a[3] + b[2] * b[3] - iw * ih
    return iw * ih / u if u > 0 else 0.0


def eslestir(ref: list, pred: list) -> list[tuple[int, int, float]]:
    """(ref_i, pred_j, IoU) — bire bir, IoU'yu en büyükleyen atama (IoU = 0 çiftler dahil)."""
    if not ref or not pred:
        return []
    m = np.array([[iou(r, p) for p in pred] for r in ref])
    ri, ci = linear_sum_assignment(-m)
    return [(int(i), int(j), float(m[i, j])) for i, j in zip(ri, ci)]


def goruntu_sayimlari(ref: list, pred: list) -> dict:
    ciftler = eslestir(ref, pred)
    s = {"n_ref": len(ref), "n_pred": len(pred)}
    for t in ESIKLER:
        s[f"tp_{t}"] = sum(1 for _, _, v in ciftler if v >= t)
    return s


def prf(sayim: list[dict], t: float) -> dict:
    tp = sum(s[f"tp_{t}"] for s in sayim)
    npred = sum(s["n_pred"] for s in sayim)
    nref = sum(s["n_ref"] for s in sayim)
    p = tp / npred if npred else float("nan")
    r = tp / nref if nref else float("nan")
    f = 2 * tp / (npred + nref) if npred + nref else float("nan")
    return {"kesinlik": p, "duyarlilik": r, "F1": f, "tp": tp, "n_pred": npred, "n_ref": nref}


def bootstrap_indeksi(n: int) -> np.ndarray:
    """Görüntü düzeyinde yeniden örnekleme indeksleri; aynı n için her çağrıda aynı (eşleştirilmiş)."""
    return np.random.default_rng(TOHUM).integers(0, n, size=(BOOT, n))


def ga(v: np.ndarray) -> list[float]:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def bootstrap(sayim: list[dict], t: float, idx: np.ndarray) -> dict:
    if not sayim:
        return {}
    tp = np.array([s[f"tp_{t}"] for s in sayim])[idx].sum(1)
    npred = np.array([s["n_pred"] for s in sayim])[idx].sum(1)
    nref = np.array([s["n_ref"] for s in sayim])[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dagilim = {"kesinlik": tp / npred, "duyarlilik": tp / nref, "F1": 2 * tp / (npred + nref)}
    return {k: ga(v) for k, v in dagilim.items()} | {"_F1_dagilim": dagilim["F1"]}


def bootstrap_p(d: np.ndarray) -> float:
    """İki yönlü yüzdelik bootstrap p değeri ((1 + sayı) / (B + 1), sıfır çıkmaz)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    alt = (1 + np.sum(d <= 0)) / (len(d) + 1)
    ust = (1 + np.sum(d >= 0)) / (len(d) + 1)
    return float(min(1.0, 2 * min(alt, ust)))


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm (1979) düzeltmeli p değerleri (adım adım, monoton)."""
    sira = sorted(p, key=p.get)
    out, onceki = {}, 0.0
    for i, k in enumerate(sira):
        onceki = max(onceki, min(1.0, (len(sira) - i) * p[k]))
        out[k] = onceki
    return out


def cohen_kappa(a: int, b: int, c: int, d: int) -> float:
    """2 × 2 tablo (a: ikisi de var, b: yalnız birinci, c: yalnız ikinci, d: ikisi de yok)."""
    n = a + b + c + d
    if not n:
        return float("nan")
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def yuvarla(x, anahtar: str = "", kayit: bool = False):
    """JSON'a uygun türler. Konsolda üç ondalık (p değerleri dört). kayit=True: kayan noktalar tam duyarlıkla
    kalır, makalenin sayıları `tur4_sayilar.py`'de bir kez yuvarlanır (üç ondalıklı kaydı ikinci kez iki ondalığa
    yuvarlamak sonucu 0,01 kaydırabilir)."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if np.isnan(x) else x if kayit else round(x, 4 if anahtar.startswith("p_") else 3)
    if isinstance(x, dict):
        return {k: yuvarla(v, str(k), kayit) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [yuvarla(v, anahtar, kayit) for v in x]
    return x


def auroc(tp_guven: list[float], fp_guven: list[float]) -> float | None:
    if not tp_guven or not fp_guven:
        return None
    u = mannwhitneyu(tp_guven, fp_guven, alternative="two-sided").statistic
    return float(u / (len(tp_guven) * len(fp_guven)))


def spearman_ozet(x: list[float], y: list[float]) -> dict:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"spearman_rho": None, "p": None, "not": "tanımsız (n < 3 ya da sabit girdi)"}
    r = spearmanr(x, y)
    return {"spearman_rho": float(r.statistic), "p": float(r.pvalue)}


def medyan_ozet(v: list[float]) -> dict:
    return {"medyan_s": statistics.median(v), "ceyrekler_s": [float(np.percentile(v, 25)), float(np.percentile(v, 75))], "n_goruntu": len(v)}


# ------------------------------------------------------------------ girdiler
def paketi_ac(yol: Path) -> Path:
    if yol.is_dir():
        return yol
    hedef = Path(tempfile.mkdtemp(prefix="degerlendirme_"))
    with tarfile.open(yol) as tf:
        tf.extractall(hedef, filter="data")
    alt = [d for d in hedef.iterdir() if d.is_dir()]
    return alt[0] if len(alt) == 1 else hedef


def etiket_oku(p: Path) -> list[tuple[int, tuple]]:
    """YOLO .txt → [(sınıf, (cx, cy, w, h))]."""
    if not p.exists():
        return []
    out = []
    for satir in p.read_text(encoding="utf-8").splitlines():
        parca = satir.split()
        if len(parca) == 5:
            out.append((int(float(parca[0])), tuple(float(v) for v in parca[1:])))
    return out


def paket_yukle(kok: Path, lezyon_sinifi: int = 0) -> dict:
    eslesme = {}
    for f in sorted(kok.glob("kurulum_*.csv")):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                eslesme[(r["kume"], r["yeni_ad"])] = r
    inceleme = {}
    with open(kok / "inceleme.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] in ("approved", "edited"):
                anahtar = (r["dataset"], r["image_name"])
                onceki = inceleme.get(anahtar)
                if onceki is None or r["updated_at"] > onceki["updated_at"]:
                    inceleme[anahtar] = r
    islem = []
    if (kok / "islem.csv").exists():
        with open(kok / "islem.csv", encoding="utf-8") as fh:
            islem = list(csv.DictReader(fh))
    kumeler, siniflar = {}, {}
    for kume, klasor in KLASOR.items():
        kayit, sayac = {}, Counter()
        for (k, yeni), r in eslesme.items():
            if k != kume or (klasor, yeni) not in inceleme:
                continue
            etiket = etiket_oku(kok / f"{klasor}_labels" / (Path(yeni).stem + ".txt"))
            sayac.update(s for s, _ in etiket)
            inc = inceleme[(klasor, yeni)]
            kayit[r["orijinal_ad"]] = {
                "yeni_ad": yeni, "sahne": r["sahne"], "koleksiyon": r["koleksiyon"],
                "uzman": inc["uzman"], "durum": inc["status"],
                "kutular": [b for s, b in etiket if s == lezyon_sinifi],
                "k_ad": r.get("k_ad"), "planlanan_cizen": r.get("cizen"),
            }
        kumeler[kume] = kayit
        siniflar[kume] = {str(s): n for s, n in sorted(sayac.items())}
    plan = {k: [r["orijinal_ad"] for (kk, _), r in sorted(eslesme.items(), key=lambda x: int(x[1]["sira"])) if kk == k] for k in KLASOR}
    return {"kumeler": kumeler, "siniflar": siniflar, "islem": islem, "plan": plan,
            "hedef": {k: sum(1 for kk, _ in eslesme if kk == k) for k in KLASOR},
            "sahne_haritasi": {(KLASOR[k], yeni): r["sahne"] for (k, yeni), r in eslesme.items() if k in KLASOR}}


def klasor_yukle(kok: Path, lezyon_sinifi: int = 0) -> dict:
    """`uzman_veriler_2/` düzenini `paket_yukle` ile aynı yapıya çevirir (bkz. --klasor)."""
    drive = {"kor": "1_kor_etiketleme", "ortak": "1b_ortak_okuma", "hitl": "2_duzeltme_onay"}
    uzman_kimligi = {"birinci": "uzman_1", "ikinci": "uzman_2"}
    with open(VERI / "uzman_veriler_manifest.csv", encoding="utf-8") as fh:
        satirlar = list(csv.DictReader(fh))
    kumeler, siniflar, sinif_ids = {}, {}, {}
    for kume, klasor in drive.items():
        lbl = kok / klasor / "labels"
        if not lbl.is_dir():
            raise SystemExit(f"HATA: etiket klasörü yok: {lbl}")
        cls = lbl / "classes.txt"
        adlar = cls.read_text(encoding="utf-8").split() if cls.exists() else []
        sinif_ids[kume] = adlar.index("Late_Blight") if "Late_Blight" in adlar else lezyon_sinifi
        kayit, sayac = {}, Counter()
        for r in satirlar:
            if r["kume"] != kume:
                continue
            p = lbl / (Path(r["yeni_ad"]).stem + ".txt")
            if not p.exists():                                    # eksik dosya = bakılmadı, sayılmaz
                continue
            etiket = etiket_oku(p)
            sayac.update(s for s, _ in etiket)
            kayit[r["orijinal_ad"]] = {
                "yeni_ad": r["yeni_ad"], "sahne": r["sahne"], "koleksiyon": r["koleksiyon"],
                "uzman": uzman_kimligi[r["cizen"]], "durum": "kayit_yok",
                "kutular": [b for s, b in etiket if s == sinif_ids[kume]],
                "k_ad": r.get("k_ad") or None, "planlanan_cizen": r["cizen"],
            }
        kumeler[kume] = kayit
        siniflar[kume] = {str(0 if s == sinif_ids[kume] else s): n for s, n in sorted(sayac.items())}
    plan = {k: [r["orijinal_ad"] for r in sorted(satirlar, key=lambda r: int(r["sira"])) if r["kume"] == k] for k in KLASOR}
    return {"kumeler": kumeler, "siniflar": siniflar, "islem": [], "plan": plan,
            "hedef": {k: sum(1 for r in satirlar if r["kume"] == k) for k in KLASOR},
            "sahne_haritasi": {(KLASOR[r["kume"]], r["yeni_ad"]): r["sahne"] for r in satirlar},
            "lezyon_sinifi_klasorde": sinif_ids}


def model_ciktilari(ad: str) -> dict[str, list[tuple[tuple, float | None]]]:
    out = {}
    for satir in (VERI / "vlm_kosulari" / f"{ad}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(satir)
        if r["durum"] == "ok":
            out[r["goruntu"]] = [(tuple(t["yolo"]), t.get("guven")) for t in r["tespitler"] if t.get("yolo") and t.get("sinif")]
    return out


def v0_ciktilari(aug: str = "original") -> dict[str, list[tuple[tuple, float | None]]]:
    out = {}
    for s in ("kaynak_a_images", "kaynak_b_images"):
        with open(VLM_KLASOR / f"{s}_autolabel_manifest.jsonl", encoding="utf-8") as fh:
            for satir in fh:
                r = json.loads(satir)
                if r["augmentation_type"] == aug:
                    out[r["source_image"]] = [(tuple(d["yolo_box"]), d.get("confidence")) for d in r["detections"] if d.get("kept")]
    return out


# ------------------------------------------------------------------ çözümlemeler
def goruntu_duzeyi(var_ref: list[bool], var_pred: list[bool], idx: np.ndarray | None = None) -> dict:
    """Görüntü düzeyinde "lezyon var" kararı: model ≥ 1 kutu ↔ uzman ≥ 1 kutu (R2-M6)."""
    r, p = np.array(var_ref, dtype=bool), np.array(var_pred, dtype=bool)
    a, b, c, d = int((r & p).sum()), int((r & ~p).sum()), int((~r & p).sum()), int((~r & ~p).sum())
    out = {"iki_de_var": a, "uzman_var_model_yok": b, "uzman_yok_model_var": c, "iki_de_yok": d,
           "uzmanca_lezyonsuz": c + d,
           "kesinlik": a / (a + c) if a + c else float("nan"),
           "duyarlilik": a / (a + b) if a + b else float("nan")}
    if idx is not None and len(r):
        R, P = r[idx], p[idx]
        tp = (R & P).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["GA95"] = {"kesinlik": ga(tp / P.sum(1)), "duyarlilik": ga(tp / R.sum(1))}
    return out


def model_degerlendir(ref: dict, pred: dict, idx: np.ndarray | None = None) -> dict:
    """ref: {ad: {"kutular", "sahne"}}, pred: {ad: [(kutu, güven)]}."""
    adlar = sorted(ref)
    idx = bootstrap_indeksi(len(adlar)) if idx is None else idx
    sayim = [goruntu_sayimlari(list(ref[a]["kutular"]), [b for b, _ in pred.get(a, [])]) for a in adlar]
    sonuc: dict = {"goruntu": len(adlar)}
    for t in ESIKLER:
        sonuc[f"iou_{t}"] = prf(sayim, t) | {"GA95": bootstrap(sayim, t, idx)}
    var_ref = [bool(ref[a]["kutular"]) for a in adlar]
    var_pred = [bool(pred.get(a)) for a in adlar]
    sonuc["goruntu_duzeyi"] = goruntu_duzeyi(var_ref, var_pred, idx)
    for s in SAHNELER:
        alt = [i for i, a in enumerate(adlar) if ref[a].get("sahne") == s]
        if not alt:
            sonuc[f"sahne_{s}"] = None
            continue
        alt_sayim, alt_idx = [sayim[i] for i in alt], bootstrap_indeksi(len(alt))
        sonuc[f"sahne_{s}"] = {"goruntu": len(alt), **prf(alt_sayim, 0.5), "GA95": bootstrap(alt_sayim, 0.5, alt_idx),
                               "goruntu_duzeyi": goruntu_duzeyi([var_ref[i] for i in alt], [var_pred[i] for i in alt], alt_idx)}
    # beyan edilen güven: doğru ve yanlış kutuları ayırıyor mu (IoU 0,5)
    tp_g, fp_g = [], []
    for a in adlar:
        p = pred.get(a, [])
        eslesen = {j for _, j, v in eslestir(list(ref[a]["kutular"]), [b for b, _ in p]) if v >= 0.5}
        for j, (_, g) in enumerate(p):
            if isinstance(g, (int, float)):
                (tp_g if j in eslesen else fp_g).append(float(g))
    sonuc["guven_auroc_05"] = auroc(tp_g, fp_g)
    sonuc["_sayim"] = sayim
    return sonuc


def ogretmen_sec(kor: dict) -> dict:
    """Önceden yazılmış kural (brif §7): K'de F1 @ 0,5 nokta tahmini en yüksek aday öğretmendir.
    En iyiyle öbür beşinin eşleştirilmiş farkı: %95 bootstrap GA ve Holm düzeltmeli bootstrap p.
    "Daha iyi" yalnız düzeltilmiş p < 0,05 ise; ikinciyle ayrışmıyorsa ikisi "not separable"."""
    F1 = {m: kor[m]["iou_0.5"]["F1"] for m in ADAYLAR}
    ust = max(F1.values())
    esit = [m for m in ADAYLAR if abs(F1[m] - ust) < 1e-12]
    # tam eşitlik (önceden yazılmadı, olasılığı çok düşük): yerinde bitki F1'i büyük olan
    en_iyi = max(esit, key=lambda m: (kor[m]["sahne_in_situ"] or {}).get("F1", -1.0))
    siralama = [en_iyi] + sorted((m for m in ADAYLAR if m != en_iyi), key=lambda m: -F1[m])
    kars, p = {}, {}
    for m in siralama[1:]:
        d = kor[en_iyi]["iou_0.5"]["GA95"]["_F1_dagilim"] - kor[m]["iou_0.5"]["GA95"]["_F1_dagilim"]
        kars[m] = {"F1_farki": F1[en_iyi] - F1[m], "GA95": ga(d)}
        p[m] = bootstrap_p(d)
    ph = holm(p)
    for m in kars:
        kars[m] |= {"p_bootstrap": p[m], "p_holm": ph[m], "ayrisiyor": ph[m] < ALFA}
    ikinci = siralama[1]
    return {"kural": "K'de F1 @ 0,5 nokta tahmini en yüksek aday; en iyiye karşı beş karşılaştırma, Holm, alfa 0,05",
            "ogretmen": en_iyi, "siralama_F1_05": siralama, "nokta_esitligi": esit if len(esit) > 1 else [],
            "en_iyiye_gore": kars, "ayrismayanlar": [m for m in kars if not kars[m]["ayrisiyor"]],
            "ikinci": ikinci, "ikinciden_ayrisiyor": kars[ikinci]["ayrisiyor"]}


def cevirme_tutarliligi(ad: str) -> dict | None:
    """Koşu ↔ yatay aynalı koşunun geri çevrilmiş kutuları: kutular üzerinden toplanmış F1 @ 0,5
    (`etiketsiz_ek_analiz.py` ile aynı tanım) ve görüntü başına (tp, kutu toplamı)."""
    if not (VERI / "vlm_kosulari" / f"{ad}_fliph.jsonl").exists():
        return None
    A, Bh = model_ciktilari(ad), model_ciktilari(f"{ad}_fliph")
    goruntu = {}
    for a in sorted(set(A) & set(Bh)):
        onc = [b for b, _ in A[a]]
        geri = [(1 - cx, cy, w, h) for (cx, cy, w, h), _ in Bh[a]]
        goruntu[a] = (sum(1 for _, _, v in eslestir(onc, geri) if v >= 0.5), len(onc) + len(geri))
    t, n = sum(v[0] for v in goruntu.values()), sum(v[1] for v in goruntu.values())
    return {"F1_05": 2 * t / n if n else float("nan"), "_goruntu": goruntu}


def uzmanlar_arasi(K: dict, O: dict, adaylar: dict) -> dict:
    """D1: ortak kümedeki kör okuma (O) ↔ aynı görüntünün K'deki okuması. K okuması referans
    sayılır; kutu F1'i simetriktir, kesinlik ile duyarlılık referans değişince yer değiştirir."""
    ciftler = [(a, K[a], O[a]) for a in sorted(O) if a in K]
    farkli = [c for c in ciftler if c[1]["uzman"] != c[2]["uzman"]]
    ayni = [c for c in ciftler if c[1]["uzman"] == c[2]["uzman"]]
    sonuc: dict = {"O_incelenen": len(O), "K_karsiligi_da_incelenen": len(ciftler),
                   "farkli_uzman": len(farkli), "ayni_uzman": len(ayni),
                   "okuma_yonu": dict(sorted(Counter(f"{k['uzman']} (K) → {o['uzman']} (O)" for _, k, o in ciftler).items()))}
    if ayni:
        s = [goruntu_sayimlari(list(k["kutular"]), list(o["kutular"])) for _, k, o in ayni]
        sonuc["ayni_uzman_tekrar_okuma"] = {"not": "atamaya uyulmamış: uzman-içi uyum, uzmanlar arası sayılmaz",
                                            "goruntu": len(ayni), "F1_05": prf(s, 0.5)["F1"]}
    if not farkli:
        return sonuc
    ref = {a: {"kutular": k["kutular"], "sahne": k["sahne"]} for a, k, _ in farkli}
    ikinci = {a: [(b, None) for b in o["kutular"]] for a, _, o in farkli}
    d = model_degerlendir(ref, ikinci)
    gd = d["goruntu_duzeyi"]
    sonuc |= {f"iou_{t}": d[f"iou_{t}"] for t in ESIKLER}
    sonuc["goruntu_duzeyi"] = gd | {
        "yuzde_uyum": (gd["iki_de_var"] + gd["iki_de_yok"]) / len(farkli),
        "cohen_kappa": cohen_kappa(gd["iki_de_var"], gd["uzman_var_model_yok"], gd["uzman_yok_model_var"], gd["iki_de_yok"])}
    sonuc["sahne"] = {s: d[f"sahne_{s}"] for s in SAHNELER}
    sonuc["kutu_sayisi"] = {"K_okumasi": sum(len(k["kutular"]) for _, k, _ in farkli),
                            "O_okumasi": sum(len(o["kutular"]) for _, _, o in farkli)}
    # aynı görüntülerde adaylar (tavan karşılaştırması): iki okumanın her birine göre F1 @ 0,5
    sonuc["adaylar_ayni_goruntulerde_F1_05"] = {
        m: {"K_okumasina_gore": prf([goruntu_sayimlari(list(k["kutular"]), [b for b, _ in pred.get(a, [])]) for a, k, _ in farkli], 0.5)["F1"],
            "O_okumasina_gore": prf([goruntu_sayimlari(list(o["kutular"]), [b for b, _ in pred.get(a, [])]) for a, _, o in farkli], 0.5)["F1"]}
        for m, pred in adaylar.items()}
    sonuc["_satirlar"] = [(a, k, sy) for (a, k, _), sy in zip(farkli, d["_sayim"])]
    return sonuc


def uzman_kutu_ozeti(kume: dict) -> dict:
    grup = defaultdict(list)
    for r in kume.values():
        grup[r["uzman"]].append(r)
    out = {}
    for u in sorted(grup):
        liste = grup[u]
        alan = [100 * w * h for r in liste for (_, _, w, h) in r["kutular"]]
        kutu = sum(len(r["kutular"]) for r in liste)
        out[u] = {"goruntu": len(liste), "kutu": kutu, "kutu_goruntu": kutu / len(liste),
                  "lezyonsuz_goruntu": sum(1 for r in liste if not r["kutular"]),
                  "duzenlendi": sum(1 for r in liste if r["durum"] == "edited"),
                  "onaylandi": sum(1 for r in liste if r["durum"] == "approved"),
                  "alan_medyan_yuzde": statistics.median(alan) if alan else None,
                  "alan_ceyrekler_yuzde": [float(np.percentile(alan, 25)), float(np.percentile(alan, 75))] if alan else None,
                  "sahne": {s: sum(1 for r in liste if r["sahne"] == s) for s in SAHNELER}}
    return out


def ioa(a, b) -> float:
    """a kutusunun alanının b kutusu içinde kalan oranı (YOLO cx, cy, w, h)."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih / (a[2] * a[3]) if a[2] * a[3] > 0 else 0.0


def eslestir_yakin(ref: list, pred: list) -> list[tuple[int, int, float]]:
    """(ref_i, pred_j, IoU) — bire bir, görüntüdeki min(n, m) kutu çiftinin hepsi eşleşir. Maliyet kesişen çiftte
    1 − IoU (< 1), kesişmeyende 1 + merkez uzaklığı (≥ 1): önce örtüşme, örtüşmeyen kutu en yakın boştakiyle."""
    if not ref or not pred:
        return []
    m = np.empty((len(ref), len(pred)))
    for i, r in enumerate(ref):
        for j, q in enumerate(pred):
            v = iou(r, q)
            m[i, j] = 1 - v if v > 0 else 1 + ((r[0] - q[0]) ** 2 + (r[1] - q[1]) ** 2) ** 0.5
    ri, ci = linear_sum_assignment(m)
    return [(int(i), int(j), iou(ref[i], pred[j])) for i, j in zip(ri, ci)]


def bolge_duzeyi(ref: dict, pred: dict, ic_esik: float = 0.5) -> dict:
    """Bölge düzeyi (2026-10-11, sonradan eklenen betimsel ölçü): istem VLM'den belirti gösteren "bölgeyi", yönerge
    uzmandan tek tek lezyonu istiyor. Uzman kutusu, alanının ≥ %50'si bir model kutusunun içindeyse "bölgesi
    bulunmuş" sayılır. Oranlar: bölgesi bulunan uzman kutusu (bölge duyarlılığı), en az bir uzman kutusu içeren model
    kutusu (bölge kesinliği), ve iki kümenin kutu alanı medyanı (görüntünün yüzdesi)."""
    n_ref = ic_ref = n_pred = ic_pred = 0
    alan_ref, alan_pred = [], []
    for a, r in ref.items():
        R = list(r["kutular"])
        P = [b for b, _ in pred.get(a, [])]
        ic = [[ioa(e, b) >= ic_esik for b in P] for e in R]
        n_ref += len(R)
        n_pred += len(P)
        ic_ref += sum(1 for i in range(len(R)) if any(ic[i]))
        ic_pred += sum(1 for j in range(len(P)) if any(ic[i][j] for i in range(len(R))))
        alan_ref += [100 * w * h for (_, _, w, h) in R]
        alan_pred += [100 * w * h for (_, _, w, h) in P]
    return {"uzman_kutusu_bolgesi_bulunan": ic_ref / n_ref if n_ref else None,
            "model_kutusu_uzman_lezyonu_iceren": ic_pred / n_pred if n_pred else None,
            "n_uzman_kutu": n_ref, "n_model_kutu": n_pred,
            "alan_medyan_yuzde_uzman": statistics.median(alan_ref) if alan_ref else None,
            "alan_medyan_yuzde_model": statistics.median(alan_pred) if alan_pred else None,
            "ic_esik": ic_esik}


def duzeltme_dokumu(ref: dict, v0: dict, ic_esik: float = 0.5, kural: str = "yakin") -> dict:
    """Uzman son etiketi ↔ VLM ön-etiketi farkının kodla dökümü (2026-10-11, proje sahibinin tarifi).

    kural "yakin" (birincil): görüntüdeki bütün VLM ve uzman kutuları bire bir eşleşir (`eslestir_yakin`), kesişmeyen
    kutu en yakın boştaki kutuyla. Uzman o görüntüde daha az kutu çizdiyse artan VLM kutuları model yanlışı (silindi),
    daha çok çizdiyse artan uzman kutuları ekleme.
      IoU ≥ 0,95 değişmedi · 0,5 ≤ IoU < 0,95 hafif düzeltildi · 0 < IoU < 0,5 büyük düzeltme · IoU = 0 taşındı
    kural "kesisen" (karşılaştırma): yalnız kesişen çift eşleşmiş sayılır, kesişmeyen kutu silindi + eklendi.
      Eşleşmeyen VLM kutusu = model yanlışı (uzman silmiş): içinde uzman kutusu varsa (uzman kutusunun alanının
        ≥ %50'si o VLM kutusunda) "bölge doğru, lezyon ayrıntısı yanlış", yoksa "yanlış pozitif".
      Eşleşmeyen uzman kutusu = uzman eklemesi: alanının ≥ %50'si bir VLM kutusunun içindeyse "VLM bölgesinin
        içinde" (VLM bölgeyi bulmuş, lezyonu ayırmamış), yoksa "VLM'in hiç görmediği lezyon".
    Bölge düzeyi (eşleştirmeden bağımsız): uzman kutularının bir VLM kutusunun içinde kalan oranı ve en az bir
    uzman kutusu içeren VLM kutularının oranı. Kaynak klasörse bunlar kaydedilmiş düzenleme değil, iki etiket
    kümesinin geometrik farkıdır."""
    kat = defaultdict(int)
    dokunulmamis = uzman_kutu = uzman_icte = vlm_iceren = 0
    for a, r in ref.items():
        onc = [b for b, _ in v0.get(a, [])]
        son = list(r["kutular"])
        ciftler = eslestir_yakin(son, onc) if kural == "yakin" else [(i, j, v) for i, j, v in eslestir(son, onc) if v > 0]
        eslesen_onc = {j for _, j, _ in ciftler}
        eslesen_son = {i for i, _, _ in ciftler}
        for _, _, v in ciftler:
            kat["degismedi" if v >= 0.95 else "hafif_duzeltildi" if v >= 0.5 else "buyuk_duzeltildi" if v > 0 else "tasindi"] += 1
        icinde = [[ioa(e, b) >= ic_esik for b in onc] for e in son]   # uzman kutusu i, VLM kutusu j içinde mi
        for j in range(len(onc)):
            if j not in eslesen_onc:
                kat["silindi_bolge_dogru" if any(icinde[i][j] for i in range(len(son))) else "silindi_yanlis_pozitif"] += 1
        for i in range(len(son)):
            if i not in eslesen_son:
                kat["eklendi_vlm_bolgesinde" if any(icinde[i]) else "eklendi_gorulmemis"] += 1
        uzman_kutu += len(son)
        uzman_icte += sum(1 for i in range(len(son)) if any(icinde[i]))
        vlm_iceren += sum(1 for j in range(len(onc)) if any(icinde[i][j] for i in range(len(son))))
        if len(onc) == len(son) and all(v >= 0.95 for _, _, v in ciftler) and len(ciftler) == len(onc):
            dokunulmamis += 1
    kat["silindi"] = kat["silindi_bolge_dogru"] + kat["silindi_yanlis_pozitif"]
    kat["eklendi"] = kat["eklendi_vlm_bolgesinde"] + kat["eklendi_gorulmemis"]
    toplam_v0 = kat["degismedi"] + kat["hafif_duzeltildi"] + kat["buyuk_duzeltildi"] + kat["tasindi"] + kat["silindi"]
    oran = lambda x, n: round(x / n, 3) if n else None
    return {"kural": ("bire bir, bütün çiftler eşleşir: önce örtüşme (1 − IoU), örtüşmeyen en yakın boştakiyle (1 + merkez "
                      "uzaklığı); artan VLM kutusu silindi, artan uzman kutusu eklendi" if kural == "yakin" else
                      "bire bir IoU eşleştirmesi, yalnız kesişen çift eşleşmiş") + "; içerme eşiği uzman kutusu alanının %50'si",
            "v0_kutu": toplam_v0, "uzman_kutu": uzman_kutu, **dict(kat),
            "oranlar_v0_kutusu": {k: oran(kat[k], toplam_v0) for k in (
                "degismedi", "hafif_duzeltildi", "buyuk_duzeltildi", "tasindi", "silindi", "silindi_bolge_dogru", "silindi_yanlis_pozitif")},
            "oranlar_uzman_kutusu": {"eslesen": oran(uzman_kutu - kat["eklendi"], uzman_kutu),
                                     "eklendi_vlm_bolgesinde": oran(kat["eklendi_vlm_bolgesinde"], uzman_kutu),
                                     "eklendi_gorulmemis": oran(kat["eklendi_gorulmemis"], uzman_kutu)},
            "bolge_duzeyi": {"uzman_kutusu_bir_vlm_kutusunun_icinde": oran(uzman_icte, uzman_kutu),
                             "uzman_kutusu_iceren_vlm_kutusu": oran(vlm_iceren, toplam_v0)},
            "dokunulmadan_kabul_goruntu": dokunulmamis, "goruntu": len(ref)}


def uzman_suresi(islem: list[dict], sahne: dict, mola_s: int = MOLA_S) -> dict:
    """R-EFFORT (brif §7): görüntü başına süre = uzmanın o görüntüdeki her kayıt/onayının, aynı
    uzmanın bir önceki kayıt/onayından geçen süresi toplamı. Öncülü olmayan (oturumun ilki) ya da
    öncülünden 10 dakikadan uzun süre sonra gelen olay (mola) görüntüyü süresiz bırakır."""
    kume_adi = {v: k for k, v in KLASOR.items()}
    olay = defaultdict(list)
    for r in islem:
        try:
            t = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if r.get("dataset") in kume_adi:
            olay[r["uzman"]].append((t, r["dataset"], r["image_name"]))
    sure, suresiz, olay_sayisi = defaultdict(float), set(), Counter()
    for u, liste in olay.items():
        liste.sort()
        for i, (t, ds, g) in enumerate(liste):
            anahtar = (u, ds, g)
            olay_sayisi[anahtar] += 1
            d = (t - liste[i - 1][0]).total_seconds() if i else None
            if d is not None and 0 <= d <= mola_s:
                sure[anahtar] += d
            else:
                suresiz.add(anahtar)
    gecerli = {k: v for k, v in sure.items() if k not in suresiz}
    g0, g1, g2, g3 = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for (u, ds, g), v in gecerli.items():
        k, s = kume_adi[ds], sahne.get((ds, g), "?")
        g0[k].append(v)
        g1[f"{u} · {k}"].append(v)
        g2[f"{k} · {s}"].append(v)
        g3[f"{u} · {k} · {s}"].append(v)
    return {"tanim": "görüntü başına toplanmış ardışık kayıt/onay aralığı; öncülsüz ya da > 10 dk boşluklu görüntü süresiz",
            "mola_esigi_s": mola_s,
            "goruntu": {"sureli": len(gecerli), "suresiz": len(set(olay_sayisi) - set(gecerli)),
                        "birden_cok_olayli": sum(1 for v in olay_sayisi.values() if v > 1)},
            "kume": {k: medyan_ozet(v) for k, v in sorted(g0.items())},
            "uzman_kume": {k: medyan_ozet(v) for k, v in sorted(g1.items())},
            "kume_sahne": {k: medyan_ozet(v) for k, v in sorted(g2.items())},
            "uzman_kume_sahne": {k: medyan_ozet(v) for k, v in sorted(g3.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    girdi = ap.add_mutually_exclusive_group(required=True)
    girdi.add_argument("--paket", type=Path, help="sunucu dışa aktarım paketi (degerlendirme_disa_aktar.sh)")
    girdi.add_argument("--klasor", type=Path, help="uzmanların klasör olarak gönderdiği etiketler (uzman_veriler_2)")
    ap.add_argument("--cikti", type=Path, default=VERI, help="sonuç klasörü (varsayılan veri/)")
    ap.add_argument("--lezyon-sinifi", type=int, default=0, help="lezyon kutusunun sınıf id'si (sunucu: Late_Blight = 0)")
    args = ap.parse_args()
    paket = (klasor_yukle(args.klasor.expanduser(), args.lezyon_sinifi) if args.klasor
             else paket_yukle(paketi_ac(args.paket.expanduser()), args.lezyon_sinifi))
    K, O, H = (paket["kumeler"][k] for k in ("kor", "ortak", "hitl"))
    kumeler = (("kor", K), ("ortak", O), ("hitl", H))
    sonuc: dict = {"olusturma": datetime.now().isoformat(timespec="seconds"), "paket": str(args.paket or args.klasor),
                   "kaynak_turu": "klasor" if args.klasor else "sunucu_paketi",
                   "kaynak_notu": ("uzmanlar etiketleri sunucu dışında çizip klasör olarak gönderdi: inceleme durumu "
                                   "(onay/düzenleme) ve süre kaydı yok") if args.klasor else None,
                   "lezyon_sinifi": args.lezyon_sinifi, "sinif_dagilimi": paket["siniflar"],
                   "n": {"kor": len(K), "ortak": len(O), "hitl": len(H), "hedef": paket["hedef"],
                         "uzman_dagilimi": {k: dict(sorted(Counter(r["uzman"] for r in d.values()).items())) for k, d in kumeler},
                         "sahne": {k: {s: sum(1 for r in d.values() if r["sahne"] == s) for s in SAHNELER} for k, d in kumeler},
                         "koleksiyon": {k: dict(sorted(Counter(r["koleksiyon"] for r in d.values()).items())) for k, d in kumeler}},
                   "uzman_kutu_ozeti": {k: uzman_kutu_ozeti(d) for k, d in kumeler if d}}
    yabanci = {k: {s: n for s, n in v.items() if s != str(args.lezyon_sinifi)} for k, v in paket["siniflar"].items()}
    yabanci = {k: v for k, v in yabanci.items() if v}
    if yabanci:
        print(f"UYARI: lezyon sınıfı ({args.lezyon_sinifi}) dışında kutular var, sayılmadı: {yabanci}")
    v0 = v0_ciktilari()
    adaylar = {m: model_ciktilari(m) for m in ADAYLAR}
    # Tutarlılık: sunucudaki K, VLM koşularının K'si (veri/kor_kume_60.txt) olmalı; H öğrenci paketinin H'si
    kor_liste = [s.strip() for s in (VERI / "kor_kume_60.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    if paket["plan"]["kor"] and sorted(paket["plan"]["kor"]) != sorted(kor_liste):
        fark = sorted(set(paket["plan"]["kor"]) ^ set(kor_liste))
        print(f"HATA: sunucudaki kör küme, VLM koşularının kör kümesiyle aynı değil ({len(fark)} fark: {fark[:6]}).")
        return 65
    eksik = {m: sorted(set(K) - set(p))[:3] for m, p in adaylar.items() if set(K) - set(p)}
    if eksik:
        print(f"HATA: adayların K koşusunda incelenmiş görüntü eksik: {eksik}")
        return 65
    ogr = KOK.parent / "colab" / "paketler" / "ogrenci.zip"
    ogr_H = json.loads(zipfile.ZipFile(ogr).read("ogrenci/bolunme.json"))["hitl"] if ogr.exists() else None
    sonuc["kume_tutarliligi"] = {"K_kor_kume_60_ile_ayni": True,
                                 "H_ogrenci_paketiyle_ayni": sorted(paket["plan"]["hitl"]) == sorted(ogr_H) if ogr_H and paket["plan"]["hitl"] else None}
    if sonuc["kume_tutarliligi"]["H_ogrenci_paketiyle_ayni"] is False:
        fark = sorted(set(paket["plan"]["hitl"]) ^ set(ogr_H))
        print(f"UYARI: sunucudaki H, öğrenci paketinin (colab/paketler/ogrenci.zip) H'siyle aynı değil ({len(fark)} fark: {fark[:4]}): "
              "Aşama 2'den önce paketle_ogrenci.py ile yeniden üretilmeli (Aşama 1 etkilenmez).")
    satirlar = []
    # ---- K: altı aday (1280 px) + bf16 kontrolü + E2B'nin ilk koşusu (v0) ve tam çözünürlük koşusu
    if K:
        idx = bootstrap_indeksi(len(K))
        ek = {m: model_ciktilari(m) for m in KONTROL} | {"gemma4-e2b_v0": v0}
        if (VERI / "vlm_kosulari" / "gemma4-e2b_tamcozunurluk.jsonl").exists():
            ek["gemma4-e2b_tamcozunurluk"] = model_ciktilari("gemma4-e2b_tamcozunurluk")
        kor = {m: model_degerlendir(K, pred, idx) for m, pred in (adaylar | ek).items()}
        for m, d in kor.items():
            for a, sy in zip(sorted(K), d["_sayim"]):
                satirlar.append({"kume": "kor", "model": m, "goruntu": a, "sahne": K[a]["sahne"], "uzman": K[a]["uzman"], **sy})
        farklar = {}
        for i, a in enumerate(ADAYLAR):
            for b in ADAYLAR[i + 1:]:
                d = kor[a]["iou_0.5"]["GA95"]["_F1_dagilim"] - kor[b]["iou_0.5"]["GA95"]["_F1_dagilim"]
                farklar[f"{a} − {b}"] = {"F1_farki": kor[a]["iou_0.5"]["F1"] - kor[b]["iou_0.5"]["F1"], "GA95": ga(d)}
        secim = ogretmen_sec(kor)
        cevirme = {m: cevirme_tutarliligi(m) for m in ADAYLAR + KONTROL}
        rho = spearman_ozet([cevirme[m]["F1_05"] for m in ADAYLAR], [kor[m]["iou_0.5"]["F1"] for m in ADAYLAR])
        # R-FLIP-CORR: görüntü başına çevirme tutarlılığı ↔ görüntü başına F1 (aday ↔ kör uzman etiketi)
        flip_iliski, x_tum, y_tum = {}, [], []
        for m in ADAYLAR:
            x, y = [], []
            for a, sy in zip(sorted(K), kor[m]["_sayim"]):
                c = cevirme[m]["_goruntu"].get(a)
                if not c or not c[1] or not (sy["n_pred"] + sy["n_ref"]):
                    continue
                x.append(2 * c[0] / c[1])
                y.append(2 * sy["tp_0.5"] / (sy["n_pred"] + sy["n_ref"]))
            flip_iliski[m] = spearman_ozet(x, y) | {"n_goruntu": len(x)}
            x_tum += x
            y_tum += y
        flip_iliski["alti_aday_birlesik"] = spearman_ozet(x_tum, y_tum) | {"n_goruntu_model": len(x_tum)}
        sonuc["kor"] = {"adaylar": ADAYLAR, "kontrol": KONTROL, "ek_satirlar": [m for m in ek if m not in KONTROL],
                        "modeller": kor, "ogretmen_secimi": secim, "ogretmen_adayi": secim["ogretmen"],
                        "siralama_F1_05": secim["siralama_F1_05"], "esli_farklar_F1_05": farklar,
                        "cevirme_tutarliligi_F1_05": {m: c["F1_05"] for m, c in cevirme.items() if c},
                        "cevirme_siralamasi_iliskisi": {"not": "n = 6; yalnız betimsel"} | rho | {"n_model": len(ADAYLAR)},
                        "cevirme_hata_iliskisi": flip_iliski,
                        "bolge_duzeyi": {m: bolge_duzeyi(K, pred) for m, pred in (adaylar | ek).items()}}
    # ---- O: uzmanlar arası uyum (D1)
    if O:
        ia = uzmanlar_arasi(K, O, adaylar)
        for a, k, sy in ia.get("_satirlar", []):
            satirlar.append({"kume": "ortak", "model": "uzman_uzman", "goruntu": a, "sahne": k["sahne"], "uzman": k["uzman"], **sy})
        ortak_ref = {a: K[a] for a in O if a in K and K[a]["uzman"] != O[a]["uzman"]}
        ia["bolge_duzeyi_uzman_uzman"] = bolge_duzeyi(ortak_ref, {a: [(b, None) for b in O[a]["kutular"]] for a in ortak_ref})
        sonuc["ortak"] = ia
    # ---- H: ilk koşu (v0) ön-etiketleri ↔ uzman son hâli
    if H:
        h = model_degerlendir(H, v0)
        sonuc["hitl"] = {"v0": h, "duzeltme": duzeltme_dokumu(H, v0), "duzeltme_kesisen": duzeltme_dokumu(H, v0, kural="kesisen"),
                         "duzeltme_sahne": {s: duzeltme_dokumu({a: r for a, r in H.items() if r["sahne"] == s}, v0) for s in SAHNELER}}
        for a, sy in zip(sorted(H), h["_sayim"]):
            satirlar.append({"kume": "hitl", "model": "gemma4-e2b_v0", "goruntu": a, "sahne": H[a]["sahne"], "uzman": H[a]["uzman"], **sy})
    sonuc["uzman_suresi"] = uzman_suresi(paket["islem"], paket["sahne_haritasi"])
    args.cikti.mkdir(parents=True, exist_ok=True)
    (args.cikti / "uzman_karsilastirma.json").write_text(json.dumps(yuvarla(sonuc, kayit=True), ensure_ascii=False, indent=1), encoding="utf-8")
    if satirlar:
        with open(args.cikti / "uzman_karsilastirma_goruntu.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(satirlar[0]))
            w.writeheader()
            w.writerows(satirlar)
    # ---- özet
    y = yuvarla
    print(f"İncelenen: kör {len(K)}/{paket['hedef']['kor']} · ortak {len(O)}/{paket['hedef']['ortak']} · HITL {len(H)}/{paket['hedef']['hitl']}")
    if K:
        print("Kör küme — F1 @ 0,5 [%95 GA] · P · R · görüntü düzeyi P / R · yerinde bitki F1 · çevirme tutarlılığı:")
        for m in sonuc["kor"]["siralama_F1_05"] + [m for m in kor if m not in ADAYLAR]:
            d = kor[m]
            ins = d["sahne_in_situ"]["F1"] if d["sahne_in_situ"] else float("nan")
            c = sonuc["kor"]["cevirme_tutarliligi_F1_05"].get(m)
            etiket = "" if m in ADAYLAR else " (kontrol)" if m in KONTROL else " (ek)"
            print(f"  {m + etiket:36s} {y(d['iou_0.5']['F1'])} {y(d['iou_0.5']['GA95']['F1'])} · P {y(d['iou_0.5']['kesinlik'])} · "
                  f"R {y(d['iou_0.5']['duyarlilik'])} · görüntü {y(d['goruntu_duzeyi']['kesinlik'])} / {y(d['goruntu_duzeyi']['duyarlilik'])} · "
                  f"yerinde {y(ins)} · çevirme {y(c) if c is not None else '–'}")
        s = sonuc["kor"]["ogretmen_secimi"]
        print(f"Öğretmen: {s['ogretmen']} (en yüksek nokta tahmini). En iyiye göre fark [%95 GA] · Holm p:")
        for m, d in s["en_iyiye_gore"].items():
            print(f"  − {m:18s} {y(d['F1_farki'])} {y(d['GA95'])} · p {y(d['p_bootstrap'], 'p_')} → Holm {y(d['p_holm'], 'p_')} "
                  f"{'ayrışıyor' if d['ayrisiyor'] else 'ayrışmıyor'}")
        if not s["ikinciden_ayrisiyor"]:
            print(f"  → {s['ogretmen']} ile {s['ikinci']} ayrışmıyor: metinde \"not separable\".")
        if s["nokta_esitligi"]:
            print(f"  UYARI: nokta tahmininde tam eşitlik {s['nokta_esitligi']} — yerinde bitki F1'iyle seçildi.")
    if O:
        ia = sonuc["ortak"]
        if "iou_0.5" in ia:
            d, g = ia["iou_0.5"], ia["goruntu_duzeyi"]
            print(f"Uzmanlar arası (O, {ia['farkli_uzman']} görüntü, iki farklı uzman): F1 @ 0,5 {y(d['F1'])} {y(d['GA95']['F1'])} · "
                  f"@ 0,3 {y(ia['iou_0.3']['F1'])} · @ 0,75 {y(ia['iou_0.75']['F1'])} · görüntü uyumu {y(g['yuzde_uyum'])}, κ {y(g['cohen_kappa'])}")
        if ia["ayni_uzman"]:
            print(f"  UYARI: {ia['ayni_uzman']} ortak görüntüyü K'deki uzmanı yeniden okumuş (atama dışı): uzmanlar arası sayılmadı.")
    if H:
        d = sonuc["hitl"]["v0"]["iou_0.5"]
        print(f"HITL — v0 kabul oranı {y(d['kesinlik'])} · bulma oranı {y(d['duyarlilik'])} · F1 {y(d['F1'])}")
        dz = sonuc['hitl']['duzeltme']
        print(f"  VLM kutusu ({dz['v0_kutu']}): {dz['oranlar_v0_kutusu']}")
        print(f"  uzman kutusu ({dz['uzman_kutu']}): {dz['oranlar_uzman_kutusu']} · bölge düzeyi {dz['bolge_duzeyi']}")
    for k, v in sonuc["uzman_suresi"]["uzman_kume"].items():
        print(f"Süre {k}: medyan {y(v['medyan_s'])} s {y(v['ceyrekler_s'])} (n {v['n_goruntu']})")
    print(f"→ {args.cikti / 'uzman_karsilastirma.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
