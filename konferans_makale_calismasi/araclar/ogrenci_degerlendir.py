"""Öğrenci koşularının yerel değerlendirmesi (S-LEARN, E1-FID, E1-ACC, E1-HITL, E1-EXP, E1-SEED) — SALT OKUMA.

Girdi: Colab'ın Drive'a yazdığı `sonuclar/ogrenci/` (her koşu klasöründe `tamam.json` ve
`tahminler.json`), öğrenci paketi (`colab/paketler/ogrenci.zip`: etiketler + bölünme) ve
isteğe bağlı uzman etiketleri: sunucunun dışa aktarım paketi (`--uzman-paket`, önerilen; K ve H
özgün adlarıyla, `uzman_vlm_karsilastir.paket_yukle`) ya da `.txt` zip/klasörü (`--uzman`).

Her koşu için (brif §7 sonu, "öğrenci değerlendirme noktası"):
  1. Güven eşiği: doğrulama bölümünde, o kolun kendi doğrulama etiketlerine göre Macar
     eşleştirmesiyle F1 @ 0,5'i en büyükleyen eşik (0,01–0,95, 0,01 adım; eşitlikte küçük eşik).
  2. K'de bu eşikle kesinlik / duyarlılık / F1 @ IoU 0,3 / 0,5 / 0,75. Eşleştirme adaylarınkiyle
     aynı kod (`uzman_vlm_karsilastir.py`). **Sadakat** öğretmenin K etiketine, **doğruluk** kör
     uzman etiketine göre (varsa). Görüntü düzeyinde lezyon var/yok kesinliği ve duyarlılığı.
  3. K'de eşiksiz AP50: Ultralytics 8.4.144 doğrulamasıyla aynı yöntem (açgözlü IoU eşleştirmesi +
     `ap_per_class`, 101 noktalı interpolasyon), sadakat ve doğruluk için.

Kullanım:
  python ogrenci_degerlendir.py --sonuclar ~/Downloads/ogrenci [--uzman-paket ~/Downloads/degerlendirme_<zaman>.tar.gz]
Çıktı: veri/ogrenci_degerlendirme.json + ekrana tablo.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uzman_vlm_karsilastir import (ESIKLER, bootstrap, bootstrap_indeksi, ga, goruntu_sayimlari,  # noqa: E402
                                   klasor_yukle, paket_yukle, paketi_ac, prf)

KOK = Path(__file__).resolve().parent.parent
PAKET_VARSAYILAN = KOK.parent / "colab" / "paketler" / "ogrenci.zip"
CIKTI = KOK / "veri" / "ogrenci_degerlendirme.json"
ESIK_IZGARASI = np.round(np.arange(0.01, 0.96, 0.01), 2)


# ------------------------------------------------------------------ girdiler
class Kaynak:
    """Zip ya da klasörden `ogrenci/...` yollarını okur."""

    def __init__(self, yol: Path):
        self.yol = yol
        self.zip = zipfile.ZipFile(yol) if yol.is_file() else None

    def oku(self, ic: str) -> str | None:
        if self.zip:
            try:
                return self.zip.read(ic).decode("utf-8")
            except KeyError:
                return None
        p = self.yol / ic if not ic.startswith("ogrenci/") else self.yol / ic.split("/", 1)[1]
        return p.read_text(encoding="utf-8") if p.exists() else None


def etiket_ayristir(metin: str | None) -> list[tuple]:
    if not metin:
        return []
    out = []
    for s in metin.splitlines():
        p = s.split()
        if len(p) == 5:
            out.append(tuple(float(v) for v in p[1:]))
    return out


def uzman_etiketleri(yol: Path | None, adlar: list[str]) -> dict[str, list] | None:
    if yol is None:
        return None
    if yol.is_file():
        z = zipfile.ZipFile(yol)
        icerik = {Path(n).stem: z.read(n).decode("utf-8") for n in z.namelist() if n.endswith(".txt")}
    else:
        icerik = {p.stem: p.read_text(encoding="utf-8") for p in yol.rglob("*.txt")}
    return {a: etiket_ayristir(icerik.get(Path(a).stem)) for a in adlar if Path(a).stem in icerik}


# ------------------------------------------------------------------ ölçüler
def esikle(tahmin: list, tau: float) -> list[tuple]:
    return [tuple(t[:4]) for t in tahmin if t[4] >= tau]


def f1_05(ref: dict, tahmin: dict, adlar: list[str], tau: float) -> float:
    s = [goruntu_sayimlari(ref.get(a, []), esikle(tahmin.get(a, []), tau)) for a in adlar]
    return prf(s, 0.5)["F1"]


def esik_sec(ref: dict, tahmin: dict, adlar: list[str]) -> tuple[float, float]:
    en_iyi = (0.01, -1.0)
    for tau in ESIK_IZGARASI:
        f = f1_05(ref, tahmin, adlar, float(tau))
        f = -1.0 if np.isnan(f) else f
        if f > en_iyi[1] + 1e-12:
            en_iyi = (float(tau), f)
    return en_iyi


def kutu_olculeri(ref: dict, tahmin: dict, adlar: list[str], tau: float) -> dict:
    s = [goruntu_sayimlari(ref.get(a, []), esikle(tahmin.get(a, []), tau)) for a in adlar]
    # Kayıtta tam duyarlık: makalenin sayıları tur4_sayilar.py'de bir kez yuvarlanır (çift yuvarlama olmasın diye)
    out = {f"iou_{t}": prf(s, t) for t in ESIKLER}
    # görüntü düzeyinde bootstrap %95 GA (10.000, tohum 42; adaylarla aynı yöntem ve aynı indeksler)
    out["iou_0.5"]["GA95"] = {k: [float(x) for x in v] for k, v in bootstrap(s, 0.5, bootstrap_indeksi(len(adlar))).items()
                              if not k.startswith("_")}
    var_ref = [bool(ref.get(a)) for a in adlar]
    var_t = [bool(esikle(tahmin.get(a, []), tau)) for a in adlar]
    tp = sum(r and t for r, t in zip(var_ref, var_t))
    out["goruntu_duzeyi"] = {"kesinlik": tp / sum(var_t) if sum(var_t) else None,
                             "duyarlilik": tp / sum(var_ref) if sum(var_ref) else None,
                             "tp": tp, "tahmin_var": sum(var_t), "referans_var": sum(var_ref), "goruntu": len(adlar)}
    return out


def xyxy(k: list) -> np.ndarray:
    a = np.array(k, dtype=float).reshape(-1, 4)
    return np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], 1)


def iou_matrisi(g: np.ndarray, p: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(g[:, None, 0], p[None, :, 0]); iy1 = np.maximum(g[:, None, 1], p[None, :, 1])
    ix2 = np.minimum(g[:, None, 2], p[None, :, 2]); iy2 = np.minimum(g[:, None, 3], p[None, :, 3])
    kes = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]); ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    return kes / np.maximum(ag[:, None] + ap[None, :] - kes, 1e-12)


def ap50(ref: dict, tahmin: dict, adlar: list[str]) -> float | None:
    """Ultralytics DetectionValidator ile aynı: açgözlü IoU eşleştirmesi (match_predictions,
    use_scipy=False) + ap_per_class (101 noktalı interpolasyon), tek sınıf, IoU 0,5."""
    from ultralytics.utils.metrics import ap_per_class
    tp, conf, n_ref = [], [], 0
    for a in adlar:
        g = ref.get(a, [])
        p = tahmin.get(a, [])
        n_ref += len(g)
        if not p:
            continue
        dogru = np.zeros(len(p), dtype=bool)
        if g:
            iou = iou_matrisi(xyxy(g), xyxy([t[:4] for t in p]))
            eslesme = np.array(np.nonzero(iou >= 0.5)).T
            if eslesme.shape[0]:
                if eslesme.shape[0] > 1:
                    eslesme = eslesme[iou[eslesme[:, 0], eslesme[:, 1]].argsort()[::-1]]
                    eslesme = eslesme[np.unique(eslesme[:, 1], return_index=True)[1]]
                    eslesme = eslesme[np.unique(eslesme[:, 0], return_index=True)[1]]
                dogru[eslesme[:, 1].astype(int)] = True
        tp.extend(dogru.tolist()); conf.extend(t[4] for t in p)
    if n_ref == 0:
        return None
    if not tp:
        return 0.0
    sonuc = ap_per_class(np.array(tp)[:, None], np.array(conf), np.zeros(len(tp)), np.zeros(n_ref))
    return float(sonuc[5][0, 0])


# ------------------------------------------------------------------ ana akış
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sonuclar", type=Path, required=True, help="Drive'dan indirilen sonuclar/ogrenci klasörü")
    ap.add_argument("--paket", type=Path, default=PAKET_VARSAYILAN)
    ap.add_argument("--uzman-paket", type=Path, help="sunucunun dışa aktarım paketi (K ve H uzman etiketleri)")
    ap.add_argument("--uzman-klasor", type=Path, help="uzmanların klasör olarak gönderdiği etiketler (uzman_veriler_2)")
    ap.add_argument("--uzman", type=Path, help="uzman etiketleri (.txt zip ya da klasörü, özgün ad köküyle)")
    ap.add_argument("--cikti", type=Path, default=CIKTI)
    args = ap.parse_args()

    paket = Kaynak(args.paket)
    B = json.loads(paket.oku("ogrenci/bolunme.json"))
    adlar_hepsi = sorted(set(B["egitim"]) | set(B["dogrulama"]) | set(B["kor"]))
    etiket = {}

    def aday_etiketi(aday: str) -> dict:
        if aday not in etiket:
            etiket[aday] = {a: etiket_ayristir(paket.oku(f"ogrenci/etiketler/{aday}/{Path(a).stem}.txt")) for a in adlar_hepsi}
        return etiket[aday]

    if args.uzman_paket or args.uzman_klasor:
        up = (klasor_yukle(args.uzman_klasor.expanduser()) if args.uzman_klasor
              else paket_yukle(paketi_ac(args.uzman_paket.expanduser())))
        uzman = {a: list(r["kutular"]) for k in ("kor", "hitl") for a, r in up["kumeler"][k].items()}
        sahne = {a: r["sahne"] for a, r in up["kumeler"]["kor"].items()}
        print(f"uzman etiketi: K {len(up['kumeler']['kor'])}/60 · H {len(up['kumeler']['hitl'])}/150")
    else:
        uzman = uzman_etiketleri(args.uzman, sorted(set(B["kor"]) | set(B["hitl"])))
        sahne = {}
    H = set(B["hitl"])
    satirlar, slearn_sayim, dogruluk_sayim = [], {}, {}
    for klasor in sorted(p for p in args.sonuclar.iterdir() if (p / "tamam.json").exists()):
        bilgi = json.loads((klasor / "tamam.json").read_text(encoding="utf-8"))
        tahmin = json.loads((klasor / "tahminler.json").read_text(encoding="utf-8"))
        ogretmen, kol = bilgi["ogretmen"], bilgi["kol"]
        og = aday_etiketi(ogretmen)
        if kol == "ogretmen":
            dogr_adlar, dogr_ref = B["dogrulama"], og
        elif kol == "ogretmen_uzman150":
            if uzman is None or any(a in H and a not in uzman for a in B["dogrulama"]):
                print(f"[atlandı] {klasor.name}: doğrulamadaki H görüntülerinin uzman etiketi gerekli"); continue
            dogr_adlar = B["dogrulama"]
            dogr_ref = {a: (uzman.get(a, []) if a in H else og[a]) for a in dogr_adlar}
        else:
            if uzman is None or any(a not in uzman for a in B["uzman150_dogrulama"]):
                print(f"[atlandı] {klasor.name}: yalnız-uzman doğrulamasının uzman etiketi gerekli"); continue
            dogr_adlar, dogr_ref = B["uzman150_dogrulama"], uzman
        tau, f_dogr = esik_sec(dogr_ref, tahmin, dogr_adlar)
        K = B["kor"]
        satir = {"ad": klasor.name, "ogretmen": ogretmen, "kol": kol, "model": bilgi["model"], "tohum": bilgi["tohum"],
                 "epoch": bilgi.get("epoch"), "esik": tau, "dogrulama_F1_05": f_dogr,
                 "sadakat": kutu_olculeri(og, tahmin, K, tau), "AP50_sadakat": ap50(og, tahmin, K)}
        if bilgi.get("asama") == 1:
            slearn_sayim[ogretmen] = [goruntu_sayimlari(og.get(a, []), esikle(tahmin.get(a, []), tau)) for a in K]
        if uzman is not None and all(a in uzman for a in K):
            satir["dogruluk"] = kutu_olculeri(uzman, tahmin, K, tau)
            dogruluk_sayim[(bilgi["model"], bilgi["tohum"], kol)] = [
                goruntu_sayimlari(uzman.get(a, []), esikle(tahmin.get(a, []), tau)) for a in K]
            satir["AP50_dogruluk"] = ap50(uzman, tahmin, K)
            yerinde = [a for a in K if sahne.get(a) == "in_situ"]
            if yerinde:   # hakem R2-MA (2026-10-12): yerinde bitki görüntüleri, aynı eşik
                satir["dogruluk_in_situ_F1_05"] = f1_05(uzman, tahmin, yerinde, tau)
        satirlar.append(satir)
        d = satir.get("dogruluk", {}).get("iou_0.5", {}).get("F1")
        k3 = lambda x: "–" if x is None else f"{x:.3f}"
        print(f"{klasor.name:48s} τ={tau:.2f} · sadakat F1@.5 {k3(satir['sadakat']['iou_0.5']['F1'])} "
              f"AP50 {k3(satir['AP50_sadakat'])} · doğruluk F1@.5 {k3(d)}")
    # S-LEARN: adayların öğrenilebilirlik farkı, aynı K görüntülerinde eşleştirilmiş bootstrap (her öğrenci kendi
    # öğretmeninin K etiketine göre; betimsel, çoklu karşılaştırma düzeltmesi yok)
    esli = {}
    if len(slearn_sayim) > 1:
        idx = bootstrap_indeksi(len(B["kor"]))
        dag = {m: bootstrap(v, 0.5, idx)["_F1_dagilim"] for m, v in slearn_sayim.items()}
        f1 = {m: prf(v, 0.5)["F1"] for m, v in slearn_sayim.items()}
        adlar = sorted(slearn_sayim)
        for i, a in enumerate(adlar):
            for b in adlar[i + 1:]:
                lo, hi = ga(dag[a] - dag[b])
                esli[f"{a} − {b}"] = {"F1_farki": f1[a] - f1[b], "GA95": [float(lo), float(hi)],
                                      "sifiri_disliyor": bool(lo > 0 or hi < 0)}
    # Kollar arası fark (2026-10-12): aynı model ve tohumda, kör uzman etiketine göre F1 @ 0,5, K görüntülerinde
    # eşleştirilmiş bootstrap (aynı indeksler; betimsel, çoklu karşılaştırma düzeltmesi yok). Aşama 1'in seçilen
    # öğretmen koşusu (slearn_<öğretmen>_yolo11n_t42) YOLO11n / öğretmen / 42 kolu sayılır.
    kol_esli = {}
    if dogruluk_sayim:
        idx = bootstrap_indeksi(len(B["kor"]))
        for (model, tohum) in sorted({(m, t) for m, t, _ in dogruluk_sayim}):
            kollar = {k: dogruluk_sayim[(model, tohum, k)] for k in ("ogretmen", "ogretmen_uzman150", "uzman150")
                      if (model, tohum, k) in dogruluk_sayim}
            dag = {k: bootstrap(v, 0.5, idx)["_F1_dagilim"] for k, v in kollar.items()}
            f1 = {k: prf(v, 0.5)["F1"] for k, v in kollar.items()}
            out = {}
            for a, b in (("ogretmen_uzman150", "ogretmen"), ("uzman150", "ogretmen_uzman150"), ("uzman150", "ogretmen")):
                if a in kollar and b in kollar:
                    lo, hi = ga(dag[a] - dag[b])
                    out[f"{a} − {b}"] = {"F1_farki": f1[a] - f1[b], "GA95": [float(lo), float(hi)],
                                         "sifiri_disliyor": bool(lo > 0 or hi < 0)}
            kol_esli[f"{model} t{tohum}"] = out
    args.cikti.write_text(json.dumps({"yontem": __doc__.strip().splitlines()[0], "kosular": satirlar,
                                      "slearn_esli_farklar_F1_05": esli, "kol_esli_farklar_F1_05": kol_esli},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {args.cikti} ({len(satirlar)} koşu)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
