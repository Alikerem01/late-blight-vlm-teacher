"""Tur 4 sayı tablosu: her `\\pending` kimliğinin ölçülmüş değeri, makalenin biçiminde — SALT OKUMA.

Analiz çıktılarını okur, uygulayıcının Tur 4'te makaleye yazacağı sayıları tek dosyada toplar. Sayıyı
kendisi hesaplamaz, yalnız biçimlendirir. Kaynak dosyalar:

  veri/uzman_karsilastirma.json      uzman_vlm_karsilastir.py (K, O, H, uzman süresi)
  veri/ogrenci_degerlendirme.json    ogrenci_degerlendir.py   (S-LEARN, Aşama 2 öğrencileri)
  veri/gecikme_lat_local.json        gecikme_olc.py           (LAT-LOCAL)
  veri/k_boyutu_benzetim_n<n>.json   k_boyutu_benzetim.py --n (gerçek n için güç; yoksa komutu yazar)

Kaynak dosyalar sayıları tam duyarlıkla tutar, her sayı burada bir kez ve yarım yukarı (ROUND_HALF_UP) yuvarlanır.
Üç ondalıklı kaydı ikinci kez iki ondalığa yuvarlamak sonucu 0,01 kaydırabilir.

Biçim: P, R, F1 ve AUROC üç ondalık; aralık tabloda "[a, b]", metinde "a--b"; yüzde bir ondalık; süre tam
saniye; eksi değer tabloda `$-0.120$`. Eksik veri "—" ve nedeni. Yorum kimlikleri (R-VLM-INTERP, E1-INTERP)
için yorum yazılmaz, ilgili sayılar listelenir. TOOL-AVAIL (depo bağlantısı) ve GUIDE proje sahibinin kararıdır.

Kullanım:  python tur4_sayilar.py            → veri/tur4_sayilar.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "veri"
ADAYLAR = ["gemma4-e2b", "gemma4-e2b-8bit", "gemma4-e4b", "gemma4-12b", "qwen3vl-4b", "qwen3vl-8b"]
KONTROL = "gemma4-e2b-bf16"
ETIKET = {"gemma4-e2b": "E2B", "gemma4-e2b-8bit": "E2B-8", "gemma4-e4b": "E4B", "gemma4-12b": "12B",
          "qwen3vl-4b": "Q4B", "qwen3vl-8b": "Q8B", KONTROL: "E2B bf16"}
AD = {"gemma4-e2b": "the 4-bit Gemma~4 E2B", "gemma4-e2b-8bit": "the 8-bit Gemma~4 E2B", "gemma4-e4b": "Gemma~4 E4B",
      "gemma4-12b": "Gemma~4 12B", "qwen3vl-4b": "Qwen3-VL-4B", "qwen3vl-8b": "Qwen3-VL-8B", KONTROL: "the bf16 Gemma~4 E2B"}
OGRENCILER = [("yolo11n", "YOLO11n", "2.59", "6.5"), ("yolo11s", "YOLO11s", "9.43", "21.7"), ("yolo11m", "YOLO11m", "20.05", "68.3"),
              ("yolo12n", "YOLO12n", "2.57", "7.5"), ("yolo12s", "YOLO12s", "9.25", "23.5"), ("yolo12m", "YOLO12m", "20.14", "71.0"),
              ("yolo26n", "YOLO26n", "2.50", "5.9"), ("yolo26s", "YOLO26s", "9.95", "22.7"), ("yolo26m", "YOLO26m", "21.77", "75.0")]
KOLLAR = ("ogretmen", "ogretmen_uzman150", "uzman150")
KOLEKSIYON = {"kaynak_a_images": "44-image collection", "kaynak_b_images": "257-image collection"}
YOK = "—"


# ------------------------------------------------------------------ biçim
def yy(x, b: int = 3) -> str:
    """Yarım yukarı yuvarlama, b ondalık. `repr` kaydın tam değerini verir (0.015 → 0.02, `f"{x:.2f}"` 0.01 verirdi)."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-b), rounding=ROUND_HALF_UP))


def isaretli(x, b: int = 3) -> str:
    return ("+" if x >= 0 else "") + yy(x, b)


def f3(x, b: int = 3) -> str:
    return YOK if x is None else yy(x, b)


def tex(x, b: int = 3) -> str:
    """Tablo hücresi: eksi değer matematik kipinde."""
    return YOK if x is None else (f"${yy(x, b)}$" if x < 0 else yy(x, b))


def ga_tablo(g, b: int = 3) -> str:
    return YOK if not g or None in g else f"[{tex(g[0], b)}, {tex(g[1], b)}]"


def ga_metin(g, b: int = 3) -> str:
    return YOK if not g or None in g else f"{yy(g[0], b)}--{yy(g[1], b)}"


def yuzde(x) -> str:
    return YOK if x is None else f"{yy(100 * x, 1)}\\%"


def oku(yol: Path):
    return json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else None


# ------------------------------------------------------------------ bölümler
class Rapor:
    def __init__(self):
        self.satir: list[str] = []
        self.uyari: list[str] = []

    def __call__(self, *s: str):
        self.satir.extend(s)

    def bolum(self, baslik: str):
        self.satir += ["", f"## {baslik}", ""]


def p1n(R: Rapor, U: dict | None):
    R.bolum("P1-N — tamamlanan görüntü sayıları")
    if not U:
        R("— uzman karşılaştırması yok (`uzman_vlm_karsilastir.py --paket …` çalıştırılmadı)."); return
    n = U["n"]
    for k, ad in (("kor", "K"), ("ortak", "O"), ("hitl", "H")):
        s, ko = n["sahne"].get(k, {}), n.get("koleksiyon", {}).get(k, {})
        kol = ", ".join(f"{v} from the {KOLEKSIYON.get(c, c)}" for c, v in ko.items())
        R(f"- {ad}: **{n[k]} of {n['hedef'][k]}** (detached {s.get('detached', 0)}, in situ {s.get('in_situ', 0)}; {kol or YOK}); "
          f"uzman başına {n['uzman_dagilimi'].get(k, {})}")
    R("", f"Önerilen yan cümle: \"{n['kor']} of the 60 images of K, {n['ortak']} of the 20 of O and {n['hitl']} of the 150 "
          f"of H were completed at the cut-off\".")
    tut = U.get("kume_tutarliligi", {})
    if tut.get("H_ogrenci_paketiyle_ayni") is False:
        R.uyari.append("Sunucunun H'si öğrenci paketinin H'siyle aynı değil (Aşama 2 bu yüzden geçersiz olabilir).")
    nk = n["kor"]
    guc = oku(VERI / f"k_boyutu_benzetim_n{nk}.json")
    if nk == 60:
        R("Güç: K tam (60), makaledeki 0.83 / 0.99 ve ±0.08 değişmez.")
    elif guc:
        g = guc.get(f"n={nk}", {})
        R(f"Güç (n = {nk}, `k_boyutu_benzetim_n{nk}.json`): fark 0.13 için {g.get('0.6/0.4', {}).get('guc')}, "
          f"0.20 için {g.get('0.65/0.35', {}).get('guc')}; tek model aralığının yarı genişliği "
          f"{g.get('0.55/0.45', {}).get('tek_model_GA95_yari')}.")
    else:
        R(f"Güç: n = {nk} < 60, önce `python araclar/k_boyutu_benzetim.py --n {nk}` çalıştırılmalı.")
        R.uyari.append(f"K eksik (n = {nk}): güç cümlesi gerçek n'e göre yeniden yazılmalı (`--n {nk}`).")


def vcomp(R: Rapor, U: dict | None):
    R.bolum("V-COMP — Tablo 2 (adaylar K'de, kör uzman etiketine göre)")
    if not U or "kor" not in U:
        R("— K sonucu yok."); return
    kor, sec = U["kor"]["modeller"], U["kor"]["ogretmen_secimi"]
    en_iyi = sec["ogretmen"]
    R("LaTeX satırları (sütunlar: P, R, F1 at 0.5 [GA], F1 at 0.3 / 0.75, F1 in situ, image-level P / R, ΔF1 to best [GA]; "
      "ΔF1 = aday − en iyi):", "", "```latex")
    for m in ADAYLAR + [KONTROL]:
        d = kor.get(m)
        if not d:
            R(f"% {ETIKET[m]}: veri yok"); continue
        i5, ins, gd = d["iou_0.5"], d.get("sahne_in_situ") or {}, d["goruntu_duzeyi"]
        if m == en_iyi or m == KONTROL:
            delta = YOK
        else:
            e = sec["en_iyiye_gore"][m]
            delta = f"{tex(-e['F1_farki'], 2)} {ga_tablo([-e['GA95'][1], -e['GA95'][0]], 2)}"
        R(f"{ETIKET[m]} & {f3(i5['kesinlik'], 2)} & {f3(i5['duyarlilik'], 2)} & {f3(i5['F1'], 2)} {ga_tablo(i5['GA95']['F1'], 2)} & "
          f"{f3(d['iou_0.3']['F1'], 2)} / {f3(d['iou_0.75']['F1'], 2)} & {f3(ins.get('F1'), 2)} & "
          f"{f3(gd['kesinlik'], 2)} / {f3(gd['duyarlilik'], 2)} & {delta} \\\\")
    R("```", "")
    ikinci = sec["ikinci"]
    e2 = sec["en_iyiye_gore"][ikinci]
    R(f"- Seçilen öğretmen (önceden yazılmış kural): **{AD[en_iyi]}**, F1 {f3(kor[en_iyi]['iou_0.5']['F1'], 2)} "
      f"(95% interval {ga_metin(kor[en_iyi]['iou_0.5']['GA95']['F1'], 2)}).",
      f"- İkinci: {AD[ikinci]}, fark {f3(e2['F1_farki'], 2)} ({ga_metin(e2['GA95'], 2)}), Holm-adjusted p {f3(e2['p_holm'])} → "
      + ("**ayrışıyor**" if e2["ayrisiyor"] else "**ayrışmıyor: \"not separable\" diye yazılır**") + ".",
      f"- Ayrışmayanlar (Holm p ≥ 0.05): {', '.join(AD[m] for m in sec['ayrismayanlar']) or 'yok'}.",
      "- En iyiye beş fark (Holm): " + "; ".join(
          f"{ETIKET[m]} {f3(v['F1_farki'], 2)} [{ga_metin(v['GA95'], 2)}] p_Holm {f3(v['p_holm'])}" for m, v in sec["en_iyiye_gore"].items()))
    if sec.get("nokta_esitligi"):
        R.uyari.append(f"Nokta tahmininde tam eşitlik {sec['nokta_esitligi']}: yerinde bitki F1'iyle seçildi, metinde belirtilmeli.")
    R("", "Sahne kırılımı (yerinde bitki birincil), F1 at 0.5 [GA]: " + "; ".join(
        f"{ETIKET[m]} in situ {f3((kor[m].get('sahne_in_situ') or {}).get('F1'))} {ga_tablo((kor[m].get('sahne_in_situ') or {}).get('GA95', {}).get('F1'))}, "
        f"detached {f3((kor[m].get('sahne_detached') or {}).get('F1'))}" for m in ADAYLAR if m in kor))


def iaa(R: Rapor, U: dict | None):
    R.bolum("IAA — uzmanlar arası uyum (O) ve Tablo 2'nin son satırı")
    O = (U or {}).get("ortak")
    if not O or "iou_0.5" not in O:
        R("— ortak küme sonucu yok."); return
    gd = O["goruntu_duzeyi"]
    n = O["farkli_uzman"]
    R("```latex",
      f"Experts (O, $n = {n}$) & {YOK} & {YOK} & {f3(O['iou_0.5']['F1'], 2)} {ga_tablo(O['iou_0.5']['GA95']['F1'], 2)} & "
      f"{f3(O['iou_0.3']['F1'], 2)} / {f3(O['iou_0.75']['F1'], 2)} & {YOK} & agreement {gd['iki_de_var'] + gd['iki_de_yok']}/{n} & {YOK} \\\\",
      "```", "",
      f"- Kutu F1 at 0.5 {f3(O['iou_0.5']['F1'], 2)} ({ga_metin(O['iou_0.5']['GA95']['F1'], 2)}), at 0.3 {f3(O['iou_0.3']['F1'], 2)}, "
      f"at 0.75 {f3(O['iou_0.75']['F1'], 2)}; görüntü düzeyinde uyum {gd['iki_de_var'] + gd['iki_de_yok']}/{n}, "
      f"Cohen κ {f3(gd.get('cohen_kappa'), 2)}; kutu sayısı K okuması {O['kutu_sayisi']['K_okumasi']}, O okuması {O['kutu_sayisi']['O_okumasi']}.",
      "- Aynı görüntülerde adaylar (F1 at 0.5, K okumasına / O okumasına göre): " + "; ".join(
          f"{ETIKET[m]} {f3(v['K_okumasina_gore'], 2)} / {f3(v['O_okumasina_gore'], 2)}" for m, v in O["adaylar_ayni_goruntulerde_F1_05"].items()))
    if O.get("ayni_uzman"):
        R.uyari.append(f"Ortak kümede {O['ayni_uzman']} görüntüyü K'deki uzmanı yeniden okumuş (atama dışı), uzmanlar arası sayılmadı.")
    if n < 20:
        R.uyari.append(f"Ortak küme eksik: uzmanlar arası uyum {n} görüntüye dayanıyor (makale \"20\" diyor).")


def etiketsiz_iliski(R: Rapor, U: dict | None):
    R.bolum("V-FLIP-RANK, R-FLIP-CORR, R-CONF-AUROC")
    kor = (U or {}).get("kor")
    if not kor:
        R("— K sonucu yok."); return
    r = kor["cevirme_siralamasi_iliskisi"]
    sira = kor["siralama_F1_05"]
    cev = kor["cevirme_tutarliligi_F1_05"]
    R(f"- V-FLIP-RANK: çevirme tutarlılığı sırası ↔ F1 sırası, Spearman ρ {f3(r.get('spearman_rho'))} (n = {r['n_model']}, betimsel). "
      f"F1 sırası: {', '.join(ETIKET[m] for m in sira)}; çevirme sırası: "
      f"{', '.join(ETIKET[m] for m in sorted(ADAYLAR, key=lambda m: -cev[m]))}.")
    h = kor["cevirme_hata_iliskisi"]
    R("- R-FLIP-CORR: görüntü başına çevirme tutarlılığı ↔ görüntü başına F1, Spearman ρ: " + "; ".join(
        f"{ETIKET[m]} {f3(h[m].get('spearman_rho'))} (n {h[m].get('n_goruntu')})" for m in ADAYLAR if m in h)
      + f"; altı aday birleşik {f3(h.get('alti_aday_birlesik', {}).get('spearman_rho'))} (n {h.get('alti_aday_birlesik', {}).get('n_goruntu_model')}).")
    R("- R-CONF-AUROC: beyan edilen güvenin doğru (IoU ≥ 0.5) ve yanlış kutuları ayırması, AUROC: " + "; ".join(
        f"{ETIKET[m]} {f3(kor['modeller'][m].get('guven_auroc_05'))}" for m in ADAYLAR if m in kor["modeller"]))


def bolge(R: Rapor, U: dict | None):
    R.bolum("R-REGION — bölge düzeyi (sonradan eklenen betimsel ölçü, K)")
    B = ((U or {}).get("kor") or {}).get("bolge_duzeyi")
    if not B:
        R("— yok."); return
    kisa = {"gemma4-e2b": "E2B", "gemma4-e2b-8bit": "E2B-8", "gemma4-e4b": "E4B", "gemma4-12b": "12B",
            "qwen3vl-4b": "Q4B", "qwen3vl-8b": "Q8B", "gemma4-e2b-bf16": "E2B bf16"}
    R("Uzman kutusu, alanının ≥ %50'si bir aday kutusunun içindeyse \"bölgesi bulunmuş\" sayılır.", "")
    for m, k in kisa.items():
        x = B.get(m)
        if x:
            R(f"- {k}: bölgesi bulunan uzman kutusu {yuzde(x['uzman_kutusu_bolgesi_bulunan'])}, uzman lezyonu içeren aday kutusu "
              f"{yuzde(x['model_kutusu_uzman_lezyonu_iceren'])} · kutu alanı medyanı aday %{yy(x['alan_medyan_yuzde_model'], 2)}, uzman %{yy(x['alan_medyan_yuzde_uzman'], 2)}")
    o = ((U or {}).get("ortak") or {}).get("bolge_duzeyi_uzman_uzman")
    if o:
        R(f"- Uzman ↔ uzman (O, tavan): {yuzde(o['uzman_kutusu_bolgesi_bulunan'])} / {yuzde(o['model_kutusu_uzman_lezyonu_iceren'])}")


def guven(R: Rapor):
    """Beyan edilen güvenin aday başına dağılımı, K (hakem R1-M2, 2026-10-13): ayırıcı mı?"""
    R.bolum("R-CONF-DIST — beyan edilen güvenin dağılımı (K, adaylar)")
    kisa = {"gemma4-e2b": "E2B", "gemma4-e2b-8bit": "E2B-8", "gemma4-e4b": "E4B", "gemma4-12b": "12B",
            "qwen3vl-4b": "Q4B", "qwen3vl-8b": "Q8B"}
    for m, k in kisa.items():
        yol = VERI / "vlm_kosulari" / f"{m}.jsonl"
        if not yol.exists():
            continue
        g = []
        for satir in yol.read_text(encoding="utf-8").splitlines():
            r = json.loads(satir)
            if r.get("durum") == "ok":
                g += [x["guven"] for x in r["tespitler"] if x.get("guven") is not None and x.get("sinif")]
        if g:
            R(f"- {k}: {len(g)} kutu · aralık {yy(min(g), 2)}--{yy(max(g), 2)} · {len(set(g))} ayrı değer · "
              f"0.80 ve üstü {yy(100 * sum(x >= 0.8 for x in g) / len(g), 0)}\\%")


def alarm(R: Rapor, U: dict | None, S: dict | None):
    """Görüntü düzeyinde yanlış alarm ve kaçırma (hakem R2-MC, 2026-10-13): robot hastalık bulduğu bitkinin tamamını
    ilaçlar, karar birimi bitki/görüntü. K'de lezyonlu görüntü payı kesinliğin taban çizgisidir."""
    R.bolum("R-IMG-ALARM — görüntü düzeyi: lezyonsuz görüntüde yanlış alarm, lezyonlu görüntüde kaçırma (K)")
    M = ((U or {}).get("kor") or {}).get("modeller") or {}
    if not M:
        R("— yok."); return
    g0 = next(iter(M.values()))["goruntu_duzeyi"]
    poz = g0["iki_de_var"] + g0["uzman_var_model_yok"]; neg = g0["uzman_yok_model_var"] + g0["iki_de_yok"]
    R(f"K: {poz} lezyonlu, {neg} lezyonsuz görüntü → her görüntüye \"var\" diyen modelin kesinliği {yy(poz / (poz + neg), 2)}.", "")
    kisa = {"gemma4-e2b": "E2B", "gemma4-e2b-8bit": "E2B-8", "gemma4-e4b": "E4B", "gemma4-12b": "12B",
            "qwen3vl-4b": "Q4B", "qwen3vl-8b": "Q8B", "gemma4-e2b-bf16": "E2B bf16"}
    R("Adaylar: " + "; ".join(f"{k} yanlış alarm {M[m]['goruntu_duzeyi']['uzman_yok_model_var']}/{neg}, kaçırma "
                               f"{M[m]['goruntu_duzeyi']['uzman_var_model_yok']}/{poz}" for m, k in kisa.items() if m in M))
    if S:
        ogr = (U.get("kor") or {}).get("ogretmen_secimi", {}).get("ogretmen")
        for kol in KOLLAR:
            v = [x["dogruluk"]["goruntu_duzeyi"] for x in S["kosular"] if x["ogretmen"] == ogr and x["kol"] == kol
                 and "dogruluk" in x and not (x["dogruluk"]["iou_0.5"]["F1"] < 0.02 and x["sadakat"]["iou_0.5"]["F1"] < 0.1)
                 and (x["ad"].startswith("ogrenci_") or x["ad"] == f"slearn_{ogr}_yolo11n_t42")]
            if v:
                fp = [g["tahmin_var"] - g["tp"] for g in v]; fn = [g["referans_var"] - g["tp"] for g in v]
                R(f"- Öğrenci, {kol} (n {len(v)}): yanlış alarm {min(fp)}–{max(fp)}/{neg}, kaçırma {min(fn)}–{max(fn)}/{poz}")


def hitl(R: Rapor, U: dict | None):
    R.bolum("H — Tablo 3 (R-VLM-PRF, R-EDIT, R-UNTOUCHED, R-VLM-IMG, R-EFFORT)")
    H = (U or {}).get("hitl")
    if not H:
        R("— H sonucu yok."); return
    v, dz, n = H["v0"], H["duzeltme"], U["n"]
    gd, o = v["goruntu_duzeyi"], dz["oranlar_v0_kutusu"]
    ozet = U.get("uzman_kutu_ozeti", {}).get("hitl", {})
    onay = sum(x.get("onaylandi", 0) for x in ozet.values())
    sure = U.get("uzman_suresi", {})
    klasor = U.get("kaynak_turu") == "klasor"                     # uzmanlar sunucu dışında etiketledi (2026-10-11)
    med = lambda anahtar: sure.get("kume_sahne", {}).get(anahtar) or {}
    def kume_medyan(k):
        vals = [x for key, x in sure.get("uzman_kume", {}).items() if key.endswith(f"· {k}")]
        return " / ".join(f"{yy(x['medyan_s'], 0)} s" for x in vals) or YOK
    R("Tablo 3 değer sütunu:", "",
      f"- Images completed (K, O, H): {n['kor']}, {n['ortak']}, {n['hitl']}",
      f"- Kept-box rate (IoU 0.5): {f3(v['iou_0.5']['kesinlik'])} {ga_tablo(v['iou_0.5']['GA95']['kesinlik'])}",
      f"- Found-lesion rate (IoU 0.5): {f3(v['iou_0.5']['duyarlilik'])} {ga_tablo(v['iou_0.5']['GA95']['duyarlilik'])}",
      f"- F1 at IoU 0.5, 0.3, 0.75: {f3(v['iou_0.5']['F1'])} {ga_tablo(v['iou_0.5']['GA95']['F1'])}, {f3(v['iou_0.3']['F1'])}, {f3(v['iou_0.75']['F1'])}",
      f"- Boxes unchanged, slightly corrected: {yuzde(o['degismedi'])}, {yuzde(o['hafif_duzeltildi'])} of {dz['v0_kutu']} pre-annotated boxes",
      f"- Boxes moved, deleted: {yuzde(o['buyuk_duzeltildi'] + o.get('tasindi', 0))}, {yuzde(o['silindi'])} "
      f"(moved = matched with IoU < 0.5: still overlapping {yuzde(o['buyuk_duzeltildi'])}, no overlap, nearest free box "
      f"{yuzde(o.get('tasindi'))})",
      f"- Boxes added by the experts: {dz.get('eklendi', 0)} of {dz.get('uzman_kutu')} expert boxes "
      f"(inside a VLM box {dz.get('eklendi_vlm_bolgesinde', 0)}, lesion the VLM did not see {dz.get('eklendi_gorulmemis', 0)})",
      f"- Region level: expert boxes lying inside a VLM box {yuzde(dz['bolge_duzeyi']['uzman_kutusu_bir_vlm_kutusunun_icinde'])}, "
      f"VLM boxes containing an expert box {yuzde(dz['bolge_duzeyi']['uzman_kutusu_iceren_vlm_kutusu'])}",
      f"- Döküm kuralı: {dz.get('kural', YOK)}",
      (f"- Images with unchanged labels: {dz['dokunulmadan_kabul_goruntu']} of {dz['goruntu']} "
       f"({yuzde(dz['dokunulmadan_kabul_goruntu'] / dz['goruntu'] if dz['goruntu'] else None)}; geometriye göre, onay kaydı yok, "
       f"\"approved\" denmez)" if klasor else
       f"- Images approved without any edit: {yuzde(dz['dokunulmadan_kabul_goruntu'] / dz['goruntu'] if dz['goruntu'] else None)} "
       f"(geometriye göre: son hâl ön-etiketle aynı; durum kaydına göre \"approved\": {onay} of {dz['goruntu']})"),
      f"- Image-level agreement: {gd['iki_de_var'] + gd['iki_de_yok']} of {v['goruntu']} images",
      ("- Time per image: ÖLÇÜLEMEDİ (işlem kaydı yok), satır tablodan çıkar" if klasor else
       f"- Time per image, median (K, H): {yy(sure['kume']['kor']['medyan_s'], 0) if 'kor' in sure.get('kume', {}) else YOK} s, "
      f"{yy(sure['kume']['hitl']['medyan_s'], 0) if 'hitl' in sure.get('kume', {}) else YOK} s "
      f"(uzman başına K {kume_medyan('kor')}, H {kume_medyan('hitl')}; O {kume_medyan('ortak')})"), "",
      f"R-EFFORT sahne kırılımı: " + "; ".join(f"{k} {yy(x['medyan_s'], 0)} s (n {x['n_goruntu']})" for k, x in sure.get("kume_sahne", {}).items()),
      f"R-EFFORT tanımı: {sure.get('tanim', YOK)}; süreli {sure.get('goruntu', {}).get('sureli')}, süresiz {sure.get('goruntu', {}).get('suresiz')} görüntü.")
    if klasor:
        R.uyari.append("Kaynak uzman klasörü (sunucu değil): H'nin döküm satırları uzmanın son etiketiyle VLM ön-etiketinin "
                       "GEOMETRİK farkıdır, kaydedilmiş düzenleme değil. \"Time per image\" (R-EFFORT) ve \"approved\" "
                       "ölçülemedi, tablodan çıkar, metinde iddia edilmez.")
    elif dz["goruntu"] and abs(onay - dz["dokunulmadan_kabul_goruntu"]) > 0:
        R.uyari.append(f"H: \"approved\" durumu ({onay}) ile geometriye göre dokunulmamış görüntü ({dz['dokunulmadan_kabul_goruntu']}) farklı; "
                       "tabloda geometri, metinde fark bir yan cümleyle.")


def ogrenciler(R: Rapor, S: dict | None, L: dict | None, U: dict | None):
    R.bolum("Öğrenciler — Tablo 4 (E1-FID, E1-ACC, E1-HITL, E1-EXP, LAT-LOCAL) ve E1-SEED")
    if not S:
        R("— öğrenci değerlendirmesi yok."); return
    ogretmen = ((U or {}).get("kor") or {}).get("ogretmen_secimi", {}).get("ogretmen")
    kos = S["kosular"]
    a2 = {x["ogretmen"] for x in kos if x["ad"].startswith("ogrenci_")}
    if a2 and ogretmen and a2 != {ogretmen}:
        R.uyari.append(f"Aşama 2 öğretmeni {a2}, K'deki seçim {ogretmen}: tutarsız.")
    ogretmen = ogretmen or (next(iter(a2)) if len(a2) == 1 else None)
    if not ogretmen:
        R("— öğretmen belli değil (K sonucu yok ve Aşama 2 koşusu yok)."); return

    def bul(model, kol, tohum=42):
        for x in kos:
            if x["ogretmen"] == ogretmen and x["model"] == model and x["kol"] == kol and x["tohum"] == tohum:
                return x
        return None
    gec = (L or {}).get("ogrenciler", {})
    def gecikme(model):
        for v in gec.values():
            if v.get("model") == model and v.get("kol") == "ogretmen" and v.get("tohum") == 42:
                return yy(v['medyan_ms'], 1)
        return YOK
    R(f"Öğretmen: {AD.get(ogretmen, ogretmen)}. Sütunlar: Params, GFLOPs, Fidelity F1, F1 on K (teacher / +150 / experts), "
      "AP50 on K (teacher kolu, kör uzman etiketine göre), Latency (ms, medyan).", "", "```latex")
    eksik = []
    for model, ad, par, gfl in OGRENCILER:
        t, h, e = bul(model, "ogretmen"), bul(model, "ogretmen_uzman150"), bul(model, "uzman150")
        dg = lambda x: f3((x or {}).get("dogruluk", {}).get("iou_0.5", {}).get("F1")) if x else YOK
        for kol, x in zip(KOLLAR, (t, h, e)):
            if x is None:
                eksik.append(f"{ad}/{kol}")
            elif "dogruluk" not in x:
                eksik.append(f"{ad}/{kol} (uzman etiketiyle değerlendirilmedi)")
        R(f"{ad} & {par} & {gfl} & {f3((t or {}).get('sadakat', {}).get('iou_0.5', {}).get('F1')) if t else YOK} & "
          f"{dg(t)} & {dg(h)} & {dg(e)} & {f3((t or {}).get('AP50_dogruluk')) if t else YOK} & {gecikme(model)} \\\\")
    R("```")
    if eksik:
        R.uyari.append("Tablo 4'te eksik hücre: " + ", ".join(eksik))
    R("", "E1-SEED (n ölçeği, tohum 42 / 43 / 44, F1 on K kör uzman etiketine göre):")
    for model, ad, _, _ in OGRENCILER:
        if not model.endswith("n"):
            continue
        for kol in KOLLAR:
            vals = [bul(model, kol, s) for s in (42, 43, 44)]
            f1 = [v["dogruluk"]["iou_0.5"]["F1"] for v in vals if v and "dogruluk" in v]
            if f1:
                R(f"- {ad} {kol}: " + " / ".join(f3(x) for x in f1) + f" → aralık {yy(min(f1))}--{yy(max(f1))} ({len(f1)} tohum)")
    # Yakınsamayan koşu: doğruluk ve sadakat neredeyse sıfır (erken durdurma öğrenme başlamadan kesti)
    for x in kos:
        if x["ad"].startswith("ogrenci_") and x.get("dogruluk", {}).get("iou_0.5", {}).get("F1", 1) < 0.02 \
                and x["sadakat"]["iou_0.5"]["F1"] < 0.1:
            R.uyari.append(f"{x['ad']}: yakınsamadı ({x.get('epoch')}. dönemde durdu, doğruluk F1 "
                           f"{yy(x['dogruluk']['iou_0.5']['F1'])}). Tabloda dipnotla gösterilir, protokol dışı yeniden eğitim yok.")
    E = S.get("kol_esli_farklar_F1_05") or {}
    if E:
        R("", "E1-HITL / E1-EXP — kollar arası eşleştirilmiş fark (F1 @ 0,5, kör uzman etiketine göre, K'de bootstrap, betimsel):")
        def ozet(anahtar, haric=()):
            v = [(k, x) for k, x in E.items() if anahtar in x and k.split()[0] not in haric]
            d = [x[anahtar]["F1_farki"] for _, x in v]
            art = sum(1 for _, x in v if x[anahtar]["sifiri_disliyor"] and x[anahtar]["F1_farki"] > 0)
            eks = sum(1 for _, x in v if x[anahtar]["sifiri_disliyor"] and x[anahtar]["F1_farki"] < 0)
            return f"{isaretli(min(d))}--{isaretli(max(d))}, {len(v)} karşılaştırma, sıfırı dışlayan: {art} artış, {eks} azalış"
        yakinsamayan = tuple(x["model"] for x in kos if x["ad"].startswith("ogrenci_") and x["kol"] == "uzman150"
                             and x.get("dogruluk", {}).get("iou_0.5", {}).get("F1", 1) < 0.02)
        R(f"- +150 − teacher (E1-HITL): {ozet('ogretmen_uzman150 − ogretmen')}",
          f"- experts − +150 (E1-EXP): {ozet('uzman150 − ogretmen_uzman150', yakinsamayan)}"
          + (f" (yakınsamayan {', '.join(yakinsamayan)} hariç)" if yakinsamayan else ""),
          f"- experts − teacher: {ozet('uzman150 − ogretmen', yakinsamayan)}")
        for k, x in E.items():
            R(f"  - {k}: " + " · ".join(f"{a} {isaretli(y['F1_farki'])} [{yy(y['GA95'][0])}, {yy(y['GA95'][1])}]" for a, y in x.items()))
    # Kol başına aralıklar (hakem Tur 2: AP50 bütün kollar, yerinde bitki F1, görüntü düzeyi, IoU 0.3); seçilen
    # öğretmenin bütün koşuları (tohumlar dahil), yakınsamayan hariç
    R("", "Kol başına aralık (seçilen öğretmen, bütün tohumlar, yakınsamayan hariç; kör uzman etiketine göre, K):")
    for kol in KOLLAR:
        v = [x for x in kos if x["ogretmen"] == ogretmen and x["kol"] == kol and "dogruluk" in x
             and not (x["dogruluk"]["iou_0.5"]["F1"] < 0.02 and x["sadakat"]["iou_0.5"]["F1"] < 0.1)
             and (x["ad"].startswith("ogrenci_") or x["ad"] == f"slearn_{ogretmen}_yolo11n_t42")]
        if not v:
            continue
        rng = lambda f: f"{yy(min(map(f, v)))}--{yy(max(map(f, v)))}"
        R(f"- {kol} (n {len(v)}): F1@0.5 {rng(lambda x: x['dogruluk']['iou_0.5']['F1'])} · F1@0.3 "
          f"{rng(lambda x: x['dogruluk']['iou_0.3']['F1'])} · AP50 {rng(lambda x: x['AP50_dogruluk'])} · in situ F1@0.5 "
          f"{rng(lambda x: x.get('dogruluk_in_situ_F1_05', float('nan')))} · görüntü P {rng(lambda x: x['dogruluk']['goruntu_duzeyi']['kesinlik'])}"
          f" R {rng(lambda x: x['dogruluk']['goruntu_duzeyi']['duyarlilik'])}")
    if not L:
        R.uyari.append("Gecikme yok: `python araclar/gecikme_olc.py --sonuclar <colab/sonuclar/ogrenci>`.")


def yorum_ve_karar(R: Rapor):
    R.bolum("Yorum ve karar kimlikleri")
    R("- R-VLM-INTERP, E1-INTERP: yorum cümleleri yazarındır, yukarıdaki sayılardan kurulur; yeni sayı türetilmez.",
      "- TOOL-AVAIL (depo bağlantısı): proje sahibinin kararı. GUIDE: uzmanların yönergeyi teyidi.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--uzman", type=Path, default=VERI / "uzman_karsilastirma.json")
    ap.add_argument("--ogrenci", type=Path, default=VERI / "ogrenci_degerlendirme.json")
    ap.add_argument("--gecikme", type=Path, default=VERI / "gecikme_lat_local.json")
    ap.add_argument("--cikti", type=Path, default=VERI / "tur4_sayilar.md")
    args = ap.parse_args()
    U, S, L = oku(args.uzman), oku(args.ogrenci), oku(args.gecikme)
    R = Rapor()
    for f in (p1n, vcomp, iaa, etiketsiz_iliski, bolge, hitl):
        f(R, U)
    ogrenciler(R, S, L, U)
    alarm(R, U, S)
    guven(R)
    yorum_ve_karar(R)
    bas = ["# Tur 4 sayı tablosu (otomatik, elle değiştirilmez)", "",
           f"Üretim: {datetime.now().isoformat(timespec='seconds')} · `araclar/tur4_sayilar.py`",
           "Kaynaklar: " + "; ".join(f"`{p.name}` {'(' + (oku(p) or {}).get('olusturma', 'var') + ')' if p.exists() else '(YOK)'}"
                                     for p in (args.uzman, args.ogrenci, args.gecikme)),
           "", "Biçim: üç ondalık, ama Tablo 2 (K) ve K/O cümleleri iki ondalık (tablo genişliği, n = 60); aralık tabloda [a, b], "
           "metinde a--b; yüzde bir ondalık; süre tam saniye.", "",
           "## Uyarılar", ""] + ([f"- {u}" for u in R.uyari] or ["- yok"])
    # LaTeX bloklarında boş hücre "--" (makalenin tablolarındaki gibi; uzun tire üslup kuralına aykırı)
    satir, latex = [], False
    for s in R.satir:
        if s.startswith("```"):
            latex = s == "```latex"
        satir.append(s.replace(YOK, "--") if latex else s)
    args.cikti.write_text("\n".join(bas + satir) + "\n", encoding="utf-8")
    print(f"→ {args.cikti} · uyarı {len(R.uyari)}")
    for u in R.uyari:
        print(f"  ! {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
