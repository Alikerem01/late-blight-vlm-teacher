"""Kıdemli uzman incelemesi (R-SENIOR) — K'nin kıdemli uzmanca incelenmiş hâline göre duyarlılık çözümlemesi.

Kıdemli uzman (sunucu dışa aktarımında anonim kimliği `--kidemli`, varsayılan uzman_2) 13 Ekim'de K'yi
ve O'yu, iki uzmanın kutuları yüklüyken araçta inceledi (VLM çıktısını görmedi). H'ye bakmadı. Ana
referans iki uzmandır. Bu betik yalnız şunu sorar: K'nin referansı kıdemli uzmanın son hâli olsaydı
öğretmen seçimi ve öğrenci kollarının sıralaması değişir miydi?

Girdi:
  --paket   sunucunun dışa aktarım paketi (K ve O kıdemli uzmanın son hâli, inceleme.csv)
  --uzman-klasor  iki uzmanın etiketleri (uzman_veriler_2: K, O, H)
H her iki çözümlemede iki uzmanındır (öğrenciler onunla eğitildi).

Çıktı (veri/kidemli/):
  ozet.json                  inceleme istatistikleri, öz tutarlılık, öğretmen seçimi, kol aralıkları
  uzman_karsilastirma.json   uzman_vlm_karsilastir.py çıktısı (K = kıdemli uzman)
  ogrenci_degerlendirme.json ogrenci_degerlendir.py çıktısı (K = kıdemli uzman)

Kullanım:
  python araclar/kidemli_inceleme.py [--paket paketler/degerlendirme_20261015T095230.tar.gz]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uzman_vlm_karsilastir import etiket_oku, goruntu_sayimlari, prf  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
ARACLAR = KOK / "araclar"
VERI = KOK / "veri"
PAKET = KOK / "paketler" / "degerlendirme_20261015T095230.tar.gz"
UZMAN = KOK / "uzman_veriler_2"
CIKTI = VERI / "kidemli"
SONUCLAR = KOK.parent / "colab" / "sonuclar" / "ogrenci"
KUMELER = {"kor": ("1_kor_etiketleme_images_labels", "1_kor_etiketleme"),
           "ortak": ("1b_ortak_okuma_images_labels", "1b_ortak_okuma")}
KOLLAR = ("ogretmen", "ogretmen_uzman150", "uzman150")
YAKINSAMAYAN = {("yolo12m", 42, "uzman150")}       # Tur 4B dipnotu: yalnız uzman kolu yakınsamadı
TOL = 1e-4


def kutular(p: Path) -> list[tuple]:
    return [b for s, b in etiket_oku(p) if s == 0] if p.exists() else []


def ayni(a: list, b: list) -> bool:
    """İki kutu listesi biçim farkı dışında aynı mı (sunucu kaydederken %.6f ile yeniden yazar)."""
    if len(a) != len(b):
        return False
    return all(max(abs(x - y) for x, y in zip(p, q)) < TOL for p, q in zip(sorted(a), sorted(b)))


def f1_05(ref: list[list], pred: list[list]) -> float:
    return prf([goruntu_sayimlari(r, p) for r, p in zip(ref, pred)], 0.5)["F1"]


def paketi_ac(paket: Path, hedef: Path) -> Path:
    with tarfile.open(paket) as t:
        t.extractall(hedef, filter="data")
    alt = [d for d in hedef.iterdir() if d.is_dir()]
    if len(alt) != 1:
        raise SystemExit(f"HATA: pakette tek bir üst klasör bekleniyordu: {[d.name for d in alt]}")
    return alt[0]


def inceleme_kaydi(paket_kok: Path, kidemli: str) -> dict[str, set[str]]:
    kayit: dict[str, set[str]] = {}
    with open(paket_kok / "inceleme.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["uzman"] == kidemli:
                kayit.setdefault(r["dataset"], set()).add(r["image_name"])
    return kayit


def k_incelemesi(paket_kok: Path, kayit: dict[str, set[str]]) -> dict:
    """K'de iki uzmanın etiketi ↔ kıdemli uzmanın son hâli."""
    lbl_p = paket_kok / KUMELER["kor"][0]
    lbl_u = UZMAN / KUMELER["kor"][1] / "labels"
    adlar = sorted(p.stem for p in lbl_u.glob("k*.txt"))
    kayitli = {Path(a).stem for a in kayit.get("1_kor_etiketleme_images", set())}
    satir = []
    for a in adlar:
        u, s = kutular(lbl_u / f"{a}.txt"), kutular(lbl_p / f"{a}.txt")
        satir.append({"ad": a, "uzman": len(u), "kidemli": len(s), "degisti": not ayni(u, s), "kayitli": a in kayitli})
    U = [kutular(lbl_u / f"{a}.txt") for a in adlar]
    S = [kutular(lbl_p / f"{a}.txt") for a in adlar]
    deg = [i for i, r in enumerate(satir) if r["degisti"]]
    kayitsiz = [r for r in satir if not r["kayitli"]]
    return {
        "goruntu": len(adlar),
        "kidemli_kaydi_olan": sum(r["kayitli"] for r in satir),
        "degisen_goruntu": len(deg),
        "kutu_uzman": sum(r["uzman"] for r in satir), "kutu_kidemli": sum(r["kidemli"] for r in satir),
        "lezyonlu_uzman": sum(r["uzman"] > 0 for r in satir), "lezyonlu_kidemli": sum(r["kidemli"] > 0 for r in satir),
        "uzman_lezyonlu_kidemli_lezyonsuz": sum(r["uzman"] > 0 and r["kidemli"] == 0 for r in satir),
        "uzman_lezyonsuz_kidemli_lezyonlu": sum(r["uzman"] == 0 and r["kidemli"] > 0 for r in satir),
        "F1_05_uzman_kidemli_tum": f1_05(U, S),
        "F1_05_uzman_kidemli_degisen": f1_05([U[i] for i in deg], [S[i] for i in deg]),
        "kaydi_olmayan": {"goruntu": len(kayitsiz), "uzman_kutulu": sum(r["uzman"] > 0 for r in kayitsiz),
                          "degisen": sum(r["degisti"] for r in kayitsiz)},
        "_goruntu": satir,
    }


def oz_tutarlilik(paket_kok: Path) -> dict:
    """O'nun 20 görüntüsü K'de de var: kıdemli uzman aynı görüntüye iki kez baktı
    (K'de bir uzmanın, O'da öbür uzmanın kutularından başlayarak)."""
    with open(next(paket_kok.glob("kurulum_ortak_*.csv")), encoding="utf-8") as fh:
        es = list(csv.DictReader(fh))
    sk, so, ua, ub = [], [], [], []
    for r in es:
        k, o = Path(r["k_ad"]).stem, Path(r["yeni_ad"]).stem
        sk.append(kutular(paket_kok / KUMELER["kor"][0] / f"{k}.txt"))
        so.append(kutular(paket_kok / KUMELER["ortak"][0] / f"{o}.txt"))
        ua.append(kutular(UZMAN / KUMELER["kor"][1] / "labels" / f"{k}.txt"))
        ub.append(kutular(UZMAN / KUMELER["ortak"][1] / "labels" / f"{o}.txt"))
    var = lambda L: [len(x) > 0 for x in L]  # noqa: E731
    return {
        "goruntu": len(es),
        "kidemli_K_O_F1_05": f1_05(sk, so),
        "kidemli_K_O_lezyon_karari_ayni": sum(a == b for a, b in zip(var(sk), var(so))),
        "iki_uzman_F1_05": f1_05(ua, ub),
        "iki_uzman_lezyon_karari_ayni": sum(a == b for a, b in zip(var(ua), var(ub))),
        "uzmanA_kidemliK_F1_05": f1_05(ua, sk), "uzmanB_kidemliO_F1_05": f1_05(ub, so),
    }


def referans_klasoru(paket_kok: Path, hedef: Path) -> None:
    """uzman_veriler_2 düzeninde: K ve O kıdemli uzmanın son hâli, H iki uzmanın."""
    for kume, (paket_ad, klasor) in KUMELER.items():
        d = hedef / klasor / "labels"
        d.mkdir(parents=True)
        for p in sorted((paket_kok / paket_ad).glob("*.txt")):
            shutil.copy2(p, d / p.name)
        (d / "classes.txt").write_text("Late_Blight\n", encoding="utf-8")   # sunucu: Late_Blight = 0
    shutil.copytree(UZMAN / "2_duzeltme_onay" / "labels", hedef / "2_duzeltme_onay" / "labels")


def calistir(betik: str, *arg: str) -> None:
    r = subprocess.run([sys.executable, str(ARACLAR / betik), *arg], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
        raise SystemExit(f"HATA: {betik} çıkış {r.returncode}")


def aralik(degerler: list[float]) -> list[float] | None:
    return [min(degerler), max(degerler)] if degerler else None


def kol_ozeti(ogr_json: Path) -> dict:
    T = {}
    for r in json.loads(ogr_json.read_text(encoding="utf-8"))["kosular"]:
        # YOLO11n / öğretmen / tohum 42 kolu Aşama 1'in koşusudur (slearn_…, yeniden eğitilmedi), o da sayılır
        # (2026-10-30 düzeltmesi: önceki süzgeç onu dışarıda bırakıyordu, eşli sayımlar 14 ve 13 çıkıyordu).
        if r.get("ogretmen") != "qwen3vl-8b" or "dogruluk" not in r or not (
                r["ad"].startswith("ogrenci_") or r["ad"] == "slearn_qwen3vl-8b_yolo11n_t42"):
            continue
        T[(r["model"], r["tohum"], r["kol"])] = r
    out = {}
    for kol in KOLLAR:
        R = [r for k, r in T.items() if k[2] == kol and k not in YAKINSAMAYAN]
        out[kol] = {
            "n_kosu": len(R),
            "F1_05": aralik([r["dogruluk"]["iou_0.5"]["F1"] for r in R]),
            "AP50": aralik([r["AP50_dogruluk"] for r in R if r.get("AP50_dogruluk") is not None]),
            "goruntu_kesinlik": aralik([r["dogruluk"]["goruntu_duzeyi"]["kesinlik"] for r in R]),
            "goruntu_duyarlilik": aralik([r["dogruluk"]["goruntu_duzeyi"]["duyarlilik"] for r in R]),
        }
    esli = {"uzman150_arti_ogretmen_gt_ogretmen": [0, 0], "uzman150_gt_ogretmen": [0, 0],
            "uzman150_arti_ogretmen_gt_uzman150": [0, 0]}
    for (m, t, k), r in T.items():
        if k != "ogretmen":
            continue
        f = r["dogruluk"]["iou_0.5"]["F1"]
        a, u = T.get((m, t, "ogretmen_uzman150")), T.get((m, t, "uzman150"))
        u = None if (m, t, "uzman150") in YAKINSAMAYAN else u
        if a:
            esli["uzman150_arti_ogretmen_gt_ogretmen"][0] += a["dogruluk"]["iou_0.5"]["F1"] > f
            esli["uzman150_arti_ogretmen_gt_ogretmen"][1] += 1
        if u:
            esli["uzman150_gt_ogretmen"][0] += u["dogruluk"]["iou_0.5"]["F1"] > f
            esli["uzman150_gt_ogretmen"][1] += 1
        if a and u:
            esli["uzman150_arti_ogretmen_gt_uzman150"][0] += a["dogruluk"]["iou_0.5"]["F1"] > u["dogruluk"]["iou_0.5"]["F1"]
            esli["uzman150_arti_ogretmen_gt_uzman150"][1] += 1
    return {"kollar": out, "esli_sayim_F1_05": esli, "not": "Qwen3-VL-8B öğretmen; yakınsamayan YOLO12m yalnız uzman koşusu hariç"}


def ogretmen_ozeti(uk_json: Path) -> dict:
    kor = json.loads(uk_json.read_text(encoding="utf-8"))["kor"]
    secim = kor["ogretmen_secimi"]
    return {
        "ogretmen": secim["ogretmen"], "siralama_F1_05": secim["siralama_F1_05"],
        "ayrismayanlar": secim["ayrismayanlar"],
        "F1_05": {m: kor["modeller"][m]["iou_0.5"]["F1"] for m in kor["adaylar"]},
        "GA95": {m: kor["modeller"][m]["iou_0.5"]["GA95"]["F1"] for m in kor["adaylar"]},
        "lezyonsuz_goruntude_alarm": {m: kor["modeller"][m]["goruntu_duzeyi"]["uzman_yok_model_var"] for m in kor["adaylar"]},
        "lezyonsuz_goruntu": kor["modeller"][kor["adaylar"][0]]["goruntu_duzeyi"]["uzmanca_lezyonsuz"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paket", type=Path, default=PAKET)
    ap.add_argument("--kidemli", default="uzman_2", help="kıdemli uzmanın paketteki anonim kimliği")
    ap.add_argument("--sonuclar", type=Path, default=SONUCLAR)
    ap.add_argument("--cikti", type=Path, default=CIKTI)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as gecici:
        g = Path(gecici)
        pk = paketi_ac(args.paket.expanduser(), g / "paket")
        kayit = inceleme_kaydi(pk, args.kidemli)
        if not kayit.get("1_kor_etiketleme_images"):
            raise SystemExit(f"HATA: {args.kidemli} için K'de inceleme kaydı yok")
        ozet = {"paket": args.paket.name, "kidemli": args.kidemli,
                "kayit_sayisi": {k: len(v) for k, v in sorted(kayit.items())},
                "K": k_incelemesi(pk, kayit), "oz_tutarlilik_O": oz_tutarlilik(pk)}
        referans_klasoru(pk, g / "ref")
        args.cikti.mkdir(parents=True, exist_ok=True)
        calistir("uzman_vlm_karsilastir.py", "--klasor", str(g / "ref"), "--cikti", str(g / "uk"))
        shutil.copy2(g / "uk" / "uzman_karsilastirma.json", args.cikti / "uzman_karsilastirma.json")
        calistir("ogrenci_degerlendir.py", "--sonuclar", str(args.sonuclar), "--uzman-klasor", str(g / "ref"),
                 "--cikti", str(args.cikti / "ogrenci_degerlendirme.json"))
    ozet["ogretmen_secimi"] = ogretmen_ozeti(args.cikti / "uzman_karsilastirma.json")
    ozet["ogrenciler"] = kol_ozeti(args.cikti / "ogrenci_degerlendirme.json")
    ozet["ana_referans_karsilastirma"] = {
        "ogretmen_secimi": ogretmen_ozeti(VERI / "uzman_karsilastirma.json"),
        "ogrenciler": kol_ozeti(VERI / "ogrenci_degerlendirme.json"),
    }
    (args.cikti / "ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")

    K, O = ozet["K"], ozet["oz_tutarlilik_O"]
    o, a = ozet["ogretmen_secimi"], ozet["ana_referans_karsilastirma"]["ogretmen_secimi"]
    print(f"K: {K['goruntu']} görüntü · kıdemli uzman kaydı {K['kidemli_kaydi_olan']} · değişen {K['degisen_goruntu']} · "
          f"kutu {K['kutu_uzman']} → {K['kutu_kidemli']} · lezyonlu görüntü {K['lezyonlu_uzman']} → {K['lezyonlu_kidemli']} "
          f"(uzman lezyonlu → kıdemli lezyonsuz {K['uzman_lezyonlu_kidemli_lezyonsuz']}, tersi {K['uzman_lezyonsuz_kidemli_lezyonlu']})")
    print(f"   iki uzman ↔ kıdemli F1@0.5: tüm {K['F1_05_uzman_kidemli_tum']:.3f} · değişen görüntülerde "
          f"{K['F1_05_uzman_kidemli_degisen']:.3f} · kaydı olmayan {K['kaydi_olmayan']}")
    print(f"O öz tutarlılık ({O['goruntu']} görüntü): kıdemli K↔O F1 {O['kidemli_K_O_F1_05']:.3f}, lezyon kararı "
          f"{O['kidemli_K_O_lezyon_karari_ayni']}/{O['goruntu']} · iki uzman F1 {O['iki_uzman_F1_05']:.3f}, "
          f"{O['iki_uzman_lezyon_karari_ayni']}/{O['goruntu']}")
    print(f"Öğretmen: {o['ogretmen']} F1 {o['F1_05'][o['ogretmen']]:.3f} (ana referans {a['ogretmen']} "
          f"{a['F1_05'][a['ogretmen']]:.3f}) · ayrışmayanlar {o['ayrismayanlar']} · lezyonsuz {o['lezyonsuz_goruntu']} "
          f"görüntüde alarm {sorted(o['lezyonsuz_goruntude_alarm'].values())}")
    for ad, s in (("kıdemli", ozet["ogrenciler"]), ("iki uzman", ozet["ana_referans_karsilastirma"]["ogrenciler"])):
        k = s["kollar"]
        f = lambda x: f"{x[0]:.3f}–{x[1]:.3f}"  # noqa: E731
        print(f"Öğrenciler ({ad}): öğretmen {f(k['ogretmen']['F1_05'])} · +150 {f(k['ogretmen_uzman150']['F1_05'])} · "
              f"uzman {f(k['uzman150']['F1_05'])} · görüntü P (+150) {f(k['ogretmen_uzman150']['goruntu_kesinlik'])} · "
              f"eşli {s['esli_sayim_F1_05']}")
    print(f"→ {args.cikti}/ozet.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
