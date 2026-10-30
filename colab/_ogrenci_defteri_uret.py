#!/usr/bin/env python3
"""`ogrenci_egitimi.ipynb` defterini tek kaynaktan üretir (ICIDST 2026 makalesi: S-LEARN ve E1).

Defteri elle düzenlemeyin: bu dosyayı değiştirip yeniden üretin.

Tasarım:
  * Aşama 1 (uzman etiketi gerekmez): altı öğretmen adayının her birinin 241 eğitim etiketiyle
    birer YOLO11n (öğrenilebilirlik denetimi, S-LEARN).
  * Aşama 2 (öğretmen seçilip uzman etiketleri gelince): YOLO11/12/26 × n/s/m × üç kol
    (yalnız öğretmen · öğretmen + H'nin 150 uzman etiketi · yalnız 150 uzman etiketi),
    n ölçeğinde tohum 42/43/44, ötekilerde 42. Seçilen öğretmenin YOLO11n / öğretmen / 42
    koşusu Aşama 1'deki koşunun kendisidir (yeniden eğitilmez).
  * Her koşu bitince Drive'a `best.pt`, `results.csv`, `args.yaml`, K ve doğrulama bölümündeki
    tahminler (güven ≥ 0,001, NMS IoU 0,7) ve `tamam.json` yazılır. Bağlantı koparsa
    "Tümünü çalıştır" bitmiş koşuları atlar.
  * Değerlendirme (eşik seçimi, Macar eşleştirmesiyle F1, görüntü düzeyi, AP50) yerelde:
    `konferans_makale_calismasi/araclar/ogrenci_degerlendir.py`.

Kullanım:  python3 _ogrenci_defteri_uret.py   → colab/ogrenci_egitimi.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

BURASI = Path(__file__).resolve().parent


def md(kaynak: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": kaynak.strip().splitlines(keepends=True)}


def kod(kaynak: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": kaynak.strip().splitlines(keepends=True)}


AYARLAR = r'''
ASAMA = 1                 # 1 = ogrenilebilirlik denetimi (6 x YOLO11n), 2 = ogrenciler
OGRETMEN = None           # Asama 2: K'de secilen ogretmen, or. "qwen3vl-8b"

DRIVE_KOK = "/content/drive/MyDrive/Colab Notebooks/1711/colab"
PAKET = "ogrenci"         # paketler/ogrenci.zip ; Asama 2 icin ayrica paketler/uzman.zip
CIHAZ = 0                 # Colab GPU
YEREL_KOK = "/content"    # paketin acildigi ve kosularin yazildigi yerel disk

ADAYLAR = ["gemma4-e2b", "gemma4-e2b-8bit", "gemma4-e4b", "gemma4-12b", "qwen3vl-4b", "qwen3vl-8b"]
AILELER = ["yolo11", "yolo12", "yolo26"]
OLCEKLER = ["n", "s", "m"]
KOLLAR = ["ogretmen", "ogretmen_uzman150", "uzman150"]
N_TOHUMLARI = [42, 43, 44]   # n olceginde uc tohum, obur olceklerde yalniz 42

EPOCHS, IMGSZ, BATCH, PATIENCE = 150, 640, 32, 30
ULTRALYTICS_SURUM = "8.4.144"
'''

KURULUM = r'''
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
%pip install -q ultralytics==8.4.144
import torch, ultralytics
print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
assert ultralytics.__version__ == ULTRALYTICS_SURUM, "Ultralytics surumu makaledekiyle ayni olmali"
'''

DRIVE = r'''
from google.colab import drive
drive.mount('/content/drive')
'''

VERI = r'''
import hashlib, json, shutil, time, zipfile
from pathlib import Path

SUNUCU_KURULUM_MD5 = "3bbbc0cdbe5c3efaa101d732f834cea9"   # K ve H sunucudaki kurulumla ayni (2026-09-30)

def zip_bul(ad):
    yol = Path(DRIVE_KOK) / "paketler" / f"{ad}.zip"
    if yol.exists():
        return yol
    adaylar = sorted(Path("/content/drive/MyDrive").glob(f"*/**/{ad}.zip"))[:3]
    return adaylar[0] if adaylar else None

paket_zip = zip_bul(PAKET)
assert paket_zip, f"'{PAKET}.zip' Drive'da bulunamadi. Beklenen: {Path(DRIVE_KOK) / 'paketler'}"
PAKET_DIZIN = Path(YEREL_KOK) / PAKET
with zipfile.ZipFile(paket_zip) as z:
    bolunme_ham = z.read(f"{PAKET}/bolunme.json")
    B = json.loads(bolunme_ham)
    assert B.get("kume_kaynagi", {}).get("sunucu_kurulum_md5") == SUNUCU_KURULUM_MD5, (
        f"Drive'daki {paket_zip.name} ESKI surum. paketler/ klasorundeki eskisini silip yenisini yukleyin.")
    yerel = PAKET_DIZIN / "bolunme.json"
    if not yerel.exists() or yerel.read_bytes() != bolunme_ham:
        if PAKET_DIZIN.exists():
            shutil.rmtree(PAKET_DIZIN)    # bu oturumda eski paket acilmis: yenisiyle degistir
        t0 = time.time()
        z.extractall(YEREL_KOK)           # zip icinde tek kok klasor: ogrenci/
        print(f"acildi: {PAKET_DIZIN} ({time.time() - t0:.0f} sn)")
BOLUNME_SHA = hashlib.sha256(bolunme_ham).hexdigest()
print("paket:", paket_zip, "| bolunme sha256:", BOLUNME_SHA[:12])
print("bolunme:", {k: v[0] for k, v in B["sayilar"].items()})
print("goruntu:", len(list((PAKET_DIZIN / "goruntuler").glob("*.jpg"))))
for a in ADAYLAR:
    assert len(list((PAKET_DIZIN / "etiketler" / a).glob("*.txt"))) == 301, f"{a}: etiket eksik"

if ASAMA == 2:
    assert OGRETMEN in ADAYLAR, "Asama 2 icin OGRETMEN ayarlanmali (K'de secilen aday)"
    uzman_zip = zip_bul("uzman")
    assert uzman_zip, "Asama 2 icin paketler/uzman.zip gerekli (uzman etiketleri)"
    if (PAKET_DIZIN / "etiketler" / "uzman").exists():
        shutil.rmtree(PAKET_DIZIN / "etiketler" / "uzman")   # her calistirmada Drive'daki uzman.zip'ten tazele
    with zipfile.ZipFile(uzman_zip) as z:
        z.extractall(YEREL_KOK)           # ogrenci/etiketler/uzman/<ad>.txt
    uzman_adlari = {p.stem for p in (PAKET_DIZIN / "etiketler" / "uzman").glob("*.txt")}
    beklenen = {Path(g).stem for g in B["hitl"]}
    print("uzman etiketi:", len(uzman_adlari))
    assert uzman_adlari == beklenen, (
        f"uzman.zip H kumesiyle ayni degil (eksik {len(beklenen - uzman_adlari)}, fazla {len(uzman_adlari - beklenen)}): "
        "paketle_uzman.py ile yeniden uretin")
'''

YARDIMCI = r'''
import os
import ultralytics
from ultralytics import YOLO

KAYIT = Path(DRIVE_KOK) / "sonuclar" / "ogrenci"
VERILER = Path(YEREL_KOK) / "veriler"
KOSULAR = Path(YEREL_KOK) / "runs"

def veri_kur(ad, egitim, dogrulama, test, etiket_sec):
    """Kosuya ozel YOLO veri klasoru. Goruntuler sembolik baglanti, etiketler kopya.
    etiket_sec(goruntu, bolum) -> etiket kaynagi (aday adi ya da 'uzman')."""
    kok = VERILER / ad
    if kok.exists():
        shutil.rmtree(kok)
    for bolum, adlar in (("train", egitim), ("val", dogrulama), ("test", test)):
        (kok / "images" / bolum).mkdir(parents=True)
        (kok / "labels" / bolum).mkdir(parents=True)
        for g in adlar:
            kok_ad = Path(g).stem
            os.symlink(PAKET_DIZIN / "goruntuler" / f"{kok_ad}.jpg", kok / "images" / bolum / f"{kok_ad}.jpg")
            shutil.copy(PAKET_DIZIN / "etiketler" / etiket_sec(g, bolum) / f"{kok_ad}.txt",
                        kok / "labels" / bolum / f"{kok_ad}.txt")
    yaml = kok / "data.yaml"
    yaml.write_text(f"path: {kok}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Late_Blight\n")
    return yaml

def kosu_tamam(ad):
    """Drive'da bitmis kosu var ve bugunku bolunmeyle uretilmisse True."""
    f = KAYIT / ad / "tamam.json"
    if not f.exists():
        return False
    if json.loads(f.read_text()).get("bolunme_sha256") != BOLUNME_SHA:
        print(f"[yeniden] {ad}: Drive'daki sonuc eski bolunmeyle uretilmis, yeniden egitilecek")
        return False
    return True

def egit_ve_kaydet(ad, agirlik, yaml, tohum, tahmin_adlari, bilgi):
    if kosu_tamam(ad):
        print(f"[atlandi] {ad} zaten Drive'da")
        return
    print(f"\n{'=' * 70}\n{ad}\n{'=' * 70}")
    t0 = time.time()
    YOLO(agirlik).train(
        data=str(yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=CIHAZ,
        seed=tohum, deterministic=True, patience=PATIENCE,
        fliplr=0.5, flipud=0.0,           # yalniz yatay aynalama (makale 3.6)
        cache=True, plots=False, verbose=False,
        project=str(KOSULAR), name=ad, exist_ok=True)
    dakika = (time.time() - t0) / 60
    kosu = KOSULAR / ad
    en_iyi = YOLO(kosu / "weights" / "best.pt")
    # Bilgi amacli: K'de, test etiketlerine (ogretmenin K etiketi) gore mAP50
    olcum = en_iyi.val(data=str(yaml), split="test", imgsz=IMGSZ, device=CIHAZ, conf=0.001, iou=0.7,
                       plots=False, verbose=False, project=str(KOSULAR), name=f"{ad}_test", exist_ok=True)
    tahmin = {}
    for adlar in tahmin_adlari.values():
        yollar = [str(PAKET_DIZIN / "goruntuler" / f"{Path(g).stem}.jpg") for g in adlar]
        for r in en_iyi.predict(source=yollar, imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300,
                                device=CIHAZ, stream=True, verbose=False):
            tahmin[Path(r.path).stem + ".jpg"] = [
                [round(float(v), 6) for v in b] + [round(float(c), 5)]
                for b, c in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist())]
    hedef = KAYIT / ad
    hedef.mkdir(parents=True, exist_ok=True)
    for f in ("weights/best.pt", "results.csv", "args.yaml"):
        if (kosu / f).exists():
            shutil.copy2(kosu / f, hedef / Path(f).name)
    (hedef / "tahminler.json").write_text(json.dumps(tahmin))
    epoch = sum(1 for _ in open(kosu / "results.csv")) - 1 if (kosu / "results.csv").exists() else None
    ozet = {"ad": ad, "agirlik": agirlik, "tohum": tohum, "epoch": epoch, "dakika": round(dakika, 1),
            "test_mAP50_ogretmen_K": round(float(olcum.box.map50), 4),
            "tahmin_goruntu": len(tahmin), "ultralytics": ultralytics.__version__, "bolunme_sha256": BOLUNME_SHA,
            "zaman": time.strftime("%Y-%m-%dT%H:%M:%S")} | bilgi
    (hedef / "tamam.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1))
    print(f"{ad}: {epoch} epoch, {dakika:.1f} dk, K'de ogretmen etiketine gore mAP50 = {ozet['test_mAP50_ogretmen_K']:.3f}")
'''

ASAMA1 = r'''
if ASAMA == 1:
    for aday in ADAYLAR:
        ad = f"slearn_{aday}_yolo11n_t42"
        yaml = veri_kur(ad, B["egitim"], B["dogrulama"], B["kor"], lambda g, bolum, a=aday: a)
        egit_ve_kaydet(ad, "yolo11n.pt", yaml, 42, {"dogrulama": B["dogrulama"], "kor": B["kor"]},
                       {"asama": 1, "ogretmen": aday, "kol": "ogretmen", "model": "yolo11n"})
    print("\nAsama 1 bitti.")
'''

ASAMA2 = r'''
def kosu_adi(aile, olcek, kol, tohum):
    # Secilen ogretmenin YOLO11n / ogretmen / 42 kosusu Asama 1'dekidir: yeniden egitilmez.
    if (aile, olcek, kol, tohum) == ("yolo11", "n", "ogretmen", 42):
        return f"slearn_{OGRETMEN}_yolo11n_t42"
    return f"ogrenci_{OGRETMEN}_{aile}{olcek}_{kol}_t{tohum}"

if ASAMA == 2:
    H = set(B["hitl"])
    for olcek in OLCEKLER:                       # once kucuk modeller: kopma olursa cok is kaybolmaz
        for aile in AILELER:
            for kol in KOLLAR:
                for tohum in (N_TOHUMLARI if olcek == "n" else [42]):
                    ad = kosu_adi(aile, olcek, kol, tohum)
                    if kol == "ogretmen":
                        egitim, dogrulama = B["egitim"], B["dogrulama"]
                        sec = lambda g, bolum: OGRETMEN
                    elif kol == "ogretmen_uzman150":
                        egitim, dogrulama = B["egitim"], B["dogrulama"]
                        sec = lambda g, bolum: "uzman" if (g in H and bolum != "test") else OGRETMEN
                    else:                        # uzman150: yalniz H'nin uzman etiketleri
                        egitim, dogrulama = B["uzman150_egitim"], B["uzman150_dogrulama"]
                        sec = lambda g, bolum: "uzman" if bolum != "test" else OGRETMEN
                    if kosu_tamam(ad):
                        print(f"[atlandi] {ad} zaten Drive'da")
                        continue
                    yaml = veri_kur(ad, egitim, dogrulama, B["kor"], sec)
                    egit_ve_kaydet(ad, f"{aile}{olcek}.pt", yaml, tohum, {"dogrulama": dogrulama, "kor": B["kor"]},
                                   {"asama": 2, "ogretmen": OGRETMEN, "kol": kol, "model": f"{aile}{olcek}"})
    print("\nAsama 2 bitti.")
'''

OZET = r'''
import pandas as pd
satirlar = [json.loads(p.read_text()) for p in sorted(KAYIT.glob("*/tamam.json"))]
if satirlar:
    tablo = pd.DataFrame(satirlar)[["ad", "asama", "ogretmen", "kol", "model", "tohum", "epoch", "dakika", "test_mAP50_ogretmen_K"]]
    display(tablo)
    print(f"{len(satirlar)} kosu Drive'da: {KAYIT}")
else:
    print("Henuz tamamlanmis kosu yok.")
'''


def defter() -> dict:
    hucreler = [
        md("""
# 🍃 Mildiyö öğrencileri — öğretmen etiketleriyle YOLO eğitimi

ICIDST 2026 makalesinin öğrenci deneyleri. **İki aşama**:

| Aşama | Ne | Ne zaman | Süre (A100, tahmin) |
|---|---|---|---|
| **1** | Altı öğretmen adayının her birinin etiketiyle birer **YOLO11n** (öğrenilebilirlik denetimi) | **Şimdi** (uzman etiketi gerekmez) | ~30–45 dk |
| **2** | Seçilen öğretmenle **YOLO11/12/26 × n/s/m × 3 kol**, n ölçeğinde 3 tohum | Öğretmen seçilip `uzman.zip` gelince | ~4–6 sa |

**Kullanım:**

1. `ogrenci.zip` dosyasını Drive'da `MyDrive/Colab Notebooks/1711/colab/paketler/` altına koyun
   (Aşama 2 için ayrıca `uzman.zip`). Başka yerdeyse defter Drive'da adıyla arar.
2. **Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU (A100)**.
3. Aşağıdaki **Ayarlar** hücresinde `ASAMA` değerine bakın (Aşama 2'de `OGRETMEN`'i de yazın), sonra
   **Çalışma zamanı → Tümünü çalıştır**.

**Bağlantı koparsa:** yeniden "Tümünü çalıştır" deyin. Bitmiş koşular Drive'da olduğu için atlanır.

**Paket güncellenirse:** Drive'daki eski `ogrenci.zip`'i silip yenisini yükleyin. Defter değişikliği
kendisi algılar: paketi yeniden açar, eski bölünmeyle üretilmiş sonuçları yeniden eğitir. Eski paket
yüklü kalırsa ilk hücrelerde açık bir mesajla durur.

**Sonuçlar:** `MyDrive/Colab Notebooks/1711/colab/sonuclar/ogrenci/<koşu>/`: `best.pt`, `results.csv`,
`args.yaml`, `tahminler.json`, `tamam.json`. Makaledeki sayılar bu tahminlerden yerelde
hesaplanır (adaylarla aynı eşleştirme), defter yalnız eğitir ve tahmin döker.
"""),
        md("## 1. Ayarlar — değiştireceğiniz tek hücre"),
        kod(AYARLAR),
        md("## 2. GPU ve kurulum"),
        kod(KURULUM),
        md("## 3. Drive ve paket"),
        kod(DRIVE),
        kod(VERI),
        md("## 4. Yardımcılar (değiştirmeyin)"),
        kod(YARDIMCI),
        md("## 5. Aşama 1 — altı aday × YOLO11n\n\n`ASAMA = 1` iken çalışır. Her aday kendi 241 eğitim etiketiyle, aynı bölünme ve aynı tohumla eğitilir."),
        kod(ASAMA1),
        md("## 6. Aşama 2 — öğrenciler\n\n`ASAMA = 2` ve `OGRETMEN` ayarlıyken çalışır. Küçük modeller önce koşar."),
        kod(ASAMA2),
        md("## 7. Drive'daki koşuların özeti"),
        kod(OZET),
    ]
    return {"cells": hucreler,
            "metadata": {"accelerator": "GPU",
                         "colab": {"provenance": [], "gpuType": "A100", "toc_visible": True},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def yerel_betik() -> str:
    """Sınama için: Colab'a özgü hücreler olmadan, ayarlar dışarıdan verilecek düz Python."""
    return "\n\n".join([VERI, YARDIMCI, ASAMA1, ASAMA2, OZET.replace("display(tablo)", "print(tablo.to_string())")])


def main() -> None:
    yol = BURASI / "ogrenci_egitimi.ipynb"
    yol.write_text(json.dumps(defter(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {yol}")


if __name__ == "__main__":
    main()
