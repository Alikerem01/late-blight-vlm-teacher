"""Öğrencilerin dizüstü işlemcisindeki gecikmesi (LAT-LOCAL, brif §7) — SALT OKUMA.

Her öğrenci (Colab'ın yazdığı `best.pt`) için, aday koşularının yapıldığı dizüstünün (Apple M5) işlemcisinde:
Ultralytics 8.4.144, PyTorch CPU, batch 1, 640 × 640 girdi, 20 ısınma + 200 ölçüm. Ölçülen, her çağrıda
ön işleme + çıkarım + NMS süresinin toplamı (Ultralytics'in kendi `speed` alanları, ms). Medyan ve p95.

- Girdi: K'nin 60 görüntüsü (öğrenci paketindeki 1280 px JPEG'ler) kare dolguyla 640 × 640'a getirilir ve
  bellekte tutulur. Böylece letterbox ek dolgu yapmaz, girdi her çağrıda tam 640 × 640'tır ve disk okuma
  ölçüme girmez. Görüntüler sırayla döner.
- Güven eşiği öğrencinin doğrulamada seçilen eşiği (`veri/ogrenci_degerlendirme.json`), yoksa 0,25. NMS IoU 0,7.
- Gecikme mimariye bağlıdır, etiketlere değil: tabloda her mimarinin öğretmen kolu, tohum 42 koşusu ölçülür.
- Göreli bir maliyet ölçüsüdür, robot donanımında ölçüm değildir. Ölçüm sırasında başka ağır iş çalışmamalı.

Kullanım:
  python gecikme_olc.py --sonuclar ~/Desktop/1711_yze/colab/sonuclar/ogrenci            # tablo: 9 öğrenci
  python gecikme_olc.py --sonuclar ... --hepsi                                          # bütün koşular
Çıktı: veri/gecikme_lat_local.json
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
PAKET = KOK.parent / "colab" / "paketler" / "ogrenci.zip"
DEGERLENDIRME = KOK / "veri" / "ogrenci_degerlendirme.json"
CIKTI = KOK / "veri" / "gecikme_lat_local.json"
ISINMA, OLCUM, BOYUT = 20, 200, 640


def kare_goruntuler(paket: Path) -> list[np.ndarray]:
    """K görüntülerini kare dolguyla 640 × 640 BGR dizilerine çevirir (Ultralytics'in numpy girdi biçimi)."""
    from PIL import Image
    z = zipfile.ZipFile(paket)
    B = json.loads(z.read("ogrenci/bolunme.json"))
    out = []
    for ad in B["kor"]:
        with Image.open(io.BytesIO(z.read(f"ogrenci/goruntuler/{Path(ad).stem}.jpg"))) as im:
            im = im.convert("RGB")
            olcek = BOYUT / max(im.size)
            im = im.resize((round(im.width * olcek), round(im.height * olcek)), Image.BILINEAR)
            kare = Image.new("RGB", (BOYUT, BOYUT), (114, 114, 114))      # Ultralytics'in dolgu rengi
            kare.paste(im, ((BOYUT - im.width) // 2, (BOYUT - im.height) // 2))
            out.append(np.ascontiguousarray(np.asarray(kare)[:, :, ::-1]))
    return out


def esikler() -> dict[str, float]:
    if not DEGERLENDIRME.exists():
        return {}
    return {k["ad"]: k["esik"] for k in json.loads(DEGERLENDIRME.read_text(encoding="utf-8"))["kosular"]}


def olc(agirlik: Path, girdiler: list[np.ndarray], esik: float) -> dict:
    import torch
    from ultralytics import YOLO
    model = YOLO(str(agirlik))
    toplam, bilesen, duvar = [], {"preprocess": [], "inference": [], "postprocess": []}, []
    for i in range(ISINMA + OLCUM):
        t0 = time.perf_counter()
        r = model.predict(girdiler[i % len(girdiler)], imgsz=BOYUT, conf=esik, iou=0.7, max_det=300,
                          device="cpu", verbose=False)[0]
        t1 = time.perf_counter()
        if i < ISINMA:
            continue
        assert tuple(r.orig_shape) == (BOYUT, BOYUT)
        for k in bilesen:
            bilesen[k].append(float(r.speed[k]))
        toplam.append(sum(float(r.speed[k]) for k in bilesen))
        duvar.append((t1 - t0) * 1000)
    yuzde = lambda v, q: round(float(np.percentile(v, q)), 1)
    return {"esik": esik, "medyan_ms": round(statistics.median(toplam), 1), "p95_ms": yuzde(toplam, 95),
            "bilesen_medyan_ms": {k: round(statistics.median(v), 2) for k, v in bilesen.items()},
            "duvar_saati_medyan_ms": round(statistics.median(duvar), 1), "olcum": len(toplam),
            "is_parcacigi": torch.get_num_threads()}     # Ultralytics işlemcide tahminde ayarlar (8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sonuclar", type=Path, required=True, help="Drive'dan indirilen sonuclar/ogrenci klasörü")
    ap.add_argument("--paket", type=Path, default=PAKET)
    ap.add_argument("--hepsi", action="store_true", help="yalnız tablo öğrencileri değil, bütün koşular")
    ap.add_argument("--cikti", type=Path, default=CIKTI)
    args = ap.parse_args()

    import torch
    import ultralytics
    tum = [(d, json.loads((d / "tamam.json").read_text(encoding="utf-8")))
           for d in sorted(p for p in args.sonuclar.iterdir() if (p / "tamam.json").exists() and (p / "best.pt").exists())]
    if args.hepsi:
        kosular = tum
    else:
        # Tablo: her mimarinin öğretmen kolu, tohum 42. Seçilen öğretmenin YOLO11n / öğretmen / 42 koşusu
        # Aşama 1'deki koşunun kendisidir (yeniden eğitilmedi).
        asama2 = [(d, t) for d, t in tum if t.get("asama") == 2 and t.get("kol") == "ogretmen" and t.get("tohum") == 42]
        ogretmen = {t.get("ogretmen") for _, t in asama2}
        kosular = [(d, t) for d, t in tum if t.get("asama") == 1 and t.get("ogretmen") in ogretmen] + asama2
    if not kosular:
        print("Ölçülecek koşu yok (Aşama 2 sonuçları gelmedi mi? Sınama için --hepsi).")
        return 1
    girdiler = kare_goruntuler(args.paket)
    es = esikler()
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    sonuc = {"yontem": __doc__.strip().splitlines()[0], "olusturma": datetime.now().isoformat(timespec="seconds"),
             "ortam": {"islemci": cpu or platform.processor(), "torch": torch.__version__,
                       "ultralytics": ultralytics.__version__, "python": platform.python_version()},
             "ayar": {"batch": 1, "girdi": f"{BOYUT}x{BOYUT}", "isinma": ISINMA, "olcum": OLCUM, "nms_iou": 0.7},
             "ogrenciler": {}}
    print(f"İşlemci {sonuc['ortam']['islemci']} · torch {torch.__version__} · ultralytics {ultralytics.__version__}")
    for d, t in kosular:
        o = olc(d / "best.pt", girdiler, es.get(d.name, 0.25))
        sonuc["ogrenciler"][d.name] = {"model": t.get("model"), "kol": t.get("kol"), "tohum": t.get("tohum")} | o
        print(f"  {d.name:46s} {t.get('model'):8s} medyan {o['medyan_ms']:6.1f} ms · p95 {o['p95_ms']:6.1f} · "
              f"bileşenler {o['bilesen_medyan_ms']} · eşik {o['esik']} · {o['is_parcacigi']} iş parçacığı")
    args.cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {args.cikti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
