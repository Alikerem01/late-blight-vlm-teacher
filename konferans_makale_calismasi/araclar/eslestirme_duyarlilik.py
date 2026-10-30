"""Eşleştirme duyarlılığı (makale 3.7) — SALT OKUMA, `veri/eslestirme_duyarlilik.json` yazar.

Makaledeki bütün kutu karşılaştırmaları Macar algoritmasıyla toplam IoU'yu en büyükleyen bire bir atamayı kurar,
eşiği (0,3 / 0,5 / 0,75) sonra uygular. Bu, eşik üstündeki çift sayısını en büyükleyen atamayla her zaman aynı
değildir. Bu betik analiz betiklerinin iki kopyasını geçici bir klasörde koşar:

  * `asil`: betikler olduğu gibi (önce `veri/` içindeki kayıtları birebir yeniden ürettiği denetlenir),
  * `alt`:  atama ağırlığı `1000 · [IoU ≥ eşik] + IoU` (önce eşik üstü çift sayısı, eşitlikte toplam IoU),
            görüntü sayımlarında her eşik ayrı ayrı.

ve iki kopyanın bütün sayısal çıktılarını yaprak yaprak karşılaştırır. Kapsam: adaylar ve uzmanlar (K, O, H),
öğrenciler (sadakat, doğruluk, eşik seçimi dahil), çevirme tutarlılığı (ilk koşu ve K), hassasiyet denetimi,
kıdemli uzman referansı. AP50 Ultralytics'in kendi açgözlü eşleştirmesini kullanır, bu karşılaştırmanın dışında.

Kullanım:  python eslestirme_duyarlilik.py   (5–10 dk)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent           # konferans_makale_calismasi/
PROJE = KOK.parent
CIKTI = KOK / "veri" / "eslestirme_duyarlilik.json"
SONUCLAR = PROJE / "colab" / "sonuclar" / "ogrenci"
UZMAN = KOK / "uzman_veriler_2"
KIDEMLI_PAKET = KOK / "paketler" / "degerlendirme_20261015T095230.tar.gz"
ZAMAN_ALANLARI = {"olusturma", "zaman", "uretim_zamani", "paket", "duzeltme_20261030"}

YAMALAR = [
    ("uzman_vlm_karsilastir.py", "    ri, ci = linear_sum_assignment(-m)\n",
     "    ri, ci = linear_sum_assignment(-(np.where(m >= 0.5, 1000.0, 0.0) + m))\n"),
    ("uzman_vlm_karsilastir.py", '''    ciftler = eslestir(ref, pred)
    s = {"n_ref": len(ref), "n_pred": len(pred)}
    for t in ESIKLER:
        s[f"tp_{t}"] = sum(1 for _, _, v in ciftler if v >= t)
    return s''', '''    s = {"n_ref": len(ref), "n_pred": len(pred)}
    for t in ESIKLER:
        if not ref or not pred:
            s[f"tp_{t}"] = 0
            continue
        m = np.array([[iou(r, p) for p in pred] for r in ref])
        ri, ci = linear_sum_assignment(-(np.where(m >= t, 1000.0, 0.0) + m))
        s[f"tp_{t}"] = int(sum(m[i, j] >= t for i, j in zip(ri, ci)))
    return s'''),
    ("cevirme_tutarliligi_vlm.py", "    ri, ci = linear_sum_assignment(-m)\n",
     "    ri, ci = linear_sum_assignment(-(np.where(m >= esik, 1000.0, 0.0) + m))\n"),
    ("etiketsiz_ek_analiz.py", '''def tp(a: list, b: list, esik: float) -> int:
    return sum(v >= esik for _, _, v in eslesme_iou(a, b))''', '''def tp(a: list, b: list, esik: float) -> int:
    if not a or not b:
        return 0
    m = np.array([[iou(x, y) for y in b] for x in a])
    ri, ci = linear_sum_assignment(-(np.where(m >= esik, 1000.0, 0.0) + m))
    return int(sum(m[i, j] >= esik for i, j in zip(ri, ci)))'''),
    ("cevirme_saglama.py", "ri, ci = linear_sum_assignment(-m)",
     "ri, ci = linear_sum_assignment(-(np.where(m >= t, 1000.0, 0.0) + m))"),
]
# (çıktı adı, kopyadaki göreli yol, gerçek kayıt)
CIKTILAR = [
    ("uzman_karsilastirma", "cikti/uk/uzman_karsilastirma.json", KOK / "veri" / "uzman_karsilastirma.json"),
    ("ogrenci_degerlendirme", "cikti/ogrenci.json", KOK / "veri" / "ogrenci_degerlendirme.json"),
    ("kor_kume_cevirme_tutarliligi", "kmc/veri/kor_kume_cevirme_tutarliligi.json", KOK / "veri" / "kor_kume_cevirme_tutarliligi.json"),
    ("cevirme_saglama", "kmc/veri/cevirme_saglama.json", KOK / "veri" / "cevirme_saglama.json"),
    ("etiketsiz_ek_analiz", "kmc/veri/etiketsiz_ek_analiz.json", KOK / "veri" / "etiketsiz_ek_analiz.json"),
    ("v_prec_analiz", "kmc/veri/v_prec_analiz.json", KOK / "veri" / "v_prec_analiz.json"),
    ("kidemli/uzman_karsilastirma", "cikti/kidemli/uzman_karsilastirma.json", KOK / "veri" / "kidemli" / "uzman_karsilastirma.json"),
    ("kidemli/ogrenci_degerlendirme", "cikti/kidemli/ogrenci_degerlendirme.json", KOK / "veri" / "kidemli" / "ogrenci_degerlendirme.json"),
    ("kidemli/ozet", "cikti/kidemli/ozet.json", KOK / "veri" / "kidemli" / "ozet.json"),
]


def yapraklar(o, yol: str = ""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k not in ZAMAN_ALANLARI:
                yield from yapraklar(v, f"{yol}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from yapraklar(v, f"{yol}[{i}]")
    else:
        yield yol, o


def farklar(a: Path, b: Path) -> tuple[int, list]:
    x = dict(yapraklar(json.loads(a.read_text(encoding="utf-8"))))
    y = dict(yapraklar(json.loads(b.read_text(encoding="utf-8"))))
    return len(x), [[k, x.get(k), y.get(k)] for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]


def kopya_kur(kok: Path) -> None:
    """Betiklerin kopyası; küçük veri dosyaları kopya (gerçek kayıtların üzerine yazılmasın), büyükler bağlantı."""
    (kok / "kmc" / "araclar").mkdir(parents=True)
    (kok / "kmc" / "veri").mkdir()
    for p in (KOK / "araclar").glob("*.py"):
        shutil.copy2(p, kok / "kmc" / "araclar" / p.name)
    for p in (KOK / "veri").iterdir():
        if p.is_file():
            shutil.copy2(p, kok / "kmc" / "veri" / p.name)
    os.symlink(KOK / "veri" / "vlm_kosulari", kok / "kmc" / "veri" / "vlm_kosulari")
    os.symlink(UZMAN, kok / "kmc" / "uzman_veriler_2")
    os.symlink(PROJE / "VLM_auto_annotation", kok / "VLM_auto_annotation")
    os.symlink(PROJE / "colab", kok / "colab")


def kos(kok: Path) -> None:
    c = kok / "cikti"
    c.mkdir()
    py, a = sys.executable, kok / "kmc" / "araclar"
    adimlar = [
        ["uzman_vlm_karsilastir.py", "--klasor", str(UZMAN), "--cikti", str(c / "uk")],
        ["ogrenci_degerlendir.py", "--sonuclar", str(SONUCLAR), "--uzman-klasor", str(UZMAN), "--cikti", str(c / "ogrenci.json")],
        ["cevirme_tutarliligi_vlm.py"], ["cevirme_saglama.py"], ["etiketsiz_ek_analiz.py"], ["v_prec_analiz.py"],
        ["kidemli_inceleme.py", "--paket", str(KIDEMLI_PAKET), "--sonuclar", str(SONUCLAR), "--cikti", str(c / "kidemli")],
    ]
    for adim in adimlar:
        r = subprocess.run([py, *adim], cwd=a, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"HATA ({kok.name}, {adim[0]}): {r.stderr.strip()[-800:]}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="eslestirme_duyarlilik_") as g:
        g = Path(g)
        for tur in ("asil", "alt"):
            kopya_kur(g / tur)
        for dosya, eski, yeni in YAMALAR:
            p = g / "alt" / "kmc" / "araclar" / dosya
            t = p.read_text(encoding="utf-8")
            if t.count(eski) != 1:
                raise SystemExit(f"HATA: yama yeri bulunamadı: {dosya}")
            p.write_text(t.replace(eski, yeni), encoding="utf-8")
        for tur in ("asil", "alt"):
            print(f"koşuluyor: {tur}", flush=True)
            kos(g / tur)
        sonuc = {"olusturma": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "yontem": __doc__.strip().splitlines()[0], "yeniden_uretim": {}, "alternatif_fark": {}}
        for ad, gorel, gercek in CIKTILAR:
            n, f = farklar(gercek, g / "asil" / gorel)
            sonuc["yeniden_uretim"][ad] = {"yaprak": n, "fark": len(f), "ornek": f[:5]}
            n, f = farklar(g / "asil" / gorel, g / "alt" / gorel)
            sonuc["alternatif_fark"][ad] = {"yaprak": n, "fark": len(f), "farklar": f}
            print(f"{ad:32s} yeniden üretim farkı {sonuc['yeniden_uretim'][ad]['fark']:3d} · alternatif farkı {len(f):3d}"
                  + "".join(f"\n    {k}: {x} → {y}" for k, x, y in f))
    CIKTI.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {CIKTI}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
