#!/usr/bin/env python3
"""Colab Aşama 2 için uzman etiketi paketi (ICIDST 2026 makalesi, E1-HITL ve E1-EXP) — SALT OKUMA.

Sunucunun dışa aktarım paketinden (`degerlendirme_disa_aktar.sh` → degerlendirme_<zaman>.tar.gz)
**yalnız H'nin** (2_duzeltme_onay, 150 görüntü) uzman son hâlini `colab/paketler/uzman.zip` olarak
yazar (+ `uzman_manifest.json`):

  ogrenci/etiketler/uzman/<özgün ad kökü>.txt   sınıf 0 = Late_Blight, yalnız lezyon sınıfı

K'nin kör uzman etiketleri bu pakete GİRMEZ: öğrencilerin test referansıdır, Drive'a çıkmaz.
Öğrencinin K'deki doğruluğu yerelde doğrudan dışa aktarım paketinden okunur
(`ogrenci_degerlendir.py --uzman-paket`).

Denetimler: H'nin 150 görüntüsünün hepsi incelenmiş olmalı; görüntüler öğrenci paketindeki
(`paketler/ogrenci.zip` → bolunme.json) H listesiyle birebir aynı olmalı; kutular geçerli olmalı.

Kullanım:
    python3 paketle_uzman.py --paket ~/Downloads/degerlendirme_<zaman>.tar.gz --dry-run
    python3 paketle_uzman.py --paket ~/Downloads/degerlendirme_<zaman>.tar.gz
Çıkış: 0 tamam · 65 veri eksik ya da tutarsız
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "konferans_makale_calismasi" / "araclar"))
from uzman_vlm_karsilastir import klasor_yukle, paket_yukle, paketi_ac  # noqa: E402

CIKTI = Path(__file__).resolve().parent / "paketler"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    girdi = ap.add_mutually_exclusive_group(required=True)
    girdi.add_argument("--paket", type=Path, help="dışa aktarım paketi (tar.gz ya da açılmış klasör)")
    girdi.add_argument("--klasor", type=Path, help="uzmanların klasör olarak gönderdiği etiketler (uzman_veriler_2)")
    ap.add_argument("--ogrenci", type=Path, default=CIKTI / "ogrenci.zip", help="öğrenci paketi (bölünme için)")
    ap.add_argument("--cikti", type=Path, default=CIKTI)
    ap.add_argument("--lezyon-sinifi", type=int, default=0, help="sunucuda Late_Blight = 0")
    ap.add_argument("--dry-run", action="store_true", help="yalnız denetim ve özet, dosya yazmaz")
    args = ap.parse_args()

    paket = (klasor_yukle(args.klasor.expanduser(), args.lezyon_sinifi) if args.klasor
             else paket_yukle(paketi_ac(args.paket.expanduser()), args.lezyon_sinifi))
    H = paket["kumeler"]["hitl"]
    with zipfile.ZipFile(args.ogrenci) as z:
        B = json.loads(z.read("ogrenci/bolunme.json"))
    beklenen = set(B["hitl"])
    eksik, fazla = sorted(beklenen - set(H)), sorted(set(H) - beklenen)
    yabanci = {s: n for s, n in paket["siniflar"]["hitl"].items() if s != str(args.lezyon_sinifi)}
    kutu = sum(len(r["kutular"]) for r in H.values())
    print(f"H: {len(H)}/{paket['hedef']['hitl']} incelenmiş · {kutu} kutu · boş {sum(1 for r in H.values() if not r['kutular'])} · "
          f"uzman {dict(sorted(Counter(r['uzman'] for r in H.values()).items()))} · "
          f"durum {dict(sorted(Counter(r['durum'] for r in H.values()).items()))}")
    if yabanci:
        print(f"UYARI: lezyon sınıfı dışında {sum(yabanci.values())} kutu var, pakete girmez: {yabanci}")
    if eksik or fazla:
        print(f"HATA: H öğrenci paketindeki listeyle aynı değil — eksik {len(eksik)} {eksik[:5]} · fazla {len(fazla)} {fazla[:5]}")
        return 65
    for a, r in H.items():
        for b in r["kutular"]:
            if not (all(0.0 <= v <= 1.0 for v in b) and b[2] > 0 and b[3] > 0):
                print(f"HATA: geçersiz kutu {a}: {b}")
                return 65
    if args.dry_run:
        print("DRY-RUN — hiçbir şey yazılmadı.")
        return 0

    args.cikti.mkdir(parents=True, exist_ok=True)
    zip_yol = args.cikti / "uzman.zip"
    gecici = zip_yol.with_suffix(".zip.tmp")
    with tempfile.TemporaryDirectory() as g:
        d = Path(g) / "ogrenci" / "etiketler" / "uzman"
        d.mkdir(parents=True)
        for a in sorted(H):
            (d / f"{Path(a).stem}.txt").write_text("".join("0 %.6f %.6f %.6f %.6f\n" % b for b in H[a]["kutular"]), encoding="utf-8")
        with zipfile.ZipFile(gecici, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(d.iterdir()):
                z.write(p, p.relative_to(Path(g)))
    gecici.replace(zip_yol)                                                    # atomik
    manifest = {"ad": "uzman", "uretim_zamani": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kaynak_paket": (args.paket or args.klasor).name, "kume": "hitl (2_duzeltme_onay_images)", "lezyon_sinifi": args.lezyon_sinifi,
                "goruntu": len(H), "kutu": kutu, "bos_goruntu": sum(1 for r in H.values() if not r["kutular"]),
                "uzman_dagilimi": dict(sorted(Counter(r["uzman"] for r in H.values()).items())),
                "durum": dict(sorted(Counter(r["durum"] for r in H.values()).items())),
                "yabanci_sinif_atildi": yabanci, "zip_sha256": sha256(zip_yol)}
    (args.cikti / "uzman_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {zip_yol} ({len(H)} etiket dosyası, sha256 {manifest['zip_sha256'][:16]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
