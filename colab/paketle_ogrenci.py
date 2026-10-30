#!/usr/bin/env python3
"""Öğrenci eğitimleri için Colab veri paketi (ICIDST 2026 makalesi, S-LEARN ve E1) — SALT OKUMA.

Paket `colab/paketler/ogrenci.zip` (+ `ogrenci_manifest.json`), içinde:

  ogrenci/goruntuler/<ad>.jpg       301 özgün görüntü, uzun kenar 1280 px. Kaynak, öğretmen
                                    adaylarının gördüğü girdinin kendisi (~/vlm_modeller/_girdi/1280,
                                    kayıpsız PNG); JPEG kalite 95 olarak saklanır.
  ogrenci/etiketler/<aday>/<ad>.txt altı adayın YOLO etiketleri (sınıf 0 = Late_Blight):
                                    241 eğitim görüntüsü `<aday>_egitim241.jsonl`'dan, kör küme K'nin
                                    60 görüntüsü `<aday>.jsonl`'dan (ikisi de 1280 px, aynı istem)
  ogrenci/bolunme.json              sabit bölünme (tohum 42, sahne türüne göre katmanlı):
                                    241 → 205 eğitim / 36 doğrulama; K (60, test); H (150, HITL,
                                    241'in alt kümesi); yalnız-uzman kolu için H → 127 / 23

K ve H sunucunun kurulumuyla birebir aynı türetilir (`degerlendirme_kur.sh`, 2026-09-25): rastgele
sıradan (`sira.csv`), protokolden önce araçta incelenmiş özgün görüntüler atlanarak ilk 60 → K,
sonraki 150 → H. Sunucu iki görüntüyü atladı (`SUNUCU_HARIC`); sonuç, sunucunun kurulum kaydındaki
(küme, özgün ad) dizisinin md5'iyle kilitlidir (`SUNUCU_KURULUM_MD5`, 2026-09-30).

Proje verisine yalnız okur; tek yazdığı `colab/paketler/` (git dışında).

Kullanım:
    python3 paketle_ogrenci.py --dry-run
    python3 paketle_ogrenci.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

KOK = Path(__file__).resolve().parent.parent
MAKALE = KOK / "konferans_makale_calismasi"
KOSU = MAKALE / "veri" / "vlm_kosulari"
GIRDI = Path.home() / "vlm_modeller" / "_girdi" / "1280"
CIKTI = Path(__file__).resolve().parent / "paketler"
TOHUM = 42
ADAYLAR = ["gemma4-e2b", "gemma4-e2b-8bit", "gemma4-e4b", "gemma4-12b", "qwen3vl-4b", "qwen3vl-8b"]
DOGRULAMA_ORANI = 0.15
# Sunucu kurulumunun atladığı (protokolden önce araçta incelenmiş) özgün görüntüler: sıra 153 ve 206.
# Doğrulama: sunucuda  awk -F, 'NR>1{print $1","$4}' degerlendirme/kayit/kurulum_20260925T112650.csv | md5sum
SUNUCU_HARIC = ("a_0007.jpg", "a_0008.jpg")
SUNUCU_KURULUM_MD5 = "3bbbc0cdbe5c3efaa101d732f834cea9"


def satirlar(p: Path) -> list[str]:
    return [s.strip() for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]


def katmanli_bol(adlar: list[str], sahne: dict[str, str], oran: float, tohum: int) -> tuple[list[str], list[str]]:
    """Her sahne türünde ayrı ayrı karıştırıp `oran` kadarını doğrulamaya ayırır."""
    rng = random.Random(tohum)
    egitim, dogrulama = [], []
    for s in sorted({sahne[a] for a in adlar}):
        grup = sorted(a for a in adlar if sahne[a] == s)
        rng.shuffle(grup)
        n = round(len(grup) * oran)
        dogrulama += grup[:n]
        egitim += grup[n:]
    return sorted(egitim), sorted(dogrulama)


def jsonl_etiketleri(yol: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    out = {}
    for r in map(json.loads, yol.read_text(encoding="utf-8").splitlines()):
        if r["durum"] != "ok":
            raise SystemExit(f"HATA: {yol.name} içinde başarısız kayıt: {r['goruntu']} ({r['durum']})")
        out[r["goruntu"]] = [tuple(t["yolo"]) for t in r["tespitler"] if t.get("yolo") and t.get("sinif")]
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="yalnız denetim ve özet, dosya yazmaz")
    args = ap.parse_args()

    kor = satirlar(MAKALE / "veri" / "kor_kume_60.txt")
    egitim241 = satirlar(MAKALE / "veri" / "egitim_241.txt")
    sira = [r["goruntu"] for r in csv.DictReader(open(KOK / "annotation_interface_local" / "degerlendirme" / "sira.csv", encoding="utf-8"))]
    uygun = [a for a in sira if a not in SUNUCU_HARIC]           # degerlendirme_kur.sh ile aynı seçim
    hitl = uygun[60:210]
    ozet = hashlib.md5(("".join(f"kor,{a}\n" for a in uygun[:60]) + "".join(f"hitl,{a}\n" for a in hitl)).encode()).hexdigest()
    sahne = {k: v["sahne"] for k, v in json.loads((MAKALE / "veri" / "sahne_turu.json").read_text(encoding="utf-8"))["goruntuler"].items()}

    # Tutarlılık denetimleri: K ve H sunucudakiyle aynı, kümeler örtüşmez, H eğitim kümesinin alt kümesi
    assert ozet == SUNUCU_KURULUM_MD5, f"K/H sunucunun kurulum kaydıyla aynı değil (md5 {ozet})"
    assert uygun[:60] == kor, "K listesi sunucunun K'siyle aynı değil"
    assert not set(kor) & set(egitim241), "K ile eğitim kümesi örtüşüyor"
    assert set(hitl) <= set(egitim241), "H eğitim kümesinin alt kümesi değil"
    assert len(set(kor)) == 60 and len(set(egitim241)) == 241 and len(set(hitl)) == 150
    tumu = sorted(set(kor) | set(egitim241))
    assert len(tumu) == 301

    egitim, dogrulama = katmanli_bol(egitim241, sahne, DOGRULAMA_ORANI, TOHUM)
    u_egitim, u_dogrulama = katmanli_bol(hitl, sahne, DOGRULAMA_ORANI, TOHUM)

    etiket: dict[str, dict[str, list]] = {}
    kaynak_ozet = {}
    for a in ADAYLAR:
        e = jsonl_etiketleri(KOSU / f"{a}_egitim241.jsonl")
        k = jsonl_etiketleri(KOSU / f"{a}.jsonl")
        assert set(e) == set(egitim241), f"{a}: eğitim koşusu 241 görüntüyü kapsamıyor"
        assert set(k) == set(kor), f"{a}: kör küme koşusu 60 görüntüyü kapsamıyor"
        etiket[a] = e | k
        for kutular in etiket[a].values():
            for b in kutular:
                assert all(0.0 <= v <= 1.0 for v in b) and b[2] > 0 and b[3] > 0, f"{a}: geçersiz kutu {b}"
        kaynak_ozet[a] = {"egitim241_jsonl_sha256": sha256(KOSU / f"{a}_egitim241.jsonl"),
                          "kor_jsonl_sha256": sha256(KOSU / f"{a}.jsonl"),
                          "kutu_egitim241": sum(len(e[x]) for x in egitim241),
                          "kutu_kor": sum(len(k[x]) for x in kor)}
    eksik = [a for a in tumu if not (GIRDI / f"{Path(a).stem}.png").exists()]
    assert not eksik, f"1280 px girdi eksik: {eksik[:5]}"

    def say(liste):
        return {s: sum(1 for x in liste if sahne[x] == s) for s in ("detached", "in_situ")}
    bolunme = {
        "tohum": TOHUM, "dogrulama_orani": DOGRULAMA_ORANI, "katman": "sahne türü",
        "kume_kaynagi": {"sira": "annotation_interface_local/degerlendirme/sira.csv", "sunucu_haric": list(SUNUCU_HARIC),
                         "sunucu_kurulum_md5": SUNUCU_KURULUM_MD5},
        "egitim": egitim, "dogrulama": dogrulama, "kor": kor, "hitl": hitl,
        "uzman150_egitim": u_egitim, "uzman150_dogrulama": u_dogrulama,
        "sayilar": {"egitim": [len(egitim), say(egitim)], "dogrulama": [len(dogrulama), say(dogrulama)],
                    "kor": [len(kor), say(kor)], "hitl": [len(hitl), say(hitl)],
                    "uzman150_egitim": [len(u_egitim), say(u_egitim)], "uzman150_dogrulama": [len(u_dogrulama), say(u_dogrulama)]},
    }
    print("Bölünme:", json.dumps(bolunme["sayilar"], ensure_ascii=False))
    for a, o in kaynak_ozet.items():
        print(f"  {a:16s} eğitim241 {o['kutu_egitim241']:4d} kutu · K {o['kutu_kor']:3d} kutu")
    if args.dry_run:
        print("DRY-RUN — hiçbir şey yazılmadı.")
        return 0

    CIKTI.mkdir(parents=True, exist_ok=True)
    zip_yol = CIKTI / "ogrenci.zip"
    with tempfile.TemporaryDirectory() as gecici:
        g = Path(gecici) / "ogrenci"
        (g / "goruntuler").mkdir(parents=True)
        for ad in tumu:
            with Image.open(GIRDI / f"{Path(ad).stem}.png") as im:
                im.convert("RGB").save(g / "goruntuler" / f"{Path(ad).stem}.jpg", format="JPEG", quality=95)
        for a in ADAYLAR:
            d = g / "etiketler" / a
            d.mkdir(parents=True)
            for ad in tumu:
                (d / f"{Path(ad).stem}.txt").write_text(
                    "".join("0 %.6f %.6f %.6f %.6f\n" % b for b in etiket[a][ad]), encoding="utf-8")
        (g / "bolunme.json").write_text(json.dumps(bolunme, ensure_ascii=False, indent=1), encoding="utf-8")
        gecici_zip = zip_yol.with_suffix(".zip.tmp")
        with zipfile.ZipFile(gecici_zip, "w", zipfile.ZIP_STORED) as z:   # JPEG zaten sıkışık
            for p in sorted(g.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(Path(gecici)))
        gecici_zip.replace(zip_yol)                                          # atomik
    manifest = {"ad": "ogrenci", "uretim_zamani": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "goruntu": len(tumu), "max_kenar": 1280, "jpeg_kalite": 95, "adaylar": ADAYLAR,
                "zip_mb": round(zip_yol.stat().st_size / 1e6, 1), "zip_sha256": sha256(zip_yol),
                "bolunme": bolunme["sayilar"], "kaynaklar": kaynak_ozet}
    (CIKTI / "ogrenci_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {zip_yol} ({manifest['zip_mb']} MB, sha256 {manifest['zip_sha256'][:16]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
