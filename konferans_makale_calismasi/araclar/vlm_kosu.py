"""Beş yerel VLM'i kör kümede (veya bir alt kümede) çalıştırır — proje verisine SALT OKUMA.

Öğretmen seçimi (brif bölüm 7, V-RUN): aynı bilgi temelli istem, modele özgü
çıktı sözleşmesi, sıcaklık 0, en çok 1024 belirteç (ilk etiketleme koşusuyla aynı).

  * Gemma 4 (E2B / E4B / 12B): istem `auto_label.GEMMA_PROMPT` ile BİREBİR aynı;
    `box_2d` = [y1, x1, y2, x2], 0–1000.
  * Qwen3-VL (4B / 8B): aynı istem, yalnız çıktı sözleşmesi Qwen3-VL'in kendi
    biçimine çevrilir: `bbox_2d` = [x1, y1, x2, y2], 0–1000.

Her görüntü için JSONL'e bir satır yazılır (kesintide kaldığı yerden devam eder):
ham yanıt, ayrıştırılmış tespitler, iki eksen yorumunda YOLO kutusu (kalibrasyon
için), süre. Kütüphane/model sürümleri `<anahtar>_meta.json` dosyasına yazılır.

Kullanım:
  python vlm_kosu.py --model qwen3vl-4b --limit 6      # kalibrasyon
  python vlm_kosu.py --model gemma4-12b                # kör kümenin tamamı
Çıktı: veri/vlm_kosulari/<anahtar>.jsonl
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
VLM_KLASOR = KOK.parent / "VLM_auto_annotation"
sys.path.insert(0, str(VLM_KLASOR))
from auto_label import GEMMA_PROMPT, box_2d_to_yolo, normalize_class_name, parse_vlm_json  # noqa: E402

MODELLER = {
    "gemma4-e2b": ("mlx-community/gemma-4-e2b-it-4bit", "gemma"),
    "gemma4-e4b": ("mlx-community/gemma-4-e4b-it-4bit", "gemma"),
    "gemma4-12b": ("mlx-community/gemma-4-12B-it-4bit", "gemma"),
    "qwen3vl-4b": ("mlx-community/Qwen3-VL-4B-Instruct-4bit", "qwen"),
    "qwen3vl-8b": ("mlx-community/Qwen3-VL-8B-Instruct-4bit", "qwen"),
    # nicemleme kontrolü (hakem R1-M7, karar D6): aynı model, daha yüksek duyarlık
    "gemma4-e2b-8bit": ("mlx-community/gemma-4-e2b-it-8bit", "gemma"),
    "gemma4-e2b-bf16": ("mlx-community/gemma-4-e2b-it-bf16", "gemma"),
}
assert GEMMA_PROMPT.count("[y1, x1, y2, x2]") == 2 and GEMMA_PROMPT.count("box_2d") >= 3
QWEN_PROMPT = GEMMA_PROMPT.replace("box_2d", "bbox_2d").replace("[y1, x1, y2, x2]", "[x1, y1, x2, y2]")
KUMELER = {"a_": "kaynak_a_images", "b_": "kaynak_b_images"}
GIRDI_ONBELLEK = Path.home() / "vlm_modeller" / "_girdi"   # küçültülmüş girdiler (deterministik, PNG)
SICAKLIK, EN_COK_BELIRTEC = 0.0, 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr, force=True)
log = logging.getLogger("vlm_kosu")


def yolo_kutulari(ham: list, aile: str) -> list[dict]:
    """Ham tespitleri iki eksen yorumunda YOLO'ya çevirir (yerel = modelin belgelenen sözleşmesi)."""
    out = []
    for d in ham:
        if not isinstance(d, dict):
            continue
        sinif = normalize_class_name(d.get("label") or d.get("class"))
        kutu = d.get("box_2d") if aile == "gemma" else d.get("bbox_2d", d.get("box_2d"))
        if not (isinstance(kutu, (list, tuple)) and len(kutu) == 4):
            out.append({"sinif": sinif, "ham_kutu": kutu, "yolo": None, "yolo_takasli": None})
            continue
        a, b, c, e = kutu
        if aile == "gemma":            # [y1, x1, y2, x2]
            yerel, takasli = [a, b, c, e], [b, a, e, c]
        else:                          # Qwen: [x1, y1, x2, y2] → box_2d_to_yolo [y1, x1, y2, x2] bekler
            yerel, takasli = [b, a, e, c], [a, b, c, e]
        try:
            guven = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            guven = None
        out.append({
            "sinif": sinif, "guven": guven, "gerekce": str(d.get("reasoning", "")), "ham_kutu": kutu,
            "yolo": box_2d_to_yolo(yerel), "yolo_takasli": box_2d_to_yolo(takasli),
            "ham_en_buyuk": max(float(v) for v in kutu) if all(isinstance(v, (int, float)) for v in kutu) else None,
        })
    return out


YEREL_KOK = Path.home() / "vlm_modeller"   # araclar/model_indir.sh'in indirdiği, SHA-256 doğrulanmış kopyalar


def model_yolu(model_id: str) -> tuple[str, str | None]:
    """Yerel doğrulanmış kopya varsa onun yolunu ve commit'ini, yoksa depo adını döner."""
    yerel = YEREL_KOK / model_id.split("/", 1)[1]
    if (yerel / ".tamam").exists():
        return str(yerel), (yerel / ".tamam").read_text(encoding="utf-8").strip()
    return model_id, None


def surum_bilgisi(model_id: str) -> dict:
    from huggingface_hub import snapshot_download
    yol, revizyon = model_yolu(model_id)
    if revizyon is None:
        try:
            revizyon = Path(snapshot_download(model_id, local_files_only=True)).name
        except Exception:  # noqa: BLE001
            revizyon = None
    paket = {}
    for p in ("mlx", "mlx-vlm", "mlx-lm", "transformers", "huggingface-hub"):
        try:
            paket[p] = md.version(p)
        except md.PackageNotFoundError:
            paket[p] = None
    return {"model_id": model_id, "yuklenen_yol": yol, "hf_revizyon": revizyon, "paketler": paket,
            "python": platform.python_version(), "makine": platform.machine(), "platform": platform.platform()}


def girdi_yolu(kaynak: Path, max_kenar: int, aynala: bool = False) -> Path:
    """Beş model aynı girdiyi görsün: uzun kenar max_kenar px (LANCZOS), kayıpsız PNG. 0 = özgün dosya.

    Qwen3-VL görüntüyü özgün çözünürlüğünde işlediği için 24–48 MP fotoğraflarda görüntü
    başına ~130 s sürüyordu (kalibrasyon, 2026-09-25); Gemma kendi girişine zaten küçültüyor.
    """
    if not max_kenar and not aynala:
        return kaynak
    from PIL import Image
    alt = (str(max_kenar) if max_kenar else "ozgun") + ("_fliph" if aynala else "")
    hedef = GIRDI_ONBELLEK / alt / (kaynak.stem + ".png")
    if not hedef.exists():
        hedef.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(kaynak) as im:
            im = im.convert("RGB")
            if max_kenar:
                im.thumbnail((max_kenar, max_kenar), Image.LANCZOS)
            if aynala:   # küçültmeden SONRA: iki koşu arasındaki tek fark aynalama olsun
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im.save(hedef.with_suffix(".tmp.png"), format="PNG")
        hedef.with_suffix(".tmp.png").rename(hedef)
    return hedef


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(MODELLER))
    ap.add_argument("--liste", type=Path, default=KOK / "veri" / "kor_kume_60.txt")
    ap.add_argument("--limit", type=int, default=0, help="ilk N görüntü (kalibrasyon)")
    ap.add_argument("--etiket", default="", help="çıktı adına ek (ör. _kalibrasyon)")
    ap.add_argument("--max-kenar", type=int, default=1280, help="girdi uzun kenarı (px); 0 = özgün dosya")
    ap.add_argument("--aynala", action="store_true", help="girdiyi yatay aynala (çevirme tutarlılığı, V-FLIP); kutular aynalı çerçevede kalır")
    args = ap.parse_args()

    model_id, aile = MODELLER[args.model]
    istem = GEMMA_PROMPT if aile == "gemma" else QWEN_PROMPT
    adlar = [s.strip() for s in args.liste.read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.limit:
        adlar = adlar[: args.limit]
    cikti_dizin = KOK / "veri" / "vlm_kosulari"
    cikti_dizin.mkdir(parents=True, exist_ok=True)
    cikti = cikti_dizin / f"{args.model}{args.etiket}.jsonl"
    bitmis = set()
    if cikti.exists():
        for line in cikti.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("durum") == "ok":
                bitmis.add(r["goruntu"])
    kalan = [a for a in adlar if a not in bitmis]
    log.info("%s (%s): %d görüntü, %d tamam, %d kalan → %s", args.model, model_id, len(adlar), len(bitmis), len(kalan), cikti.name)
    if not kalan:
        return 0

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    t0 = time.time()
    yuklenecek, _ = model_yolu(model_id)
    model, processor = load(yuklenecek)
    config = load_config(yuklenecek)
    yukleme_s = time.time() - t0
    meta = surum_bilgisi(model_id) | {"anahtar": args.model, "aile": aile, "sicaklik": SICAKLIK,
                                      "en_cok_belirtec": EN_COK_BELIRTEC, "yukleme_s": round(yukleme_s, 1), "girdi_max_kenar": args.max_kenar, "girdi_aynali": args.aynala,
                                      "istem_sha": __import__("hashlib").sha256(istem.encode()).hexdigest()[:16],
                                      "baslangic": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (cikti_dizin / f"{args.model}{args.etiket}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Model yüklendi (%.1f s). mlx-vlm %s", yukleme_s, meta["paketler"]["mlx-vlm"])
    bicimli = apply_chat_template(processor, config, istem, num_images=1)

    with open(cikti, "a", encoding="utf-8") as f:
        for i, ad in enumerate(kalan, 1):
            kume = next(v for k, v in KUMELER.items() if ad.startswith(k))
            yol = girdi_yolu(VLM_KLASOR / kume / ad, args.max_kenar, args.aynala)
            bas = time.time()
            kayit = {"goruntu": ad, "model": args.model}
            try:
                sonuc = generate(model, processor, bicimli, image=str(yol), max_tokens=EN_COK_BELIRTEC,
                                 temperature=SICAKLIK, verbose=False)
                kayit["sure_ms"] = round((time.time() - bas) * 1000, 1)
                kayit["ham_yanit"] = sonuc.text
                ham = parse_vlm_json(sonuc.text)
                if ham is None:
                    kayit["durum"] = "parse_hatasi"
                else:
                    kayit["durum"] = "ok"
                    kayit["tespitler"] = yolo_kutulari(ham, aile)
            except Exception as exc:  # noqa: BLE001 — tek görüntü hatası koşuyu durdurmaz
                kayit["durum"] = "hata"
                kayit["hata"] = f"{type(exc).__name__}: {exc}"
                kayit["sure_ms"] = round((time.time() - bas) * 1000, 1)
            f.write(json.dumps(kayit, ensure_ascii=False) + "\n")
            f.flush()
            n = len(kayit.get("tespitler", []))
            log.info("[%d/%d] %s · %s · %d tespit · %.1f s", i, len(kalan), ad, kayit["durum"], n, kayit["sure_ms"] / 1000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
