"""Faz 1+2 — Augmentation + Gemma 4 ile otomatik YOLO etiketleme.

Akış:
  1. (opsiyonel, parametrik) Augmentation: flip_h / flip_v / darker / lighter.
     Her biri ayrı bir boolean flag. Üretilen görüntüler AYNI images_dir'e,
     "{orijinal_ad}_{teknik}{uzantı}" olarak kaydedilir (örn. a_0001_fliph.jpg).
     İdempotent: zaten var olan bir augment dosyası tekrar üretilmez.
  2. Etiketleme: images_dir'deki HER görüntü (orijinal + augment'lar) Gemma 4
     (mlx-vlm) ile etiketlenir. Eksen sırası kalibre edildi (bkz. proje
     hafızası, 2026-08-11) — box_2d = [y1,x1,y2,x2], 0-1000 normalize.
     Sonuç YOLO formatında (class cx cy w h, 0-1 normalize) images_dir'in
     YANINDA (aynı üst klasörde) "{images_dir_adı}_labels/" klasörüne yazılır.
     İdempotent: zaten .txt'si olan bir görüntü --force verilmedikçe atlanır.

Confidence threshold altındaki tespitler YOLO .txt'ye YAZILMAZ (elenir) ama
manifest'te ham haliyle (confidence + reasoning ile) kalır — hangi tespitlerin
neden elendiği sonradan incelenebilir.

Kullanım:
    .venv/bin/python3 auto_label.py --images-dir greenhouse_images --dry-run
    .venv/bin/python3 auto_label.py --images-dir greenhouse_images --limit 10
    .venv/bin/python3 auto_label.py --images-dir greenhouse_images

Exit code'lar:
    0 - Tüm görüntüler başarıyla işlendi (parse hatası dahil hiç yoksa).
    1 - En az bir görüntüde parse/inference hatası oluştu (rapora bakın).
    2 - Config/kaynak klasör hatası (hiçbir görüntü işlenmedi).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler (kalibre edilmiş / onaylanmış değerler — bkz. proje hafızası)
# ---------------------------------------------------------------------------

MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AUG_SUFFIXES = ("_fliph", "_flipv", "_darker", "_lighter")

# SADECE Late_Blight etiketleniyor (kullanıcı kararı, 2026-08-11): bu veri seti
# zaten hastalık-ağırlıklı, Healthy sınıfının değeri yok. Class id kasıtlı
# olarak 1 (0 DEĞİL) - projenin `labeled/dataset_main` setinde 0=Healthy,
# 1=Late_Blight konvansiyonu var; ileride iki seti birleştirmek gerekirse
# id çakışması olmasın diye tutarlılık korunuyor.
CLASS_NAME_TO_ID = {"Late_Blight": 1}
CLASS_ALIASES = {
    "late_blight": "Late_Blight", "lateblight": "Late_Blight",
    "mildiyö": "Late_Blight", "mildiyo": "Late_Blight",
    "hasta": "Late_Blight", "diseased": "Late_Blight",
}

GEMMA_PROMPT = """Sen bir bitki patolojisi uzmanısın, domates yaprağı hastalıklarında uzmanlaşmışsın. Görüntüdeki domates yaprağında late blight (mildiyö, Phytophthora infestans) belirtilerini tespit et.

LATE BLIGHT BELİRTİLERİ (bu kriterlere göre değerlendir):
- Açık yeşilden kahverengi-siyaha uzanan, su emmiş görünümlü yaprak lekeleri; hızla büyüyerek yaprağın geniş bölümlerini kaplar.
- Yaprağın alt yüzeyinde gri-beyaz küf/sporangium tabakası.
- Büyük lekelerin çevresinde, nemli havada oluşan halka (ring) şeklinde küf büyümesi.
- İlerleyen aşamada yaprak sararır, ardından kahverengileşir, kıvrılır, buruşur ve ölür.

AYIRICI TANI (bunlarla karıştırma):
- Yaprak altındaki gri-beyaz küf tabakası powdery mildew (külleme) ile karıştırılabilir. Ayrım noktası: late blight'ta bu tabakaya su emmiş, yağlı görünümlü kahverengi-siyah lekeler eşlik eder ve hızla yayılır; sadece beyaz pudra görünümü tek başına late blight kanıtı sayılmaz.

GÖREV:
Görüntüde late blight belirtisi gösteren HER bölgeyi işaretle. Sağlıklı bölgeleri
işaretlemene gerek YOK — sadece hastalıklı/lezyonlu alanları tespit et. Her tespit için:
- label: her zaman "Late_Blight"
- box_2d: bounding box
- confidence: bulgu güvenin (0.0-1.0)
- reasoning: kısa gerekçe

ÇIKTI FORMATI (sadece geçerli JSON dizisi döndür, başka metin yazma):
[
  {
    "box_2d": [y1, x1, y2, x2],
    "label": "Late_Blight",
    "confidence": 0.85,
    "reasoning": "Su emmiş kahverengi lezyon, kenarında sarımsı halka"
  }
]

ÖNEMLİ:
- box_2d koordinatları 0-1000 aralığına NORMALİZE edilmiş tam sayılar, [y1, x1, y2, x2] sırasında
- SADECE hastalık belirtisi olan bölgeleri işaretle, sağlıklı yaprak/bölge için kutu ÇİZME
- Görüntüde hiç hastalık belirtisi yoksa boş dizi [] döndür
- Emin olmadığın tespitlere düşük confidence ver
- Sadece JSON dizisi döndür, öncesinde/sonrasında açıklama yazma
"""


@dataclass(frozen=True)
class PipelineConfig:
    """Bir çalıştırmanın tüm ayarları."""

    images_dir: Path
    augment_flip_h: bool = True
    augment_flip_v: bool = True
    augment_darker: bool = False
    augment_lighter: bool = False
    brightness_factor: float = 0.12  # ±%12 (darker/lighter aktif olunca kullanılır)
    confidence_threshold: float = 0.3
    max_tokens: int = 1024
    temperature: float = 0.0
    dry_run: bool = False
    force: bool = False
    limit: Optional[int] = None

    @property
    def labels_dir(self) -> Path:
        return self.images_dir.parent / f"{self.images_dir.name}_labels"


@dataclass
class DetectionRecord:
    """Tek bir tespit (threshold uygulanmadan önce, ham)."""

    class_name: str
    confidence: float
    reasoning: str
    box_2d_raw: list
    yolo_box: Optional[tuple[float, float, float, float]]  # (cx, cy, w, h) veya None (geçersiz)
    kept: bool  # confidence_threshold'u geçti mi


@dataclass
class ImageRecord:
    """Bir görüntü için manifest satırı."""

    image_name: str
    augmentation_type: str  # "original" | "fliph" | "flipv" | "darker" | "lighter"
    source_image: str  # augment ise hangi orijinalden üretildi
    status: str  # "ok" | "atlandi" | "parse_hatasi" | "hata"
    num_detections_raw: int
    num_detections_kept: int
    elapsed_ms: float
    error_message: str = ""
    detections: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Augmentation (Faz 1)
# ---------------------------------------------------------------------------


def is_augmented_name(stem: str) -> bool:
    """Bir dosya adının zaten bir augmentation çıktısı olup olmadığını kontrol eder."""
    return any(stem.endswith(suf) for suf in AUG_SUFFIXES)


def plan_augmentations(config: PipelineConfig) -> list[tuple[Path, str, Path]]:
    """Üretilmesi gereken (henüz diskte olmayan) augment dosyalarını planlar.

    Returns:
        (kaynak_dosya, teknik_adı, hedef_dosya) üçlülerinin listesi.
    """
    base_files = sorted(
        p for p in config.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not is_augmented_name(p.stem)
    )
    techniques: list[str] = []
    if config.augment_flip_h:
        techniques.append("fliph")
    if config.augment_flip_v:
        techniques.append("flipv")
    if config.augment_darker:
        techniques.append("darker")
    if config.augment_lighter:
        techniques.append("lighter")

    planned = []
    for src in base_files:
        for tech in techniques:
            target = config.images_dir / f"{src.stem}_{tech}{src.suffix}"
            if not target.exists():
                planned.append((src, tech, target))
    return planned


def apply_augmentation(src: Path, technique: str, brightness_factor: float) -> Image.Image:
    """Bir görüntüye tek bir augmentation tekniğini uygular."""
    img = Image.open(src).convert("RGB")
    if technique == "fliph":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if technique == "flipv":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if technique == "darker":
        return ImageEnhance.Brightness(img).enhance(1.0 - brightness_factor)
    if technique == "lighter":
        return ImageEnhance.Brightness(img).enhance(1.0 + brightness_factor)
    raise ValueError(f"Bilinmeyen augmentation tekniği: {technique}")


def run_augmentation_phase(config: PipelineConfig, planned: list) -> int:
    """Planlanan tüm augmentation dosyalarını üretir. Üretilen dosya sayısını döner."""
    logger.info("Augmentation planı: %d yeni dosya üretilecek (zaten var olanlar atlandı).", len(planned))
    if config.dry_run:
        for src, tech, target in planned[:10]:
            logger.info("  [dry-run] %s -> %s", src.name, target.name)
        if len(planned) > 10:
            logger.info("  ... ve %d dosya daha", len(planned) - 10)
        return len(planned)

    for src, tech, target in planned:
        img = apply_augmentation(src, tech, config.brightness_factor)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        img.save(tmp_path, format="JPEG" if target.suffix.lower() in (".jpg", ".jpeg") else "PNG", quality=95)
        tmp_path.rename(target)
    logger.info("Augmentation tamamlandı: %d dosya üretildi.", len(planned))
    return len(planned)


# ---------------------------------------------------------------------------
# Gemma çıktı ayrıştırma + YOLO dönüşümü
# ---------------------------------------------------------------------------


def parse_vlm_json(response: str) -> Optional[list]:
    """VLM yanıtından JSON dizisini çıkarır (fence/preamble tolere edilir)."""
    candidates = []
    fence = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", response, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    for opener, closer in (("[", "]"), ("{", "}")):
        start = response.find(opener)
        end = response.rfind(closer)
        if start != -1 and end > start:
            candidates.append(response[start:end + 1])
    for c in candidates:
        try:
            parsed = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return parsed.get("detections", [])
    return None


def normalize_class_name(raw) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    if raw in CLASS_NAME_TO_ID:
        return raw
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIASES.get(key)


def box_2d_to_yolo(box: list) -> Optional[tuple[float, float, float, float]]:
    """Gemma box_2d ([y1,x1,y2,x2], 0-1000 normalize, swap_axes=False - kalibre edildi)
    -> YOLO (cx, cy, w, h), 0-1 normalize.

    Görüntü piksel boyutuna ihtiyaç YOK: box_2d zaten görüntünün kendi
    boyutuna göre normalize edilmiş, 0-1'e /1000 ile doğrudan ölçeklenir.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        y1, x1, y2, x2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = max(0.0, min(x1, 1000.0)), max(0.0, min(x2, 1000.0))
    y1, y2 = max(0.0, min(y1, 1000.0)), max(0.0, min(y2, 1000.0))
    w, h = (x2 - x1) / 1000.0, (y2 - y1) / 1000.0
    if w <= 0 or h <= 0:
        return None
    cx, cy = (x1 + x2) / 2000.0, (y1 + y2) / 2000.0
    return (cx, cy, w, h)


def detections_from_raw(raw_list: list, confidence_threshold: float) -> list[DetectionRecord]:
    """Ham VLM tespit listesini DetectionRecord'lara çevirir (threshold uygulanır)."""
    records = []
    for det in raw_list:
        if not isinstance(det, dict):
            continue
        cname = normalize_class_name(det.get("label") or det.get("class"))
        if cname is None:
            continue
        yolo_box = box_2d_to_yolo(det.get("box_2d"))
        try:
            conf = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        records.append(DetectionRecord(
            class_name=cname,
            confidence=conf,
            reasoning=str(det.get("reasoning", "")),
            box_2d_raw=det.get("box_2d"),
            yolo_box=yolo_box,
            kept=(yolo_box is not None and conf >= confidence_threshold),
        ))
    return records


# ---------------------------------------------------------------------------
# Etiketleme (Faz 2)
# ---------------------------------------------------------------------------


def collect_images_to_label(config: PipelineConfig, planned_augments: Optional[list] = None) -> list[Path]:
    """Etiketlenecek görüntüleri döner (zaten .txt'si olanlar --force yoksa atlanır).

    Args:
        planned_augments: dry-run'da henüz diske YAZILMAMIŞ augment dosyalarının
            listesi (plan_augmentations çıktısı) - gerçek çalıştırmada None
            geçilir çünkü o an augment dosyaları zaten diskte mevcuttur.
    """
    existing = sorted(
        p for p in config.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    all_images = existing + [target for _, _, target in (planned_augments or [])]
    all_images = sorted(set(all_images), key=lambda p: p.name)

    if config.force:
        todo = all_images
    else:
        todo = [p for p in all_images if not (config.labels_dir / f"{p.stem}.txt").exists()]
    if config.limit is not None:
        todo = todo[: config.limit]
    return todo


def augmentation_type_of(image_path: Path) -> tuple[str, str]:
    """Dosya adından augmentation türünü ve kaynak (orijinal) adını çıkarır."""
    stem = image_path.stem
    for suf in AUG_SUFFIXES:
        if stem.endswith(suf):
            tech = suf.lstrip("_")
            source = stem[: -len(suf)] + image_path.suffix
            return tech, source
    return "original", image_path.name


def label_one_image(model, processor, config_obj, image_path: Path, cfg: PipelineConfig) -> ImageRecord:
    """Bir görüntüyü Gemma ile etiketler, YOLO .txt yazar, ImageRecord döner."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    aug_type, source_image = augmentation_type_of(image_path)
    start = time.time()

    try:
        prompt = apply_chat_template(processor, config_obj, GEMMA_PROMPT, num_images=1)
        result = generate(
            model, processor, prompt, image=str(image_path),
            max_tokens=cfg.max_tokens, temperature=cfg.temperature, verbose=False,
        )
        elapsed_ms = (time.time() - start) * 1000.0
        raw_list = parse_vlm_json(result.text)

        if raw_list is None:
            return ImageRecord(
                image_name=image_path.name, augmentation_type=aug_type, source_image=source_image,
                status="parse_hatasi", num_detections_raw=0, num_detections_kept=0,
                elapsed_ms=elapsed_ms, error_message=f"JSON parse edilemedi. Ham yanıt (ilk 200 kr.): {result.text[:200]!r}",
            )

        detections = detections_from_raw(raw_list, cfg.confidence_threshold)
        kept = [d for d in detections if d.kept]

        if not cfg.dry_run:
            lines = [
                f"{CLASS_NAME_TO_ID[d.class_name]} {d.yolo_box[0]:.6f} {d.yolo_box[1]:.6f} {d.yolo_box[2]:.6f} {d.yolo_box[3]:.6f}"
                for d in kept
            ]
            txt_path = cfg.labels_dir / f"{image_path.stem}.txt"
            tmp_path = txt_path.with_suffix(".txt.tmp")
            tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            tmp_path.rename(txt_path)

        return ImageRecord(
            image_name=image_path.name, augmentation_type=aug_type, source_image=source_image,
            status="ok", num_detections_raw=len(detections), num_detections_kept=len(kept),
            elapsed_ms=elapsed_ms, detections=[asdict(d) for d in detections],
        )
    except Exception as exc:  # noqa: BLE001 - tek görüntü hatası batch'i durdurmamalı
        elapsed_ms = (time.time() - start) * 1000.0
        logger.error("HATA (%s): %s: %s", image_path.name, type(exc).__name__, exc)
        return ImageRecord(
            image_name=image_path.name, augmentation_type=aug_type, source_image=source_image,
            status="hata", num_detections_raw=0, num_detections_kept=0,
            elapsed_ms=elapsed_ms, error_message=f"{type(exc).__name__}: {exc}",
        )


def run_labeling_phase(config: PipelineConfig, planned_augments: list) -> list[ImageRecord]:
    """Tüm etiketleme işini yapar (dry-run'da sadece plan gösterir)."""
    todo = collect_images_to_label(config, planned_augments if config.dry_run else None)
    existing_count = len(sorted(p for p in config.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))
    total_available = existing_count + (len(planned_augments) if config.dry_run else 0)
    logger.info(
        "Etiketleme planı: %d/%d görüntü işlenecek (%d zaten etiketli, atlanıyor).",
        len(todo), total_available, (total_available - len(todo)) if not config.force else 0,
    )

    if config.dry_run:
        return [
            ImageRecord(
                image_name=p.name, augmentation_type=augmentation_type_of(p)[0],
                source_image=augmentation_type_of(p)[1], status="dry_run_plan",
                num_detections_raw=0, num_detections_kept=0, elapsed_ms=0.0,
            )
            for p in todo
        ]

    if not todo:
        return []

    config.labels_dir.mkdir(parents=True, exist_ok=True)

    from mlx_vlm import load
    from mlx_vlm.utils import load_config as mlx_load_config

    logger.info("Model yükleniyor: %s", MODEL_ID)
    model, processor = load(MODEL_ID)
    config_obj = mlx_load_config(MODEL_ID)
    logger.info("Model yüklendi. Etiketleme başlıyor...")

    manifest_path = config.labels_dir.parent / f"{config.images_dir.name}_autolabel_manifest.jsonl"
    records: list[ImageRecord] = []
    t_start = time.time()

    with manifest_path.open("a", encoding="utf-8") as manifest_f:
        for i, image_path in enumerate(todo, start=1):
            record = label_one_image(model, processor, config_obj, image_path, config)
            records.append(record)
            manifest_f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            manifest_f.flush()

            if i % 10 == 0 or i == len(todo):
                elapsed = time.time() - t_start
                rate = elapsed / i
                remaining = rate * (len(todo) - i)
                logger.info(
                    "  [%d/%d] %s | tespit: %d/%d | %.0f sn geçti, ~%.0f sn kaldı",
                    i, len(todo), image_path.name, record.num_detections_kept, record.num_detections_raw,
                    elapsed, remaining,
                )

    return records


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------


def write_summary(config: PipelineConfig, records: list[ImageRecord], num_augmented: int) -> None:
    ok = [r for r in records if r.status == "ok"]
    parse_err = [r for r in records if r.status == "parse_hatasi"]
    other_err = [r for r in records if r.status == "hata"]
    zero_det = [r for r in ok if r.num_detections_kept == 0]

    class_counts = {"Healthy": 0, "Late_Blight": 0}
    for r in ok:
        for d in r.detections:
            if d["kept"]:
                class_counts[d["class_name"]] += 1

    mode_label = "DRY-RUN" if config.dry_run else "GERÇEK ÇALIŞTIRMA"
    logger.info("=" * 60)
    logger.info("ÖZET [%s]", mode_label)
    logger.info("=" * 60)
    logger.info("Yeni üretilen augment dosyası : %d", num_augmented)
    logger.info("İşlenen görüntü                : %d", len(records))
    logger.info("Başarılı                       : %d", len(ok))
    logger.info("JSON parse hatası               : %d", len(parse_err))
    logger.info("Diğer hata                      : %d", len(other_err))
    logger.info("Sıfır tespit (threshold sonrası): %d", len(zero_det))
    logger.info("Sınıf dağılımı (kept)          : %s", class_counts)

    if not config.dry_run and records:
        summary_path = config.labels_dir.parent / f"{config.images_dir.name}_autolabel_summary.json"
        csv_path = config.labels_dir.parent / f"{config.images_dir.name}_autolabel_manifest.csv"
        summary = {
            "olusturulma_zamani": datetime.now(timezone.utc).isoformat(),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
            "yeni_augment": num_augmented,
            "islenen": len(records),
            "basarili": len(ok),
            "parse_hatasi": len(parse_err),
            "diger_hata": len(other_err),
            "sifir_tespit": len(zero_det),
            "sinif_dagilimi": class_counts,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "augmentation_type", "source_image", "status", "num_detections_raw", "num_detections_kept", "elapsed_ms"])
            for r in records:
                writer.writerow([r.image_name, r.augmentation_type, r.source_image, r.status, r.num_detections_raw, r.num_detections_kept, f"{r.elapsed_ms:.0f}"])

        logger.info("Özet JSON : %s", summary_path)
        logger.info("CSV       : %s", csv_path)
        logger.info("JSONL (ham, her tespitin reasoning'i dahil): %s", config.labels_dir.parent / f"{config.images_dir.name}_autolabel_manifest.jsonl")


def run(config: PipelineConfig) -> int:
    if not config.images_dir.is_dir():
        logger.error("Config hatası: images_dir yok: %s", config.images_dir)
        return 2

    planned_augments = plan_augmentations(config)
    num_augmented = run_augmentation_phase(config, planned_augments)
    records = run_labeling_phase(config, planned_augments)
    write_summary(config, records, num_augmented)

    if any(r.status in ("parse_hatasi", "hata") for r in records):
        return 1
    return 0


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Faz 1+2: augmentation + Gemma ile otomatik YOLO etiketleme.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Etiketlenecek görüntülerin klasörü.")
    parser.add_argument("--no-flip-h", action="store_false", dest="flip_h", help="Horizontal flip'i kapat.")
    parser.add_argument("--no-flip-v", action="store_false", dest="flip_v", help="Vertical flip'i kapat.")
    parser.add_argument("--darker", action="store_true", help="Darker augmentation'ı aktive et.")
    parser.add_argument("--lighter", action="store_true", help="Lighter augmentation'ı aktive et.")
    parser.add_argument("--brightness-factor", type=float, default=0.12, help="Darker/lighter için oran (varsayılan: 0.12).")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="YOLO .txt'ye yazılacak min. confidence (varsayılan: 0.3).")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true", help="Dosya yazmadan/model yüklemeden sadece plan göster.")
    parser.add_argument("--force", action="store_true", help="Zaten etiketli görüntüleri de yeniden işle.")
    parser.add_argument("--limit", type=int, default=None, help="Sadece ilk N görüntüyü etiketle (test amaçlı).")
    parser.set_defaults(flip_h=True, flip_v=True)
    args = parser.parse_args()

    return PipelineConfig(
        images_dir=args.images_dir,
        augment_flip_h=args.flip_h,
        augment_flip_v=args.flip_v,
        augment_darker=args.darker,
        augment_lighter=args.lighter,
        brightness_factor=args.brightness_factor,
        confidence_threshold=args.confidence_threshold,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        force=args.force,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(run(parse_args()))
