"""Faz 0 — Ham görüntü klasörlerini VLM etiketleme için normalize eder.

Yapılanlar (her görüntü için):
  1. HEIC ise JPEG'e çevrilir (pillow-heif ile okunur).
  2. EXIF Orientation etiketi varsa piksel düzeyinde uygulanır (fiziksel
     döndürme) ve etiket kaldırılır — aksi halde EXIF'i dikkate almayan
     hiçbir tüketici (VLM inference, web arayüzü canvas'ı, YOLO eğitimi)
     görüntüyü doğru yönde göremez. Bu projede kaynak fotoğrafların
     %66'sında (198/301) Orientation=6 (90° döndürme gerekiyor) bulundu —
     bu adım atlanırsa etiketleme büyük ölçüde yanlış yönde yapılır.
  3. RGB'ye çevrilir, JPEG (quality=95) olarak kaydedilir.
  4. Kaynak klasöre göre önekli, sıralı, padded isim verilir (örn. a_0001.jpg).
  5. Kaynak dosyanın SHA-256'sı hesaplanır (tam eşleşen duplicate tespiti).

Orijinal kaynak klasörlere KESİNLİKLE dokunulmaz (okunur, yazılmaz/silinmez).
Çıktı ayrı bir klasöre (--output) yazılır. Two-phase atomic write (.tmp ->
rename) ile yarım kalan bir çalıştırma çıktı klasörünü bozmaz.

Kullanım:
    .venv/bin/python3 normalize_dataset.py --dry-run
    .venv/bin/python3 normalize_dataset.py

Exit code'lar:
    0 - Tüm görüntüler başarıyla işlendi.
    1 - En az bir görüntüde hata oluştu (diğerleri işlendi, rapora bakın).
    2 - Config/kaynak klasör hatası (hiçbir görüntü işlenmedi).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}
ORIENTATION_EXIF_TAG = 274

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceSpec:
    """Bir kaynak görüntü klasörü ve çıktıda kullanılacak dosya adı öneki."""

    directory: Path
    prefix: str


@dataclass(frozen=True)
class NormalizeConfig:
    """Faz 0 çalıştırma ayarları."""

    sources: tuple[SourceSpec, ...]
    output_dir: Path
    jpeg_quality: int = 95
    dry_run: bool = False


@dataclass
class FileRecord:
    """Tek bir görüntü için manifest/CSV satırı."""

    new_name: str
    source_folder: str
    original_filename: str
    original_ext: str
    original_size_bytes: int
    original_sha256: str
    exif_orientation_before: Optional[int]
    final_width: int
    final_height: int
    output_size_bytes: int
    status: str  # "ok" | "hata"
    error_message: str = ""


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        directory=Path(__file__).resolve().parent / "kaynak_a",
        prefix="a",
    ),
    SourceSpec(
        directory=Path(__file__).resolve().parent / "kaynak_b",
        prefix="b",
    ),
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "greenhouse_images"


def collect_source_files(spec: SourceSpec) -> list[Path]:
    """Bir kaynak klasördeki görüntü dosyalarını deterministik sırayla döner."""
    if not spec.directory.is_dir():
        raise FileNotFoundError(f"Kaynak klasör yok: {spec.directory}")
    return sorted(
        p for p in spec.directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_one_image(
    src_path: Path,
    new_name: str,
    output_dir: Path,
    jpeg_quality: int,
    dry_run: bool,
) -> FileRecord:
    """Bir görüntüyü okur, EXIF-transpose uygular, JPEG olarak yazar.

    Args:
        src_path: Kaynak dosya (orijinal, değiştirilmez).
        new_name: Çıktı dosya adı (örn. "a_0001.jpg").
        output_dir: Çıktı klasörü.
        jpeg_quality: JPEG kayıt kalitesi (0-100).
        dry_run: True ise dosya YAZILMAZ, sadece analiz yapılır.

    Returns:
        FileRecord: işlem sonucu (başarı/hata dahil).
    """
    original_bytes = src_path.read_bytes()
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()

    try:
        with Image.open(src_path) as img:
            orientation = img.getexif().get(ORIENTATION_EXIF_TAG)
            transposed = ImageOps.exif_transpose(img)
            rgb = transposed.convert("RGB")
            final_width, final_height = rgb.size

            output_path = output_dir / new_name
            if dry_run:
                output_size_bytes = 0
            else:
                tmp_path = output_dir / f"{new_name}.tmp"
                rgb.save(tmp_path, format="JPEG", quality=jpeg_quality)
                tmp_path.rename(output_path)  # atomic (aynı dosya sistemi içinde)
                output_size_bytes = output_path.stat().st_size

        return FileRecord(
            new_name=new_name,
            source_folder=src_path.parent.name,
            original_filename=src_path.name,
            original_ext=src_path.suffix.lower(),
            original_size_bytes=len(original_bytes),
            original_sha256=original_sha256,
            exif_orientation_before=orientation,
            final_width=final_width,
            final_height=final_height,
            output_size_bytes=output_size_bytes,
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001 - tek dosya hatası tüm batch'i durdurmamalı
        logger.error("HATA (%s): %s: %s", src_path.name, type(exc).__name__, exc)
        return FileRecord(
            new_name=new_name,
            source_folder=src_path.parent.name,
            original_filename=src_path.name,
            original_ext=src_path.suffix.lower(),
            original_size_bytes=len(original_bytes),
            original_sha256=original_sha256,
            exif_orientation_before=None,
            final_width=0,
            final_height=0,
            output_size_bytes=0,
            status="hata",
            error_message=f"{type(exc).__name__}: {exc}",
        )


def find_exact_duplicates(records: list[FileRecord]) -> dict[str, list[str]]:
    """SHA-256'sı aynı olan (byte-identical) kaynak dosyaları gruplar."""
    by_hash: dict[str, list[str]] = {}
    for r in records:
        by_hash.setdefault(r.original_sha256, []).append(r.new_name)
    return {h: names for h, names in by_hash.items() if len(names) > 1}


def run(config: NormalizeConfig) -> int:
    """Tüm kaynakları işler, manifest/CSV/log yazar, özet raporu basar.

    Returns:
        Exit code (0=tam başarı, 1=kısmi hata, 2=config hatası).
    """
    try:
        per_source_files = {spec: collect_source_files(spec) for spec in config.sources}
    except FileNotFoundError as exc:
        logger.error("Config hatası: %s", exc)
        return 2

    total_files = sum(len(files) for files in per_source_files.values())
    if total_files == 0:
        logger.error("Hiçbir kaynak klasörde görüntü bulunamadı.")
        return 2

    if not config.dry_run:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[FileRecord] = []
    for spec, files in per_source_files.items():
        logger.info("%s: %d görüntü işlenecek (önek: %s_)", spec.directory.name, len(files), spec.prefix)
        for i, src_path in enumerate(files, start=1):
            new_name = f"{spec.prefix}_{i:04d}.jpg"
            record = process_one_image(
                src_path, new_name, config.output_dir, config.jpeg_quality, config.dry_run,
            )
            records.append(record)

    error_records = [r for r in records if r.status == "hata"]
    ok_records = [r for r in records if r.status == "ok"]
    duplicates = find_exact_duplicates(records)
    orientation_counts: dict[str, int] = {}
    for r in ok_records:
        key = str(r.exif_orientation_before)
        orientation_counts[key] = orientation_counts.get(key, 0) + 1
    heic_count = sum(1 for r in ok_records if r.original_ext == ".heic")

    mode_label = "DRY-RUN (hiçbir dosya yazılmadı)" if config.dry_run else "GERÇEK ÇALIŞTIRMA"
    logger.info("=" * 60)
    logger.info("ÖZET [%s]", mode_label)
    logger.info("=" * 60)
    logger.info("Toplam görüntü         : %d", total_files)
    logger.info("Başarılı               : %d", len(ok_records))
    logger.info("Hatalı                 : %d", len(error_records))
    logger.info("HEIC -> JPEG dönüştürülen: %d", heic_count)
    logger.info("EXIF Orientation dağılımı (None=etiket yok, 1=düz): %s", orientation_counts)
    logger.info("Tam eşleşen (byte-identical) duplicate grup sayısı: %d", len(duplicates))
    if duplicates:
        for h, names in duplicates.items():
            logger.warning("  DUPLICATE (hash %s...): %s", h[:12], names)
    if error_records:
        for r in error_records:
            logger.warning("  HATA: %s (%s) -> %s", r.original_filename, r.source_folder, r.error_message)

    if not config.dry_run:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = config.output_dir.parent / "greenhouse_images_manifest.json"
        csv_path = config.output_dir.parent / "greenhouse_images_manifest.csv"

        manifest = {
            "olusturulma_zamani": datetime.now(timezone.utc).isoformat(),
            "toplam": total_files,
            "basarili": len(ok_records),
            "hatali": len(error_records),
            "heic_donusturulen": heic_count,
            "exif_orientation_dagilimi": orientation_counts,
            "duplicate_gruplar": duplicates,
            "kayitlar": [asdict(r) for r in records],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))

        logger.info("Manifest yazıldı: %s", manifest_path)
        logger.info("CSV yazıldı     : %s", csv_path)

    return 1 if error_records else 0


def parse_args() -> NormalizeConfig:
    parser = argparse.ArgumentParser(description="Faz 0: ham görüntüleri VLM etiketleme için normalize eder.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Çıktı klasörü (varsayılan: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG kayıt kalitesi (varsayılan: 95)")
    parser.add_argument("--dry-run", action="store_true", help="Dosya yazmadan sadece analiz/rapor üret")
    args = parser.parse_args()
    return NormalizeConfig(
        sources=DEFAULT_SOURCES,
        output_dir=args.output,
        jpeg_quality=args.jpeg_quality,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(run(parse_args()))
