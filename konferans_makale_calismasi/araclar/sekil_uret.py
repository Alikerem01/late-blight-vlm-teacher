"""Makale şekillerini veriden üretir (İngilizce etiketler) — proje verisine SALT OKUMA.

Çıktı: makale/figures/{fig_pipeline.pdf, fig_examples.pdf, fig_diagnostics.pdf,
fig_training_vlm_labels.pdf, fig_tool.jpg}

Genişlik LNCS metin genişliği (12,2 cm ≈ 4,8 in). Yazı tipleri PDF'e gömülür
(Type 42). Her sayı veri/ altındaki JSON/CSV'den okunur, elle yazılmaz.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

PROJE = Path(__file__).resolve().parents[2]
VLM = PROJE / "VLM_auto_annotation"
KOK = Path(__file__).resolve().parent.parent
VERI, CIKTI = KOK / "veri", KOK / "makale" / "figures"
SETS = ["kaynak_a_images", "kaynak_b_images"]
W_IN = 4.8
KIRMIZI, CAMGOBEGI = "#d62728", "#17becf"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})


def manifest() -> dict[str, tuple[str, dict]]:
    """image_name -> (set, kayıt)."""
    out = {}
    for s in SETS:
        with open(VLM / f"{s}_autolabel_manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["image_name"]] = (s, r)
    return out


# ---------------------------------------------------------------- Şekil 1
def fig_pipeline(ortak: bool = False, h2: bool = False) -> None:
    """Şekil 1: öğretmen–öğrenci hattı.

    `ortak` (karar D1: ortak okuma kümesi O) ve `h2` (karar D5: ikinci HITL kümesi) uzman
    kutusuna ilgili kümeyi ekler. Uzmandan öğrenciye giden ok (kör kümede test) alttan
    dolaşır ve şeklin içinde etiketlidir (hakem notu, Tur 3).
    """
    uzman = ["blind set K (60):", "teacher selection, test"]
    if ortak:
        uzman.append("overlap set O (20)")
    uzman.append("HITL set H (150)")
    if h2:
        uzman.append("HITL set H2 (30)")
    ek = max(0, len(uzman) - 3)
    fig, ax = plt.subplots(figsize=(W_IN, 2.45 + 0.13 * ek))
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(-0.17 - 0.06 * ek, 1)
    ax.axis("off")
    bw, bh = 0.226, 0.34
    bh2 = 0.30 + 0.075 * ek

    def kutu(x, y, baslik, govde, h=bh):
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - h / 2), bw, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    linewidth=0.8, edgecolor="#333333", facecolor="#f4f4f4"))
        ax.text(x, y + h / 2 - 0.04, baslik, ha="center", va="top", fontsize=5.05, fontweight="bold")
        ax.text(x, y + h / 2 - 0.115, govde, ha="center", va="top", fontsize=4.95, linespacing=1.35)

    def ok(x1, y1, x2, y2, etiket=None, dx=0.0, **kw):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6.5, linewidth=0.8,
                                     color="#333333", shrinkA=0, shrinkB=0, **kw))
        if etiket:
            ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, etiket, ha="left", va="center", fontsize=5.1, style="italic")

    xs, y1, y0 = [0.115, 0.372, 0.628, 0.885], 0.605, 0.99
    y2 = 0.175 - 0.0375 * ek
    kutu(xs[0], y1, "Greenhouse images", "301 originals\n112 detached leaves\n189 plants in situ")
    kutu(xs[1], y1, "Preprocessing", "EXIF rotation\nHEIC\u2192JPEG")
    kutu(xs[2], y1, "VLM teacher", "6 on-device candidates:\nGemma 4 E2B, 4/8-bit\nGemma 4 E4B, 12B\nQwen3-VL 4B, 8B")
    kutu(xs[3], y1, "Label-free checks", "flip consistency\ncoordinate grid\nconfidence values")
    kutu(xs[2], y2, "Expert verification", "\n".join(uzman), h=bh2)
    kutu(xs[1], y2, "YOLO student", "trained on teacher\nlabels, scored against\nteacher and experts", h=bh2)
    kutu(xs[0], y2, "Decision support", "for an agricultural\nrobot (future work)", h=bh2)
    ax.text(xs[2], y0, "Extension factsheet (Malawi): leaf symptoms", ha="center", va="center", fontsize=5.3,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#777777", lw=0.6))
    g = 0.006
    for a, b in ((0, 1), (1, 2), (2, 3)):
        ok(xs[a] + bw / 2 + g, y1, xs[b] - bw / 2 - g, y1)
    ok(xs[2], y0 - 0.05, xs[2], y1 + bh / 2 + g, "prompt", dx=0.012)
    ok(xs[2], y1 - bh / 2 - g, xs[2], y2 + bh2 / 2 + g, "pre-annotations", dx=0.012)
    ok(xs[2] - bw / 2 + 0.01, y1 - bh / 2 - g, xs[1] + bw / 2 - 0.01, y2 + bh2 / 2 + g)
    ax.text((xs[1] + xs[2]) / 2 - 0.035, (y1 + y2 + (bh2 - bh) / 2) / 2 + 0.02, "teacher labels", ha="right",
            va="center", fontsize=5.1, style="italic")
    alt = y2 - bh2 / 2 - g
    ok(xs[2], alt, xs[1], alt, connectionstyle="bar,fraction=-0.22")
    ax.text((xs[1] + xs[2]) / 2, alt - 0.085, "blind-set test", ha="center", va="top", fontsize=5.1, style="italic")
    ok(xs[1] - bw / 2 - g, y2, xs[0] + bw / 2 + g, y2)
    fig.savefig(CIKTI / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- Şekil 2
def fig_examples(man: dict) -> None:
    ornek = [("b_0191.jpg", "(a)"), ("b_0150.jpg", "(b)"), ("a_0014.jpg", "(c)")]
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, 2.35), gridspec_kw={"wspace": 0.04})
    for ax, (ad, harf) in zip(axes, ornek):
        s, r = man[ad]
        im = Image.open(VLM / s / ad).convert("RGB")
        im.thumbnail((900, 900))
        w, h = im.size
        ax.imshow(im)
        for d in r["detections"]:
            cx, cy, bw, bh = d["yolo_box"]
            ax.add_patch(Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h, fill=False,
                                   edgecolor=KIRMIZI, linewidth=1.3))
        _, rf = man[ad[:-4] + "_fliph.jpg"]
        for d in rf["detections"]:
            cx, cy, bw, bh = d["yolo_box"]
            cx = 1 - cx
            ax.add_patch(Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h, fill=False,
                                   edgecolor=CAMGOBEGI, linewidth=1.2, linestyle=(0, (3, 2))))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(harf, fontsize=7.5, labelpad=2)
    fig.savefig(CIKTI / "fig_examples.pdf", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- Şekil 3
def fig_diagnostics(man: dict) -> None:
    kutular = [d for _, r in man.values() for d in r["detections"] if d.get("kept")]
    guven = [round(float(d["confidence"]), 2) for d in kutular]
    koord = [int(round(v)) for d in kutular for v in d["box_2d_raw"]]
    sag = json.loads((VERI / "cevirme_saglama.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, 1.55), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1, 1.25]})

    ax = axes[0]
    c = Counter(guven)
    xs = sorted(c)
    ax.bar(xs, [c[x] for x in xs], width=0.008, color="#555555")
    ax.set_xlim(0.70, 1.0)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Verbalised confidence")
    ax.set_ylabel("Boxes")
    ax.set_title("(a)", loc="left")

    ax = axes[1]
    m = Counter(v % 50 for v in koord)
    ax.bar(range(50), [100 * m[i] / len(koord) for i in range(50)], width=0.85, color="#555555")
    ax.axhline(100 / 50, color=KIRMIZI, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coordinate value mod 50")
    ax.set_ylabel("% of coordinates")
    ax.set_title("(b)", loc="left")

    ax = axes[2]
    etiket = ["Observed", "No\ninverse", "Swapped\naxes"]
    anahtar = ["no-swap · {a} · geri-çevrilmiş", "no-swap · {a} · geri-çevirmeden", "swap · {a} · geri-çevrilmiş"]
    x = range(len(etiket))
    yat = [sag[k.format(a="fliph")] for k in anahtar]
    dik = [sag[k.format(a="flipv")] for k in anahtar]
    ax.bar([i - 0.19 for i in x], yat, width=0.36, color="#444444", label="horizontal")
    ax.bar([i + 0.19 for i in x], dik, width=0.36, color="#aaaaaa", label="vertical")
    lo, hi = sag["şans tabanı min-max"]
    ax.axhspan(lo, hi, color=KIRMIZI, alpha=0.15, linewidth=0)
    ax.axhline(sag["şans tabanı (farklı görüntü, 20 permütasyon, seed 42) ort"], color=KIRMIZI, linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(etiket)
    ax.set_ylabel("F1@0.5 (flip)")
    ax.set_ylim(0, 0.25)
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.06, 1.09), handlelength=1.0, ncol=2, columnspacing=0.6)
    ax.set_title("(c)", loc="left")
    fig.savefig(CIKTI / "fig_diagnostics.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- Şekil 4
def fig_training() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 1.6), gridspec_kw={"wspace": 0.35})
    renk = {"yolo11n": "#1f77b4", "yolo11s": "#ff7f0e", "yolo11m": "#2ca02c"}
    for m, rk in renk.items():
        with open(PROJE / "colab" / "sonuclar" / "mildiyo" / f"{m}_640px_150e" / "results.csv", encoding="utf-8") as f:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        ep = [int(float(r["epoch"])) for r in rows]
        ad = m.replace("yolo", "YOLO")
        axes[0].plot(ep, [float(r["train/box_loss"]) for r in rows], color=rk, linewidth=0.9, label=f"{ad} train")
        axes[0].plot(ep, [float(r["val/box_loss"]) for r in rows], color=rk, linewidth=0.9, linestyle="--", label=f"{ad} val")
        axes[1].plot(ep, [float(r["metrics/mAP50(B)"]) for r in rows], color=rk, linewidth=0.9, label=ad)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Box loss")
    axes[0].set_title("(a)", loc="left")
    axes[0].set_ylim(1.6, 4.2)
    axes[0].legend(frameon=False, ncol=3, fontsize=5.2, handlelength=1.4, columnspacing=0.6, loc="upper right")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation mAP50")
    axes[1].set_ylim(0, 0.12)
    axes[1].set_title("(b)", loc="left")
    axes[1].legend(frameon=False, fontsize=6)
    fig.savefig(CIKTI / "fig_training_vlm_labels.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- Şekil 5
def fig_tool() -> None:
    kaynak = PROJE / "izleyici_sunum" / "donem_1_sunumu" / "gorseller" / "etiketleme_arayuzu_ana_ekran.jpg"
    im = Image.open(kaynak).convert("RGB")
    olcek = im.width / 1671  # koordinatlar 1671 px genişlikteki görünüme göre ölçüldü
    d = ImageDraw.Draw(im)
    zemin = im.getpixel((int(1300 * olcek), int(12 * olcek)))
    # kişisel bilgi: kullanıcı adı + profil fotoğrafı (üst çubuk, sağ)
    d.rectangle([int(1362 * olcek), int(6 * olcek), int(1562 * olcek), int(50 * olcek)], fill=zemin)
    im.thumbnail((1800, 1800))
    im.save(CIKTI / "fig_tool.jpg", quality=88, optimize=True)


# ---------------------------------------------------------------- Şekil: beş VLM, etiketsiz
def fig_vlm_etiketsiz() -> None:
    ozet = json.loads((VERI / "kor_kume_vlm_etiketsiz_ozet.json").read_text(encoding="utf-8"))
    cev = json.loads((VERI / "kor_kume_cevirme_tutarliligi.json").read_text(encoding="utf-8"))
    sira = ["gemma4-e2b", "gemma4-e4b", "gemma4-12b", "qwen3vl-4b", "qwen3vl-8b"]
    ad = {"gemma4-e2b": "E2B", "gemma4-e4b": "E4B", "gemma4-12b": "12B", "qwen3vl-4b": "Q4B", "qwen3vl-8b": "Q8B"}
    renk = ["#8c2d04", "#d94801", "#fd8d3c", "#08519c", "#4292c6"]
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, 1.75), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1, 1.05]})
    x = range(len(sira))
    ax = axes[0]
    ax.bar(x, [cev[m]["F1_05"] for m in sira], color=renk, width=0.7)
    for i, m in enumerate(sira):
        ax.plot([i - 0.35, i + 0.35], [cev[m]["sans_F1_05_ort"]] * 2, color="black", linewidth=0.9)
    ax.set_xticks(list(x)); ax.set_xticklabels([ad[m] for m in sira], fontsize=5.8)
    ax.set_ylabel("Flip consistency F1@0.5"); ax.set_ylim(0, 0.7); ax.set_title("(a)", loc="left")
    ax = axes[1]
    ax.bar(x, [100 * ozet["ozet"][m]["kat50_orani"] for m in sira], color=renk, width=0.7)
    ax.axhline(100 * 21 / 1001, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x)); ax.set_xticklabels([ad[m] for m in sira], fontsize=5.8)
    ax.set_ylabel("% coordinates on 50-grid"); ax.set_ylim(0, 75); ax.set_title("(b)", loc="left")
    ax = axes[2]
    n = len(sira)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(sira):
        for j, b in enumerate(sira):
            if i < j:
                v = ozet["modeller_arasi_uyum_F1_05"][f"{a}~{b}"]
                mat[i, j] = mat[j, i] = v
    cmap = matplotlib.colormaps["Greys"].copy()
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(mat), vmin=0, vmax=1, cmap=cmap)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=4.8, color="white" if mat[i, j] > 0.5 else "black")
    kisa = [ad[m] for m in sira]
    ax.set_xticks(range(n)); ax.set_xticklabels(kisa, fontsize=5.8); ax.set_yticks(range(n)); ax.set_yticklabels(kisa, fontsize=5.8)
    for sp in ax.spines.values():
        sp.set_visible(True)
    ax.set_title("(c) pairwise agreement F1@0.5", loc="left")
    fig.savefig(CIKTI / "fig_vlm_label_free.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--yalniz-hat", action="store_true", help="yalnız Şekil 1'i üret")
    ap.add_argument("--ortak", action="store_true", help="karar D1: ortak okuma kümesi O")
    ap.add_argument("--h2", action="store_true", help="karar D5: ikinci HITL kümesi H2")
    args = ap.parse_args()
    CIKTI.mkdir(parents=True, exist_ok=True)
    fig_pipeline(args.ortak, args.h2)
    if args.yalniz_hat:
        return 0
    man = manifest()
    fig_examples(man)
    fig_diagnostics(man)
    fig_training()
    fig_tool()
    fig_vlm_etiketsiz()
    for p in sorted(CIKTI.iterdir()):
        print(f"{p.name:32s} {p.stat().st_size / 1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
