"""Çevirme tutarlılığı bulgusunun sağlaması — SALT OKUMA.
H0: düşük tutarlılık bizim eksen yorumumuzdan (swap) kaynaklanıyor olabilir.
Kontroller: (a) eksen-takası hipotezi, (b) şans tabanı (farklı görüntüyle eşleştirme),
(c) geri-çevirmeden eşleştirme (model konumsal önsel mi basıyor?)."""
import json, random, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT = Path(__file__).resolve().parents[2] / "VLM_auto_annotation"
SETS = ["kaynak_a_images", "kaynak_b_images"]
def to_yolo(b, swap):
    a, b_, c, d = b
    y1, x1, y2, x2 = (b_, a, d, c) if swap else (a, b_, c, d)
    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
    return ((x1+x2)/2000, (y1+y2)/2000, (x2-x1)/1000, (y2-y1)/1000)
def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw, ih = max(0, min(ax2,bx2)-max(ax1,bx1)), max(0, min(ay2,by2)-max(ay1,by1))
    u = a[2]*a[3]+b[2]*b[3]-iw*ih
    return iw*ih/u if u > 0 else 0.0
def tp(ref, pred, t=0.5):
    if not ref or not pred: return 0
    m = np.array([[iou(r, p) for p in pred] for r in ref]); ri, ci = linear_sum_assignment(-m)
    return int(sum(m[i, j] >= t for i, j in zip(ri, ci)))
def unflip(b, aug):
    cx, cy, w, h = b
    return (1-cx, cy, w, h) if aug == "fliph" else (cx, 1-cy, w, h)
recs = [json.loads(l) | {"_set": s} for s in SETS for l in open(ROOT/f"{s}_autolabel_manifest.jsonl", encoding="utf-8")]
def gruplar(swap):
    g = defaultdict(dict)
    for r in recs:
        g[r["_set"]+"/"+r["source_image"]][r["augmentation_type"]] = [to_yolo(d["box_2d_raw"], swap) for d in r["detections"] if d.get("kept")]
    return [v for v in g.values() if {"original","fliph","flipv"} <= v.keys()]
def f1(pairs):
    t = sum(tp(a, b) for a, b in pairs); n = sum(len(a)+len(b) for a, b in pairs)
    return round(2*t/n, 4)
sonuc = {}
for swap in (False, True):
    G = gruplar(swap)
    for aug in ("fliph", "flipv"):
        sonuc[f"{'swap' if swap else 'no-swap'} · {aug} · geri-çevrilmiş"] = f1([(g["original"], [unflip(b, aug) for b in g[aug]]) for g in G])
        sonuc[f"{'swap' if swap else 'no-swap'} · {aug} · geri-çevirmeden"] = f1([(g["original"], g[aug]) for g in G])
G = gruplar(False); rng = random.Random(42); taban = []
for _ in range(20):
    perm = list(range(len(G))); rng.shuffle(perm)
    for aug in ("fliph", "flipv"):
        taban.append(f1([(G[i]["original"], [unflip(b, aug) for b in G[j][aug]]) for i, j in zip(range(len(G)), perm) if i != j]))
sonuc["şans tabanı (farklı görüntü, 20 permütasyon, seed 42) ort"] = round(statistics.mean(taban), 4)
sonuc["şans tabanı min-max"] = [min(taban), max(taban)]
# Kayıt defterindeki yolo_box ile yeniden hesaplanan no-swap aynı mı (dönüşüm doğrulaması)?
fark = sum(1 for r in recs for d in r["detections"] if d.get("kept") and max(abs(a-b) for a, b in zip(to_yolo(d["box_2d_raw"], False), d["yolo_box"])) > 1e-6)
sonuc["yolo_box ≠ no-swap dönüşümü (adet)"] = fark
OUT = Path(__file__).resolve().parent.parent / "veri" / "cevirme_saglama.json"
OUT.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(sonuc, ensure_ascii=False, indent=1))
