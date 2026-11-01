# late-blight-vlm-teacher

Code and prompt for the paper *Choosing an On-Device VLM Teacher for YOLO Students: Label-Free Diagnostics and
Expert Verification for Greenhouse Tomato Late Blight* (ICIDST 2026).

On-device vision-language models (VLMs) pre-annotate late blight lesions from a factsheet-derived prompt, experts
verify the labels, and YOLO students learn from the selected teacher. The repository holds the pre-annotation
prompt, the VLM runs, the label-free diagnostics, the teacher selection against blind expert labels and the student
training and evaluation.

**Not included:** the greenhouse images, the expert labels and the annotation tool. The images are not publicly
available, so the scripts document the procedure but cannot be re-run without them.

Code comments and console messages are in Turkish.

The images come from two collections, one per expert who photographed the greenhouse. They are named neutrally
here: `kaynak_a` (44 images, file prefix `a_`) and `kaynak_b` (257 images, file prefix `b_`).

## Layout

| Path | Paper | What it does |
|---|---|---|
| `prompt/` | Sect. 3.2, Fig. 2 | The prompt as given to the models (Turkish) and its English translation |
| `VLM_auto_annotation/normalize_dataset.py` | Sect. 3.1 | HEIC to JPEG, EXIF orientation applied, sequential names, duplicate check |
| `VLM_auto_annotation/auto_label.py` | Sects. 3.2, 3.5 | Pre-annotation with 4-bit Gemma 4 E2B on the full-resolution originals, the source of the pre-annotations of the HITL set H |
| `konferans_makale_calismasi/araclar/vlm_kosu.py` | Sect. 3.3 | Teacher candidates under common running conditions (1280 px input) |
| `.../model_indir.sh`, `.../v_prec_kosu.sh` | Sect. 3.3 | Model download with SHA-256 check, runs of the Gemma 4 E2B precision conversions (the 8-bit one is the sixth candidate) |
| `.../cevirme_tutarliligi_vlm.py`, `.../etiketsiz_ek_analiz.py`, `.../vlm_istatistik.py`, `.../cevirme_saglama.py`, `.../v_prec_analiz.py` | Sects. 3.4, 4.1 | Label-free diagnostics: flip consistency, coordinate quantisation, the `confidence` field of the output, the 8-bit conversion |
| `.../orneklem_uret.py`, `.../sahne_turu.py`, `.../k_boyutu_benzetim.py` | Sects. 3.1, 3.5 | Random order of the originals (seed 42), scene type, size of the blind set by simulation |
| `.../uzman_vlm_karsilastir.py` | Sects. 3.7, 4.1–4.2 | Teacher selection on the blind set (paired bootstrap, Holm), expert agreement, corrections in the HITL set |
| `.../eslestirme_duyarlilik.py` | Sect. 3.7 | Re-runs every comparison with a matching that first maximises the number of pairs at or above the IoU threshold |
| `colab/` | Sect. 3.6 | Student packages and the Colab notebook (Ultralytics, three arms, seeds 42–44 at the n scale) |
| `.../ogrenci_degerlendir.py`, `.../gecikme_olc.py` | Sects. 4.1, 4.3 | Student evaluation (learnability check, threshold on validation, F1 and AP50 on the blind set), CPU latency |
| `.../kidemli_inceleme.py` | Sect. 4.3 | Sensitivity check with the senior expert's review as the reference |
| `.../tur4_sayilar.py` | Sect. 4 | Collects the reported numbers and table rows. The analysis records keep full precision and each reported value is rounded once, half up |

`...` stands for `konferans_makale_calismasi/araclar`. The scripts find each other through these relative paths and
expect the private data next to them (`VLM_auto_annotation/*_images`, `konferans_makale_calismasi/veri/`).

## Environment

- Apple Silicon (M3 and M5), macOS, Python 3.12 managed with [uv](https://docs.astral.sh/uv/).
- VLM runs: `mlx` 0.32.2, `mlx-vlm` 0.7.0, `transformers` 5.16.1, MLX conversions from Hugging Face:
  `mlx-community/gemma-4-e2b-it-4bit`, `gemma-4-e2b-it-8bit`, `gemma-4-e4b-it-4bit`, `gemma-4-12B-it-4bit`,
  `Qwen3-VL-4B-Instruct-4bit` and `Qwen3-VL-8B-Instruct-4bit`.
- Students: Ultralytics 8.4.144 on Google Colab (GPU), 640 px, one class.
- Seed 42 throughout.

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt
```

## Licence of the prompt

The prompt files in `prompt/` are derived from a farmer factsheet of the Ministry of Agriculture of Malawi (MaDiPHS
project), licensed CC BY-NC-SA 4.0, and are shared under the same licence. See `prompt/README.md`.
