# Pre-annotation prompt

| File | Used for |
|---|---|
| `prompt_gemma_tr.txt` | Gemma 4 models (pre-annotation of the HITL set and the Gemma candidates). Boxes as `box_2d`, `[y1, x1, y2, x2]`, normalised to 0–1000 |
| `prompt_qwen_tr.txt` | Qwen3-VL models. Identical except the output contract follows Qwen3-VL's documented convention (`bbox_2d`, `[x1, y1, x2, y2]`, same 0–1000 scale) |
| `prompt_en.md` | English translation, as printed in the paper (Fig. 2) |

The Turkish files are the exact strings given to the models (`GEMMA_PROMPT` in
`VLM_auto_annotation/auto_label.py`, `QWEN_PROMPT` in `konferans_makale_calismasi/araclar/vlm_kosu.py`).

## Source and licence

The four leaf-symptom statements follow:

> Ministry of Agriculture, Malawi. *Tomato late blight disease (Phytophthora infestans).* MaDiPHS project farmer
> factsheet, 2025. https://ckan.madiphs.org/dataset/tomato-late-blight-disease-phytophthora-infestans-english-version
> Licensed under CC BY-NC-SA 4.0.

The statements were translated into Turkish and adapted to a detection prompt. The differential-diagnosis rule
against powdery mildew, the task description and the output contract were written by the authors. As an adaptation
of CC BY-NC-SA 4.0 material, the prompt files in this folder are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
