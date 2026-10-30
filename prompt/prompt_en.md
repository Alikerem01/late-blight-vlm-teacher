# Pre-annotation prompt (English translation)

Translated from the Turkish original (`prompt_gemma_tr.txt`). The numbering replaces the original bullet points.

---

You are a plant pathology expert specialised in tomato leaf diseases. Detect late blight (*Phytophthora infestans*)
symptoms on the tomato leaf in the image.

**LATE BLIGHT SYMPTOMS** (evaluate according to these criteria):
(1) Water-soaked leaf spots ranging from pale green to brownish-black. They grow rapidly and cover large areas of the
leaf. (2) A grey-white mould/sporangium layer on the underside of the leaf. (3) A ring of mould growth around larger
spots, forming in humid weather. (4) At later stages the leaf turns yellow, then brown, curls, shrivels and dies.

**DIFFERENTIAL DIAGNOSIS** (do not confuse with): The grey-white mould layer on the leaf underside can be confused
with powdery mildew. Distinguishing point: in late blight this layer is accompanied by water-soaked, greasy-looking
brown-black spots that spread rapidly. A white powdery appearance alone is not evidence of late blight.

**TASK:** Mark EVERY region in the image that shows late blight symptoms. Do not mark healthy regions. For each
detection give: label (always "Late_Blight"), box_2d, confidence (0.0–1.0) and a short reasoning.

**OUTPUT FORMAT** (return only a valid JSON array, no other text):

```json
[{"box_2d": [y1, x1, y2, x2], "label": "Late_Blight", "confidence": 0.85, "reasoning": "Water-soaked brown lesion with a yellowish halo at its edge"}]
```

**IMPORTANT:** box_2d coordinates are integers normalised to 0–1000, in [y1, x1, y2, x2] order. Mark only regions
with disease symptoms. Do not draw boxes around healthy leaves or regions. If the image shows no disease symptoms,
return an empty array []. Give low confidence to detections you are unsure about. Return only the JSON array, with
no explanation before or after it.

---

Licence: CC BY-NC-SA 4.0 (see `README.md` in this folder).
