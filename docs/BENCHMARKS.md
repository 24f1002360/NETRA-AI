# BENCHMARKS.md
Owner: Anshika Maurya (Clinical XAI & Validation Lead)
Single source of truth for every number in the deck/report. If it isn't
here, it doesn't go on a slide. Regenerate with: `python eval/run_all.py`
Last updated: 2026-09-01 · commit: e80695f · **numbers frozen: NO (n=3, real pipeline, predicted lesion masks)**
---
## 1. Grading (Kanchan's EfficientNet-B0)
Final model: EfficientNet-B0, 384×384 input, NETRA/Muskan preprocessing.
Best checkpoint: Epoch 5 (best validation QWK; QWK decreased after epoch 5).
Checkpoint file: `netra_dr_effb0_muskan_preproc.pth` (46.41 MB)
Target layer for Grad-CAM: `features.8` (confirmed by inspecting checkpoint
state_dict — standard torchvision `efficientnet_b0`).

| Metric | Value | Split | n |
|---|---|---|---|
| Quadratic weighted κ | 0.8544 | Validation | TBD |
| Validation accuracy | 77.35% | Validation | TBD |
| Validation loss | 0.8283 | Validation | TBD |
| Referable DR (grade ≥2) sensitivity | TBD | APTOS held-out | TBD |
| Referable DR (grade ≥2) specificity | TBD | APTOS held-out | TBD |
| External κ | TBD | Messidor-2 / IDRiD held-out | TBD |
| κ drop (internal → external) | TBD | — | — |
| GPU inference latency | 3.208 ms/image | — | — |
| CPU inference latency | 168.145 ms/image (batch=16, workers=0) | PHC deployment | — |

ROC: `eval/results/grading_roc.png` · Calibration: `eval/results/grading_calibration.png`

**Why still TBD:** `eval/run_all.py::run_grading_eval()` is currently a
stub (prints a message, computes nothing). QWK/sensitivity/specificity
need a labeled held-out set (e.g. APTOS `train.csv` with ground-truth
`diagnosis` column) to compare predictions against — the 3 repo fixtures
have no ground-truth labels, so nothing can be computed from them alone.

## 2. Segmentation (Kanchan's U-Net)
| Lesion class | Dice | IoU |
|---|---|---|
| Microaneurysms | TBD | TBD |
| Haemorrhages | TBD | TBD |
| Hard exudates | TBD | TBD |
| Soft exudates | TBD | TBD |
n = 27 (IDRiD test split). Microaneurysm Dice expected low — report honestly.

Kanchan's own notebook numbers (V8 official test, not yet independently
re-verified by this eval/ harness): MA=0.0152, HE=0.2349, EX=0.5209,
SE=0.4831, mean=0.3136.

**Why still TBD:** same reason as Section 1 — `run_segmentation_eval()`
is a stub; needs IDRiD test-split ground-truth masks to compute Dice/IoU
independently.

## 3. XAI
| Metric | Value |
|---|---|
| Deletion AUC | 0.1917 (n=3, mean, real pipeline) |
| Insertion AUC | 0.5683 (n=3, mean, real pipeline) |
| Lesion-localisation hit rate | 0.0% (n=3, predicted masks not ground truth) |
| Grad-CAM vs Grad-CAM++ chosen | TBD (~20-image comparison, not yet run) |

### Real-pipeline verification (n=3, all repo fixtures)

Ran `explain()` end-to-end via `eval/run_all.py` against the real
EfficientNet-B0 checkpoint (`features.8`) and real DRSegmenter (4-channel
lesion masks):

| Fixture | Grade | Confidence | Guard | Agreement | Del AUC | Ins AUC |
|---|---|---|---|---|---|---|
| good.png | 0 (No DR) | 0.980 | OK | n/a (no lesions to compare) | 0.2481 | 0.9086 |
| blurry.png | 0 (No DR) | 0.414 | LOW_AGREEMENT | 0.107 | 0.1553 | 0.3410 |
| severe.png | 4 (PDR) | 0.527 | LOW_AGREEMENT | 0.094 | 0.1716 | 0.4553 |

**Bug found and fixed:** `cam_lesion_agreement()` originally returned
`0.0` (hard disagreement) whenever the segmented lesion mask was
genuinely empty (e.g. a real No-DR image with nothing to segment) —
this looked like guard failure but was actually "nothing to compare
against." Fixed to return `None` (not evaluable) in that case; `good.png`
now correctly reports guard `OK` instead of a false `LOW_AGREEMENT`.

**On blurry.png / severe.png:** the low agreement (9–11%) is a genuine
finding, not a bug — it's consistent with Kanchan's own reported weak
segmentation performance (microaneurysm Dice = 0.0152). Two independent
models disagreeing on lesion location is exactly the signal this guard
was built to catch, and it's catching it.

n=3 is a plumbing check, not a statistically meaningful sample. A larger
run (~20+ images per the guide) is needed before these percentages go
on a deck slide as a real result.

## 4. Guard trigger rates (n=3, real pipeline)
| Guard | Trigger rate | On inspection |
|---|---|---|
| OK | 33.3% (1/3) | good.png — No-DR image, no lesions to evaluate agreement against |
| CAM_OFF_RETINA | 0.0% (0/3) | — |
| LOW_AGREEMENT | 66.7% (2/3) | blurry.png, severe.png — genuine grading/segmentation disagreement, consistent with Kanchan's reported low segmentation Dice |

n=3 is too small to be a real trigger-rate estimate — need ~20+ images
before this is deck-ready.

## 5. Clinical alignment
See `docs/CLINICAL_ALIGNMENT.md` for the full ICDR grade → NETRA label →
routing action table, including the guard-status override rules.

## 6. Limitations (for the honest-limitations slide)
- Trained on 3,662 APTOS images; external κ drops by TBD on Messidor-2.
- Segmentation trained on 54 annotated images (IDRiD), n=27 test split.
- Microaneurysm Dice is low (0.0152 per Kanchan); exudates/haemorrhages
  favoured by construction — reflected directly in the LOW_AGREEMENT
  guard firing on real images above.
- CDR-based glaucoma flagging is a screening trigger, not a diagnosis.
- No prospective clinical validation.
- Grading and segmentation accuracy metrics (κ, sensitivity,
  specificity, Dice, IoU) are still TBD — blocked on a labeled held-out
  set, not yet run.
- All XAI numbers above are preliminary (n=3 fixtures) — not final.
- Next steps: run grading/segmentation eval against labeled held-out
  data; run guard trigger rates on ~20+ images; compare Grad-CAM vs
  Grad-CAM++.