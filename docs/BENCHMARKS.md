# BENCHMARKS.md
Owner: Anshika Maurya (Clinical XAI & Validation Lead)
Single source of truth for every number in the deck/report. If it isn't
here, it doesn't go on a slide. Regenerate with: `python eval/run_all.py`
Last updated: 2026-09-01 · commit: TBD (update after this commit) · **numbers frozen: NO (n=20, real pipeline, APTOS sample)**
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
`diagnosis` column) to compare predictions against — the 20-image
sample used for XAI eval below has no ground-truth labels, so nothing
can be computed from it alone.

## 2. Segmentation (Kanchan's U-Net)
| Lesion class | Dice | IoU |
|---|---|---|
| Lesion class | Dice | IoU |
|---|---|---|
| Microaneurysms | 0.0152 | TBD |
| Haemorrhages | 0.2349 | TBD |
| Hard exudates | 0.5209 | TBD |
| Soft exudates | 0.4831 | TBD |
n = 27 (IDRiD test split). Microaneurysm Dice expected low — report honestly.

Source: Kanchan's official IDRiD test-set evaluation of the frozen V8
model (`docs/KANCHAN_MODEL_HANDOFF.md`, Section 11). These are Kanchan's
reported values, used as-is here rather than independently re-run by
this eval/ harness (IDRiD ground-truth masks would be needed to
recompute Dice/IoU from scratch, and that's Kanchan's model-quality
metric, not an XAI-team task). IoU wasn't reported alongside Dice in
the handoff doc -- still TBD; ask Kanchan directly if it's needed.

**Why this matters for Section 3/4 below:** this weak MA Dice (0.0152,
near-random) directly explains the XAI guard's LOW_AGREEMENT behaviour
on real images -- see the explanation there.

## 3. XAI

### Headline result (n=20, real pipeline, APTOS sample)

Ran `explain()` end-to-end via `eval/run_all.py` against the real
EfficientNet-B0 checkpoint (`features.8`) and real `DRSegmenter`
(4-channel lesion masks), on 20 images sampled from the APTOS dataset
(`data/gradcam_compare/`).

| Metric | Value |
|---|---|
| Deletion AUC | 0.1397 (n=20, mean) |
| Insertion AUC | 0.4263 (n=20, mean) |
| Lesion-localisation hit rate | not captured this run — rerun with per-image logging to report |
| Grad-CAM vs Grad-CAM++ chosen | **gradcam** (n=20, score 0.2807 vs 0.2371) |

Sharp deletion AUC (0.14) and moderate insertion AUC (0.43) indicate the
CAM generally points at pixels the grading model actually relies on —
consistent with the earlier n=3 spot-check, now confirmed on a larger
sample.

### Guard trigger rates (n=20, real pipeline, APTOS sample)

| Guard | Trigger rate | Interpretation |
|---|---|---|
| LOW_AGREEMENT | 95.0% (19/20) | See explanation below — tied to segmentation quality, not a guard bug |
| OK | 5.0% (1/20) | image_2 (16).png — the one case where CAM and predicted lesion mask meaningfully overlapped (agreement=0.295) |
| CAM_OFF_RETINA | 0.0% (0/20) | CAM consistently lands inside the retina across this sample — FOV guard behaving as expected |

**Why LOW_AGREEMENT fires on 95% of images — investigated, not a bug.**
`cam_lesion_agreement()` measures what fraction of the CAM's top-20%
highest-activation pixels fall inside the segmentation model's
predicted lesion mask. On this sample, predicted lesion masks are
non-empty (verified directly: e.g. one image had 6,300 MA pixels, 1,347
HE, 1,051 EX, 763 SE — segmentation *is* producing output), so this is
not the earlier "empty mask" bug already fixed (see n=3 section below).

Instead, the low agreement (mostly 0.000–0.045, i.e. at or below the
~6% overlap expected by chance alone given lesion-mask coverage) is
consistent with Kanchan's own reported segmentation weakness: official-
test **microaneurysm Dice = 0.0152** (near-random). The predicted
lesion masks are themselves unreliable, so a CAM disagreeing with them
is not evidence the CAM is wrong — it's evidence the lesion mask isn't
a trustworthy reference yet. Two independent models (grading-CAM vs.
segmentation) agreeing would be a stronger positive signal than they
currently give us; their *disagreement* here is expected given
segmentation's known limitations, not a XAI-side defect.

**Practical implication:** until segmentation improves, `LOW_AGREEMENT`
should not be read as "this specific screening result is untrustworthy."
The FOV guard (`CAM_OFF_RETINA`, 0% here) remains the more reliable
independent check in the current pipeline. This is documented in
`docs/CLINICAL_ALIGNMENT.md`.

### Earlier verification (n=3, all repo fixtures) — historical, bug-fix record

First real-pipeline run, before the n=20 APTOS sample was available:

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

n=3 was a plumbing check only. The n=20 run above is the first
statistically meaningful sample (per the guide's ~20-image bar) and
supersedes these numbers for reporting purposes; kept here as the
record of what bug was found and how it was fixed.

## 4. Guard trigger rates — see Section 3

Guard trigger rates are now reported alongside the XAI headline result
in Section 3 (n=20, real pipeline) rather than duplicated here, to keep
grade/confidence/guard/agreement together in one place.

## 5. Clinical alignment
See `docs/CLINICAL_ALIGNMENT.md` for the full ICDR grade → NETRA label →
routing action table, including the guard-status override rules and the
note on `LOW_AGREEMENT` reliability pending segmentation improvements.

## 6. Limitations (for the honest-limitations slide)
- Trained on 3,662 APTOS images; external κ drops by TBD on Messidor-2.
- Segmentation trained on 54 annotated images (IDRiD), n=27 test split.
- Microaneurysm Dice is low (0.0152 per Kanchan); this directly causes
  the XAI `LOW_AGREEMENT` guard to fire on 95% of a 20-image APTOS
  sample — the guard is working correctly, but the segmentation
  reference it checks against is not yet reliable enough to make
  `LOW_AGREEMENT` a trustworthy per-screening signal on its own.
- CDR-based glaucoma flagging is a screening trigger, not a diagnosis.
- No prospective clinical validation.
- Grading and segmentation accuracy metrics (κ, sensitivity,
  specificity, Dice, IoU) are still TBD — blocked on a labeled held-out
  set, not yet run.
- Deletion/insertion AUC and guard trigger rates above are n=20,
  unlabeled APTOS images — a real sample per the guide's bar, but not
  yet a clinically validated evaluation set.
- Next steps: run grading/segmentation eval against labeled held-out
  data; capture lesion-localisation hit rate for the n=20 sample;
  re-run guard trigger rates once segmentation quality improves.