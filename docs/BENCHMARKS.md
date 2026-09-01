# BENCHMARKS.md
**Owner:** Anshika Maurya (Clinical XAI & Validation Lead)
Single source of truth for every number in the deck/report. Regenerate with `python eval/run_all.py`.

**Last updated:** 2026-09-01 · **Status:** NOT FROZEN (n=20, real pipeline, APTOS sample)

---

## 1. Grading — EfficientNet-B0 (Kanchan)

- Checkpoint: `netra_dr_effb0_muskan_preproc.pth` (46.41 MB), best epoch 5
- Input: 384×384, NETRA/Muskan preprocessing
- Grad-CAM target layer: `features.8`

| Metric | Value | Split |
|---|---|---|
| Quadratic weighted κ | 0.8544 | Validation |
| Validation accuracy | 77.35% | Validation |
| Validation loss | 0.8283 | Validation |
| Referable DR (grade ≥2) sensitivity | TBD | APTOS held-out |
| Referable DR (grade ≥2) specificity | TBD | APTOS held-out |
| External κ (Messidor-2 / IDRiD) | TBD | — |
| GPU inference latency | 3.208 ms/image | — |
| CPU inference latency | 168.145 ms/image (batch=16) | PHC deployment |

**Why sensitivity/specificity/external κ are TBD:** needs a labeled held-out set (ground-truth grades) to score against. Not yet run.

---

## 2. Segmentation — U-Net V8 (Kanchan)

| Lesion class | Dice | IoU |
|---|---|---|
| Microaneurysms | 0.0152 | TBD |
| Haemorrhages | 0.2349 | TBD |
| Hard exudates | 0.5209 | TBD |
| Soft exudates | 0.4831 | TBD |

n=27 (IDRiD test split). Source: Kanchan's official V8 evaluation (`KANCHAN_MODEL_HANDOFF.md`, §11).

**IoU:** confirmed with Kanchan (1 Sep) — not calculated, no verified number. Left as TBD rather than deriving an unverified value from Dice.

**Note:** the very low microaneurysm Dice (0.0152, near-random) is the direct cause of the XAI guard finding below.

---

## 3. XAI — Real Pipeline Results (n=20, APTOS sample)

Ran `explain()` end-to-end via `eval/run_all.py`: real EfficientNet-B0 + real DRSegmenter, on 20 images from `data/gradcam_compare/`.

| Metric | Value |
|---|---|
| Deletion AUC | 0.1397 (mean) |
| Insertion AUC | 0.4263 (mean) |
| Grad-CAM vs Grad-CAM++ | **Grad-CAM wins** (score 0.2807 vs 0.2371) |
| Lesion-localisation hit rate | Not captured this run — pending |

Sharp deletion AUC + moderate insertion AUC → the CAM reliably points at pixels the model actually uses.

### Guard trigger rates

| Guard | Rate | Note |
|---|---|---|
| LOW_AGREEMENT | 95.0% (19/20) | Explained below — segmentation quality issue, not a guard bug |
| OK | 5.0% (1/20) | Only case with real CAM–lesion overlap |
| CAM_OFF_RETINA | 0.0% (0/20) | FOV guard working as expected |

**Why LOW_AGREEMENT fires 95% of the time:** verified the segmentation masks aren't empty (e.g. 6,300 MA pixels on one image), so this isn't the earlier empty-mask bug. The real cause: microaneurysm Dice is 0.0152 (near-random) — the segmentation reference itself is unreliable, so CAM disagreeing with it isn't proof the CAM is wrong. It means the lesion mask can't yet be trusted as ground truth.

**Practical implication:** until segmentation improves, don't read `LOW_AGREEMENT` as "this result is untrustworthy." `CAM_OFF_RETINA` (0% here) remains the more reliable guard for now.

### Bugs found & fixed during real-pipeline verification
1. CAM (384×384) vs FOV/lesion masks (different resolution) — added resize step.
2. `cam_outside_fov_fraction` was measured *after* zeroing the CAM, always returning 0.0 — fixed order.
3. `cam_lesion_agreement` returned 0.0 (false disagreement) on genuinely empty lesion masks — now returns `None` (not evaluable).
4. `np.trapz` deprecated in current NumPy — switched to `np.trapezoid`.

---

## 4. Clinical Alignment
See `docs/CLINICAL_ALIGNMENT.md` — full ICDR grade → NETRA label → routing action table, including guard-status overrides.

---

## 5. Limitations (honest-limitations slide)
- Trained on 3,662 APTOS images; external κ on Messidor-2 not yet measured.
- Segmentation trained on only 54 annotated IDRiD images (n=27 test).
- Microaneurysm Dice is low (0.0152) → `LOW_AGREEMENT` fires often; not yet a reliable per-screening signal.
- CDR-based glaucoma flagging is a screening trigger, not a diagnosis.
- No prospective clinical validation.
- Grading/segmentation accuracy metrics (κ, sensitivity, specificity, IoU) — TBD, blocked on labeled held-out data.
- XAI numbers above (n=20) are real but not yet a clinically validated evaluation set.

**Next steps:** run grading/segmentation eval on labeled data · capture lesion-localisation hit rate · re-check guard rates once segmentation improves.