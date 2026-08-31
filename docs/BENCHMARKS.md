# BENCHMARKS.md
Owner: Anshika Maurya (Clinical XAI & Validation Lead)
Single source of truth for every number in the deck/report. If it isn't
here, it doesn't go on a slide. Regenerate with: `python eval/run_all.py`
Last updated: TBD · commit: TBD · **numbers frozen: NO**
---
## 1. Grading (Kanchan's EfficientNet-B0)
Final model: EfficientNet-B0, 384×384 input, NETRA/Muskan preprocessing.
Best checkpoint: Epoch 5 (best validation QWK; QWK decreased after epoch 5).
Checkpoint file: `netra_dr_effb0_muskan_preproc.pth` (46.41 MB)

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
## 2. Segmentation (Kanchan's U-Net)
| Lesion class | Dice | IoU |
|---|---|---|
| Microaneurysms | TBD | TBD |
| Haemorrhages | TBD | TBD |
| Hard exudates | TBD | TBD |
| Soft exudates | TBD | TBD |
n = 27 (IDRiD test split). Microaneurysm Dice expected low — report honestly.
## 3. XAI
| Metric | Value |
|---|---|
| Deletion AUC | TBD |
| Insertion AUC | TBD |
| Lesion-localisation hit rate | TBD |
| Grad-CAM vs Grad-CAM++ chosen | TBD (~20-image comparison) |
## 4. Guard trigger rates
| Guard | Trigger rate | On inspection |
|---|---|---|
| CAM_OFF_RETINA | TBD % | TBD |
| LOW_AGREEMENT | TBD % | TBD |
## 5. Clinical alignment
ICDR grade → NETRA label → routing action — TODO.
## 6. Limitations (for the honest-limitations slide)
- Trained on 3,662 APTOS images; external κ drops by TBD on Messidor-2.
- Segmentation trained on 54 annotated images (IDRiD), n=27 test split.
- Microaneurysm Dice is low; exudates/haemorrhages favoured by construction.
- CDR-based glaucoma flagging is a screening trigger, not a diagnosis.
- No prospective clinical validation.
- Next steps: TBD