# BENCHMARKS.md
Owner: Anshika Maurya (Clinical XAI & Validation Lead)

Single source of truth for every number in the deck/report. If it isn't
here, it doesn't go on a slide. Regenerate with: `python eval/run_all.py`

Last updated: TBD · commit: TBD · **numbers frozen: NO**

---

## 1. Grading (Kanchan's ResNet-50)
| Metric | Value | Split | n |
|---|---|---|---|
| Quadratic weighted κ | TBD | APTOS held-out | TBD |
| Referable DR (grade ≥2) sensitivity | TBD | APTOS held-out | TBD |
| Referable DR (grade ≥2) specificity | TBD | APTOS held-out | TBD |
| External κ | TBD | Messidor-2 / IDRiD held-out | TBD |
| κ drop (internal → external) | TBD | — | — |

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