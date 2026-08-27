# Guide — Anshika Maurya
**Role:** Clinical XAI & Validation Lead
**Mission:** when the system says "moderate NPDR", a doctor can see why — and when the evidence doesn't hold up, the system says so instead of pretending.

You own the two things judges push hardest on: *does the explanation mean anything*, and *are your numbers real*. Your "anatomically guarded" idea is the strongest genuinely-novel claim in the deck. Make it a mechanism, not an adjective.

---

## What you own
`core/xai/`, `docs/BENCHMARKS.md`, the evaluation harness
Contract function: `explain(bgr, model_handle, grading, lesion_mask, fov_mask) -> dict`

---

## Part 1 — Grad-CAM, done properly

Baseline: Grad-CAM on the last convolutional block of Kanchan's grading backbone (get the exact layer name from him on Day 7). Upsample to input resolution, normalise, colourmap, alpha-blend over the original.

Consider **Grad-CAM++** or **Score-CAM** if plain Grad-CAM produces blobs too coarse to be useful — DR lesions are small and diffuse, and plain Grad-CAM on a 512² input with a 16² feature map gives you 32-pixel resolution. Compare the two on ~20 images and pick with evidence.

## Part 2 — The guard (this is your contribution)

Three checks, each producing a number that goes into the contract:

**1. FOV guard.** Compute the fraction of CAM energy falling outside Muskan's FOV mask (her Day-8 handoff to you).
```
cam_outside_fov_fraction = sum(CAM * (1 - fov_mask)) / sum(CAM)
```
If it exceeds ~0.15, the model is attending to the black surround, the eyelid, or a camera artefact — not the retina. That's `guard_status = "CAM_OFF_RETINA"`. Zero out the CAM outside the FOV before display either way.

**2. Lesion agreement.** Compare the top-quantile CAM region against Kanchan's U-Net lesion mask:
```
cam_lesion_agreement = |top20%_CAM ∩ lesion_mask| / |top20%_CAM|
```
Two independent models — one trained for grading, one for lesion localisation — pointing at the same pixels is real corroborating evidence. When they disagree, one of them is wrong and you don't know which. That's `guard_status = "LOW_AGREEMENT"` and it routes to a specialist.

**3. Anatomical plausibility.** DR lesions cluster in the posterior pole (macula and around the vessel arcades). CAM energy concentrated on the optic disc alone, or in the far periphery, is suspicious. A simple radial/regional prior is enough — you don't need a segmentation model for this.

**Frame it for judges like this:** *"Every published DR system shows a heatmap. None of them check whether the heatmap makes anatomical sense. We do, and when it doesn't we escalate rather than report a grade we can't justify."* That's a defensible novelty claim, unlike "we used Grad-CAM."

## Part 3 — Multi-disease flags (Phase 2, cut if behind)

- Glaucoma suspect: consume Muskan's CDR, flag above threshold
- Hypertensive retinopathy: arteriovenous nipping / vessel tortuosity from the vessel mask. This is genuinely hard with your data — an honest "flagged as a research direction, not validated" is better than an unvalidated claim on a slide.

## Part 4 — Validation (start this in week 1, not week 3)

You own every number in the deck. Build `eval/` early so results are reproducible with one command.

**Grading evaluation**
- Quadratic weighted κ, per-class recall, confusion matrix
- **Referable DR (grade ≥2) sensitivity and specificity with an ROC curve** — lead with this; it's the metric a clinician cares about
- Calibration plot (predicted confidence vs observed accuracy) — validates that the routing thresholds mean something
- External validation on Messidor-2 (or, if access doesn't arrive, a held-out IDRiD grading split). Report the drop.

**Segmentation evaluation**
- Per-class Dice and IoU on the IDRiD test split, with the caveat that n=27

**XAI evaluation** — this is what makes your section publishable-feeling:
- **Deletion/insertion test:** progressively mask the highest-CAM pixels and measure how fast the predicted probability drops. A sharp drop means the CAM is pointing at pixels the model actually uses. Cheap to implement, and it's a real quantitative XAI metric rather than "the heatmap looks right."
- **Lesion-localisation hit rate:** on IDRiD images with masks, what fraction of the time does the peak CAM land within an annotated lesion? Report it.
- **Guard trigger rate:** on your test set, how often does each guard fire, and on inspection, is it firing on genuinely problematic images? A guard that fires on 40% of good images is broken; one that fires on 8% and those 8% are mostly poor captures is working.

**Clinical alignment.** Map your grades to ICDR (Wilkinson et al. 2003, already in your references) and state the referral criteria you're using. One slide showing ICDR grade → your label → your routing action makes the whole system look clinically grounded.

---

## Timeline

| Days | Deliverable |
|---|---|
| 1–2 | Stub returning a valid `xai` block. Read Grad-CAM + Grad-CAM++ papers. |
| 3–6 | Build `eval/` harness against Kanchan's stub — metrics code, plotting, `BENCHMARKS.md` template |
| 7 | **Receive checkpoint + layer name from Kanchan.** Grad-CAM producing real overlays same day. |
| 8 | **Receive FOV mask function from Muskan.** FOV guard implemented. |
| 9–11 | Lesion agreement score (needs Kanchan's U-Net, ~Day 12 — use his stub mask until then), anatomical prior |
| 12–13 | Deletion/insertion test, localisation hit rate. **Day 13: hand guard status to Divyanshu for routing.** |
| 14–15 | Multi-disease flags, external validation run |
| 16–18 | Guard threshold tuning, qualitative review of ~50 cases |
| 19 | **Final numbers into `docs/BENCHMARKS.md`. Numbers freeze.** Hand table to Abhishek. |
| 20–21 | Prepare your two slides + the honest-limitations slide |

---

## The limitations slide (you write it — it's an asset, not a liability)

Something like: trained on 3,662 APTOS images, external κ drops by X on Messidor-2; segmentation trained on 54 annotated images; microaneurysm Dice is low and we show exudates and haemorrhages by preference; CDR-based glaucoma flagging is a screening trigger, not a diagnosis; no prospective clinical validation. Then what you'd do next.

Every strong team gets asked "what are your limitations?" Most improvise badly. Having this pre-written and confident is a visible differentiator.

---

## Failure modes
- Waiting for a "good" model before building the eval harness. Build it against stubs in week 1.
- Grad-CAM on the wrong layer — too early gives noise, too late gives a single blob. Ask Kanchan; try two or three.
- Reporting a heatmap as evidence without any quantitative XAI metric. That's the gap you're filling.
- Numbers in the deck that don't come from `BENCHMARKS.md`. You are the single source of truth.

## Reading
- Selvaraju et al. (2016), Grad-CAM (already in your references) + the Grad-CAM++ paper
- Petsiuk et al. (2018), RISE — for the deletion/insertion metric
- Wilkinson et al. (2003), ICDR severity scales (already in your references)
