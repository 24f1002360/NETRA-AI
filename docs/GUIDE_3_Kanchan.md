# Guide — Kanchan
**Role:** Deep Learning & Optimization Lead
**Mission:** a model that is accurate enough to be useful, small enough to ship, and fast enough that nobody waits.

You are the critical path. Two people (Anshika, Ishank) are blocked on your outputs at specific dates. Hitting those dates matters more than squeezing out another 0.02 κ.

---

## What you own
`core/models/`, `matlab/training/`, the model registry
Contract functions: `grade(bgr) -> dict`, `segment(bgr) -> (dict, ndarray)`

---

## Start today: get the data

This takes longer than you think. Start on Day 1, before anything else.

| Dataset | Size | Use | Access |
|---|---|---|---|
| APTOS 2019 | 3,662 train images, 5-class ICDR | **Primary grading training set** | Kaggle, instant |
| IDRiD | 516 graded, **54 train + 27 test with pixel lesion masks** | Segmentation + Indian-camera grading | IEEE DataPort, quick registration |
| DRIVE | 40 images, vessel masks | Vessel extraction (Muskan's Frangi validation) | Free, instant |
| Messidor-2 | 1,748 images | **Held-out external test — never train on this** | Requires an agreement; can take days. Start the request on Day 1. |

Note the number that should worry you: **54 images with lesion masks.** That is your entire segmentation training set. Plan accordingly (patch-based training, heavy augmentation, and honest expectations — see §Metrics).

---

## Model 1 — Grading

**Backbone decision (Decision B in the master plan, yours to resolve by Day 12).**

Train both, report both:
- ResNet-50 — the deck's current claim, your baseline
- **EfficientNet-B0** — the one you'll almost certainly deploy

Why: ResNet-50 at INT8 is roughly 26 MB, not the "under 15 MB" on your slide. EfficientNet-B0 is ~5.5 MB at INT8 and typically matches ResNet-50's κ on APTOS. Produce a table — params, size FP32, size INT8, CPU latency, val κ — and let it make the decision. That table is a slide.

**Training recipe that works on this dataset:**
- Input 512×512 (higher resolution matters a lot for microaneurysms; 224 loses them entirely)
- Preprocessing: **Muskan's `enhance()`, unchanged.** Get it on Day 5 and use exactly that function.
- Augmentation: random rotation (full 360° — retinas have no canonical orientation), horizontal/vertical flip, brightness/contrast jitter, mild blur, random FOV crop. The blur augmentation is what makes the model survive real field images.
- Loss: treat grading as **ordinal, not categorical.** Grade 0 mistaken for 4 is far worse than 0 for 1. Options: regression head + thresholds, or cross-entropy with label smoothing plus an ordinal penalty. Regression + optimised thresholds is the simplest thing that lifts κ noticeably.
- Class imbalance: APTOS is dominated by grade 0. Use weighted sampling or class-weighted loss, and report per-class recall, not just accuracy.
- Optimiser: AdamW, cosine schedule, 15–30 epochs, early stop on val κ.
- 5-fold CV is nice; if time is short, one stratified 80/20 split is acceptable — just say so.

**Calibration matters here.** Your `confidence` field drives the routing rules. Raw softmax is overconfident. Apply temperature scaling on the validation set (one scalar, ten lines of code) so that "0.68 confidence" means something. If you have time, MC-dropout at inference (10 passes) gives better uncertainty — but check it against the latency budget.

---

## Model 2 — Lesion segmentation

- Architecture: **width-reduced U-Net** (base 16 or 32 filters instead of 64), or U-Net with a MobileNet encoder. Target ~4 MB INT8.
- Classes: microaneurysms, haemorrhages, hard exudates, soft exudates → 4-channel output mask.
- With 54 images: train on **patches** (e.g. 512×512 crops from the 4288×2848 originals), which turns 54 images into thousands of samples. Sample patches preferentially around annotated lesions or you'll train on almost pure background.
- Loss: Dice + BCE combined. Pure BCE fails badly on lesions occupying <1% of pixels.
- Post-process: connected components with a minimum-area filter to produce the `counts` in the contract.

---

## Optimisation & export

1. Train FP32 in PyTorch (or MATLAB Deep Learning Toolbox if you prefer — both are fine)
2. Export ONNX (`torch.onnx.export`, opset 17), verify numerics match FP32 within tolerance
3. Post-training **static** INT8 quantization with a calibration set of ~200 representative images (dynamic quantization gives less speedup for convnets)
4. **Re-measure κ after quantization.** Expect a small drop; if it's more than ~0.02, use per-channel quantization or fall back to FP16.
5. Benchmark on CPU with threads capped to 4 — that's your target laptop, not your dev machine

For the MATLAB path (Decision A): `importNetworkFromONNX` brings your trained model into MATLAB, and MATLAB Compiler SDK packages the inference function as a Python module. Attempt this in Phase 2, not before — it's a packaging problem, not a modelling problem, and it must not eat your training time.

---

## Timeline

| Days | Deliverable |
|---|---|
| 1 | Request Messidor-2 access. Download APTOS, IDRiD, DRIVE. |
| 2 | Stub returning valid `grading` + `lesions` blocks |
| 3–4 | Data loaders, split strategy, baseline ResNet-50 training run started |
| 5 | Receive `enhance()` from Muskan; rebuild the preprocessing pipeline around it |
| 6–7 | **Day 7: hand first checkpoint + Grad-CAM target layer name to Anshika.** Even if κ is mediocre. |
| 8–10 | EfficientNet-B0 training, ordinal loss, calibration. ONNX export working. |
| 10 | **Hand measured latency + model size + peak memory to Ishank** |
| 11–12 | U-Net on IDRiD patches. **Day 12: publish the backbone benchmark table.** |
| 13–14 | INT8 quantization, re-benchmark, integrate into `core/models/` |
| 15–16 | MATLAB Compiler SDK path attempt. Freeze models Day 16. |
| 17–19 | Final held-out evaluation. **Numbers into `docs/BENCHMARKS.md` by Day 19.** |

---

## Metrics — and what to promise

| Metric | Realistic | Do not claim |
|---|---|---|
| APTOS val quadratic weighted κ | 0.83–0.90 | >0.93 |
| Referable DR (≥2) sensitivity | ≥0.90 | — |
| Messidor-2 external κ | 0.05–0.15 lower than internal | "no degradation" |
| Hard exudate Dice | 0.60–0.72 | >0.80 |
| Microaneurysm Dice | 0.25–0.45 | >0.5 |
| Grading latency (INT8, 4-core CPU) | 70–200 ms | — |

Report the external validation drop openly. A team that measures generalisation failure and says so looks more credible than one claiming perfection, and the judges will probe for exactly this.

---

## Failure modes
- **Preprocessing mismatch between training and inference.** The number-one silent killer. Import Muskan's function; don't reimplement it.
- Training at 224×224. Microaneurysms are a few pixels wide; you'll erase them.
- Reporting plain accuracy. On APTOS, always-predict-0 gets ~50%. Report κ and per-class recall.
- Perfectionism past Day 16. A frozen 0.86 model beats an unfrozen 0.89 one.
- Missing the Day 7 handoff to Anshika. She loses a week and cannot recover it.

## Reading
- Gulshan et al. (2016), JAMA — already in your references, read the methods section
- Ronneberger et al. (2015), U-Net
- Any top-5 APTOS 2019 Kaggle solution write-up — free, concrete recipes for exactly this dataset
