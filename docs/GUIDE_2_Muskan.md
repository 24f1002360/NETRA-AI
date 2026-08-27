# Guide — Muskan Jadon
**Role:** Image Processing — Quality Gate, enhancement, FOV, optic disc
**Mission:** the system never grades an image it cannot grade honestly.

Your module is the one that makes this project different from every academic DR classifier. Standard models take whatever photo they're given and confidently produce a number. Yours stops the pipeline and tells the health worker *what to fix*. Present it that way.

---

## What you own
`core/iqa/` and `matlab/iqa/`
Contract functions: `assess_quality(bgr) -> dict`, `enhance(bgr, quality) -> ndarray`, `cup_disc_ratio(...) -> float`

You also produce the **three shared test fixtures** on Day 3 — one clean image, one blurry/dark, one severe-DR case. Everyone's tests use them.

---

## The Quality Gate — what to actually compute

Five scores, each normalised to 0–1, each with a threshold in `configs/thresholds.yaml`:

| Score | Method | Notes |
|---|---|---|
| `blur` | Variance of Laplacian, **computed inside the FOV mask only** | Computing it over the whole frame is the classic bug — the black surround drags the variance down and every image looks sharp/blurry inconsistently |
| `illumination` | Mean intensity of green channel within FOV + fraction of clipped pixels (>250 and <5) | Two failure directions: too dark, and blown-out flash |
| `fov_coverage` | Area of the retinal disc mask ÷ expected area, plus centroid offset from image centre | Catches partial and off-centre captures |
| `contrast` | RMS contrast on the CLAHE-ready channel, or vessel-visibility proxy: energy of the Frangi filter response inside FOV | Frangi energy is a better proxy for "can you actually see structure" than raw contrast |
| `artefact` | Fraction of pixels that are specular highlights (very high intensity, low saturation) near the macula | Glare and dust |

**FOV mask:** threshold the red channel or intensity, take the largest connected component, fill holes, fit a circle. Erode by ~5% before computing anything so you don't include the boundary ring. Export it — Anshika needs it for the CAM guard, and this is her Day-8 dependency on you.

**Verdict logic:**
```
if any score below hard_threshold      -> RETAKE  + reason codes
elif any score below soft_threshold    -> AUTO_CORRECTED (run enhance, re-score once)
else                                    -> PASS
```
Re-score exactly once after enhancement. Never loop.

**Emit reason codes, never English.** `BLUR_HIGH`, `TOO_DARK`, `TOO_BRIGHT`, `FOV_PARTIAL`, `GLARE`, `OFF_CENTRE`. Abhishek maps them to Hindi/Tamil text and audio. Add codes freely; tell him when you do.

---

## Enhancement

CLAHE on the **green channel** (where retinal vessels and haemorrhages have the most contrast) or on L in LAB, then recombine. Typical: `clipLimit=2.0–3.0`, `tileGridSize=(8,8)` — tune on your fixtures.

Optional: illumination correction by subtracting a large-kernel Gaussian blur estimate of the background before CLAHE. Helps a lot with vignetting.

**The single most important rule in your entire module:** `enhance()` is the *same function* used during Kanchan's training and during live inference. If training sees raw images and inference sees CLAHE'd images (or vice versa), accuracy quietly collapses and nobody will know why. **Hand this function to Kanchan on Day 5.** Freeze its behaviour after that; if you must change it, tell him so he can retrain.

---

## Optic disc & cup-to-disc ratio (Phase 2, cut if behind)

Disc: brightest region in the red/green channel, refine with morphological closing + Hough circle or a fitted ellipse. Cup: brighter inner region within the disc, thresholded relative to disc intensity.
CDR = cup diameter / disc diameter (use vertical). Flag as glaucoma-suspect above ~0.6.

Be honest in the deck: intensity-based CDR is approximate and unreliable on poor-quality images. Frame it as a *screening flag that triggers specialist review*, not a diagnosis. That framing is both true and defensible.

---

## MATLAB vs Python

Prototype in **MATLAB** (`imadjust`, `adapthisteq`, `fibermetric` for Frangi, `imfindcircles`, `regionprops`) — it's much faster to iterate visually and it's a real MathWorks contribution. Then port the settled algorithm to OpenCV in `core/iqa/` for the runtime. Keep both; the MATLAB scripts go in the deck as your development evidence.

Do not leave MATLAB in the live IQA path — it's the one stage that must be instant.

---

## Timeline

| Days | Deliverable |
|---|---|
| 1–2 | Stub returning a valid `quality` block. Environment set up. |
| 3 | **Three shared fixtures published** to `tests/fixtures/` |
| 4–6 | FOV mask + blur + illumination scores, validated visually on ~50 APTOS images |
| 5 | **Hand `enhance()` to Kanchan** |
| 7–8 | Contrast + artefact scores, verdict logic, reason codes. **Hand FOV mask function to Anshika (Day 8).** |
| 9–10 | Hand-label ~100 images as good/bad, measure retake precision & recall, tune thresholds |
| 11 | **Hand measured retake rate to Ishank** — it's a key parameter in his queue model |
| 12–15 | Optic disc / CDR (if time). Latency optimisation to hit the 1.5 s budget. |
| 16–21 | Freeze. Support integration. Prepare your two slides. |

---

## Your acceptance test

Build a labelled set of ~100 fundus images (mix APTOS good ones with deliberately degraded versions — add Gaussian blur, reduce gamma, crop the FOV, overlay synthetic glare). Report:
- Retake precision ≥ 0.85 (when you say retake, you're right)
- Retake recall on genuinely ungradable images ≥ 0.80
- Median IQA latency < 1500 ms

Put that confusion matrix in the deck. It's evidence, and almost no competing team will have it.

---

## Failure modes
- Computing blur over the whole frame including the black surround → meaningless scores. Mask first.
- Tuning thresholds on five images. You need ~100 and both classes.
- Over-enhancing: aggressive CLAHE amplifies noise into things that look like microaneurysms. Check with Kanchan whether enhancement helps or hurts his validation κ — if it hurts, dial it back.
- Changing `enhance()` after Day 5 without telling Kanchan.

## Reading
- Frangi et al. (1998), *Multiscale vessel enhancement filtering* — the vessel filter you're using
- MATLAB docs: `adapthisteq`, `fibermetric`
- Any recent survey on retinal image quality assessment, for the metric taxonomy and to cite one in the deck
