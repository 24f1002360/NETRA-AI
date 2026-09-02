# NETRA AI — Explainable AI for Diabetic Retinopathy Screening in Rural India

**Team Aim Nexus** · Smart India Hackathon — **SIH26038** (MathWorks) · IIT Madras BS Degree Programme

> A single-laptop, offline screening tool that lets a health worker capture a fundus photo, get an AI-assisted diabetic retinopathy grade with visual evidence, and receive a defined routing recommendation — with no internet connection required. The under-30-second target is being measured on the demo laptop.

---

## Table of Contents

- [The Problem](#the-problem)
- [Our Approach](#our-approach)
- [What We Have Built](#what-we-have-built)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Current Results](#current-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Team](#team)
- [Honest Limitations](#honest-limitations)
- [Roadmap](#roadmap)

---

## The Problem

Diabetic retinopathy (DR) is a leading cause of preventable blindness in India, but screening capacity is critically short — roughly **one ophthalmologist for every 100,000 people**, concentrated heavily in cities. In rural Primary Health Centres (PHCs):

- Fundus photos are often too dark, blurry, or poorly framed to be gradable, and this is usually discovered only after the patient has already left.
- There is no offline, low-cost way for a non-specialist health worker (ASHA/ANM) to get an immediate, trustworthy read on a retinal image.
- Existing AI screening tools tend to be cloud-dependent, opaque "black box" predictions with no visual evidence a doctor can verify, and no clear escalation path for uncertain cases.

**SIH26038** asks for an explainable, offline-capable AI system — with a genuine MathWorks/MATLAB component — that closes this gap.

## Our Approach

NETRA AI is designed around one idea: **a health worker with no medical training should be able to screen a patient in one sitting, and a doctor should be able to trust the result because they can see the evidence.**

The pipeline:

```
ASHA/ANM                                                    District specialist
   │                                                                  ▲
   ▼                                                                  │
┌───────────┐  ┌───────────┐  ┌────────────────┐  ┌───────────┐  ┌───┴────────┐
│ 1 Capture │─▶│ 2 Quality │─▶│ 3 Grade +       │─▶│ 4 Explain │─▶│ 5 Report   │
│  visual   │  │   Gate    │  │   Segment       │  │  + Guard  │  │  + Route   │
└───────────┘  └────┬──────┘  └─────────────────┘  └───────────┘  └────┬───────┘
                     │ RETAKE                                          │
                     └──── back to Capture                             ▼
                                                                ┌──────────────┐
                                                                │ SQLite +     │
                                                                │ offline sync │
                                                                └──────┬───────┘
                                                                       │
                                                        anonymised timings, retake
                                                        rate, grade mix
                                                                       ▼
                                                        ┌───────────────────────┐
                                                        │ Simulink / SimEvents  │
                                                        │ district capacity     │
                                                        │ model                 │
                                                        └───────────────────────┘
```

Key design decisions:

1. **Quality Gate before grading, not after.** A dedicated image-quality module scores blur, illumination, contrast, artefacts, and field-of-view coverage, and rejects ungradable photos on the spot with a specific reason — so the retake happens while the patient is still there, instead of a wasted screening.
2. **Every AI verdict ships with visual evidence.** Grad-CAM heatmaps and a lesion-segmentation overlay are shown alongside the grade, not hidden behind a score.
3. **The system checks its own evidence before it trusts it.** A guard layer compares the Grad-CAM attention against the segmented lesions and the retinal field-of-view. If they disagree, or the model's attention falls outside the retina, the case is automatically routed to a human specialist instead of being shown as a confident result.
4. **Offline-first.** The entire pipeline (quality check → grading → segmentation → explainability → report → routing) runs locally on a basic PHC laptop, with results queued in SQLite and synced to a district server whenever connectivity returns.
5. **A real MathWorks/MATLAB story, not a bolted-on one.** Image Processing Toolbox is used for IQA/enhancement prototyping, Deep Network Designer for architecture and quantization analysis, and a Simulink/SimEvents model simulates district-level screening capacity using the pipeline's own measured timings — closing the loop between the clinical tool and the operations question of *how many cameras, laptops and specialists a district actually needs.*
6. **Clinically traceable output.** Every grade maps to the standard ICDR severity scale (no DR → mild → moderate → severe → PDR) and a defined routing action (routine / non-urgent referral / urgent referral / emergency alert), documented in `docs/CLINICAL_ALIGNMENT.md`.

## What We Have Built

The project has moved past the scaffold stage into a working, testable, offline end-to-end application.

| Area | Owner | Status |
|---|---|---|
| **Backend, persistence & orchestration** — SQLite DAO, offline sync queue/worker, alert queue, screening API routes, and the single orchestrator (`core/inference.py`) that joins every module under one screening contract | Divyanshu | Delivered — production-style plumbing, wired end to end |
| **Model integration & grading** — EfficientNet-B0 DR grader and a 4-channel U-Net lesion segmenter, strict checkpoint loading, preprocessing/post-processing, model handoff docs | Kanchan | Delivered — both checkpoints load and run through the real pipeline |
| **Explainability & clinical safety** — Real Grad-CAM integration, FOV guard and CAM–lesion agreement guard, mask-resolution fixes, `docs/CLINICAL_ALIGNMENT.md` | Anshika | Delivered — guard logic actively demoted low-confidence cases to manual review during testing |
| **Image quality assessment** — FOV detection, CLAHE-based enhancement, blur/contrast/illumination scoring, PASS/AUTO/RETAKE verdicts with reason codes | Muskan | Delivered — a working quality gate, not just a placeholder score |
| **Simulation & QA tooling** — District-capacity SimEvents-style simulator, capacity/retake-sensitivity/specialist-load charts, synthetic degraded-image generator for stress-testing the quality gate | Ishank | Delivered — simulator runs and produces charts (see `sim/results/`) |
| **UI/UX & integration** — Two-eye capture, quality, combined-result, printable report and history screens; camera/upload capture with visual-first design (usable without audio); real upload → screening → persisted-history flow; evidence images served with traversal protection | Abhishek | Delivered — connected to real team output, survives a server restart |

**Validated locally:**
- Full test suite passes: **38/38 tests**, ~102 s, on the demo laptop (Python 3.12.9, `torch 2.6.0+cpu`, `torchvision 0.21.0+cpu`).
- Both production model artifacts (grading + segmentation checkpoints) load and run through the real inference path, integrity-checked with SHA-256.
- The pipeline **fails safely to a "review needed" state** if the local model runtime is unavailable — it never invents a positive diagnosis.
- End-to-end flow (two-eye capture → quality → grading → segmentation → explainability → report → SQLite persistence) runs offline, and offline sync primitives are in place for reconnect scenarios.

## System Architecture

```
netra-ai/
├── app/          Flask UI — capture, quality, result, history, report (PDF)
├── core/
│   ├── contracts.py     Shared ScreeningResult schema (single source of truth)
│   ├── inference.py     Orchestrator — wires every module into one screening
│   ├── iqa/              Image quality + enhancement
│   ├── models/            Grading (EfficientNet-B0) + segmentation (U-Net)
│   └── xai/                Grad-CAM + FOV / CAM–lesion guards
├── db/           SQLite DAO, offline sync worker
├── alerts/       Emergency-referral alert queue/sender
├── matlab/       IQA prototyping, Deep Network Designer analysis, Simulink district model
├── sim/          District-capacity simulation + result chart
├── eval/         Benchmark & XAI evaluation harness (`eval/run_all.py`)
├── configs/      app.yaml (runtime/module switches), thresholds.yaml, i18n
├── tests/        Contract, grading, segmentation, and XAI tests (run on every PR)
├── docs/         Master plan, interface contracts, benchmarks, clinical alignment
└── notebooks/    Exploration only — never imported by app/
```

## Tech Stack

- **Backend / orchestration:** Python, Flask
- **Modeling:** PyTorch, Torchvision (EfficientNet-B0 grading backbone, U-Net segmentation)
- **Explainability:** Grad-CAM, custom guard logic (FOV + lesion-agreement checks)
- **Image processing:** OpenCV, scikit-image, MATLAB Image Processing Toolbox
- **MATLAB/Simulink:** Deep Network Designer (architecture/quantization analysis), SimEvents (district capacity model)
- **Data & persistence:** SQLite, JSON Schema (contract validation)
- **Reporting:** self-contained HTML report, printable to PDF in the browser
- **Testing:** pytest (contract, grading, segmentation, XAI tests)
- **Simulation/analysis:** NumPy, Matplotlib, SciPy

## Current Results

Full details and the evaluation harness live in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`docs/CLINICAL_ALIGNMENT.md`](docs/CLINICAL_ALIGNMENT.md) — this is our single source of truth, kept in sync with whatever we present.

| Metric | Value | Note |
|---|---|---|
| Quadratic weighted κ (grading, validation) | 0.854 | EfficientNet-B0, best epoch 5 |
| Validation accuracy | 77.4% | — |
| CPU inference latency | ~168 ms/image (batch=16) | Target PHC deployment hardware |
| Hard exudate segmentation Dice | 0.52 | IDRiD test split, n=27 |
| Soft exudate segmentation Dice | 0.48 | IDRiD test split, n=27 |
| Microaneurysm segmentation Dice | 0.015 | Genuinely hard; honestly reported as near-random |
| Test suite | 38/38 passing | Contracts, grading, segmentation, XAI |

We report metrics honestly, including the ones that aren't good yet — e.g. microaneurysm segmentation is currently unreliable, which is *why* the CAM–lesion agreement guard fires often, and we treat that as a safety feature (route to a human) rather than hide it. Referable-DR sensitivity/specificity and external (Messidor-2) validation are marked **TBD** pending a labelled held-out set — we'd rather say "not yet measured" than quote an invented number.


## Getting Started

```bash
# 1. Clone the repository
git clone <repo-url>
cd netra-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the model checkpoints (not committed — see docs/KANCHAN_MODEL_HANDOFF.md)
#    into artifacts/

# 4. Run the app
python run.py
```

The app launches locally on `http://localhost:5000` with no network requirement. Configuration (runtime, module toggles, alerts, sync) lives in `configs/app.yaml`.

Run the test suite:

```bash
pytest
```

## Team — Aim Nexus

| Member | Role |
|---|---|
| Abhishek Dhama | Team Lead · UI/UX, vernacular HCI |
| Muskan Jadon | Image quality & enhancement |
| Kanchan | Deep learning & model optimization |
| Anshika Maurya | Clinical XAI & validation |
| Divyanshu Kumar | Backend, database, orchestration |
| Ishank Gupta | Simulation, compression, hardware |

## Honest Limitations

- Grading model trained on 3,662 APTOS images; external validation on Messidor-2 not yet measured.
- Segmentation trained on only 54 annotated IDRiD images — enough for a working prototype, not a clinical-grade result.
- Microaneurysm detection is currently unreliable (Dice ≈ 0.015); we surface this rather than mask it, and route affected cases to manual review.
- Referable-DR sensitivity/specificity and external κ are pending a labelled held-out evaluation set.
- No prospective clinical validation has been performed — this is a hackathon prototype, not a certified medical device.
- CDR-based glaucoma flagging (if shown) is a screening trigger only, not a diagnosis.

## Roadmap

- [ ] Verify the real-model environment on the exact demo laptop (PyTorch/Torchvision install, both checkpoints load)
- [ ] Run grading/segmentation evaluation on labelled held-out data to close out TBD metrics
- [ ] Rehearse and measure real device offline-sync and alert-routing behaviour
- [ ] Land the MATLAB Compiler SDK inference path as an optional runtime (`runtime: matlab` in `configs/app.yaml`), with `runtime: onnx`/PyTorch as the tested fallback
- [ ] Freeze benchmark numbers and rehearse the full demo script end to end, offline, on battery

---

*Built for Smart India Hackathon — Problem Statement SIH26038 (MathWorks): Explainable AI for Diabetic Retinopathy Screening in Rural India.*
