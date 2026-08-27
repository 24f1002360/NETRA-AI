# NETRA AI — Prototype Round Implementation Plan
**Team Aim Nexus · SIH26038 · Explainable AI for Diabetic Retinopathy Screening in Rural India**
Master document. Every member reads this once, fully, before writing any code.

---

## 0. How to use this pack

| File | Who reads it | When |
|---|---|---|
| `00_MASTER_PLAN.md` | Everyone | Once, now, fully |
| `01_INTERFACE_CONTRACTS.md` | Everyone | Now, then daily as reference |
| `GUIDE_1_Abhishek.md` … `GUIDE_6_Ishank.md` | Your own + skim the two you hand off to | Now, then per phase |

**The single most important idea in this pack:** nobody waits for anybody. On Day 2 the entire pipeline runs end-to-end with *fake* outputs. After that, each of you replaces your fake block with a real one, independently, in any order. If you understand only one thing, understand that.

---

## 1. Two decisions that must be made in the first 48 hours

These are blocking. Do not start Phase 1 until both are settled and written into `configs/decisions.md`.

### Decision A — Where does inference actually run?

SIH26038 is a **MathWorks** problem statement. MATLAB/Simulink usage is part of how you're judged, so "we did it all in PyTorch" is a weak answer. But a MATLAB call in the live demo path is the classic way hackathon demos die on stage.

Resolve it by building **one interface with two implementations**:

```python
# core/inference.py
class Inferencer(Protocol):
    def grade(self, img) -> GradingBlock: ...
    def segment(self, img) -> LesionBlock: ...

# selected by configs/app.yaml -> runtime: "onnx" | "matlab"
```

| Path | What it is | Pros | Cons |
|---|---|---|---|
| **ONNX (safety net)** | Train in PyTorch → export ONNX → run with `onnxruntime` in Python | Always works, fast, zero licensing, tiny install | MATLAB story lives only in Simulink + IQA |
| **MATLAB (headline)** | Train/import in MATLAB Deep Learning Toolbox, package with **MATLAB Compiler SDK** into a Python module, ship with free MATLAB Runtime | Strong MathWorks story, still "offline on a laptop" | Compile step is fiddly; Runtime is a large install |

**Recommended sequencing:** build ONNX first (Days 1–10, guarantees a working demo), then land the MATLAB path by Day 14 and make it the default *if and only if* it survives two full clean-machine rehearsals. Keep `runtime: onnx` as a one-line rollback.

**Avoid** the MATLAB Engine API for Python as your final answer — it needs a full MATLAB install on the demo laptop and adds ~10 s of startup, which directly contradicts your "runs on a basic PHC laptop" claim. It's fine as a Days 4–13 development crutch.

MATLAB stays genuinely central regardless: IQA and enhancement prototyping (Image Processing Toolbox), Deep Network Designer for architecture analysis and quantization, and the entire SimEvents district model. That's a real MathWorks story, not a bolted-on one.

### Decision B — Which backbone are you actually deploying?

The deck says ResNet-50, INT8, under 15 MB. Those three claims are mutually inconsistent:

| Backbone | Params | FP32 | INT8 | CPU latency (est., 512², 4-core i5) |
|---|---|---|---|---|
| ResNet-50 | 25.6 M | ~98 MB | **~26 MB** | ~250–400 ms |
| ResNet-18 | 11.7 M | ~45 MB | ~12 MB | ~90–150 ms |
| **EfficientNet-B0** | 5.3 M | ~20 MB | **~5.5 MB** | ~70–130 ms |
| MobileNetV3-L | 5.4 M | ~21 MB | ~5.5 MB | ~50–100 ms |

**Recommendation:** train ResNet-50 as your *benchmark baseline*, deploy **EfficientNet-B0** + a width-reduced U-Net (~4 MB INT8). Total footprint ≈ 10 MB, comfortably under 15 MB, and roughly 3× faster on CPU.

This is a *stronger* slide, not a weaker one: "we benchmarked ResNet-50 against EfficientNet-B0 and found equal κ at one-fifth the size" is an engineering result. "We used ResNet-50" is an assumption. Kanchan owns producing that comparison table by Day 12.

---

## 2. What you are actually building

A single-laptop, offline application. One health worker, one fundus photo, one report, under 30 seconds, with visual evidence a doctor can check.

```
  ASHA / ANM                                              District specialist
      │                                                            ▲
      ▼                                                            │
┌───────────┐   ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌──┴────────┐
│ 1 Capture │──▶│ 2 Quality│──▶│ 3 Grade +  │──▶│ 4 Explain │──▶│ 5 Report  │
│  + voice  │   │   Gate   │   │  Segment   │   │  + Guard  │   │  + Route  │
└───────────┘   └────┬─────┘   └────────────┘   └───────────┘   └──┬────────┘
   Abhishek          │ RETAKE       Kanchan        Anshika          │ Divyanshu
                     └──────────────┐                               ▼
                       back to (1)  │                        ┌────────────┐
                          Muskan    │                        │ 6 SQLite + │
                                    │                        │  sync/SMS  │
                                    │                        └─────┬──────┘
                                    │                              │
                     ┌──────────────┴──────────────────────────────┘
                     ▼   anonymised timings, retake rate, grade mix
              ┌──────────────────────────┐
              │ Simulink / SimEvents     │  Ishank
              │ district capacity model  │
              └──────────────────────────┘
```

Ishank's simulation is not a side project. It consumes the real measured numbers your app produces (service times, retake rate, grade distribution, payload sizes) and answers the question no other team will answer: *how many cameras, laptops and specialists does a district actually need, and what does the Quality Gate save?* Keep that feedback loop visible.

---

## 3. Repository

```
netra-ai/
├── app/                    # Flask UI          → Abhishek, Divyanshu
│   ├── routes/  templates/  static/  audio/
├── core/
│   ├── contracts.py        # dataclasses + JSON schema   [SHARED — PR needs 2 approvals]
│   ├── inference.py        # single orchestrator entry   → Divyanshu
│   ├── iqa/                # quality + enhancement       → Muskan
│   ├── models/             # grading + segmentation      → Kanchan
│   ├── xai/                # Grad-CAM + guards           → Anshika
│   └── stubs/              # fake implementations of every block  [Day 2, everyone]
├── db/                     # schema.sql, migrations, dao.py → Divyanshu
├── matlab/
│   ├── iqa/                # Muskan (MATLAB versions)
│   ├── training/           # Kanchan (Deep Network Designer, quantization)
│   ├── simulink/           # Ishank — NETRA_district.slx
│   └── compiled/           # Compiler SDK output (gitignored)
├── configs/                # app.yaml, thresholds.yaml, decisions.md
├── tests/                  # test_contracts.py runs on every PR
├── artifacts/              # .gitignore'd — models live on shared Drive
├── data/                   # .gitignore'd — never commit fundus images
├── docs/                   # this pack + benchmark results
└── notebooks/              # exploration only, never imported by app/
```

**Rules that prevent 90% of merge pain:**
1. `data/` and `artifacts/` are gitignored. Models and datasets go to a shared Drive folder with a `MODEL_REGISTRY.md` listing filename, date, val metrics, who trained it.
2. No absolute paths. Ever. Everything relative to repo root or read from `configs/app.yaml`.
3. Nothing in `app/` imports from `notebooks/`.
4. `core/contracts.py` is shared property. Changing it needs a message in the group **before** the PR, plus two approvals.
5. Every module is a pure function where possible: image in, dict out. No module writes to the DB except Divyanshu's DAO layer.

**Branching:** `main` (always demo-able) ← `dev` ← `feat/<name>/<thing>`.
Merge to `dev` daily. `dev` → `main` every Wednesday and Saturday after a green integration call. Never commit directly to `main`.

---

## 4. Timeline — 21 days, four phases

If your prototype window is shorter, see §4.5 for the compression rule. Day numbers are relative to your kickoff.

### Phase 0 — Skeleton & contracts (Days 1–3)
**Goal: on Day 3 the whole app runs end to end and produces a complete, plausible, entirely fake report.**

| Day | Everyone | Individual |
|---|---|---|
| 1 | Read this pack. Kickoff call: settle Decisions A & B. Repo created, everyone pushes a hello-world commit. | Install your stack (see your guide) |
| 2 | Write your **stub** in `core/stubs/` returning a hardcoded but schema-valid block | Abhishek: 3 static HTML pages. Divyanshu: `inference.py` wiring stubs together. |
| 3 | **Integration call #1** — run `python -m app` and screenshot the fake report | Kanchan starts data download (it's slow, start now) |

Exit criteria: `pytest tests/test_contracts.py` green; fake report renders; everyone has pushed to `dev`.

Do not skip this phase. Three days spent here buys you the right to work in parallel for eighteen.

### Phase 1 — Vertical slices (Days 4–10)
Everyone replaces their stub with a real implementation. Nobody needs anybody.

| | Deliverable by Day 10 |
|---|---|
| Abhishek | Real capture → result flow, Hindi voice prompts wired, PDF generator on fake data |
| Muskan | Real IQA scoring + CLAHE + FOV mask, verdict PASS/AUTO/RETAKE with reason codes |
| Kanchan | Trained grading model, val κ reported, ONNX export working |
| Anshika | Grad-CAM on Kanchan's checkpoint, FOV guard, overlay PNG |
| Divyanshu | SQLite schema live, screenings persisted, patient history query |
| Ishank | SimEvents model running with placeholder parameters, first capacity chart |

Checkpoint: **Integration call #2, Day 7.** Half-real pipeline must still run.

### Phase 2 — Integration & depth (Days 11–16)
Now the seams matter. This is where hackathon projects usually break.

- Day 11–12: real IQA + real grading in the same run. Timing budget enforced (§6).
- Day 12: Kanchan publishes the backbone benchmark table → Decision B locked.
- Day 13: Anshika's CAM–lesion agreement score wired into the routing rule.
- Day 13–14: Divyanshu's SMS/alert trigger + offline sync queue.
- Day 14: MATLAB runtime path attempted (Decision A); go/no-go on Day 16.
- Day 15: Ishank swaps placeholder sim parameters for *measured* ones. Retake-rate sensitivity chart produced.
- Day 15–16: multi-disease (CDR-based glaucoma flag) and longitudinal comparison — **cut these first if you're behind.**
- Day 16: **Integration call #3 — feature freeze.**

### Phase 3 — Harden & present (Days 17–21)
- Day 17: bug bash. Everyone tries to break someone else's module. Log everything.
- Day 18: clean-machine test. Fresh laptop, fresh clone, follow `docs/SETUP.md` verbatim, time it. Whatever breaks, fix.
- Day 19: benchmark run on the held-out set; final numbers go into the deck. **Numbers freeze here.**
- Day 20: demo rehearsal ×3, full run each time. Deck rewritten around real results.
- Day 21: buffer. Do not add features on Day 21.

### 4.5 If your window is shorter
Compress proportionally but **never compress Phase 0 below one full day**, and never compress Phase 3 below two days. A 10-day version: Phase 0 = day 1, Phase 1 = days 2–5, Phase 2 = days 6–8, Phase 3 = days 9–10. First things cut: multi-disease detection, languages beyond Hindi, longitudinal tracking UI.

---

## 5. Working rhythm

- **Daily async standup**, fixed time, three lines each: *done / doing / blocked*. If you're blocked, name the person.
- **Daily merge window** — open your PR by 9 pm, reviewer merges before midnight.
- **Integration calls** Wednesday & Saturday, 60 min, camera on, one person shares screen and runs the full demo script. If it fails, that's the agenda.
- **The green rule:** `main` must always be demo-able. If you break it, you fix it that night or you revert.
- **Blocked > 90 minutes = escalate.** Post the exact error. Nobody gets to be quietly stuck for a day.

**Definition of Done for any PR:**
1. Runs from a fresh clone on a machine that isn't yours
2. Output validates against `core/contracts.py`
3. Emits its timing into `timings_ms`
4. No absolute paths, no hardcoded dataset paths, no secrets
5. One-paragraph note in `docs/` on what changed

---

## 6. Performance budget

Total wall clock, capture → report displayed: **under 30 s** (you promised this). Target on a 4-core CPU, 8 GB RAM, no GPU:

| Stage | Budget | Owner |
|---|---|---|
| IQA + enhancement | 1.5 s | Muskan |
| Grading inference | 1.5 s | Kanchan |
| Lesion segmentation | 3.0 s | Kanchan |
| Grad-CAM + guards | 2.0 s | Anshika |
| Report render + PDF | 2.0 s | Abhishek |
| DB write + routing | 0.5 s | Divyanshu |
| **Compute subtotal** | **10.5 s** | |
| Slack for cold start, disk, UI | 19.5 s | |

Every module logs its own timing from Day 2. Ishank feeds the measured values into the SimEvents model — that's the link between the two halves of the project.

---

## 7. Metrics you will be asked about

Pick these now and report them honestly. Judges respect a measured 0.86 far more than a claimed 0.97.

| Metric | Realistic target | Notes |
|---|---|---|
| Quadratic weighted kappa (APTOS val) | 0.83–0.90 | The standard DR grading metric |
| Referable DR (grade ≥ 2) sensitivity | ≥ 0.90 | This is the clinically meaningful one — lead with it |
| Referable DR specificity | ≥ 0.80 | Sensitivity/specificity trade-off is a design choice; say so |
| External test (Messidor-2) κ | expect a 0.05–0.15 drop | Reporting the drop honestly is a *strength* |
| Hard exudate Dice (IDRiD) | 0.60–0.72 | Achievable |
| Microaneurysm Dice (IDRiD) | 0.25–0.45 | Genuinely hard. Do not promise more. |
| IQA retake precision | ≥ 0.85 | On a hand-labelled set of ~100 images |
| End-to-end latency, median | < 12 s | Measure on the actual demo laptop |

**Dataset reality check:** IDRiD gives only 54 training images with pixel-level lesion masks. That is enough for a prototype with heavy patch-based augmentation and nothing more. Say that on the slide before a judge says it to you.

---

## 8. Risk register

| # | Risk | Trigger to watch | Owner | Mitigation |
|---|---|---|---|---|
| R1 | MATLAB runtime path breaks the demo | Compile fails or startup > 5 s by Day 16 | Kanchan + Divyanshu | `runtime: onnx` one-line rollback, already tested |
| R2 | Model too big / too slow on demo laptop | Latency > 20 s on Day 18 clean test | Kanchan | EfficientNet-B0 fallback, already trained |
| R3 | Segmentation Dice too low to show | MA Dice < 0.2 by Day 14 | Kanchan | Show exudates + haemorrhages only; drop MA from the visual |
| R4 | Messidor-2 access not granted in time | No download by Day 8 | Anshika | Hold out an APTOS split + IDRiD grading set as external proxy; say so plainly |
| R5 | Grad-CAM lands off-retina / on artefacts | Guard flags > 15% of images | Anshika | That's a *result*, not a bug — route those to specialist and present it as the trust mechanism |
| R6 | Integration debt piles up | Any member skips two merge windows | Abhishek (lead) | Standup escalation; pair session same day |
| R7 | Deck numbers ≠ code numbers | Any | Abhishek | Numbers freeze Day 19; single source is `docs/BENCHMARKS.md` |
| R8 | One member goes dark (exams, illness) | Missed 2 standups | Abhishek | Every module has a named backup (see §9) |

---

## 9. Ownership and backup

| Area | Primary | Backup |
|---|---|---|
| UI, voice, PDF | Abhishek | Divyanshu |
| IQA, enhancement, FOV, CDR | Muskan | Anshika |
| Grading, segmentation, quantization | Kanchan | Anshika |
| XAI, guards, validation, benchmarks | Anshika | Kanchan |
| Backend, DB, orchestration, sync/SMS | Divyanshu | Abhishek |
| Simulink, compression, hardware spec | Ishank | Divyanshu |

Backup means: you've read their guide, you can run their module, and you know where their artifacts live.

---

## 10. Demo script (7 minutes) — rehearse this, don't improvise

1. **0:00–0:45** — The gap. One eye doctor per 100,000 people; a fundus photo that's too dark to grade. Show the bad photo.
2. **0:45–1:30** — Capture with voice guidance in Hindi. Let the judges hear it.
3. **1:30–2:15** — Quality Gate rejects the bad photo, states *why*, worker retakes. **This is your differentiator — spend real time here.**
4. **2:15–3:15** — Good photo → grade, confidence, lesion overlay, Grad-CAM. Point at the heatmap and the lesion mask agreeing.
5. **3:15–3:45** — Show a case where they *disagree* → low confidence → auto-routed to specialist. This is the trust story.
6. **3:45–4:15** — Stage 4 case → emergency alert fires.
7. **4:15–5:00** — Pull the network cable. Run another case. It still works. Plug back in, watch it sync.
8. **5:00–6:00** — Simulink: district capacity chart, and the retake-rate sensitivity showing what the Quality Gate is worth in patients/day.
9. **6:00–7:00** — Honest numbers slide: what works, what doesn't, what's next.

Rehearse with the laptop you'll actually use, on battery, without internet.

---

## 11. What "done" looks like on submission day

- [ ] Fresh clone + `docs/SETUP.md` gets a stranger to a working app in under 20 minutes
- [ ] Demo runs offline, on battery, on the demo laptop, three times in a row
- [ ] `docs/BENCHMARKS.md` numbers match the deck exactly
- [ ] Simulink model runs and exports its charts from a clean MATLAB
- [ ] 3-minute demo video recorded (insurance against live failure)
- [ ] Every member can explain any other member's module for 60 seconds
