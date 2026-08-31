# NETRA-AI — Work Review, Abhishek's Contribution, and Phase 3 Plan

**Review date:** 31 August 2026  
**Reviewed remote revision:** `origin/main` at `619d601` — *Add inference pipeline (#10)*  
**Review basis:** current GitHub `main`, project contracts, committed tests, and an isolated source inspection.

## 1. Executive review

The team has delivered meaningful module-level work: IQA, synthetic quality-test tooling, simulation, XAI guard utilities, grading/segmentation wrappers, a result schema, fixtures, and individual tests. The strongest contributions are the IQA implementation and the candid model handoff documentation.

However, the remote `main` branch is **not demo-ready**. It contains a merge-history regression that removed the Flask UI, `run.py`, all translations, and `configs/app.yaml`. As a result, it cannot start the application or run the default inference pipeline. The immediate goal is therefore **integration recovery**, not new features.

### Validation performed

- Refreshed `origin/main` from GitHub and reviewed commits through `619d601`.
- Inspected the complete diff from the shared base and all production paths in `core/`, `sim/`, `tests/`, and `docs/`.
- Ran Python bytecode compilation successfully on an isolated checkout.
- Ran the real IQA module successfully on `tests/fixtures/good.png`; it returned `AUTO_CORRECTED` at a capped working size of `1024 × 686`.
- Tried the default `run_screening(...)` path. It fails with `FileNotFoundError` because `configs/app.yaml` is absent from `origin/main`.
- Could not run the committed pytest suite in the current environment because dependencies are not installed. More importantly, the committed `requirements.txt` omits packages required by the model and simulation code (`torch`, `torchvision`, and `matplotlib`).

### Release-blocking findings

| Priority | Finding | Impact | Required resolution |
|---|---|---|---|
| P0 | The merge to remote `main` deleted `run.py`, the Flask app, all four templates, static CSS/JS, report code, i18n loader, and `configs/i18n/*`. | The user-facing screening application, Hindi workflow, PDF/report work, and launch command are gone from GitHub `main`. | Restore Abhishek's UI commits into `main` before further feature work. Preserve the teammate modules while restoring the deleted files. |
| P0 | `configs/app.yaml` was deleted, but `core/inference.py` loads it by default. | Every default `run_screening(...)` call fails before IQA or model inference. | Restore a versioned config with explicit `stub`/`real` module choices and a default demo-safe configuration. |
| P0 | The inference orchestrator does not match real module APIs. It calls module-level `grade()` and `segment()` and expects segmentation to return `(lesions, mask)`. The real files expose `DRGrader.grade()` and `DRSegmenter.segment()` classes, while the latter returns one dictionary. | Selecting the real models cannot complete the pipeline even after restoring config and artifacts. | Create adapters or a shared factory with one contract: `grade(bgr) -> grading block`; `segment(bgr) -> (lesions block, mask)`. Add a real-mode integration test. |
| P0 | `core/xai/explain.py` is referenced by the module map but is not in the repository; the pipeline currently never invokes XAI, routing, persistence, report generation, or timing aggregation. | The promised anatomical XAI, safety routing, history, database, PDF, and <30 s measured result are not integrated. | Implement/wire the XAI entry point, routing, DAO, report, and timing steps before claiming an end-to-end pipeline. |
| P1 | `requirements.txt` omits `torch`, `torchvision`, and `matplotlib`; model files also require external checkpoints that are not part of a fresh checkout. | A clean-machine setup cannot run the models or simulation using the declared requirements. | Add pinned runtime dependencies and a documented artifact bootstrap/checksum step. Make tests skip clearly when optional artifacts are unavailable. |
| P1 | `core/__init__.py` was deleted. | `core` becomes a namespace package and can collide with unrelated installed packages named `core`; this occurred during isolated validation. | Restore `core/__init__.py` and run all commands as a package from the repository root. |
| P1 | IQA reason/message keys do not match the restored UI translations: for example IQA emits `iqa.corrected` and `iqa.retake.blur_high`, whereas the existing UI had `iqa.auto_corrected` and `iqa.retake.blur`. | Once the UI is restored, operators can see raw keys instead of understandable text/audio. | Define one reason-code-to-i18n-key mapping and test every emitted IQA key in `en.json` and `hi.json`. |
| P2 | Simulation charts are useful scenario analysis but should not yet be presented as measured operational findings. The model charges full inference time on every retake and continues specialist processing past the clinic-day boundary; committed results also depend on placeholders. | Capacity claims can be overstated or internally inconsistent. | Separate capture/IQA/pass-through stages, stop/report at the day boundary, and regenerate charts with measured parameters. |
| P2 | Formatting checks report trailing whitespace in thresholds, inference code, and generated CSV. | Low risk, but it obscures meaningful review noise. | Add a formatting/lint check and regenerate CSVs cleanly. |

## 2. Review of work completed by the team

### Muskan / Ishank — IQA, enhancement, fixtures, and QA tooling

**Delivered well**

- `core/iqa/fov.py` creates a retinal field-of-view mask so quality scoring is not distorted by the black border.
- `core/iqa/quality.py` evaluates blur, illumination, FOV coverage, contrast, and centre offset; it returns structured verdicts and reason codes rather than UI text.
- `core/iqa/enhance.py` uses illumination correction and green-channel CLAHE, and the later resolution cap protects the latency budget.
- Quality fixtures, schema checks, synthetic degradation generation, and compression support provide a sensible foundation for reproducible tuning.

**Review notes**

- The module is the most operationally mature piece currently pushed. The isolated fixture test completed successfully.
- Its thresholds still need tuning against a labelled/clinically reviewed set; synthetic degradation is useful for development but does not prove real-world retake precision.
- Reason code/message-key alignment must be resolved with the restored vernacular UI before integration.

### Ishank — district simulation and MATLAB/SimEvents path

**Delivered well**

- `sim/netra_sim.py` is a readable discrete-event model with configurable clinic capacity, retakes, referrals, and specialist review.
- `sim/run_experiments.py` produces reproducible charts and CSV outputs, while MATLAB scripts provide a credible SimEvents narrative.
- The simulation README honestly identifies placeholder parameters that must become measured values.

**Review notes**

- Add `matplotlib` to requirements or a dedicated simulation extras file.
- Treat current graphs as sensitivity scenarios, not evidence of field capacity, until all `MEASURE` values are replaced from the integrated app.
- Correct retake/service-time and end-of-day specialist queue logic before the charts are used in the final deck.
- The MATLAB builder is explicitly untested against the target MATLAB session; run and record one successful build before presentation day.

### Anshika — XAI guards and validation scaffolding

**Delivered well**

- `core/xai/guards.py` implements useful safety logic: CAM energy outside the FOV, CAM-lesion agreement, and an anatomical plausibility utility.
- The XAI stub and guard tests give downstream work a defined shape.
- `docs/BENCHMARKS.md` establishes the right principle: only measured values should reach the deck.

**Review notes**

- There is no real Grad-CAM/explain module in `main`, despite the inference module mapping `real` XAI to `core.xai.explain`.
- All XAI metrics and guard trigger rates remain `TBD`; keep the UI in “evidence unavailable” mode rather than showing fabricated overlays.
- The missing implementation must be added and connected to routing before the system can claim anatomically guarded explanations.

### Kanchan — grading and lesion segmentation

**Delivered well**

- The EfficientNet-B0 grading wrapper validates input, loads a named checkpoint strictly, returns a clear grading block, and reuses the IQA preprocessing path.
- The segmentation wrapper includes a concrete U-Net architecture, per-lesion thresholds, post-processing, output statistics, and direct tests.
- `docs/KANCHAN_MODEL_HANDOFF.md` is unusually transparent about model size, lack of deployed ONNX/INT8/MATLAB paths, artifact locations, and the weak microaneurysm result. That honesty should be retained in the presentation.

**Review notes**

- The model wrappers do not expose the module-level functions expected by `core/inference.py`; this is a P0 integration issue, not a model-quality issue.
- Both production artifacts are external downloads. Their version, checksum, availability, and clean-machine loading must be verified.
- The 46.41 MB grading checkpoint and ~89 MB segmentation checkpoint exceed the original compact/offline target. Do not claim <15 MB or a completed ONNX/INT8/MATLAB deployment.
- Grading tests skip when its artifact is absent, but segmentation tests instantiate the real segmenter without an equivalent skip; adjust the test policy for fresh clones.

### Divyanshu — contract and inference foundation

**Delivered well**

- `core/contracts.py` provides a complete result skeleton and `core/schema/screening_result.json` creates a common data vocabulary.
- `core/inference.py` correctly begins with image loading, metadata, IQA, and an early return on `RETAKE`.

**Review notes**

- This is a partial orchestration foundation, not yet a complete backend: the repository has no DB/DAO implementation, routing engine, sync queue, or app endpoints.
- The default config is absent and real module interfaces are incompatible with the orchestration calls.
- `timings_ms` is initialized but never measured; the project cannot validate its <30 s claim yet.
- Add one integration test that starts from a fixture and validates the complete result schema in both all-stub and all-real modes.

## 3. Abhishek Dhama's part

### Build contribution

Abhishek completed the Checkpoint 1 UI scaffold locally in commits `e8463d2` and `98592a1`:

- Flask application factory, routes, fake-data stubs, and launch entry point.
- Four ASHA-focused screens: capture, quality feedback/retake, result, and patient history.
- High-contrast responsive CSS and vanilla JavaScript designed for a 1366×768, difficult-lighting environment.
- Translation layer plus English, Hindi, Tamil, and Telugu JSON files.
- Audio directory scaffold for offline recorded prompts.
- Result-report/PDF function stubs and template endpoint corrections.

This work satisfies the correct early-stage objective: the UI can be built against stable fake outputs before model delivery. It is **not present on current GitHub `main`** because the remote merge history deleted the files. The next action is recovery and integration, not re-creating the design from scratch.

### Team-lead contribution and immediate responsibilities

- Restore the “green main” rule: no merge to `main` without a launch and fixture smoke test.
- Own the seam between IQA reason codes, UI translations, offline audio, and the retake loop.
- Run integration calls around an explicit demo script: capture → IQA → retake or result → evidence → report → history.
- Keep claims in the deck aligned with `docs/BENCHMARKS.md` and label unmeasured values as assumptions.
- Enforce a recovery-first feature freeze until the application can complete one all-stub and one real-model run on a clean machine.

## 4. Phase 3 — next implementation plan

### Goal

By the end of this phase, a fresh laptop can run one complete offline screening using a real image, produces a contract-valid result and human-readable report, and has an honest fallback when a model artifact is unavailable.

### Work sequence

| Order | Work item | Owner(s) | Definition of done |
|---|---|---|---|
| 1 | Recover application baseline | Abhishek + Divyanshu | Restore `app/`, `run.py`, `configs/app.yaml`, i18n, report code, and `core/__init__.py` into a branch based on current remote `main`. Launching the Flask app works with all modules set to stubs. |
| 2 | Freeze the executable contract | Divyanshu + Kanchan + Anshika + Muskan | Add adapters so stub and real modules implement identical functions and outputs. Add the missing `core.xai.explain` entry point. Validate full results against `screening_result.json`. |
| 3 | Make deployment reproducible | Kanchan + Ishank | Pin required dependencies, document artifact download/checksums, make tests handle absent optional artifacts clearly, and perform a clean-machine setup test. |
| 4 | Wire the operator flow | Abhishek + Divyanshu | Map every IQA reason to translated text and offline audio; implement the retake short-circuit; render real grading/lesion/XAI data safely; generate a one-page PDF from the same report view. |
| 5 | Add safety routing and persistence | Divyanshu + Anshika | Implement routing from grade, confidence, and XAI guard status; persist screening/history records; show a clear error/fallback state when models, camera, or artifacts are unavailable. |
| 6 | Measure and harden | Entire team, led by Abhishek | Run all fixtures through the real path, record timings and IQA retake metrics, regenerate simulation inputs/charts from measured values, then rehearse the full demo three times. |

### Non-negotiable acceptance gates

1. `python run.py` starts the local application from a fresh clone.
2. The all-stub flow completes capture to report without network access.
3. The real-model flow either completes successfully or displays a safe, translated fallback; it never crashes on a missing artifact.
4. Every emitted IQA message key exists in English and Hindi, with offline audio for the critical Hindi prompts.
5. `pytest tests/ -v` passes in documented clean-machine conditions; real-model tests may skip only with explicit artifact-missing messages.
6. One fixture result validates against the shared schema and records all stage timings.
7. Only then should the team update the deck with measured figures and move to polish/new capabilities.

## 5. Decision

**Do not add new disease features, languages, or presentation claims yet.** First restore the removed UI/configuration, repair the module interfaces, and prove a single end-to-end offline screening. This is the shortest route to a credible, demo-safe NETRA-AI submission.
