# NETRA-AI — Current Work Review and Next Implementation Plan

**Review date:** 1 September 2026
**Reviewed revision:** `origin/main` at `b800867`
**Scope:** latest merged backend, model, XAI, IQA, simulation, and UI work.

## 1. Current assessment

NETRA-AI now has a credible offline end-to-end foundation. The application has a visual two-eye capture workflow, real inference orchestration, local SQLite persistence, offline sync primitives, quality gating, clinical-routing output, and Grad-CAM/XAI support. The main remaining risk is deployment verification: PyTorch must be installed on the target laptop and the exact supplied checkpoints must be exercised through the real path before presentation claims are final.

### Validation completed locally

- Both supplied production artifacts were extracted into the ignored `artifacts/` directory.
- The complete test suite passes: **38/38 tests** in 102.48 seconds on the demo laptop.
- The demo environment uses Python 3.12.9 with `torch 2.6.0+cpu` and `torchvision 0.21.0+cpu`; both production model suites pass.
- Flask route, two-eye upload, durable history, and persisted result smoke tests pass.
- Generated retinal media is served only from `data/captures/`; model files are not exposed by the web application.
- Python compilation passes for `app`, `core`, `db`, `alerts`, `sync`, and `run.py`.
- The pipeline fails safely to `REVIEW` when the local PyTorch runtime is unavailable; it does not invent a positive diagnosis.

### Artifact integrity

| Artifact | SHA-256 |
|---|---|
| `netra_dr_effb0_muskan_preproc.pth` | `33ADC78704BA2A53794CA8286097F29693592EFA75E71250DB4211860261E519` |
| `FINAL_V8_SEGMENTATION.pth` | `B1F157459D3570ADF3139B80C11BAA7DA6E2E2B5526D5910540B831091776A6E` |

The weights remain intentionally untracked, so they must be provided alongside any demo laptop.

## 2. Team review

### Divyanshu — backend, persistence, and offline sync

Delivered the core SQLite DAO, sync queue/worker, alerts queue/sender, screening API routes, and the inference orchestration layer. The orchestrator now joins IQA, grading, segmentation, XAI, routing, longitudinal comparison, artifact output, and persistence under one screening contract.

**Review:** this is now useful production-style plumbing, not merely a scaffold. The next check is a real-device sync rehearsal against the intended server endpoint; avoid representing queued sync as proven transmission until that is measured.

### Kanchan — model integration and grading

Delivered the EfficientNet-B0 grader and four-channel U-Net segmenter, strict checkpoint loading, preprocessing, post-processing, metadata, and model handoff documentation. The supplied grading and segmentation weights match the paths expected by the code.

**Review:** the model packaging is clear and the pipeline is wired correctly. The exact frozen segmentation checkpoint still needs official-test evaluation before quoting an exact Dice score for that artifact. Model sizes also exceed the earlier compact target, so do not claim a lightweight/quantised deployment.

### Anshika — explainability and clinical alignment

Delivered real Grad-CAM integration, FOV and CAM-lesion guard logic, mask-resolution fixes, benchmark updates, and `docs/CLINICAL_ALIGNMENT.md`.

**Review:** the clinical-alignment document is presentation-ready and the guard is an important safety feature. In the interface, missing or failed XAI must remain visibly unavailable and route to review—not be shown as evidence.

### Muskan — image-quality assessment

Delivered FOV detection, enhancement, blur/contrast/illumination assessment, and structured quality outputs.

**Review:** this is a strong operational gate. Thresholds should be tuned on clinically labelled field images; synthetic degradations prove robustness engineering, not real-world retake accuracy.

### Ishank — simulation and QA tools

Delivered the district-capacity simulator, result charts, and the degraded-image generator for quality-gate testing.

**Review:** the simulator is valuable scenario analysis. Replace placeholder timings with measurements from this integrated pipeline before using the charts as operational evidence.

## 3. Abhishek — UI/UX and integration contribution

The original UI scaffold is restored and now being connected to team work rather than fake results:

- Polished, high-contrast, responsive two-eye capture, quality, combined-result, report, and history screens.
- Visual-first interaction with camera/upload fallback, clear quality retake feedback, large touch targets, and a judge-friendly evidence layout.
- No audio dependency: the workflow remains usable without Hindi narration.
- Flask blueprints repaired so both the web UI and backend API are registered.
- Real upload flow: each eye is analysed by `run_screening`, persisted to SQLite, then shown in the combined screen.
- Real result/history links survive a Flask restart by loading the complete result contract from the offline database.
- Evidence images are served from the generated capture folder with traversal protection; external model weights are never exposed through a route.
- Dependency manifest now declares PyTorch, Torchvision, and Matplotlib.

## 4. Immediate implementation plan

| Order | Work | Owner | Done when |
|---|---|---|---|
| 1 | Finish the real-model environment | Abhishek + Kanchan | PyTorch/Torchvision install cleanly, both checkpoints load, and `tests/test_grading.py` plus `tests/test_segmentation.py` pass. |
| 2 | Run the full visual workflow | Abhishek | A real OD/OS upload produces quality, clinical result, evidence, printable report, and persisted history without a browser/server restart. |
| 3 | Harden result safety | Abhishek + Anshika | Missing model/XAI, low confidence, and non-OK guard states show a clear manual-review outcome with no misleading image evidence. |
| 4 | Verify offline sync and alerts | Divyanshu | Queue, retry, duplicate protection, and alert routing are demonstrated on a disconnected/reconnected laptop. |
| 5 | Measure and document | Entire team | Record end-to-end timing, real retake observations, checkpoint metadata, and the exact test commands used for the demo machine. |
| 6 | Rehearse and freeze | Entire team | Three complete visual-only demo rehearsals pass; presentation claims match the measured benchmarks and clinical-alignment document. |

## 5. Demo-quality acceptance gates

1. `python run.py` launches the interface locally with no network requirement.
2. A user can complete the two-eye flow using visuals alone—no audio prompts required.
3. Both real checkpoints load from `artifacts/` and a result validates against the shared contract.
4. A restart does not lose result pages or patient history.
5. Any unavailable model, poor image, or XAI disagreement produces a plainly visible safe next action.
6. The presentation states only measured model, timing, and capacity figures.

## 6. Decision

Prioritise real-model verification and demo rehearsal over new features. The visual UI is already strong enough to make a good first impression; the highest-value improvement now is making every screen demonstrably backed by the actual offline pipeline.
