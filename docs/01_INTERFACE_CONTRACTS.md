# NETRA AI — Interface Contracts
**This is the document that lets six people build one system without meetings.**

Each of you owns exactly one block of one JSON object. You never call each other's code directly — you fill in your block and hand the object on. That's the entire collaboration model.

---

## 1. The object: `ScreeningResult`

```json
{
  "schema_version": "1.0",
  "screening_id": "b7c1e2f0-...",
  "patient_id": "PHC12-0043",
  "captured_at": "2026-09-04T10:32:11+05:30",
  "eye": "OD",
  "operator_id": "ASHA-0192",
  "phc_id": "PHC12",

  "image": {
    "raw_path": "data/captures/b7c1e2f0_raw.jpg",
    "processed_path": "data/captures/b7c1e2f0_proc.png",
    "width": 1536, "height": 1536,
    "fov_mask_path": "data/captures/b7c1e2f0_fov.png"
  },

  "quality": {                                          // OWNER: Muskan
    "verdict": "PASS",                                  // PASS | AUTO_CORRECTED | RETAKE
    "scores": {
      "blur": 0.82, "illumination": 0.91,
      "fov_coverage": 0.94, "contrast": 0.77, "artefact": 0.05
    },
    "reasons": [],                                      // e.g. ["BLUR_HIGH","FOV_PARTIAL"]
    "operator_message_key": "iqa.pass",
    "enhancement_applied": ["CLAHE_GREEN", "GAMMA_1.2"]
  },

  "grading": {                                          // OWNER: Kanchan
    "icdr_grade": 2,                                    // 0..4
    "grade_label": "Moderate NPDR",
    "probabilities": [0.03, 0.11, 0.68, 0.15, 0.03],
    "referable_dr": true,                               // grade >= 2
    "confidence": 0.68,
    "uncertain": false,
    "model_id": "effnetb0-dr", "model_version": "v0.4-int8"
  },

  "lesions": {                                          // OWNER: Kanchan
    "mask_path": "data/captures/b7c1e2f0_lesions.png",
    "counts": {"microaneurysms": 14, "haemorrhages": 6,
               "hard_exudates": 9, "soft_exudates": 1},
    "area_fraction": {"hard_exudates": 0.011, "haemorrhages": 0.004},
    "model_version": "unet-lite-v0.3"
  },

  "xai": {                                              // OWNER: Anshika
    "gradcam_path": "data/captures/b7c1e2f0_cam.png",
    "overlay_path": "data/captures/b7c1e2f0_overlay.png",
    "cam_lesion_agreement": 0.71,                       // IoU-style, 0..1
    "cam_outside_fov_fraction": 0.03,
    "guard_status": "OK"                                // OK | CAM_OFF_RETINA | LOW_AGREEMENT
  },

  "other_conditions": {                                 // OWNER: Muskan (CDR) + Anshika (flags)
    "glaucoma_suspect": {"cup_disc_ratio": 0.51, "flag": false},
    "hypertensive_retinopathy": {"flag": false, "evidence": []}
  },

  "longitudinal": {                                     // OWNER: Divyanshu
    "prior_screening_id": "a11f...", "prior_grade": 1,
    "delta": "+1", "trend": "WORSENING"                 // IMPROVING | STABLE | WORSENING | FIRST_VISIT
  },

  "routing": {                                          // OWNER: Divyanshu
    "action": "REVIEW",                                 // ROUTINE | REVIEW | URGENT_REFERRAL
    "reason": "REFERABLE_DR",
    "alert_sent": false, "sync_status": "PENDING"       // PENDING | SYNCED | FAILED
  },

  "timings_ms": {                                       // EVERYONE fills their own key
    "iqa": 780, "grading": 1120, "segmentation": 2410,
    "xai": 1650, "report": 1300, "db": 90, "total": 7350
  }
}
```

**Rules:**
- Your block is yours. Never write into someone else's block.
- Never *remove* a key. Adding an optional key is fine; announce it in the group.
- Every field must be present even when empty (`[]`, `null`, `0.0`). Downstream code should never need a `try/except KeyError`.
- Changing this file needs a group message first and two PR approvals.

---

## 2. Function signatures

These six functions are the entire API surface between you. Implement yours as a **pure function**: inputs in, dict out, no database, no globals, no file writes except the paths you return.

```python
# core/iqa/quality.py                                    MUSKAN
def assess_quality(bgr: np.ndarray) -> dict
    """Returns the `quality` block + writes fov_mask_path. Must run < 1500 ms."""

def enhance(bgr: np.ndarray, quality: dict) -> np.ndarray
    """Returns the corrected image. Idempotent. No-op if verdict == RETAKE."""

def cup_disc_ratio(bgr: np.ndarray, fov_mask: np.ndarray) -> float
    """Optional (Phase 2). Feeds other_conditions.glaucoma_suspect."""


# core/models/grading.py                                 KANCHAN
def grade(bgr: np.ndarray) -> dict
    """Returns the `grading` block. Must run < 1500 ms on CPU."""

# core/models/segmentation.py                            KANCHAN
def segment(bgr: np.ndarray) -> tuple[dict, np.ndarray]
    """Returns (lesions block, HxWx4 uint8 mask stack). < 3000 ms."""


# core/xai/gradcam.py                                    ANSHIKA
def explain(bgr, model_handle, grading: dict,
            lesion_mask: np.ndarray, fov_mask: np.ndarray) -> dict
    """Returns the `xai` block, writes cam + overlay PNGs. < 2000 ms."""


# db/dao.py                                              DIVYANSHU
def save_screening(result: dict) -> str
def get_history(patient_id: str, limit: int = 10) -> list[dict]
def compute_routing(result: dict) -> dict     # returns the `routing` block
def enqueue_sync(screening_id: str) -> None


# app/report.py                                          ABHISHEK
def render_result(result: dict) -> str        # HTML for the browser
def generate_pdf(result: dict) -> str         # returns path to PDF


# core/net/compression.py                                ISHANK
def compress_for_sync(image_path: str, bandwidth_kbps: float) -> tuple[str, dict]
    """Returns (compressed path, {"quality":int,"bytes":int,"est_seconds":float})."""
```

---

## 3. The orchestrator (Divyanshu owns this file)

```python
# core/inference.py
def run_screening(image_path, patient_id, eye, cfg) -> dict:
    r = new_result(patient_id, eye)                      # contracts.py
    bgr = imread(image_path)

    r["quality"] = iqa.assess_quality(bgr)
    if r["quality"]["verdict"] == "RETAKE":
        r["routing"] = {"action": "ROUTINE", "reason": "RETAKE_REQUESTED", ...}
        return r                                          # early exit, ~1.5 s

    bgr = iqa.enhance(bgr, r["quality"])
    r["grading"] = models.grade(bgr)
    r["lesions"], mask = models.segment(bgr)
    r["xai"] = xai.explain(bgr, handle, r["grading"], mask, fov)
    r["longitudinal"] = dao.compare_with_prior(patient_id, r)
    r["routing"] = dao.compute_routing(r)
    dao.save_screening(r)
    return r
```

Note the early exit on RETAKE. That path must be fast — it's the one the health worker hits most often, and it's the one that impresses judges.

---

## 4. Routing rules (agree these on Day 1, tune in Phase 2)

Lives in `configs/thresholds.yaml` so nobody has to edit code to change a threshold.

```yaml
routing:
  urgent_referral:    # SMS fires
    - grade >= 4
    - grade == 3 and confidence >= 0.7
  review:             # specialist queue, no SMS
    - grade >= 2
    - confidence < 0.55                    # model unsure at any grade
    - xai.guard_status != "OK"             # evidence doesn't line up
    - other_conditions.glaucoma_suspect.flag == true
  routine:
    - everything else
```

The third `review` rule is your novelty. When the heatmap and the lesion mask disagree, the system says "I'm not sure why I said this" and escalates instead of pretending. Anshika owns proving that this catches real failures.

---

## 5. Reason codes and message keys

Never put user-facing English in your module. Emit a key; Abhishek maps it to text and audio in every language.

| Code | Meaning | Voice prompt (Hindi example) |
|---|---|---|
| `BLUR_HIGH` | Laplacian variance below threshold | "तस्वीर धुंधली है — कैमरा स्थिर रखें" |
| `TOO_DARK` | Mean intensity in FOV too low | "रोशनी कम है — फ्लैश जाँचें" |
| `TOO_BRIGHT` | Clipped pixel fraction too high | "बहुत तेज़ रोशनी है" |
| `FOV_PARTIAL` | Retinal disc partly outside frame | "आँख को बीच में रखें" |
| `GLARE` | Specular artefact over macula | "कैमरा थोड़ा हिलाएँ" |
| `OFF_CENTRE` | FOV centroid far from image centre | "कैमरा सीधा रखें" |

Muskan adds codes; Abhishek adds translations and audio. Neither blocks the other.

---

## 6. Stub protocol (Day 2 — this is what makes parallel work possible)

Every module ships a stub in `core/stubs/` before it ships anything real:

```python
# core/stubs/iqa_stub.py
def assess_quality(bgr):
    return {"verdict": "PASS",
            "scores": {"blur": 0.85, "illumination": 0.9,
                       "fov_coverage": 0.95, "contrast": 0.8, "artefact": 0.02},
            "reasons": [], "operator_message_key": "iqa.pass",
            "enhancement_applied": []}
```

`configs/app.yaml` selects real vs stub per module:

```yaml
modules:
  iqa: stub          # flip to "real" when Muskan lands
  grading: stub
  segmentation: stub
  xai: stub
```

Now Abhishek can build the entire report page on Day 3 without a trained model existing, and Kanchan can train for a week without ever opening the UI. **Flip your own flag to `real` the day your module passes its acceptance test — that's your public "I'm done" signal.**

---

## 7. Contract test (runs on every PR)

```python
# tests/test_contracts.py
import jsonschema, json
from core.inference import run_screening

def test_result_schema():
    r = run_screening("tests/fixtures/sample_fundus.jpg", "TEST-001", "OD", cfg)
    jsonschema.validate(r, json.load(open("core/schema/screening_result.json")))

def test_timing_budget():
    r = run_screening(...)
    assert r["timings_ms"]["total"] < 30000

def test_retake_short_circuits():
    r = run_screening("tests/fixtures/blurry.jpg", "TEST-002", "OD", cfg)
    assert r["quality"]["verdict"] == "RETAKE"
    assert r["timings_ms"]["total"] < 3000
```

Three fixtures go in `tests/fixtures/` on Day 2: one good image, one blurry, one severe case. Everyone tests against the same three. Muskan sources them.

---

## 8. Handoff calendar — who unblocks whom, and when

| Day | From → To | What is handed over |
|---|---|---|
| 2 | Everyone → Divyanshu | Your stub, importable |
| 3 | Muskan → Everyone | Three test fixtures |
| 5 | Muskan → Kanchan | Preprocessing function (so training and inference preprocess identically — **critical, don't get this wrong**) |
| 7 | Kanchan → Anshika | First trained checkpoint + layer name for Grad-CAM |
| 8 | Muskan → Anshika | FOV mask function (needed for the CAM guard) |
| 9 | Divyanshu → Abhishek | Real `ScreeningResult` from the DB |
| 10 | Kanchan → Ishank | Measured inference latency, model size, memory |
| 11 | Muskan → Ishank | Measured retake rate on the fixture set |
| 12 | Kanchan → Everyone | Backbone benchmark table → Decision B locked |
| 13 | Anshika → Divyanshu | Guard status feeding routing rules |
| 14 | Ishank → Abhishek | Simulation charts as PNG for the deck |
| 15 | Divyanshu → Ishank | Real payload sizes + sync timings |
| 19 | Anshika → Abhishek | Final benchmark table for the deck |

**The Day-5 handoff is the one that silently ruins projects.** If Kanchan trains on images preprocessed one way and the app preprocesses them another way, accuracy quietly collapses at inference time and you'll waste three days finding it. Muskan's `enhance()` must be a single shared function used by *both* training and serving.
