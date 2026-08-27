# Guide — Divyanshu Kumar
**Role:** Systems & Backend Lead
**Mission:** everything holds together, offline, and nothing is ever lost.

You own the orchestrator, which means you're the first person to feel every integration problem. That's uncomfortable but useful: you'll know two days before anyone else which seam is about to fail. Say so loudly in standup when you do.

---

## What you own
`core/inference.py`, `db/`, `app/routes/`, sync queue, alerting
Contract functions: `save_screening`, `get_history`, `compute_routing`, `enqueue_sync`, `compare_with_prior`

---

## The orchestrator

`core/inference.py` (spelled out in `01_INTERFACE_CONTRACTS.md` §3) is the spine of the application. Build it on **Day 2 against everyone's stubs** and get the whole pipeline running end to end before a single real model exists. That single act is what lets five other people work in parallel for three weeks.

Two things it must do well:
1. **Module selection from config.** `configs/app.yaml` decides real vs stub per module, and ONNX vs MATLAB for the runtime. Everyone flips their own flag when they land. No code changes to switch.
2. **Fail soft.** If segmentation crashes, you still return a grade and a heatmap with `lesions: null` and a warning — you do not 500 the request. A demo that degrades gracefully survives; one that throws a stack trace on stage does not. Wrap each stage, log the exception, keep going.

Also enforce the early exit on `verdict == RETAKE`. That path skips inference entirely and must come back in under 3 seconds.

---

## Database

SQLite. Not Postgres, not MySQL — you need a single file that works with zero setup on a PHC laptop, and SQLite is genuinely the right engineering choice here, not a compromise. Say that on the slide.

```sql
CREATE TABLE patients (
  patient_id TEXT PRIMARY KEY, phc_id TEXT, name_hash TEXT,
  age INTEGER, sex TEXT, diabetes_years INTEGER, created_at TEXT);

CREATE TABLE screenings (
  screening_id TEXT PRIMARY KEY, patient_id TEXT REFERENCES patients,
  eye TEXT, captured_at TEXT, operator_id TEXT,
  quality_verdict TEXT, icdr_grade INTEGER, confidence REAL,
  referable INTEGER, guard_status TEXT,
  routing_action TEXT, sync_status TEXT,
  result_json TEXT,                        -- full ScreeningResult
  raw_image_path TEXT, overlay_path TEXT);

CREATE TABLE sync_queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT, screening_id TEXT,
  payload_bytes INTEGER, attempts INTEGER DEFAULT 0,
  last_attempt TEXT, status TEXT);

CREATE TABLE alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, screening_id TEXT,
  channel TEXT, recipient_hash TEXT, sent_at TEXT, status TEXT);

CREATE INDEX idx_screen_patient ON screenings(patient_id, captured_at);
CREATE INDEX idx_sync_status ON sync_queue(status);
```

Store the full `ScreeningResult` as JSON in `result_json` *and* denormalise the fields you query on. You get flexibility (the schema can evolve) plus fast queries. This is the right pattern for a project where the contract is still moving.

**Privacy:** hash patient names and phone numbers; never store raw identifiers. Note in the deck that you're aligned with DPDP Act principles — data minimisation, local storage, no cloud dependency. It's a real point and it's free.

---

## Longitudinal tracking

`compare_with_prior(patient_id, result)` pulls the most recent prior screening for the same patient and eye, and returns the `longitudinal` block: prior grade, delta, trend. Simple, but it's what a follow-up visit at a PHC actually needs, and no competing team will have it.

---

## Routing and alerts

Rules live in `configs/thresholds.yaml` (see contracts §4) so Anshika and Abhishek can tune them without touching your code.

**SMS/WhatsApp:** for the prototype, implement against an interface with two backends —
- `MockSender` — writes to a log file and shows on screen. **This is what you demo.**
- `TwilioSender` (or Gupshup/MSG91) — real, if a free trial number is available.

Do not make your live demo depend on an SMS API and a working network. Show the mock firing, and mention the real backend exists. Judges understand this; a failed live SMS is a disaster and a mocked one is a non-event.

Include a **kill switch**: `alerts.enabled: false` in config. If anything goes wrong on stage, you flip one line.

---

## Offline-first sync

The claim on your slide is "works offline, syncs when connectivity returns." Make it literally true and then *demonstrate* it:

1. Everything writes to SQLite first, always. Never block on network.
2. `sync_queue` gets a row per screening with payload size.
3. A background worker retries with exponential backoff whenever connectivity is detected.
4. Idempotent upload — use `screening_id` as the key so a retry never duplicates.

**Demo moment (step 7 in the demo script): pull the network cable, run a screening, plug it back in, watch the queue drain.** Rehearse it. It's forty seconds and it proves more than any slide.

Ishank needs your real payload sizes and sync timings on Day 15 for his bandwidth model — instrument the queue to log bytes and elapsed time from the start.

---

## Timeline

| Days | Deliverable |
|---|---|
| 1–2 | Repo scaffolding, `configs/app.yaml`, **`core/inference.py` wired to everyone's stubs** |
| 3 | Integration call #1 — full fake pipeline runs. `tests/test_contracts.py` green in CI. |
| 4–6 | SQLite schema, DAO layer, screenings persisted, Flask routes for capture/result |
| 7–8 | Patient history query, `compare_with_prior`, longitudinal block |
| 9 | **Hand a real `ScreeningResult` from the DB to Abhishek** |
| 10–12 | Swap in real modules as they land; fail-soft wrappers; timing instrumentation |
| 13–14 | Routing rules from config, alert interface + mock sender, sync queue with retry |
| 15 | **Hand payload sizes + sync timings to Ishank** |
| 16–18 | Clean-machine test — fresh laptop, fresh clone, `docs/SETUP.md`, time it. You own `SETUP.md`. |
| 19–21 | Demo rehearsals; you drive the offline/sync demo moment |

---

## Your acceptance test

On a machine that has never seen the project: clone, follow `docs/SETUP.md` verbatim, reach a completed screening in under 20 minutes without asking anyone a question. Do this on Day 16 with a friend outside the team watching you not help them.

Then: run 50 screenings in a loop, kill the process mid-run, restart. Nothing lost, nothing duplicated.

---

## Failure modes
- Building the orchestrator after the modules exist. Build it first, against stubs, on Day 2.
- Hard failures on one module taking down the whole request. Wrap every stage.
- Storing only denormalised columns — then the contract changes and you've lost data. Keep `result_json`.
- Depending on a live SMS API in the demo.
- `SETUP.md` written from memory on Day 20. Write it as you go; test it on someone else.

## Reading
- SQLite "When to use SQLite" docs — one paragraph justifies your architecture choice to a judge
- Flask blueprints, for keeping `app/routes/` from becoming one 900-line file
- DPDP Act 2023, data minimisation principles — skim, cite one line
