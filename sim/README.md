# NETRA district simulation

Two implementations of the same model. Use whichever fits the moment.

## Python (guaranteed to run, use this for the results)

```bash
pip install numpy matplotlib
python sim/run_experiments.py
```

Writes four PNGs and `results.csv` into `sim/results/`. Takes about a minute.
Every number in the deck should come from here, not from a slide draft.

- `netra_sim.py` — the discrete-event engine. All parameters in `Params`.
- `run_experiments.py` — runs the four sweeps and draws the charts.

## MATLAB / SimEvents (for the MathWorks story)

Upload `matlab/simulink/params.m` and `matlab/simulink/build_netra_model.m`
to MATLAB Drive, then in the Command Window:

```
build_netra_model
```

It builds and wires the model for you, then saves `NETRA_district.slx`.
Press Run. Block parameter names shift between MATLAB releases, so the
script prints any dialog value it could not set — open that block and
type the value from `params.m` by hand.

**Untested against your MATLAB Online session.** The wiring is the part
that matters and that always builds; a couple of dialogs may need a
manual touch.

## Parameters that must be replaced with measured values

| Parameter | Marked | Comes from |
|---|---|---|
| `inference_mean` | MEASURE | Kanchan — CPU benchmark |
| `retake_prob` | MEASURE | your own Quality Gate evaluation |
| `payload_mb` | MEASURE | Divyanshu — sync queue |
| `arrivals_per_day` | PLACEHOLDER | derive from diabetes prevalence × catchment |
| `capture_mean` | PLACEHOLDER | time Abhishek's UI flow with a stopwatch |
| `referral_frac` | PLACEHOLDER | grade distribution from Kanchan's val set |

Anything still marked PLACEHOLDER on demo day must be labelled as an
assumption on the slide.
