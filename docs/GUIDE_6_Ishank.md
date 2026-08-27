# Guide — Ishank Gupta
**Role:** Systems Simulation Engineer
**Mission:** answer the question no other team will answer — *what does it actually take to run this across a district, and what is the Quality Gate worth in patients screened per day?*

SIH26038 is a **MathWorks** problem statement. Your Simulink/SimEvents work isn't a side exhibit; it's the part of the submission that most directly speaks to the sponsor. Treat it as a first-class deliverable with its own results, not a diagram.

---

## What you own
`matlab/simulink/NETRA_district.slx`, `core/net/compression.py`, `docs/HARDWARE_SPEC.md`

The trap in your role is that you can drift into building a pretty model that nobody's numbers feed into. Avoid it by fixing three dates in your calendar right now: **Day 10** (Kanchan's latency), **Day 11** (Muskan's retake rate), **Day 15** (Divyanshu's payload sizes). Everything before those dates uses placeholder parameters; everything after uses measured ones. When you present, the line that lands is *"these aren't assumed numbers, they're measured from our own system."*

---

## Part 1 — The SimEvents district model

### What you're modelling

```
Patient arrivals (Poisson, λ per PHC)
        │
        ▼
  ASHA capture station  ──── service ~90 s ────┐
        │                                       │
        ▼                                       │
   Quality Gate  ── RETAKE (prob p) ────────────┘   ← the feedback loop that matters
        │ PASS
        ▼
  Edge inference  ── service ~8 s (measured) ──
        │
        ▼
  Routing split ──── ROUTINE (~70%) ──▶ done, sync later
        │
        ├──────────── REVIEW (~25%) ──▶ ┐
        └──────────── URGENT (~5%) ────▶ │  district specialist queue
                                          │  (n servers, priority discipline)
                                          ▼
                                     report signed off
Parallel: sync queue ──▶ bandwidth-gated upload window
```

### Build it in this order
1. **Single PHC, no retakes** — get arrivals, one server, a queue, and statistics working. Day 4–6.
2. **Add the retake feedback loop** — an `Entity Output Switch` on the quality outcome, routing failures back to the capture station. This is the heart of your model.
3. **Add the district layer** — N PHCs feeding one specialist queue with a priority discipline (urgent jumps the line).
4. **Add the sync/bandwidth branch** — uploads held until a connectivity window opens.

Key SimEvents blocks: `Entity Generator` (Poisson inter-arrival), `Entity Server`, `Entity Queue` (FIFO and priority), `Entity Output Switch` / `Input Switch`, `Entity Terminator`, and `Simulink Function` blocks for the probabilistic routing decisions. Use `Simulation Data Inspector` for the plots.

### Parameters — placeholders now, measured later

| Parameter | Placeholder (Days 1–10) | Source of the real value |
|---|---|---|
| Arrival rate λ per PHC | 12 patients/day | Diabetes prevalence × PHC catchment (~30k) × screening coverage — derive it and show your arithmetic |
| Capture service time | 90 s ± 30 s | Time Abhishek's UI flow yourself with a stopwatch |
| **Retake probability p** | 0.15 | **Muskan, Day 11** — and the *counterfactual* p without the Quality Gate |
| Inference latency | 8 s | **Kanchan, Day 10** |
| Specialist review time | 30 s (with NETRA) vs 180 s (manual) | Your deck already claims this — model it |
| Grade distribution | APTOS: ~50/10/27/8/5% | Kanchan's val set distribution |
| Payload per screening | 2 MB | **Divyanshu, Day 15** |
| Connectivity window | 4 h/day at 256 kbps | Assumption — state it |

### The results that matter

Produce these four charts. They are your slides.

1. **Patients screened per day vs number of camera stations per PHC** — with the knee of the curve marked. This is the district planning answer.
2. **Retake-rate sensitivity: p = 0.05 vs 0.15 vs 0.30** → throughput and mean patient wait time. **This is your headline result.** It converts Muskan's Quality Gate from a feature into a quantified operational benefit: *"cutting the retake rate from 30% to 10% adds N patients/day per PHC without adding any hardware."* No other team will have this.
3. **Specialist queue backlog vs number of specialists**, at 30 s review vs 180 s review. This is where your "80% doctor time saving" claim gets a number behind it instead of an assertion.
4. **Sync backlog vs available bandwidth** — how many hours of connectivity per day are needed to clear a district's daily volume, and what compression buys.

Run each as a parameter sweep (`sim` with a `SimulationInput` array, or `parsim` if you have Parallel Computing Toolbox). Export as PNG for Abhishek, and keep the generating script so it's reproducible.

---

## Part 2 — Bandwidth-adaptive compression

Small, concrete, and it feeds both the model and the product.

```python
# core/net/compression.py
def compress_for_sync(image_path, bandwidth_kbps) -> (path, meta)
```

Build a quality ladder: full-resolution PNG → JPEG q90 → q75 → q50 → downscaled JPEG. Measure actual bytes at each rung, and estimate upload seconds at a given bandwidth.

**Then measure the thing that matters:** does compression change the grade? Take ~100 images, run Kanchan's model on each rung, and plot **κ (or grade agreement) vs payload size**. That gives you a defensible answer to *"how much can we compress before it costs clinical accuracy?"* — and a recommended operating point. Do this around Day 13–15, once Kanchan's model is stable.

That plot is a genuinely good slide and it's yours alone.

---

## Part 3 — Hardware deployment scaling

Produce `docs/HARDWARE_SPEC.md`: a one-page minimum-spec sheet a district health officer could actually procure against.

Method: get Kanchan's inference profiling, then run the app yourself under constrained conditions — cap CPU threads to 2 and 4, cap available RAM, run on battery. Record:
- Median and p95 end-to-end latency per configuration
- Peak RAM
- Disk footprint (models + app + 1 year of local screening data — compute the storage growth rate from Divyanshu's payload numbers)
- Whether it degrades gracefully or falls over

Output a small table: **minimum spec** (works, slower), **recommended spec** (comfortable), and cost per station in ₹, multiplied out to a district. Pair it with your throughput curve and you have a costed deployment plan, which is exactly what the "scalability" judging criterion is asking for.

---

## Timeline

| Days | Deliverable |
|---|---|
| 1–3 | MATLAB/Simulink environment verified, SimEvents licensed. Read the SimEvents getting-started example end to end. Stub `compress_for_sync`. |
| 4–6 | Single-PHC model running with placeholder parameters. First throughput chart. |
| 7–9 | Retake feedback loop + district layer with specialist queue. Parameter sweep script. |
| 10 | **Receive latency/size/memory from Kanchan** → update model |
| 11 | **Receive retake rate from Muskan** → run the sensitivity sweep. Headline chart. |
| 12–13 | Bandwidth/sync branch. Compression ladder implemented and measured. |
| 14 | **Hand simulation charts as PNG to Abhishek** for the deck |
| 13–15 | κ-vs-payload experiment with Kanchan's model |
| 15 | **Receive payload sizes + sync timings from Divyanshu** → final sync model run |
| 16–18 | Hardware profiling under constrained CPU/RAM. `HARDWARE_SPEC.md`. |
| 19–21 | Freeze results. Rehearse your 60 seconds of the demo (script step 8). |

---

## Your acceptance test

Someone else opens MATLAB on a clean machine, opens `NETRA_district.slx`, runs `run_all_experiments.m`, and gets all four charts without editing anything. Parameters come from `params.m` (or a `.mat`), not from values typed into block dialogs — that's what makes a sweep possible and a model reviewable.

---

## Failure modes
- **Building a beautiful model with invented parameters.** The value is entirely in it being fed by your team's measured numbers. Chase those three handoffs.
- Modelling in isolation until week 3 and discovering the numbers don't exist. Ask for them early and repeatedly.
- Parameters buried in block dialogs — you can't sweep, and nobody can review it.
- Presenting the model as a diagram instead of as results. Nobody is impressed by a block diagram; they're impressed by "10% retake rate buys you N more patients per day."
- Treating the compression work as trivial. The κ-vs-payload curve is one of the more original things in the whole submission.

## Reading
- MathWorks: *SimEvents Getting Started* + the "Explore Statistics and Visualize Simulation Results" example
- Any queueing-theory primer on M/M/c — enough to sanity-check that your simulated mean wait matches the analytical result for the simple case. Doing that check and saying you did it is a strong credibility signal.
- MathWorks blog posts on hospital/clinic discrete-event models — good structural templates
