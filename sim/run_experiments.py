"""
NETRA AI - run all district-capacity experiments and produce the charts.

    python sim/run_experiments.py

Writes four PNGs and a CSV into sim/results/.
Every number printed here comes from sim/netra_sim.py, not from a slide.
"""

import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netra_sim import Params, run_replications

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
REPS = 40

plt.rcParams.update({
    "figure.figsize": (7.5, 4.4), "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})

rows = []


def record(experiment, x_label, x, out):
    rows.append({"experiment": experiment, x_label: x, **out})


# ---------------------------------------------------------------------
# 1. HEADLINE: what is the Quality Gate worth?
# ---------------------------------------------------------------------
def exp_retake_sensitivity():
    ps = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    wait, screened = [], []
    for p in ps:
        o = run_replications(Params(retake_prob=p, arrivals_per_day=70), REPS)
        wait.append(o["mean_wait_capture_min"])
        screened.append(o["screened"])
        record("retake_sensitivity", "retake_prob", p, o)

    fig, ax1 = plt.subplots()
    ax1.plot([x * 100 for x in ps], wait, "o-", color="#c0392b", lw=2,
             label="Mean patient wait")
    ax1.set_xlabel("Retake rate (%)")
    ax1.set_ylabel("Mean wait for capture (min)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")

    ax2 = ax1.twinx()
    ax2.plot([x * 100 for x in ps], screened, "s--", color="#2471a3", lw=2,
             label="Patients screened")
    ax2.set_ylabel("Patients screened per day", color="#2471a3")
    ax2.tick_params(axis="y", labelcolor="#2471a3")
    ax2.grid(False)

    ax1.axvspan(5, 12, alpha=0.10, color="green")
    ax1.text(8.5, max(wait) * 0.93, "with\nQuality Gate", ha="center",
             fontsize=9, color="green")
    ax1.axvspan(28, 40, alpha=0.10, color="red")
    ax1.text(34, max(wait) * 0.93, "without", ha="center",
             fontsize=9, color="#c0392b")

    plt.title("Impact of the Quality Gate on PHC throughput and waiting time")
    fig.tight_layout()
    fig.savefig(f"{OUT}/1_retake_sensitivity.png")
    plt.close(fig)

    lo = run_replications(Params(retake_prob=0.10, arrivals_per_day=70), REPS)
    hi = run_replications(Params(retake_prob=0.30, arrivals_per_day=70), REPS)
    return lo, hi


# ---------------------------------------------------------------------
# 2. How many camera stations does a PHC need?
# ---------------------------------------------------------------------
def exp_stations():
    ns = [1, 2, 3, 4]
    loads = [60, 90, 120]
    fig, ax = plt.subplots()
    for load in loads:
        y = []
        for n in ns:
            o = run_replications(
                Params(n_capture_stations=n, arrivals_per_day=load), REPS)
            y.append(o["screened"])
            record("stations", "n_stations", n, o)
        ax.plot(ns, y, "o-", lw=2, label=f"{load} patients/day arriving")
    ax.set_xlabel("Camera stations per PHC")
    ax.set_ylabel("Patients screened per day")
    ax.set_xticks(ns)
    ax.legend()
    plt.title("Screening capacity vs camera stations per PHC")
    fig.tight_layout()
    fig.savefig(f"{OUT}/2_capacity_vs_stations.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# 3. District specialists: what does the visual report buy?
# ---------------------------------------------------------------------
def exp_specialists():
    ns = [1, 2, 3, 4]
    fig, ax = plt.subplots()
    for netra, label, colour in ((True, "With NETRA visual report (0.5 min/case)", "#2471a3"),
                                 (False, "Manual review (3 min/case)", "#c0392b")):
        y = []
        for n in ns:
            o = run_replications(
                Params(n_specialists=n, use_netra_review=netra), REPS)
            y.append(o["mean_wait_specialist_min"])
            record("specialists", "n_specialists", n, o)
        ax.plot(ns, y, "o-", lw=2, label=label, color=colour)
    ax.set_xlabel("Ophthalmologists at the district hospital")
    ax.set_ylabel("Mean wait for specialist review (min)")
    ax.set_xticks(ns)
    ax.legend()
    plt.title("District specialist load (8 PHCs feeding one district hospital)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/3_specialist_load.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# 4. Where does a single-station PHC break?
# ---------------------------------------------------------------------
def exp_load():
    loads = [30, 45, 60, 75, 90, 105, 120]
    fig, ax1 = plt.subplots()
    screened, wait = [], []
    for load in loads:
        o = run_replications(Params(arrivals_per_day=load), REPS)
        screened.append(o["screened"])
        wait.append(o["mean_wait_capture_min"])
        record("load", "arrivals_per_day", load, o)

    ax1.plot(loads, screened, "o-", color="#2471a3", lw=2)
    ax1.plot(loads, loads, "--", color="grey", lw=1, label="ideal (all screened)")
    ax1.set_xlabel("Patients arriving per day")
    ax1.set_ylabel("Patients actually screened", color="#2471a3")
    ax1.tick_params(axis="y", labelcolor="#2471a3")
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(loads, wait, "s--", color="#c0392b", lw=2)
    ax2.set_ylabel("Mean wait (min)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.grid(False)

    plt.title("Single-station PHC: where capacity runs out")
    fig.tight_layout()
    fig.savefig(f"{OUT}/4_load_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    print("Running NETRA district simulation experiments...\n")

    lo, hi = exp_retake_sensitivity()
    exp_stations()
    exp_specialists()
    exp_load()

    keys = sorted({k for r in rows for k in r})
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("=" * 66)
    print("HEADLINE RESULT - what the Quality Gate is worth")
    print("=" * 66)
    print(f"  Retake rate 30% (no gate) : {hi['screened']:5.1f} screened/day, "
          f"{hi['mean_wait_capture_min']:5.1f} min mean wait")
    print(f"  Retake rate 10% (gate)    : {lo['screened']:5.1f} screened/day, "
          f"{lo['mean_wait_capture_min']:5.1f} min mean wait")
    print(f"  Gain                      : "
          f"+{lo['screened'] - hi['screened']:.1f} patients/day per PHC, "
          f"wait cut {hi['mean_wait_capture_min'] / max(lo['mean_wait_capture_min'], 0.1):.1f}x")
    print(f"  Across 8 PHCs             : "
          f"+{(lo['screened'] - hi['screened']) * 8:.0f} patients/day, "
          f"no extra hardware")
    print()
    print(f"Charts and results.csv written to {OUT}/")
