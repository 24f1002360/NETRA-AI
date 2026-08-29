"""
NETRA AI - district capacity simulation.

Discrete-event model of a diabetic retinopathy screening day at a PHC,
including the Quality Gate retake loop and the district specialist queue.

    arrivals (Poisson)
        |
        v
    capture station(s)  <----------+
        |                          |
        v                          | retake (prob p)
    Quality Gate ------------------+
        | pass
        v
    edge inference (grading + segmentation + XAI)
        |
        +--> routine (no specialist)
        |
        +--> referred --> district specialist queue --> reviewed

The retake loop is the point of the whole model: it is what turns
Muskan/Ishank's Quality Gate from a "feature" into a number of
patients per day.

All parameters live in Params. Replace the PLACEHOLDER ones with
measured values as they arrive from the team.
"""

from dataclasses import dataclass, field
import heapq
import numpy as np


@dataclass
class Params:
    # --- clinic day -------------------------------------------------
    day_minutes: float = 480.0          # 8-hour PHC day
    arrivals_per_day: float = 60.0      # PLACEHOLDER: patients arriving to screen
    n_capture_stations: int = 1         # cameras + operators at the PHC
    n_phcs: int = 8                     # PHCs feeding one district hospital
    n_specialists: int = 1              # ophthalmologists at district level

    # --- service times, minutes (lognormal, mean +- spread) ---------
    capture_mean: float = 5.0           # PLACEHOLDER: register + position + shoot
    capture_cv: float = 0.35            # coefficient of variation

    inference_mean: float = 0.5         # MEASURE: from Kanchan's benchmark
    inference_cv: float = 0.15

    review_mean_netra: float = 0.5      # specialist review WITH visual report
    review_mean_manual: float = 3.0     # specialist review WITHOUT the system
    review_cv: float = 0.4

    # --- behaviour --------------------------------------------------
    retake_prob: float = 0.15           # MEASURE: from the Quality Gate eval
    max_retakes: int = 2                # after this, refer out anyway
    referral_frac: float = 0.30         # PLACEHOLDER: fraction needing a specialist

    use_netra_review: bool = True
    seed: int = 0


def _lognormal(rng, mean, cv):
    """Positive service time with the given mean and coefficient of variation."""
    if mean <= 0:
        return 0.0
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return float(rng.lognormal(mu, sigma))


@dataclass
class Results:
    screened: int = 0                # patients who completed a good capture
    arrived: int = 0
    abandoned: int = 0               # still in queue when the day ended
    retakes: int = 0
    referred: int = 0
    reviewed: int = 0
    wait_capture: list = field(default_factory=list)
    wait_specialist: list = field(default_factory=list)
    capture_busy: float = 0.0
    spec_left: int = 0
    specialist_busy: float = 0.0
    p: Params = None

    def summary(self):
        pd = self.p.day_minutes
        return {
            "arrived": self.arrived,
            "screened": self.screened,
            "screened_per_day_per_phc": self.screened,
            "retake_events": self.retakes,
            "referred": self.referred,
            "reviewed_by_specialist": self.reviewed,
            "specialist_backlog": self.referred - self.reviewed,
            "mean_wait_capture_min": float(np.mean(self.wait_capture)) if self.wait_capture else 0.0,
            "p90_wait_capture_min": float(np.percentile(self.wait_capture, 90)) if self.wait_capture else 0.0,
            "mean_wait_specialist_min": float(np.mean(self.wait_specialist)) if self.wait_specialist else 0.0,
            "capture_utilisation": self.capture_busy / (pd * self.p.n_capture_stations),
            "specialist_utilisation": self.specialist_busy / (pd * self.p.n_specialists),
        }


# ---------------------------------------------------------------- engine

ARRIVAL, CAPTURE_DONE, REVIEW_DONE, SPEC_ARRIVAL = 0, 1, 2, 3


def simulate(p: Params) -> Results:
    rng = np.random.default_rng(p.seed)
    r = Results(p=p)

    events = []                       # (time, seq, kind, payload)
    seq = 0

    def push(t, kind, payload=None):
        nonlocal seq
        heapq.heappush(events, (t, seq, kind, payload))
        seq += 1

    # Poisson arrivals over the clinic day
    rate = p.arrivals_per_day / p.day_minutes
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t > p.day_minutes:
            break
        push(t, ARRIVAL, {"attempts": 0, "arrived": t})
        r.arrived += 1

    # The district specialist also receives referrals from the other PHCs.
    # We model this PHC in full and the remaining (n_phcs - 1) as a Poisson
    # stream of referrals arriving at the district hospital.
    if p.n_phcs > 1:
        other_rate = ((p.n_phcs - 1) * p.arrivals_per_day * p.referral_frac
                      / p.day_minutes)
        t = 0.0
        while True:
            t += rng.exponential(1.0 / other_rate)
            if t > p.day_minutes:
                break
            push(t, SPEC_ARRIVAL, {"spec_queued_at": t, "external": True})

    capture_free = p.n_capture_stations
    capture_q = []                     # patients waiting for a camera
    spec_free = p.n_specialists
    spec_q = []

    review_mean = p.review_mean_netra if p.use_netra_review else p.review_mean_manual

    def start_capture(now, patient):
        nonlocal capture_free
        capture_free -= 1
        dur = _lognormal(rng, p.capture_mean, p.capture_cv)
        dur += _lognormal(rng, p.inference_mean, p.inference_cv)
        r.capture_busy += dur
        push(now + dur, CAPTURE_DONE, patient)

    def start_review(now, patient):
        nonlocal spec_free
        spec_free -= 1
        dur = _lognormal(rng, review_mean, p.review_cv)
        r.specialist_busy += dur
        push(now + dur, REVIEW_DONE, patient)

    while events:
        now, _, kind, patient = heapq.heappop(events)

        if kind == ARRIVAL:
            if capture_free > 0:
                r.wait_capture.append(0.0)
                start_capture(now, patient)
            else:
                patient["queued_at"] = now
                capture_q.append(patient)

        elif kind == CAPTURE_DONE:
            capture_free += 1
            patient["attempts"] += 1

            failed = rng.random() < p.retake_prob
            if failed and patient["attempts"] < p.max_retakes:
                # Quality Gate rejected the image -> straight back to capture.
                r.retakes += 1
                if capture_free > 0 and now < p.day_minutes:
                    start_capture(now, patient)
                else:
                    patient["queued_at"] = now
                    capture_q.insert(0, patient)   # retakes get priority
            else:
                r.screened += 1
                if rng.random() < p.referral_frac:
                    r.referred += 1
                    if spec_free > 0:
                        r.wait_specialist.append(0.0)
                        start_review(now, patient)
                    else:
                        patient["spec_queued_at"] = now
                        spec_q.append(patient)

            # pull the next waiting patient into the freed camera,
            # but only while the clinic is still open
            if capture_q and capture_free > 0 and now < p.day_minutes:
                nxt = capture_q.pop(0)
                r.wait_capture.append(now - nxt["queued_at"])
                start_capture(now, nxt)

        elif kind == SPEC_ARRIVAL:
            r.referred += 1
            if spec_free > 0:
                r.wait_specialist.append(0.0)
                start_review(now, patient)
            else:
                patient["spec_queued_at"] = now
                spec_q.append(patient)

        elif kind == REVIEW_DONE:
            spec_free += 1
            r.reviewed += 1
            if spec_q and spec_free > 0:
                nxt = spec_q.pop(0)
                r.wait_specialist.append(now - nxt["spec_queued_at"])
                start_review(now, nxt)

    r.abandoned = len(capture_q)
    r.spec_left = len(spec_q)
    return r


def run_replications(p: Params, n_reps=30):
    """Average over independent days so the numbers are not one lucky run."""
    rows = []
    for i in range(n_reps):
        q = Params(**{**p.__dict__})
        q.seed = p.seed + i
        rows.append(simulate(q).summary())
    keys = rows[0].keys()
    return {k: float(np.mean([row[k] for row in rows])) for k in keys}
