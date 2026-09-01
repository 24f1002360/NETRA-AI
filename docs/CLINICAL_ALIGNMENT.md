# Clinical Alignment — ICDR Grade → NETRA Label → Routing Action

Owner: Anshika Maurya (Clinical XAI & Validation Lead)

This maps NETRA AI's output to the standard clinical severity scale so
the system's decisions are traceable to established medical practice,
not an arbitrary internal label. Reference: Wilkinson et al. (2003),
"Proposed International Clinical Diabetic Retinopathy Severity Scales"
(already in project references).

---

## The Mapping

| ICDR Grade | Clinical Name | Clinical Definition (ICDR) | NETRA Label | Routing Action |
|---|---|---|---|---|
| 0 | No apparent retinopathy | No abnormalities | `no_dr` | No referral. Routine annual screening. |
| 1 | Mild NPDR | Microaneurysms only | `mild_dr` | No urgent referral. Re-screen in 12 months. |
| 2 | Moderate NPDR | More than microaneurysms, less than severe | `moderate_dr` | **Referable.** Non-urgent specialist review within weeks. |
| 3 | Severe NPDR | Extensive haemorrhages / venous beading / IRMA (4-2-1 rule) | `severe_dr` | **Urgent referral** — specialist review within days. |
| 4 | Proliferative DR (PDR) | Neovascularisation, vitreous/preretinal haemorrhage | `pdr` | **Emergency.** Immediate automated alert to district doctor. |

**Referable DR threshold: grade ≥ 2.** This is the clinically standard
cutoff (moderate NPDR or worse) and is what `docs/BENCHMARKS.md`'s
sensitivity/specificity numbers are measured against.

---

## Guard Override — When XAI Doesn't Trust Its Own Answer

Independent of the grade above, if `core/xai/guards.py` returns a
non-`"OK"` status, routing changes regardless of the predicted grade:

| `guard_status` | Meaning | Routing Override |
|---|---|---|
| `CAM_OFF_RETINA` | Model's evidence is outside the retina | Route to specialist for manual review |
| `LOW_AGREEMENT` | Grading model and lesion-segmentation model disagree | Route to specialist for manual review |
| `OK` | Evidence checks out | Follow grade-based routing normally |

**Real example (see BENCHMARKS.md):** `severe.png` predicted Grade 4
(PDR, emergency) but `guard_status = LOW_AGREEMENT` — the system flags
it for specialist review rather than silently issuing an emergency
alert on unverified evidence.

---

## Consumed By

- **Divyanshu** (routing) — `guard_status` + `icdr_grade` decides queue
  vs. automated emergency alert.
- **Abhishek** (deck) — basis for the clinical alignment slide.

---

## Open Question

Should DME (Diabetic Macular Edema) also trigger "referable" status
independent of ICDR grade? NETRA AI has no dedicated DME detector —
known limitation, listed on the honest-limitations slide.