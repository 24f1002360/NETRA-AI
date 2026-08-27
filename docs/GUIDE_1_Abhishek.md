# Guide — Abhishek Dhama
**Role:** UI/UX & Vernacular HCI Developer · **Also:** Team Leader
**Mission:** an ASHA worker with no technical training completes a screening without being told what to do.

You have two jobs and the second one is heavier than it looks. Budget roughly 60% build, 40% lead.

---

## Part A — The build

### What you own
`app/templates/`, `app/static/`, `app/audio/`, `app/report.py`, `configs/i18n/`
Contract functions: `render_result(result) -> str`, `generate_pdf(result) -> str`

### Stack
Flask + Jinja2 templates, vanilla JS (no React — it adds a build step you don't need), plain CSS. For PDF: **WeasyPrint** (renders your existing HTML report to PDF, so you write the layout once). For audio: pre-recorded MP3s, not runtime TTS — TTS needs internet or a big offline model, and you promised offline.

### Screens (four, no more)
1. **Capture** — patient ID field, eye selector (OD/OS), big capture button, live voice prompt area
2. **Quality feedback** — pass/retake verdict, the *reason* in words and audio, retake button. This screen is your differentiator; make it fast and unambiguous.
3. **Result** — grade with a plain-language label, confidence, side-by-side original / lesion overlay / Grad-CAM, lesion counts, routing action
4. **History** — this patient's prior screenings, trend arrow

Design for a 1366×768 laptop screen, high contrast, minimum 16px text, touch-target-sized buttons. Assume a dusty screen in bad light.

### The vernacular layer
Structure it so adding a language is a data change, never a code change:

```
configs/i18n/
├── hi.json      {"iqa.retake.blur": "तस्वीर धुंधली है — कैमरा स्थिर रखें", ...}
├── ta.json
├── te.json
└── en.json
app/audio/hi/iqa.retake.blur.mp3
```

Every string in the UI is a key lookup. Record audio for the ~15 keys that matter during capture (positioning, retake reasons, "hold still", "done"). Record with team members' voices — a real human voice in Hindi beats a robotic TTS voice, and judges notice.

**Cut order if time runs short:** Hindi complete → English complete → Tamil/Telugu text only, audio omitted. Say so honestly on the slide.

### Timeline
| Days | Deliverable |
|---|---|
| 1–3 | Three static HTML screens rendering the stub `ScreeningResult`. Nothing dynamic. |
| 4–6 | Flask routes, real capture upload, i18n key system, `en.json` + `hi.json` |
| 7–8 | Hindi audio recorded and wired; retake loop working against Muskan's stub codes |
| 9–10 | `generate_pdf()` producing a clean one-page report from fake data |
| 11–14 | Swap to real data from Divyanshu; history screen; Tamil/Telugu text |
| 15–16 | Polish, empty states, error states ("model not loaded", "camera not found") |
| 17–21 | Deck rebuild around real numbers; demo rehearsal |

### Your acceptance test
Hand the laptop to someone outside the team who has never seen the project. Say nothing. If they can't complete a screening, the UI isn't done. Do this on Day 12 and again on Day 18.

### Failure modes to avoid
- Building the UI against imagined data instead of the stub — you'll rewrite it. Use the stub.
- Runtime TTS. It will fail offline on demo day.
- A PDF generator written separately from the HTML report. Two layouts = two things to maintain and one of them will be wrong.
- Putting Hindi strings inside Python files. Everything goes in `i18n/`.

---

## Part B — Leading the team

Nobody made you responsible for other people's code. You're responsible for the *seams*.

### Your weekly loop
- **Daily:** read the standup thread. If someone's "blocked" line is the same two days running, call them. Don't wait for them to escalate.
- **Wednesday & Saturday:** run the integration call. You share screen and execute the demo script yourself. If you can't run it, that's the meeting.
- **Every merge window:** confirm `main` is still green. If it's red at midnight, revert first, debug tomorrow.

### The four things that will actually go wrong
1. **Kanchan disappears into training.** Model work expands to fill all available time. Hold him to the Day 7 checkpoint handoff even if the model is bad. A bad checkpoint unblocks Anshika; a perfect one on Day 14 doesn't.
2. **Integration is deferred.** Everyone's module works alone on Day 15 and nothing works together on Day 16. This is why Phase 0 and the twice-weekly calls exist. Don't let them slip.
3. **The deck drifts from the code.** Someone writes "95% accuracy" from a slide draft. Numbers freeze Day 19 and come only from `docs/BENCHMARKS.md`. You enforce that.
4. **Scope creep in week 3.** Someone wants to add OCT support. The answer after Day 16 is no.

### Decisions you own
- Feature freeze on Day 16 — call it out loud, in the group, by name.
- Cut list, in this order: multi-disease detection → languages beyond Hindi → longitudinal UI → WhatsApp alerts (keep SMS).
- Go/no-go on the MATLAB runtime path, Day 16, based on Kanchan and Divyanshu's clean-machine test.

### Presentation ownership
You present. Which means you must be able to answer, for 60 seconds each, without help:
- Why EfficientNet-B0 and not ResNet-50
- What "anatomically guarded Grad-CAM" actually does mechanically
- What the Quality Gate rejects and what its precision is
- What the Simulink model says about district capacity
- What your system does *wrong* — have a real answer ready, it's the question that separates teams

Book 30 minutes with each member in Phase 3 and have them teach you their module.

### Reading
- Nielsen's 10 usability heuristics (20 min, worth it for the design defence)
- WHO guidance on task-shifting to community health workers — one paragraph in the deck about designing for ASHA workflow constraints reads as real domain grounding
