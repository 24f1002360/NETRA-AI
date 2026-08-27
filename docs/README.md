# NETRA AI — Prototype Round Working Pack
**Team Aim Nexus · SIH26038 · IIT Madras BS Degree Programme**

## Read in this order

| # | File | Who | Time |
|---|---|---|---|
| 1 | `00_MASTER_PLAN.md` | Everyone, fully | 20 min |
| 2 | `01_INTERFACE_CONTRACTS.md` | Everyone, then keep open daily | 15 min |
| 3 | Your own guide below | You | 10 min |
| 4 | The guide of whoever you hand off to | You | 5 min |

| Member | Guide |
|---|---|
| Abhishek Dhama — UI/UX, vernacular HCI, team lead | `GUIDE_1_Abhishek.md` |
| Muskan Jadon — image quality & enhancement | `GUIDE_2_Muskan.md` |
| Kanchan — deep learning & optimization | `GUIDE_3_Kanchan.md` |
| Anshika Maurya — clinical XAI & validation | `GUIDE_4_Anshika.md` |
| Divyanshu Kumar — backend, DB, orchestration | `GUIDE_5_Divyanshu.md` |
| Ishank Gupta — simulation, compression, hardware | `GUIDE_6_Ishank.md` |

## First 48 hours — the whole team

- [ ] Everyone reads files 1–3
- [ ] Kickoff call: settle **Decision A** (inference runtime) and **Decision B** (backbone). Write both into `configs/decisions.md`.
- [ ] Repo created, everyone pushes a hello-world commit
- [ ] Kanchan requests Messidor-2 access and starts all dataset downloads
- [ ] Everyone writes their stub in `core/stubs/`
- [ ] Divyanshu wires `core/inference.py` to the stubs
- [ ] Day 3: integration call — the full pipeline produces a complete fake report

## Two things in the deck that need correcting

1. **"AI model under 15 MB" with ResNet-50 is not achievable.** ResNet-50 at INT8 is ~26 MB. Deploy EfficientNet-B0 (~5.5 MB) and keep ResNet-50 as a benchmarked baseline. See master plan §1, Decision B.
2. **MATLAB in the live inference path is your biggest demo risk.** Resolve with one interface, two implementations, and a config flag. See master plan §1, Decision A.

## The one rule

`main` is always demo-able. If you break it, you fix it that night or you revert.
