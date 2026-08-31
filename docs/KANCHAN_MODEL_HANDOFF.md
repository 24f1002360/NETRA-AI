# Kanchan — Model Training & Integration

This section documents the ML work completed for NETRA-AI: diabetic-retinopathy grading, four-lesion segmentation, model selection, checkpoint packaging, and Python integration.

> **Status:** model-level engineering integration is complete and the repository test suite passes. One scientific verification remains before calling the segmentation artifact fully frozen: evaluate the exact frozen `FINAL_V8_SEGMENTATION.pth` on the official IDRiD test set.

## Models

| Block | Dataset | Model | Input | Final reported metric | Artifact |
|---|---|---|---|---|---|
| DR grading | APTOS 2019 | EfficientNet-B0 | 384×384 | Val QWK **0.8544** | `netra_dr_effb0_muskan_preproc.pth` |
| Lesion segmentation | IDRiD | U-Net | 512×512 | V8 official-test mean Dice **0.3136*** | `FINAL_V8_SEGMENTATION.pth` |

\* The **0.3136** score comes from the V8 evaluation candidate/ensemble in the training notebook. The frozen production file was packaged from `best_unet_v5_class_weighted.pth`; therefore the frozen file itself must be independently evaluated on the official test set before claiming that exact score for the frozen artifact.

## Grading

The final grading checkpoint uses EfficientNet-B0 with NETRA preprocessing and 384×384 input.

Recorded checkpoint metadata:

```text
epoch: 5
validation QWK: 0.854444
validation accuracy: 0.773533
checkpoint size: 46.41 MB
CPU benchmark: 168.145 ms/image
```

Checkpoint:

urlKaggle — NETRA grading model checkpointshttps://www.kaggle.com/datasets/kanchandalal123/netra-ai-training-muskan-dr-model-checkpoints

Place it locally at:

```text
artifacts/netra_dr_effb0_muskan_preproc.pth
```

## Segmentation

IDRiD contains only **54 annotated training images**, split into **43 train + 11 validation**, with **27 official held-out test images**.

The final U-Net predicts:

```text
MA — Microaneurysms
HE — Hemorrhages
EX — Hard Exudates
SE — Soft Exudates
```

Production thresholds:

```text
MA = 0.40
HE = 0.65
EX = 0.80
SE = 0.70
```

Final public artifact:

urlKaggle — FINAL V8 segmentation modelhttps://www.kaggle.com/datasets/kanchandalal123/final-v8-segmentation

Place the checkpoint at:

```text
artifacts/FINAL_V8_SEGMENTATION.pth
```

Python API:

```python
from core.models.segmentation import DRSegmenter

segmenter = DRSegmenter()
result = segmenter.segment(image)
```

Input:

```text
H × W × 3 uint8 BGR NumPy array
```

Output:

```text
probabilities
masks
thresholds
statistics
image_size
model_id
model_version
```

## Model evolution

The segmentation work went through multiple experiments:

```text
V1 → V2 → V3 → V4 → V5 → V7/V8 → V9 → V11 → V8/V11 ensemble
```

Important decisions:

- class imbalance required more than plain BCE;
- threshold optimization materially affected results;
- microaneurysm segmentation remained the hardest problem;
- V9 MA-focused training was rejected;
- V11 improved validation behaviour but did not beat V8 on the official test;
- V8+V11 achieved **0.3304 validation mean Dice** but only **0.3079 official-test mean Dice**, so it was rejected;
- V8 achieved **0.3136 official-test mean Dice** in the notebook evaluation and was selected as the candidate.

## Integration

The segmentation model was moved from notebook-only code into:

```text
core/models/segmentation.py
```

and tested with:

```text
tests/test_segmentation.py
```

The final test run reported:

```text
8/8 segmentation tests passed
38/38 repository tests passed
```

The integration includes exact checkpoint architecture matching, preprocessing, sigmoid probabilities, lesion thresholds, connected-component cleanup, statistics, input validation, and model metadata.

## What differs from the original Kanchan guide?

The original guide proposed a very small U-Net/MobileNet-style model, ONNX export, static INT8 quantization, and a MATLAB Compiler SDK path.

The actual implementation currently uses:

- full U-Net;
- ~7.76M parameters;
- ~89 MB PyTorch checkpoint;
- PyTorch inference;
- no final ONNX artifact;
- no final INT8 artifact;
- no completed MATLAB Compiler SDK deployment.

These differences are documented intentionally rather than hidden.

## Known limitations

### Grading

- 46.41 MB checkpoint is larger than the original <15 MB target.
- No final INT8 checkpoint has been produced.
- External clinical validation is not part of this workstream.
- Validation confidence has not been clinically calibrated.

### Segmentation

The main weakness is microaneurysm segmentation.

The V8 official-test evaluation reported:

| Lesion | Dice |
|---|---:|
| MA | 0.0152 |
| HE | 0.2349 |
| EX | 0.5209 |
| SE | 0.4831 |
| Mean | **0.3136** |

This should be reported honestly. The model is considerably stronger on larger lesions than on microaneurysms.

## Next steps

1. Re-evaluate the exact frozen `FINAL_V8_SEGMENTATION.pth` on the official 27-image test set.
2. Record the frozen-checkpoint score separately from ensemble scores.
3. Integrate grading + segmentation through the shared `core/inference.py` orchestrator.
4. Benchmark the actual deployment machine.
5. Export/verify ONNX if required.
6. Quantize only after verifying numerical equivalence.
7. Complete the MATLAB/Compiler SDK path if required by the final submission.
8. Improve MA segmentation using high-resolution/patch-based sampling and MA-focused losses.
9. Perform broader external validation before making clinical performance claims.

For the complete experiment history, model-selection decisions, artifact layout, integration details, failure modes, and submission checklist, see:

`docs/KANCHAN_MODEL_HANDOFF.md`
