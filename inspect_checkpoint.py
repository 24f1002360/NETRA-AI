"""
Run once: python inspect_checkpoint.py

Finds the exact target_layer name for GradCAM(model, target_layer=...)
from Kanchan's checkpoint, without needing her grading.py to be pushed.

If this fails to auto-load into a standard torchvision EfficientNet-B0,
it falls back to printing the raw checkpoint keys so you can compare
them by hand against candidate architectures (torchvision / timm /
efficientnet_pytorch all name layers differently).
"""
import torch

CKPT_PATH = "artifacts/netra_dr_effb0_muskan_preproc.pth"


def main():
    # weights_only=False: safe here because this checkpoint is from a
    # trusted teammate (Kanchan), not a random internet download.
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    print("=" * 60)
    print("Checkpoint top-level keys:")
    print("=" * 60)
    if isinstance(ckpt, dict):
        for k in ckpt.keys():
            print(" ", k)
    else:
        print("  (checkpoint is not a dict -- it's a raw state_dict or full model)")

    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    print()
    print("=" * 60)
    print("First 5 and last 15 parameter names in the checkpoint:")
    print("=" * 60)
    keys = list(state_dict.keys())
    for k in keys[:5]:
        print(" ", k)
    print("  ...")
    for k in keys[-15:]:
        print(" ", k)

    # Try the most common case: torchvision EfficientNet-B0 with a
    # replaced 5-class classifier head (matches "class_names=[0,1,2,3,4]"
    # in the handoff doc).
    print()
    print("=" * 60)
    print("Attempting to load into torchvision efficientnet_b0...")
    print("=" * 60)
    try:
        from torchvision.models import efficientnet_b0
        import torch.nn as nn

        model = efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys:    {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

        if len(missing) == 0 and len(unexpected) == 0:
            print()
            print(">>> MATCH: this is a standard torchvision EfficientNet-B0.")
            print(">>> Use this as your target_layer for GradCAM:")
            print(">>>     target_layer = 'features.8'")
            print(">>> (the final Conv2dNormActivation block, right before avgpool)")
        else:
            print()
            print("Partial/no match -- architecture may differ from plain")
            print("torchvision efficientnet_b0. First few missing/unexpected keys:")
            print("  missing:", missing[:5])
            print("  unexpected:", unexpected[:5])
            print()
            print("Compare these against the raw checkpoint keys printed above")
            print("to figure out which library/layer naming was used, or ask")
            print("Kanchan directly for the exact target_layer string.")
    except Exception as e:
        print(f"Could not attempt torchvision load: {e}")


if __name__ == "__main__":
    main()
