from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from core.iqa.enhance import enhance


NUM_CLASSES = 5
IMAGE_SIZE = 384

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "netra_dr_effb0_muskan_preproc.pth"
)

# Last convolutional block for Grad-CAM (torchvision efficientnet_b0:
# features[8] is the final 1x1 ConvNormActivation before avgpool).
# Best current guess per GUIDE_4_Anshika.md Part 1 -- confirm with Kanchan
# before treating this as final.
GRADCAM_LAYER = "features.8"


def build_model() -> nn.Module:
    """
    Build the exact EfficientNet-B0 architecture
    expected by the trained checkpoint.

    The checkpoint uses Torchvision-style keys:
        features.0.0.weight
        features.1.0.block...
    """

    model = efficientnet_b0(
        weights=None,
        num_classes=NUM_CLASSES,
    )

    return model


class DRGrader:
    """
    Production NETRA diabetic-retinopathy grading model.

    Contract:
        grade(bgr) -> dict

    Input:
        H x W x 3 uint8 BGR image.

    Output:
        Dictionary containing ICDR grade, probabilities,
        referable DR status, confidence and model metadata.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.device = torch.device(
            device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.weights_path = Path(
            weights_path or DEFAULT_WEIGHTS
        )

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Grading checkpoint not found: "
                f"{self.weights_path}"
            )

        # ----------------------------------------------------------
        # Build exact training architecture
        # ----------------------------------------------------------
        self.model = build_model()

        # ----------------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------------
        checkpoint = torch.load(
            self.weights_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Unexpected grading checkpoint format."
            )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict"),
        )

        if state_dict is None:
            raise KeyError(
                "Checkpoint does not contain "
                "'model_state_dict' or 'state_dict'."
            )

        # Remove DataParallel prefix if present.
        clean_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]

            clean_state_dict[key] = value

        # ----------------------------------------------------------
        # Strict load — MUST pass
        # ----------------------------------------------------------
        self.model.load_state_dict(
            clean_state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_val_qwk = checkpoint.get("val_qwk")
        self.checkpoint_image_size = checkpoint.get(
            "img_size",
            IMAGE_SIZE,
        )

        self.class_names = checkpoint.get(
            "class_names",
            list(range(NUM_CLASSES)),
        )

        self.gradcam_layer = GRADCAM_LAYER

    def preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """
        Shared NETRA preprocessing: enhance -> BGR2RGB -> training-compatible
        resize -> normalize -> CHW batch tensor on self.device.

        Exposed separately from predict() so core/xai/explain.py's GradCAM
        can run the exact same preprocessing on the exact same model
        (model_handle["preprocess"] in the real XAI contract).
        """

        if image is None:
            raise ValueError(
                "Input image cannot be None."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a numpy.ndarray."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Expected BGR image with shape HxWx3."
            )

        if image.dtype != np.uint8:
            raise ValueError(
                "Expected uint8 BGR image."
            )

        # ----------------------------------------------------------
        # Shared NETRA preprocessing
        # ----------------------------------------------------------
        enhanced = enhance(image)

        # ----------------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------------
        rgb = cv2.cvtColor(
            enhanced,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------------
        # Training-compatible resize
        # ----------------------------------------------------------
        rgb = cv2.resize(
            rgb,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # ----------------------------------------------------------
        # Normalize
        # ----------------------------------------------------------
        rgb = rgb.astype(np.float32) / 255.0

        rgb = (rgb - MEAN) / STD

        # HWC -> CHW -> batch
        tensor = torch.from_numpy(
            rgb.transpose(2, 0, 1)
        ).float().unsqueeze(0)

        return tensor.to(self.device)

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> dict:
        """
        Run production grading inference.

        Input:
            uint8 BGR image, H x W x 3.
        """

        tensor = self.preprocess(image)

        # ----------------------------------------------------------
        # Inference
        # ----------------------------------------------------------
        logits = self.model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        probs = probabilities.cpu().numpy()

        # ----------------------------------------------------------
        # Prediction
        # ----------------------------------------------------------
        grade = int(np.argmax(probs))

        confidence = float(probs[grade])

        result = {
            "icdr_grade": grade,
            "grade_label": GRADE_LABELS[grade],
            "probabilities": [
                round(float(p), 6)
                for p in probs
            ],
            "referable_dr": grade >= 2,
            "confidence": round(confidence, 6),
            "uncertain": confidence < 0.55,
            "model_id": "netra-dr-grader",
            "model_version": "muskan-preproc-effb0-v1",
        }

        return result

    def grade(self, image: np.ndarray) -> dict:
        """
        NETRA contract:

            grade(bgr) -> dict
        """

        return self.predict(image)