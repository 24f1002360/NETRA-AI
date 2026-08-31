from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 4
MODEL_IMAGE_SIZE = 512
OUTPUT_IMAGE_SIZE = 384

LESION_LABELS = {
    0: "MA",
    1: "HE",
    2: "EX",
    3: "SE",
}

LESION_THRESHOLDS = {
    "MA": 0.40,
    "HE": 0.65,
    "EX": 0.80,
    "SE": 0.70,
}

# Minimum connected-component size after thresholding.
# These values are deliberately small because MA lesions are tiny.
MIN_COMPONENT_AREAS = {
    "MA": 3,
    "HE": 5,
    "EX": 5,
    "SE": 5,
}

# MA is currently the unstable V8 channel.
# If thresholding produces an unrealistically large portion of
# the image as MA, treat it as an unreliable detection instead
# of returning a giant false-positive mask.
MAX_COVERAGE = {
    "MA": 5.0,
    "HE": 20.0,
    "EX": 20.0,
    "SE": 20.0,
}

DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "FINAL_V8_SEGMENTATION.pth"
)


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d + BatchNorm2d + ReLU blocks.

    This architecture matches FINAL_V8_SEGMENTATION.pth.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Exact U-Net architecture matching FINAL_V8_SEGMENTATION.pth.

    Channels:
        3 -> 32 -> 64 -> 128 -> 256 -> 512
        -> 256 -> 128 -> 64 -> 32 -> 4
    """

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2,
        )
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(
            32,
            NUM_CLASSES,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)
        d4 = torch.cat(
            [d4, e4],
            dim=1,
        )
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )
        d1 = self.dec1(d1)

        return self.final(d1)


def build_model() -> UNet:
    return UNet()


class DRSegmenter:
    """
    Production NETRA lesion segmentation model.

    Contract:
        segment(bgr) -> dict

    Input:
        H x W x 3 uint8 BGR fundus image.

    Output:
        Dictionary containing lesion probability maps,
        binary masks and lesion statistics.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.weights_path = Path(
            weights_path or DEFAULT_WEIGHTS
        )

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Segmentation checkpoint not found: "
                f"{self.weights_path}"
            )

        self.model = build_model()

        checkpoint = torch.load(
            self.weights_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Unexpected segmentation checkpoint format."
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

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_mean_dice = checkpoint.get(
            "mean_dice"
        )
        self.checkpoint_dice_per_channel = checkpoint.get(
            "dice_per_channel"
        )

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        rgb = cv2.resize(
            rgb,
            (
                MODEL_IMAGE_SIZE,
                MODEL_IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )

        rgb = rgb.astype(
            np.float32
        ) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        rgb = (
            rgb - mean
        ) / std

        tensor = torch.from_numpy(
            rgb.transpose(2, 0, 1)
        ).float()

        return tensor.unsqueeze(0).to(
            self.device
        )

    @staticmethod
    def _remove_small_components(
        mask: np.ndarray,
        min_area: int,
    ) -> np.ndarray:
        """
        Remove connected components smaller than min_area.
        """
        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask.astype(np.uint8),
                connectivity=8,
            )
        )

        cleaned = np.zeros_like(
            mask,
            dtype=np.uint8,
        )

        for label in range(1, num_labels):
            area = stats[
                label,
                cv2.CC_STAT_AREA,
            ]

            if area >= min_area:
                cleaned[labels == label] = 1

        return cleaned

    @staticmethod
    def _stats(
        mask: np.ndarray,
    ) -> dict:
        """
        Calculate pixel count and percentage coverage.
        """
        pixels = int(mask.sum())

        total = int(
            mask.shape[0] * mask.shape[1]
        )

        percentage = (
            100.0 * pixels / total
            if total > 0
            else 0.0
        )

        return {
            "pixels": pixels,
            "percentage": round(
                percentage,
                4,
            ),
        }

    @staticmethod
    def _coverage_percentage(
        mask: np.ndarray,
    ) -> float:
        total = mask.shape[0] * mask.shape[1]

        if total == 0:
            return 0.0

        return 100.0 * float(mask.sum()) / float(total)

    def _postprocess_mask(
        self,
        mask: np.ndarray,
        lesion: str,
    ) -> np.ndarray:
        """
        Production post-processing.

        1. Remove isolated tiny components.
        2. Reject obviously pathological full-image detections.

        The coverage guard is especially important for V8 MA,
        whose probability distribution can sit around its threshold.
        """
        cleaned = self._remove_small_components(
            mask,
            MIN_COMPONENT_AREAS[lesion],
        )

        coverage = self._coverage_percentage(
            cleaned
        )

        if coverage > MAX_COVERAGE[lesion]:
            cleaned = np.zeros_like(
                cleaned,
                dtype=np.uint8,
            )

        return cleaned

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray,
    ) -> dict:
        if image is None:
            raise ValueError(
                "Input image cannot be None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "Input image must be a numpy.ndarray."
            )

        if image.ndim != 3:
            raise ValueError(
                "Expected HxWx3 image."
            )

        if image.shape[2] != 3:
            raise ValueError(
                "Expected 3-channel BGR image."
            )

        if image.dtype != np.uint8:
            raise ValueError(
                "Expected uint8 BGR image."
            )

        original_height, original_width = (
            image.shape[:2]
        )

        tensor = self._preprocess(
            image
        )

        logits = self.model(tensor)

        probabilities = torch.sigmoid(
            logits
        )[0]

        probabilities = F.interpolate(
            probabilities.unsqueeze(0),
            size=(
                OUTPUT_IMAGE_SIZE,
                OUTPUT_IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )[0]

        probs = (
            probabilities
            .cpu()
            .numpy()
        )

        masks = {}
        statistics = {}

        for index, lesion in LESION_LABELS.items():
            probability_map = probs[index]

            threshold = LESION_THRESHOLDS[
                lesion
            ]

            binary_mask = (
                probability_map >= threshold
            ).astype(
                np.uint8
            )

            binary_mask = self._postprocess_mask(
                binary_mask,
                lesion,
            )

            masks[lesion] = binary_mask

            statistics[lesion] = self._stats(
                binary_mask
            )

        return {
            "masks": masks,
            "probabilities": {
                lesion: probs[index]
                for index, lesion
                in LESION_LABELS.items()
            },
            "thresholds": LESION_THRESHOLDS.copy(),
            "statistics": statistics,
            "image_size": {
                "height": original_height,
                "width": original_width,
            },
            "model_id": "netra-dr-segmentation",
            "model_version": "final-v8",
        }

    def segment(
        self,
        image: np.ndarray,
    ) -> dict:
        """
        NETRA segmentation contract.

        segment(bgr) -> dict
        """
        return self.predict(image)
