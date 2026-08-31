import os
import sys

import cv2
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from core.models.segmentation import (
    DRSegmenter,
    LESION_LABELS,
    LESION_THRESHOLDS,
)


FIXTURES = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "fixtures",
)


def _load(name):
    path = os.path.join(
        FIXTURES,
        name,
    )

    image = cv2.imread(path)

    assert image is not None, (
        f"fixture missing: {path}"
    )

    return image


def test_segmentation_checkpoint_loads():

    segmenter = DRSegmenter()

    assert segmenter.model is not None


def test_segmentation_checkpoint_metadata():

    segmenter = DRSegmenter()

    assert segmenter.checkpoint_epoch == 48
    assert segmenter.checkpoint_mean_dice is not None


def test_segmentation_real_inference():

    segmenter = DRSegmenter()

    image = _load("good.png")

    result = segmenter.segment(image)

    assert "masks" in result
    assert "probabilities" in result
    assert "statistics" in result


def test_segmentation_has_four_lesions():

    segmenter = DRSegmenter()

    image = _load("good.png")

    result = segmenter.segment(image)

    assert set(
        result["masks"].keys()
    ) == {"MA", "HE", "EX", "SE"}


def test_segmentation_masks_are_binary():

    segmenter = DRSegmenter()

    image = _load("good.png")

    result = segmenter.segment(image)

    for lesion, mask in result["masks"].items():

        assert isinstance(
            mask,
            np.ndarray,
        )

        assert mask.dtype == np.uint8

        assert set(
            np.unique(mask)
        ).issubset({0, 1})


def test_segmentation_probabilities_are_valid():

    segmenter = DRSegmenter()

    image = _load("good.png")

    result = segmenter.segment(image)

    for lesion, probability_map in (
        result["probabilities"].items()
    ):

        assert probability_map.shape == (
            384,
            384,
        )

        assert np.all(
            probability_map >= 0
        )

        assert np.all(
            probability_map <= 1
        )


def test_production_thresholds():

    assert LESION_THRESHOLDS == {
        "MA": 0.40,
        "HE": 0.65,
        "EX": 0.80,
        "SE": 0.70,
    }


def test_segmentation_output_size():

    segmenter = DRSegmenter()

    image = _load("good.png")

    result = segmenter.segment(image)

    for mask in result["masks"].values():

        assert mask.shape == (
            384,
            384,
        )
