from pathlib import Path

import cv2
import numpy as np

from app.pose import PoseEstimator
from app.alignment import align_garment


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PERSON_PATH = (
    DATASET_DIR
    / "persons"
    / "person_001.jpg"
)

# IMPORTANT:
# These two files are the matching garment + mask pair.
GARMENT_PATH = (
    DATASET_DIR
    / "garments"
    / "garment_processed.png"
)

MASK_PATH = (
    DATASET_DIR
    / "masks"
    / "garment_mask.png"
)

ALIGNED_GARMENT_PATH = (
    OUTPUT_DIR
    / "aligned_garment.png"
)

ALIGNED_MASK_PATH = (
    OUTPUT_DIR
    / "aligned_garment_mask.png"
)

DEBUG_PATH = (
    OUTPUT_DIR
    / "garment_control_points.png"
)


# ============================================================
# HELPERS
# ============================================================

def ensure_rgba(image):
    """Ensure garment is RGBA."""

    if image is None:
        raise ValueError(
            "Alignment returned None."
        )

    if image.ndim != 3:
        raise ValueError(
            f"Aligned garment must be 3D. "
            f"Got {image.shape}"
        )

    if image.shape[2] == 4:
        return image

    if image.shape[2] == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2BGRA,
        )

    raise ValueError(
        f"Unsupported channel count: "
        f"{image.shape[2]}"
    )


def ensure_mask(mask):
    """Ensure mask is single-channel uint8."""

    if mask is None:
        raise ValueError(
            "Alignment returned no mask."
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Aligned mask must be 2D. "
            f"Got {mask.shape}"
        )

    if mask.dtype != np.uint8:
        mask = np.clip(
            mask,
            0,
            255,
        ).astype(np.uint8)

    return mask


# ============================================================
# TEST
# ============================================================

def test_garment_alignment():

    print()
    print("=" * 60)
    print("Raritone VTON - Alignment Test")
    print("=" * 60)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Person
    # --------------------------------------------------------

    print()
    print("Loading person...")

    person = cv2.imread(
        str(PERSON_PATH),
        cv2.IMREAD_COLOR,
    )

    assert person is not None, (
        f"Unable to load person:\n"
        f"{PERSON_PATH}"
    )

    print(
        "Person:",
        person.shape,
    )

    # --------------------------------------------------------
    # 2. Processed garment
    # --------------------------------------------------------

    print()
    print("Loading PROCESSED garment...")

    garment = cv2.imread(
        str(GARMENT_PATH),
        cv2.IMREAD_UNCHANGED,
    )

    assert garment is not None, (
        f"Unable to load garment:\n"
        f"{GARMENT_PATH}"
    )

    print(
        "Processed garment:",
        garment.shape,
    )

    # --------------------------------------------------------
    # 3. Garment mask
    # --------------------------------------------------------

    print()
    print("Loading garment mask...")

    garment_mask = cv2.imread(
        str(MASK_PATH),
        cv2.IMREAD_GRAYSCALE,
    )

    assert garment_mask is not None, (
        f"Unable to load garment mask:\n"
        f"{MASK_PATH}"
    )

    print(
        "Mask:",
        garment_mask.shape,
    )

    # --------------------------------------------------------
    # 4. Validate garment/mask
    # --------------------------------------------------------

    print()
    print(
        "Validating garment/mask dimensions..."
    )

    assert garment.shape[:2] == garment_mask.shape[:2], (
        "Garment and mask dimensions do not match:\n"
        f"Garment = {garment.shape[:2]}\n"
        f"Mask    = {garment_mask.shape[:2]}"
    )

    print(
        "Garment/mask dimensions: OK"
    )

    # --------------------------------------------------------
    # 5. Pose
    # --------------------------------------------------------

    print()
    print("Estimating person pose...")

    pose = PoseEstimator()

    try:
        landmarks = pose.estimate(
            str(PERSON_PATH)
        )
    finally:
        pose.close()

    assert landmarks is not None, (
        "Pose estimation returned None."
    )

    print(
        "Landmarks:",
        list(landmarks.keys()),
    )

    required = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ]

    for name in required:

        assert name in landmarks, (
            f"Required landmark missing: {name}"
        )

    # --------------------------------------------------------
    # 6. Alignment
    # --------------------------------------------------------

    print()
    print("Running garment alignment...")

    result = align_garment(
        garment_image=garment,
        garment_mask=garment_mask,
        person_landmarks=landmarks,
        person_image_shape=person.shape[:2],
    )

    assert result is not None, (
        "align_garment() returned None."
    )

    # --------------------------------------------------------
    # 7. Extract aligned garment
    # --------------------------------------------------------

    if hasattr(result, "image"):

        aligned_garment = result.image

    elif hasattr(result, "aligned_garment"):

        aligned_garment = result.aligned_garment

    elif isinstance(result, dict):

        if "image" in result:
            aligned_garment = result["image"]

        elif "aligned_garment" in result:
            aligned_garment = result[
                "aligned_garment"
            ]

        else:
            raise AssertionError(
                "Result does not contain "
                "'image' or 'aligned_garment'."
            )

    else:

        raise AssertionError(
            f"Unsupported alignment result: "
            f"{type(result)}"
        )

    # --------------------------------------------------------
    # 8. Extract aligned mask
    # --------------------------------------------------------

    if hasattr(result, "mask"):

        aligned_mask = result.mask

    elif hasattr(result, "aligned_mask"):

        aligned_mask = result.aligned_mask

    elif isinstance(result, dict):

        if "mask" in result:
            aligned_mask = result["mask"]

        elif "aligned_mask" in result:
            aligned_mask = result[
                "aligned_mask"
            ]

        else:
            aligned_mask = None

    else:

        aligned_mask = None

    # --------------------------------------------------------
    # 9. Fallback mask from alpha
    # --------------------------------------------------------

    if aligned_mask is None:

        print()
        print(
            "No aligned mask returned."
        )

        print(
            "Using aligned garment alpha..."
        )

        if (
            aligned_garment.ndim == 3
            and aligned_garment.shape[2] == 4
        ):

            aligned_mask = aligned_garment[
                :, :, 3
            ]

        else:

            raise AssertionError(
                "Cannot create aligned mask: "
                "garment has no alpha channel."
            )

    # --------------------------------------------------------
    # 10. Normalize outputs
    # --------------------------------------------------------

    aligned_garment = ensure_rgba(
        aligned_garment
    )

    aligned_mask = ensure_mask(
        aligned_mask
    )

    # --------------------------------------------------------
    # 11. Validate dimensions
    # --------------------------------------------------------

    print()
    print(
        "Validating alignment output..."
    )

    print(
        "Aligned garment:",
        aligned_garment.shape,
    )

    print(
        "Aligned mask:",
        aligned_mask.shape,
    )

    expected_shape = person.shape[:2]

    assert aligned_garment.shape[:2] == expected_shape, (
        "Aligned garment dimensions do not match person:\n"
        f"Aligned = {aligned_garment.shape[:2]}\n"
        f"Person  = {expected_shape}"
    )

    assert aligned_mask.shape == expected_shape, (
        "Aligned mask dimensions do not match person:\n"
        f"Aligned = {aligned_mask.shape}\n"
        f"Person  = {expected_shape}"
    )

    assert aligned_garment.shape[2] == 4, (
        "Aligned garment must be RGBA."
    )

    # --------------------------------------------------------
    # 12. Validate mask
    # --------------------------------------------------------

    mask_pixels = np.count_nonzero(
        aligned_mask > 127
    )

    print()
    print(
        "Aligned mask foreground:",
        mask_pixels,
        "pixels",
    )

    assert mask_pixels > 0, (
        "Aligned garment mask is empty."
    )

    # --------------------------------------------------------
    # 13. Validate alpha
    # --------------------------------------------------------

    alpha = aligned_garment[:, :, 3]

    alpha_pixels = np.count_nonzero(
        alpha > 127
    )

    print(
        "Aligned alpha foreground:",
        alpha_pixels,
        "pixels",
    )

    assert alpha_pixels > 0, (
        "Aligned garment alpha is empty."
    )

    # --------------------------------------------------------
    # 14. Save aligned garment
    # --------------------------------------------------------

    print()
    print("Saving aligned garment...")

    success = cv2.imwrite(
        str(ALIGNED_GARMENT_PATH),
        aligned_garment,
    )

    assert success, (
        f"Failed to save:\n"
        f"{ALIGNED_GARMENT_PATH}"
    )

    # --------------------------------------------------------
    # 15. Save aligned mask
    # --------------------------------------------------------

    print(
        "Saving aligned mask..."
    )

    success = cv2.imwrite(
        str(ALIGNED_MASK_PATH),
        aligned_mask,
    )

    assert success, (
        f"Failed to save:\n"
        f"{ALIGNED_MASK_PATH}"
    )

    # --------------------------------------------------------
    # 16. Debug image
    # --------------------------------------------------------

    print(
        "Creating debug image..."
    )

    debug = person.copy()

    binary_mask = np.where(
        aligned_mask > 127,
        255,
        0,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        debug,
        contours,
        -1,
        (0, 255, 0),
        2,
    )

    # Bounding box
    ys, xs = np.where(
        binary_mask > 0
    )

    if len(xs) > 0:

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

    cv2.imwrite(
        str(DEBUG_PATH),
        debug,
    )

    # --------------------------------------------------------
    # 17. Verify files
    # --------------------------------------------------------

    assert ALIGNED_GARMENT_PATH.exists(), (
        "aligned_garment.png was not created."
    )

    assert ALIGNED_MASK_PATH.exists(), (
        "aligned_garment_mask.png was not created."
    )

    assert DEBUG_PATH.exists(), (
        "Debug image was not created."
    )

    # --------------------------------------------------------
    # 18. Final report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ALIGNMENT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        "Aligned garment:",
        ALIGNED_GARMENT_PATH,
    )

    print(
        "Aligned mask   :",
        ALIGNED_MASK_PATH,
    )

    print(
        "Debug image    :",
        DEBUG_PATH,
    )

    print(
        "Garment shape  :",
        aligned_garment.shape,
    )

    print(
        "Mask shape     :",
        aligned_mask.shape,
    )

    print(
        "Mask pixels    :",
        mask_pixels,
    )

    print("=" * 60)
    print("ALIGNMENT TEST PASSED")
    print("=" * 60)