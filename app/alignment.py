from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import cv2
import numpy as np


# ============================================================
# TYPES
# ============================================================

Point = Tuple[float, float]
Landmarks = Dict[str, Any]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class GarmentGeometry:
    neckline: Point
    left_shoulder: Point
    right_shoulder: Point
    left_sleeve: Point
    right_sleeve: Point
    left_hem: Point
    right_hem: Point

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.neckline,
                self.left_shoulder,
                self.right_shoulder,
                self.left_sleeve,
                self.right_sleeve,
                self.left_hem,
                self.right_hem,
            ],
            dtype=np.float32,
        )

    def as_dict(self) -> Dict[str, Point]:
        return {
            "neckline": self.neckline,
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder,
            "left_sleeve": self.left_sleeve,
            "right_sleeve": self.right_sleeve,
            "left_hem": self.left_hem,
            "right_hem": self.right_hem,
        }


@dataclass
class BodyGeometry:
    neck: Point
    left_shoulder: Point
    right_shoulder: Point
    left_sleeve: Point
    right_sleeve: Point
    left_hip: Point
    right_hip: Point

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.neck,
                self.left_shoulder,
                self.right_shoulder,
                self.left_sleeve,
                self.right_sleeve,
                self.left_hip,
                self.right_hip,
            ],
            dtype=np.float32,
        )

    def as_dict(self) -> Dict[str, Point]:
        return {
            "neck": self.neck,
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder,
            "left_sleeve": self.left_sleeve,
            "right_sleeve": self.right_sleeve,
            "left_hip": self.left_hip,
            "right_hip": self.right_hip,
        }


@dataclass
class SimilarityTransform:
    matrix: np.ndarray
    scale: float
    rotation_degrees: float
    translation: Tuple[float, float]


@dataclass
class AlignedGarment:
    image: np.ndarray
    mask: np.ndarray
    garment_points: np.ndarray
    body_points: np.ndarray
    similarity: SimilarityTransform
    aligned_points: np.ndarray
    debug_image: Optional[np.ndarray] = None


# ============================================================
# LANDMARK HELPERS
# ============================================================

def _extract_xy(point: Any) -> np.ndarray:
    if point is None:
        raise ValueError("Landmark point is None.")

    if hasattr(point, "pixel_x") and hasattr(point, "pixel_y"):
        return np.array(
            [float(point.pixel_x), float(point.pixel_y)],
            dtype=np.float32,
        )

    if isinstance(point, dict):
        if "pixel_x" in point and "pixel_y" in point:
            return np.array(
                [float(point["pixel_x"]), float(point["pixel_y"])],
                dtype=np.float32,
            )

        if "x" in point and "y" in point:
            return np.array(
                [float(point["x"]), float(point["y"])],
                dtype=np.float32,
            )

    if hasattr(point, "x") and hasattr(point, "y"):
        return np.array(
            [float(point.x), float(point.y)],
            dtype=np.float32,
        )

    if isinstance(point, (tuple, list, np.ndarray)):
        if len(point) < 2:
            raise ValueError(f"Invalid landmark: {point}")
        return np.array(
            [float(point[0]), float(point[1])],
            dtype=np.float32,
        )

    raise TypeError(
        f"Unsupported landmark type: {type(point)}"
    )


def _to_pixel_coordinates(
    point: Any,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    h, w = image_shape[:2]

    if hasattr(point, "pixel_x") and hasattr(point, "pixel_y"):
        return np.array(
            [float(point.pixel_x), float(point.pixel_y)],
            dtype=np.float32,
        )

    if isinstance(point, dict):
        if "pixel_x" in point and "pixel_y" in point:
            return np.array(
                [float(point["pixel_x"]), float(point["pixel_y"])],
                dtype=np.float32,
            )

    xy = _extract_xy(point)
    x = float(xy[0])
    y = float(xy[1])

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= w
        y *= h

    return np.array([x, y], dtype=np.float32)


def _require_landmark(
    landmarks: Landmarks,
    name: str,
) -> Any:
    if name not in landmarks:
        raise ValueError(
            f"Required person landmark missing: {name}"
        )

    if landmarks[name] is None:
        raise ValueError(
            f"Required person landmark is None: {name}"
        )

    return landmarks[name]


# ============================================================
# MASK
# ============================================================

def validate_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("Garment mask is None.")

    mask = np.asarray(mask)

    if mask.size == 0:
        raise ValueError("Garment mask is empty.")

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.ndim != 2:
        raise ValueError(
            f"Garment mask must be 2D. Got {mask.shape}"
        )

    if mask.dtype != np.uint8:
        mask = cv2.normalize(
            mask,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    # Keep the largest connected foreground component.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        raise ValueError(
            "Garment mask contains no foreground."
        )

    largest_label = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )

    binary = np.where(
        labels == largest_label,
        255,
        0,
    ).astype(np.uint8)

    if cv2.countNonZero(binary) == 0:
        raise ValueError(
            "Garment mask contains no usable foreground."
        )

    return binary


def extract_main_contour(mask: np.ndarray) -> np.ndarray:
    mask = validate_mask(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError("No garment contour found.")

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(contour) < 100:
        raise ValueError(
            "Garment foreground is too small."
        )

    return contour.reshape(-1, 2).astype(np.float32)


# ============================================================
# GARMENT GEOMETRY
# ============================================================

def _x_at_y_band(
    contour: np.ndarray,
    y_low: float,
    y_high: float,
    side: str,
) -> np.ndarray:
    region = contour[
        (contour[:, 1] >= y_low)
        & (contour[:, 1] <= y_high)
    ]

    if len(region) == 0:
        region = contour

    if side == "left":
        return region[np.argmin(region[:, 0])]

    if side == "right":
        return region[np.argmax(region[:, 0])]

    raise ValueError("side must be left or right")


def _estimate_neckline(
    contour: np.ndarray,
) -> np.ndarray:
    """
    Estimate the neck/collar center from the upper garment area.

    We intentionally do NOT use a hole in the mask as the neckline,
    because the collar may be completely solid in the segmentation.
    """

    y_min = float(np.min(contour[:, 1]))
    y_max = float(np.max(contour[:, 1]))
    height = max(y_max - y_min, 1.0)

    # Collar/neck target is normally close to the top-center.
    upper = contour[
        contour[:, 1] <= y_min + 0.18 * height
    ]

    if len(upper) == 0:
        upper = contour

    # Use the center of the upper garment instead of an extreme.
    x = float(np.median(upper[:, 0]))
    y = float(np.percentile(upper[:, 1], 35))

    return np.array([x, y], dtype=np.float32)


def extract_garment_geometry(
    garment_mask: np.ndarray,
) -> GarmentGeometry:
    """
    Extract garment points using stable image geometry.

    Front-facing garment convention:

        image-left  -> wearer's RIGHT
        image-right -> wearer's LEFT
    """

    mask = validate_mask(garment_mask)
    contour = extract_main_contour(mask)

    x_min = float(np.min(contour[:, 0]))
    x_max = float(np.max(contour[:, 0]))
    y_min = float(np.min(contour[:, 1]))
    y_max = float(np.max(contour[:, 1]))

    width = x_max - x_min
    height = y_max - y_min

    if width < 20 or height < 20:
        raise ValueError(
            "Garment is too small for reliable alignment."
        )

    neckline = _estimate_neckline(contour)

    # --------------------------------------------------------
    # Shoulder line.
    # --------------------------------------------------------
    shoulder_band = (
        y_min + 0.08 * height,
        y_min + 0.30 * height,
    )

    # Anatomical left = image right.
    left_shoulder = _x_at_y_band(
        contour,
        shoulder_band[0],
        shoulder_band[1],
        "right",
    )

    # Anatomical right = image left.
    right_shoulder = _x_at_y_band(
        contour,
        shoulder_band[0],
        shoulder_band[1],
        "left",
    )

    # --------------------------------------------------------
    # Sleeve points.
    # --------------------------------------------------------
    sleeve_band = (
        y_min + 0.18 * height,
        y_min + 0.48 * height,
    )

    left_sleeve = _x_at_y_band(
        contour,
        sleeve_band[0],
        sleeve_band[1],
        "right",
    )

    right_sleeve = _x_at_y_band(
        contour,
        sleeve_band[0],
        sleeve_band[1],
        "left",
    )

    # --------------------------------------------------------
    # Hem points.
    # --------------------------------------------------------
    hem_band = (
        y_min + 0.78 * height,
        y_max + 1.0,
    )

    left_hem = _x_at_y_band(
        contour,
        hem_band[0],
        hem_band[1],
        "right",
    )

    right_hem = _x_at_y_band(
        contour,
        hem_band[0],
        hem_band[1],
        "left",
    )

    return GarmentGeometry(
        neckline=tuple(neckline),
        left_shoulder=tuple(left_shoulder),
        right_shoulder=tuple(right_shoulder),
        left_sleeve=tuple(left_sleeve),
        right_sleeve=tuple(right_sleeve),
        left_hem=tuple(left_hem),
        right_hem=tuple(right_hem),
    )


# ============================================================
# BODY GEOMETRY
# ============================================================

def build_body_geometry(
    landmarks: Landmarks,
    image_shape: Tuple[int, int],
) -> BodyGeometry:

    left_shoulder = _to_pixel_coordinates(
        _require_landmark(
            landmarks,
            "left_shoulder",
        ),
        image_shape,
    )

    right_shoulder = _to_pixel_coordinates(
        _require_landmark(
            landmarks,
            "right_shoulder",
        ),
        image_shape,
    )

    left_hip = _to_pixel_coordinates(
        _require_landmark(
            landmarks,
            "left_hip",
        ),
        image_shape,
    )

    right_hip = _to_pixel_coordinates(
        _require_landmark(
            landmarks,
            "right_hip",
        ),
        image_shape,
    )

    shoulder_mid = (
        left_shoulder + right_shoulder
    ) / 2.0

    hip_mid = (
        left_hip + right_hip
    ) / 2.0

    # --------------------------------------------------------
    # Neck.
    # --------------------------------------------------------
    if "neck" in landmarks:
        neck = _to_pixel_coordinates(
            landmarks["neck"],
            image_shape,
        )
    elif "nose" in landmarks:
        nose = _to_pixel_coordinates(
            landmarks["nose"],
            image_shape,
        )

        # Neck is below the nose and above shoulder midpoint.
        neck = (
            0.72 * shoulder_mid
            +
            0.28 * nose
        )
    else:
        neck = (
            0.82 * shoulder_mid
            +
            0.18 * hip_mid
        )

    # --------------------------------------------------------
    # Sleeve targets.
    #
    # Use elbows only for sleeve placement, NOT for global
    # rotation estimation.
    # --------------------------------------------------------
    if "left_elbow" in landmarks:
        left_elbow = _to_pixel_coordinates(
            landmarks["left_elbow"],
            image_shape,
        )

        left_sleeve = (
            0.55 * left_shoulder
            +
            0.45 * left_elbow
        )
    else:
        left_sleeve = left_shoulder.copy()

    if "right_elbow" in landmarks:
        right_elbow = _to_pixel_coordinates(
            landmarks["right_elbow"],
            image_shape,
        )

        right_sleeve = (
            0.55 * right_shoulder
            +
            0.45 * right_elbow
        )
    else:
        right_sleeve = right_shoulder.copy()

    return BodyGeometry(
        neck=tuple(neck),
        left_shoulder=tuple(left_shoulder),
        right_shoulder=tuple(right_shoulder),
        left_sleeve=tuple(left_sleeve),
        right_sleeve=tuple(right_sleeve),
        left_hip=tuple(left_hip),
        right_hip=tuple(right_hip),
    )


# ============================================================
# CORRESPONDENCE
# ============================================================

def build_correspondences(
    garment: GarmentGeometry,
    body: BodyGeometry,
) -> Tuple[np.ndarray, np.ndarray]:

    garment_points = garment.as_array()
    body_points = body.as_array()

    if garment_points.shape != body_points.shape:
        raise ValueError(
            "Garment/body correspondence shapes do not match."
        )

    return (
        garment_points.astype(np.float32),
        body_points.astype(np.float32),
    )


# ============================================================
# ROBUST TORSO-AWARE SIMILARITY
# ============================================================

def _angle_degrees(
    vector: np.ndarray,
) -> float:
    return float(
        np.degrees(
            np.arctan2(
                vector[1],
                vector[0],
            )
        )
    )


def _normalize_angle(
    angle: float,
) -> float:
    while angle > 180.0:
        angle -= 360.0

    while angle < -180.0:
        angle += 360.0

    return angle


def estimate_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> SimilarityTransform:
    """
    Estimate a stable torso-oriented similarity transform.

    IMPORTANT DIFFERENCE FROM THE OLD IMPLEMENTATION:

    The old implementation fed shoulders, sleeves and hems into
    RANSAC. Because the sleeve points have very different geometry
    from the person's elbow-derived points, RANSAC could produce
    a large incorrect rotation such as -40 degrees.

    This implementation determines:

        1. rotation ONLY from shoulder lines
        2. scale from BOTH shoulder width and torso height
        3. translation from the neck/shoulder region

    This prevents the garment from flipping or rotating sideways.
    """

    source_points = np.asarray(
        source_points,
        dtype=np.float32,
    )

    target_points = np.asarray(
        target_points,
        dtype=np.float32,
    )

    if source_points.shape != target_points.shape:
        raise ValueError(
            "Source and target point shapes must match."
        )

    if source_points.shape[0] < 4:
        raise ValueError(
            "At least four semantic points are required."
        )

    # Point order:
    # 0 neck
    # 1 left shoulder
    # 2 right shoulder
    # 3 left sleeve
    # 4 right sleeve
    # 5 left hem
    # 6 right hem

    src_neck = source_points[0]
    src_left_shoulder = source_points[1]
    src_right_shoulder = source_points[2]
    src_left_hem = source_points[5]
    src_right_hem = source_points[6]

    dst_neck = target_points[0]
    dst_left_shoulder = target_points[1]
    dst_right_shoulder = target_points[2]
    dst_left_hip = target_points[5]
    dst_right_hip = target_points[6]

    # --------------------------------------------------------
    # 1. Shoulder widths.
    # --------------------------------------------------------
    src_shoulder_vec = (
        src_left_shoulder
        -
        src_right_shoulder
    )

    dst_shoulder_vec = (
        dst_left_shoulder
        -
        dst_right_shoulder
    )

    src_shoulder_width = float(
        np.linalg.norm(src_shoulder_vec)
    )

    dst_shoulder_width = float(
        np.linalg.norm(dst_shoulder_vec)
    )

    if src_shoulder_width < 5.0:
        raise ValueError(
            "Garment shoulder width is too small."
        )

    if dst_shoulder_width < 5.0:
        raise ValueError(
            "Person shoulder width is too small."
        )

    # --------------------------------------------------------
    # 2. Torso heights.
    # --------------------------------------------------------
    src_shoulder_mid = (
        src_left_shoulder
        +
        src_right_shoulder
    ) / 2.0

    src_hem_mid = (
        src_left_hem
        +
        src_right_hem
    ) / 2.0

    dst_shoulder_mid = (
        dst_left_shoulder
        +
        dst_right_shoulder
    ) / 2.0

    dst_hip_mid = (
        dst_left_hip
        +
        dst_right_hip
    ) / 2.0

    src_torso_height = float(
        np.linalg.norm(
            src_hem_mid
            -
            src_shoulder_mid
        )
    )

    dst_torso_height = float(
        np.linalg.norm(
            dst_hip_mid
            -
            dst_shoulder_mid
        )
    )

    # Avoid pathological values.
    src_torso_height = max(
        src_torso_height,
        10.0,
    )

    dst_torso_height = max(
        dst_torso_height,
        10.0,
    )

    # --------------------------------------------------------
    # 3. Scale.
    #
    # Shoulder width gets slightly more weight because the
    # garment should sit on the shoulders correctly.
    # --------------------------------------------------------
    width_scale = (
        dst_shoulder_width
        /
        src_shoulder_width
    )

    height_scale = (
        dst_torso_height
        /
        src_torso_height
    )

    scale = (
        0.65 * width_scale
        +
        0.35 * height_scale
    )

    # Prevent obviously catastrophic scaling.
    scale = float(
        np.clip(
            scale,
            0.15,
            1.50,
        )
    )

    # --------------------------------------------------------
    # 4. Rotation ONLY from shoulder line.
    # --------------------------------------------------------
    src_angle = _angle_degrees(
        src_shoulder_vec
    )

    dst_angle = _angle_degrees(
        dst_shoulder_vec
    )

    rotation = _normalize_angle(
        dst_angle - src_angle
    )

    # The garment in this VTON stage should stay upright.
    # A human torso will rarely require more than ~25 degrees.
    rotation = float(
        np.clip(
            rotation,
            -25.0,
            25.0,
        )
    )

    # --------------------------------------------------------
    # 5. Construct rotation matrix.
    # --------------------------------------------------------
    theta = np.radians(
        rotation
    )

    cos_t = float(
        np.cos(theta)
    )

    sin_t = float(
        np.sin(theta)
    )

    R = np.array(
        [
            [cos_t, -sin_t],
            [sin_t, cos_t],
        ],
        dtype=np.float32,
    )

    A = (
        scale * R
    )

    # --------------------------------------------------------
    # 6. Translation.
    #
    # Align shoulder midpoint and neckline jointly.
    # --------------------------------------------------------
    src_neck_transformed = (
        A
        @
        src_neck
    )

    src_shoulder_mid_transformed = (
        A
        @
        src_shoulder_mid
    )

    dst_reference = (
        0.65 * dst_shoulder_mid
        +
        0.35 * dst_neck
    )

    src_reference = (
        0.65 * src_shoulder_mid_transformed
        +
        0.35 * src_neck_transformed
    )

    translation_vec = (
        dst_reference
        -
        src_reference
    )

    matrix = np.array(
        [
            [
                A[0, 0],
                A[0, 1],
                translation_vec[0],
            ],
            [
                A[1, 0],
                A[1, 1],
                translation_vec[1],
            ],
        ],
        dtype=np.float32,
    )

    return SimilarityTransform(
        matrix=matrix,
        scale=scale,
        rotation_degrees=rotation,
        translation=(
            float(translation_vec[0]),
            float(translation_vec[1]),
        ),
    )


def apply_similarity_to_points(
    points: np.ndarray,
    transform: SimilarityTransform,
) -> np.ndarray:

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    ones = np.ones(
        (len(points), 1),
        dtype=np.float32,
    )

    homogeneous = np.concatenate(
        [
            points,
            ones,
        ],
        axis=1,
    )

    return (
        homogeneous
        @
        transform.matrix.T
    ).astype(np.float32)


# ============================================================
# WARP
# ============================================================

def warp_image_similarity(
    image: np.ndarray,
    transform: SimilarityTransform,
    output_size: Tuple[int, int],
    border_value=0,
) -> np.ndarray:

    width, height = output_size

    return cv2.warpAffine(
        image,
        transform.matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


# ============================================================
# DEBUG
# ============================================================

def draw_control_points(
    image: np.ndarray,
    points: np.ndarray,
    labels,
) -> np.ndarray:

    debug = image.copy()

    if (
        debug.ndim == 3
        and debug.shape[2] == 4
    ):
        # Composite RGBA over black for debugging.
        alpha = (
            debug[:, :, 3:4].astype(np.float32)
            / 255.0
        )

        rgb = (
            debug[:, :, :3].astype(np.float32)
        )

        debug = (
            rgb * alpha
        ).astype(np.uint8)

    for point, label in zip(
        points,
        labels,
    ):
        x = int(
            round(
                float(point[0])
            )
        )

        y = int(
            round(
                float(point[1])
            )
        )

        cv2.circle(
            debug,
            (x, y),
            6,
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            debug,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return debug


# ============================================================
# ALIGN GARMENT
# ============================================================

def align_garment(
    garment_image: np.ndarray,
    garment_mask: np.ndarray,
    person_landmarks: Landmarks,
    person_image_shape: Tuple[int, int],
) -> AlignedGarment:

    if garment_image is None:
        raise ValueError(
            "Garment image is None."
        )

    if garment_image.size == 0:
        raise ValueError(
            "Garment image is empty."
        )

    if garment_image.ndim != 3:
        raise ValueError(
            f"Garment image must be 3D. Got {garment_image.shape}"
        )

    if garment_image.shape[2] not in (3, 4):
        raise ValueError(
            "Garment image must have 3 or 4 channels."
        )

    garment_mask = validate_mask(
        garment_mask
    )

    if garment_image.shape[:2] != garment_mask.shape[:2]:
        raise ValueError(
            "Garment image and mask dimensions do not match:\n"
            f"Garment = {garment_image.shape[:2]}\n"
            f"Mask    = {garment_mask.shape[:2]}"
        )

    # --------------------------------------------------------
    # Geometry.
    # --------------------------------------------------------
    garment_geometry = extract_garment_geometry(
        garment_mask
    )

    body_geometry = build_body_geometry(
        person_landmarks,
        person_image_shape,
    )

    garment_points, body_points = build_correspondences(
        garment_geometry,
        body_geometry,
    )

    # --------------------------------------------------------
    # Stable torso-aware transform.
    # --------------------------------------------------------
    similarity = estimate_similarity_transform(
        garment_points,
        body_points,
    )

    aligned_points = apply_similarity_to_points(
        garment_points,
        similarity,
    )

    output_height = int(
        person_image_shape[0]
    )

    output_width = int(
        person_image_shape[1]
    )

    output_size = (
        output_width,
        output_height,
    )

    # --------------------------------------------------------
    # Warp garment.
    # --------------------------------------------------------
    if garment_image.shape[2] == 4:
        aligned_garment = warp_image_similarity(
            garment_image,
            similarity,
            output_size,
            border_value=(0, 0, 0, 0),
        )
    else:
        aligned_garment = warp_image_similarity(
            garment_image,
            similarity,
            output_size,
            border_value=(0, 0, 0),
        )

    # --------------------------------------------------------
    # Warp mask with EXACT SAME transform.
    # --------------------------------------------------------
    aligned_mask = warp_image_similarity(
        garment_mask,
        similarity,
        output_size,
        border_value=0,
    )

    aligned_mask = np.clip(
        aligned_mask,
        0,
        255,
    ).astype(np.uint8)

    _, aligned_mask = cv2.threshold(
        aligned_mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    # Small closing to avoid holes caused by interpolation.
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    aligned_mask = cv2.morphologyEx(
        aligned_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # --------------------------------------------------------
    # Keep RGBA alpha synchronized with the mask.
    # --------------------------------------------------------
    if (
        aligned_garment.ndim == 3
        and aligned_garment.shape[2] == 4
    ):
        aligned_garment[:, :, 3] = aligned_mask

    # --------------------------------------------------------
    # Debug.
    # --------------------------------------------------------
    debug_image = draw_control_points(
        garment_image,
        garment_points,
        [
            "neck",
            "L_shoulder",
            "R_shoulder",
            "L_sleeve",
            "R_sleeve",
            "L_hem",
            "R_hem",
        ],
    )

    return AlignedGarment(
        image=aligned_garment,
        mask=aligned_mask,
        garment_points=garment_points,
        body_points=body_points,
        similarity=similarity,
        aligned_points=aligned_points,
        debug_image=debug_image,
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_alignment_debug(
    result: AlignedGarment,
    output_dir: str,
) -> None:

    output = __import__("pathlib").Path(
        output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if result.debug_image is not None:
        cv2.imwrite(
            str(
                output
                /
                "garment_control_points.png"
            ),
            result.debug_image,
        )

    cv2.imwrite(
        str(
            output
            /
            "aligned_garment.png"
        ),
        result.image,
    )

    cv2.imwrite(
        str(
            output
            /
            "aligned_garment_mask.png"
        ),
        result.mask,
    )


def alignment_summary(
    result: AlignedGarment,
) -> Dict[str, Any]:

    return {
        "scale": float(
            result.similarity.scale
        ),
        "rotation_degrees": float(
            result.similarity.rotation_degrees
        ),
        "translation": (
            float(
                result.similarity.translation[0]
            ),
            float(
                result.similarity.translation[1]
            ),
        ),
        "garment_points": (
            result.garment_points.tolist()
        ),
        "body_points": (
            result.body_points.tolist()
        ),
        "aligned_points": (
            result.aligned_points.tolist()
        ),
    }
