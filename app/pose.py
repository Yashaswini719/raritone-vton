from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp


@dataclass
class Landmark:
    """One Raritone body landmark."""

    name: str
    x: float
    y: float
    z: float
    visibility: float
    pixel_x: int
    pixel_y: int


class PoseEstimator:
    """
    Raritone pose estimation using the MediaPipe Tasks API.

    Detects one person and extracts the body landmarks needed
    for garment alignment.
    """

    REQUIRED_LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }

    def __init__(
        self,
        model_path: str = "models/pose_landmarker_full.task",
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model = Path(model_path)

        if not model.exists():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}\n"
                "Place pose_landmarker_full.task inside models/."
            )

        base_options = mp.tasks.BaseOptions

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options(
                model_asset_path=str(model.resolve())
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = (
            mp.tasks.vision.PoseLandmarker.create_from_options(options)
        )

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    @staticmethod
    def _load_image(image_path: str):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return image

    @staticmethod
    def _to_mp_image(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

    @staticmethod
    def _pixel_coordinates(
        landmark,
        width: int,
        height: int,
    ):
        """
        Convert normalized MediaPipe coordinates into
        safe image pixel coordinates.
        """

        x = float(landmark.x)
        y = float(landmark.y)

        pixel_x = round(x * width)
        pixel_y = round(y * height)

        # Keep coordinates inside image boundaries.
        pixel_x = max(0, min(width - 1, pixel_x))
        pixel_y = max(0, min(height - 1, pixel_y))

        return x, y, pixel_x, pixel_y

    def _extract_landmarks(
        self,
        detected,
        width: int,
        height: int,
    ) -> Dict[str, Landmark]:

        landmarks = {}

        for name, index in self.REQUIRED_LANDMARKS.items():

            landmark = detected[index]

            x, y, pixel_x, pixel_y = self._pixel_coordinates(
                landmark,
                width,
                height,
            )

            visibility = float(
                getattr(landmark, "visibility", 1.0)
            )

            z = float(landmark.z)

            landmarks[name] = Landmark(
                name=name,
                x=x,
                y=y,
                z=z,
                visibility=visibility,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
            )

        return landmarks

    # ---------------------------------------------------------
    # Pose estimation
    # ---------------------------------------------------------

    def estimate(
        self,
        image_path: str,
    ) -> Optional[Dict[str, Landmark]]:
        """
        Detect one person and return the required landmarks.

        Returns None if no person/pose is detected.
        """

        image = self._load_image(image_path)

        height, width = image.shape[:2]

        mp_image = self._to_mp_image(image)

        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        detected = result.pose_landmarks[0]

        return self._extract_landmarks(
            detected,
            width,
            height,
        )

    # ---------------------------------------------------------
    # Debug visualization
    # ---------------------------------------------------------

    def save_debug_image(
        self,
        image_path: str,
        output_path: str,
    ) -> Dict[str, Landmark]:
        """
        Detect pose and save an annotated debug image.

        The debug image contains:
        - all detected landmarks
        - pose connections
        - important Raritone landmark names
        """

        image = self._load_image(image_path)

        height, width = image.shape[:2]

        mp_image = self._to_mp_image(image)

        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            raise ValueError(
                f"No human pose detected in image: {image_path}"
            )

        detected = result.pose_landmarks[0]

        landmarks = self._extract_landmarks(
            detected,
            width,
            height,
        )

        # -----------------------------------------------------
        # Draw complete detected skeleton
        # -----------------------------------------------------

        connections = (
            mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        )

        for connection in connections:

            start = detected[connection.start]
            end = detected[connection.end]

            _, _, x1, y1 = self._pixel_coordinates(
                start,
                width,
                height,
            )

            _, _, x2, y2 = self._pixel_coordinates(
                end,
                width,
                height,
            )

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # -----------------------------------------------------
        # Draw all landmarks
        # -----------------------------------------------------

        for landmark in detected:

            _, _, x, y = self._pixel_coordinates(
                landmark,
                width,
                height,
            )

            cv2.circle(
                image,
                (x, y),
                4,
                (0, 0, 255),
                -1,
            )

        # -----------------------------------------------------
        # Draw important Raritone landmarks + labels
        # -----------------------------------------------------

        for name, landmark in landmarks.items():

            x = landmark.pixel_x
            y = landmark.pixel_y

            cv2.circle(
                image,
                (x, y),
                6,
                (255, 0, 0),
                -1,
            )

            cv2.putText(
                image,
                name,
                (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        output = Path(output_path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output),
            image,
        )

        if not success:
            raise IOError(
                f"Unable to save pose debug image: {output}"
            )

        return landmarks

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(self):
        """Release MediaPipe resources."""

        self.landmarker.close()