from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp


class PersonSegmenter:
    """
    Raritone person segmentation using
    MediaPipe Tasks ImageSegmenter.

    Output:
        Binary person mask:
        255 = person
        0   = background
    """

    def __init__(
        self,
        model_path: str = "models/selfie_segmenter.tflite",
        threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Segmentation model not found: {self.model_path}\n"
                "Place selfie_segmenter.tflite inside models/."
            )

        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = (
            mp.tasks.vision.ImageSegmenterOptions
        )

        options = ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=str(
                    self.model_path.resolve()
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False,
        )

        self.segmenter = (
            ImageSegmenter.create_from_options(options)
        )

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------

    @staticmethod
    def _load_image(image_path: str):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return image

    # ---------------------------------------------------------
    # Convert OpenCV image to MediaPipe image
    # ---------------------------------------------------------

    @staticmethod
    def _to_mp_image(image):

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

    # ---------------------------------------------------------
    # Generate mask
    # ---------------------------------------------------------

    def segment(
        self,
        image_path: str,
    ) -> np.ndarray:
        """
        Generate a binary person mask.

        Returns:
            numpy array with:
                255 = person
                0   = background
        """

        image = self._load_image(image_path)

        mp_image = self._to_mp_image(image)

        result = self.segmenter.segment(mp_image)

        if result.category_mask is None:
            raise RuntimeError(
                "MediaPipe did not return a category mask."
            )

        category_mask = result.category_mask

        mask = category_mask.numpy_view()

        # Convert to numpy array.
        mask = np.asarray(mask)

        # Remove unnecessary dimensions.
        mask = np.squeeze(mask)

        # MediaPipe selfie segmentation is a two-class
        # person/background segmentation model.
        #
        # We inspect the actual values rather than assuming
        # a particular output dtype.

        if mask.dtype == np.uint8:
            binary_mask = np.where(
                mask > 0,
                255,
                0,
            ).astype(np.uint8)

        else:
            # Float/category values.
            binary_mask = np.where(
                mask >= self.threshold,
                255,
                0,
            ).astype(np.uint8)

        # Resize mask back to original image size.
        height, width = image.shape[:2]

        if binary_mask.shape != (height, width):

            binary_mask = cv2.resize(
                binary_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        return binary_mask

    # ---------------------------------------------------------
    # Save mask
    # ---------------------------------------------------------

    def save_mask(
        self,
        image_path: str,
        output_path: str,
    ) -> np.ndarray:
        """
        Generate and save the person mask.
        """

        mask = self.segment(image_path)

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output),
            mask,
        )

        if not success:
            raise IOError(
                f"Unable to save person mask: {output}"
            )

        return mask

    # ---------------------------------------------------------
    # Create foreground preview
    # ---------------------------------------------------------

    def save_foreground(
        self,
        image_path: str,
        output_path: str,
    ):
        """
        Save the person extracted from the background.

        The output has a white background for easy debugging.
        """

        image = self._load_image(image_path)

        mask = self.segment(image_path)

        foreground = image.copy()

        background = np.full_like(
            image,
            255,
        )

        mask_3ch = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR,
        )

        foreground = np.where(
            mask_3ch > 0,
            foreground,
            background,
        ).astype(np.uint8)

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not cv2.imwrite(
            str(output),
            foreground,
        ):
            raise IOError(
                f"Unable to save foreground image: {output}"
            )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(self):
        """Release MediaPipe resources."""

        self.segmenter.close()