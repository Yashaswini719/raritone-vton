from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from rembg import remove


class GarmentProcessor:
    """
    Raritone garment preprocessing pipeline.

    Pipeline:
        Garment image
            ↓
        Background removal
            ↓
        Garment mask
            ↓
        Crop to garment
            ↓
        Resize while preserving aspect ratio
            ↓
        Center on fixed canvas
            ↓
        Save processed garment + mask
    """

    TARGET_WIDTH = 512
    TARGET_HEIGHT = 384

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 384),
        alpha_threshold: int = 10,
    ):
        self.target_width = target_size[0]
        self.target_height = target_size[1]
        self.alpha_threshold = alpha_threshold

    def _read_image(self, image_path: str) -> Image.Image:
        """
        Read garment image and convert to RGBA.
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Garment image not found: {image_path}"
            )

        try:
            image = Image.open(path).convert("RGBA")
        except Exception as exc:
            raise ValueError(
                f"Unable to read garment image: {image_path}"
            ) from exc

        return image

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the garment background using rembg.

        Returns:
            RGBA image where the alpha channel represents
            the garment foreground.
        """

        result = remove(image)

        if not isinstance(result, Image.Image):
            result = Image.open(result)

        return result.convert("RGBA")

    def _create_mask(
        self,
        rgba_image: Image.Image,
    ) -> np.ndarray:
        """
        Extract a binary garment mask from the alpha channel.

        Returns:
            uint8 mask containing only 0 and 255.
        """

        rgba = np.array(rgba_image)

        alpha = rgba[:, :, 3]

        mask = np.where(
            alpha >= self.alpha_threshold,
            255,
            0,
        ).astype(np.uint8)

        # Remove tiny isolated noise.
        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        # Fill small holes inside the garment.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return mask

    def _crop_to_garment(
        self,
        rgba_image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Crop image and mask to the garment bounding box.
        """

        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            raise ValueError(
                "No garment foreground detected after background removal."
            )

        x_min = int(xs.min())
        x_max = int(xs.max()) + 1
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1

        cropped_image = rgba_image.crop(
            (x_min, y_min, x_max, y_max)
        )

        cropped_mask = mask[
            y_min:y_max,
            x_min:x_max,
        ]

        return cropped_image, cropped_mask

    def _resize_and_center(
        self,
        rgba_image: Image.Image,
        mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Resize the garment while preserving its aspect ratio,
        then center it on a fixed 512x384 transparent canvas.
        """

        source_width, source_height = rgba_image.size

        if source_width <= 0 or source_height <= 0:
            raise ValueError("Invalid garment dimensions.")

        scale = min(
            self.target_width / source_width,
            self.target_height / source_height,
        )

        new_width = max(1, int(round(source_width * scale)))
        new_height = max(1, int(round(source_height * scale)))

        resized_image = rgba_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

        resized_mask = cv2.resize(
            mask,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )

        # Fixed transparent canvas.
        canvas = Image.new(
            "RGBA",
            (self.target_width, self.target_height),
            (0, 0, 0, 0),
        )

        mask_canvas = np.zeros(
            (self.target_height, self.target_width),
            dtype=np.uint8,
        )

        offset_x = (self.target_width - new_width) // 2
        offset_y = (self.target_height - new_height) // 2

        canvas.alpha_composite(
            resized_image,
            (offset_x, offset_y),
        )

        mask_canvas[
            offset_y:offset_y + new_height,
            offset_x:offset_x + new_width,
        ] = resized_mask

        return canvas, mask_canvas

    def _normalize(
        self,
        rgba_image: Image.Image,
    ) -> np.ndarray:
        """
        Convert processed RGBA image into float32 [0, 1]
        representation.

        This is useful for later model inference.
        """

        array = np.asarray(
            rgba_image,
            dtype=np.float32,
        )

        return array / 255.0

    def process(
        self,
        image_path: str,
        output_image_path: str,
        output_mask_path: str,
    ):
        """
        Run the complete garment preprocessing pipeline.

        Saves:
            processed garment PNG
            garment binary mask PNG

        Returns:
            Dictionary containing useful processing information.
        """

        # 1. Read
        image = self._read_image(image_path)

        original_size = image.size

        # 2. Background removal
        foreground = self._remove_background(image)

        # 3. Create binary garment mask
        mask = self._create_mask(foreground)

        # 4. Crop
        cropped_image, cropped_mask = self._crop_to_garment(
            foreground,
            mask,
        )

        # 5. Resize + center
        processed_image, processed_mask = self._resize_and_center(
            cropped_image,
            cropped_mask,
        )

        # 6. Normalize for future inference
        normalized = self._normalize(processed_image)

        if normalized.dtype != np.float32:
            raise RuntimeError(
                "Garment normalization failed."
            )

        # Ensure output directories exist.
        image_output = Path(output_image_path)
        mask_output = Path(output_mask_path)

        image_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 7. Save processed garment.
        processed_image.save(
            image_output,
            format="PNG",
        )

        # 8. Save binary mask.
        if not cv2.imwrite(
            str(mask_output),
            processed_mask,
        ):
            raise IOError(
                f"Unable to save garment mask: {output_mask_path}"
            )

        garment_pixels = int(
            np.count_nonzero(processed_mask)
        )

        total_pixels = (
            self.target_width *
            self.target_height
        )

        coverage = garment_pixels / total_pixels

        return {
            "original_width": original_size[0],
            "original_height": original_size[1],
            "processed_width": self.target_width,
            "processed_height": self.target_height,
            "garment_pixels": garment_pixels,
            "mask_coverage": coverage,
            "image_path": str(image_output),
            "mask_path": str(mask_output),
        }