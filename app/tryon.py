from pathlib import Path

import cv2
import numpy as np


class TryOnPipeline:
    """
    Raritone VTON final compositing pipeline.

    Inputs:
        1. Person image
        2. Aligned garment RGBA image
        3. Person segmentation mask
        4. Aligned garment mask

    Output:
        Final virtual try-on image

    All inputs must use the same:
        (H, W)
    coordinate system.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        feather_radius: int = 2,
        garment_opacity: float = 1.0,
    ):
        """
        Parameters
        ----------
        feather_radius:
            Amount of edge feathering.
            0 disables feathering.

        garment_opacity:
            Final garment opacity.
            1.0 = completely opaque.
            0.0 = invisible.
        """

        if feather_radius < 0:
            raise ValueError(
                "feather_radius must be >= 0"
            )

        if not 0.0 <= garment_opacity <= 1.0:
            raise ValueError(
                "garment_opacity must be between 0.0 and 1.0"
            )

        self.feather_radius = int(
            feather_radius
        )

        self.garment_opacity = float(
            garment_opacity
        )

    # =========================================================
    # IMAGE LOADING
    # =========================================================

    @staticmethod
    def _load_image(
        path: str,
        flags: int,
    ) -> np.ndarray:
        """
        Load an image using OpenCV.
        """

        image_path = Path(path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"File not found: {image_path}"
            )

        image = cv2.imread(
            str(image_path),
            flags,
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return image

    # =========================================================
    # DIMENSION VALIDATION
    # =========================================================

    @staticmethod
    def _validate_dimensions(
        person: np.ndarray,
        garment: np.ndarray,
        person_mask: np.ndarray,
        garment_mask: np.ndarray,
    ) -> None:
        """
        Verify that all images belong to the same
        person-sized coordinate system.
        """

        person_size = person.shape[:2]

        if garment.shape[:2] != person_size:
            raise ValueError(
                "Aligned garment dimensions do not match "
                "person dimensions: "
                f"{garment.shape[:2]} vs "
                f"{person_size}"
            )

        if person_mask.shape[:2] != person_size:
            raise ValueError(
                "Person mask dimensions do not match "
                "person image: "
                f"{person_mask.shape[:2]} vs "
                f"{person_size}"
            )

        if garment_mask.shape[:2] != person_size:
            raise ValueError(
                "Garment mask dimensions do not match "
                "person image: "
                f"{garment_mask.shape[:2]} vs "
                f"{person_size}"
            )

    # =========================================================
    # BINARY MASK
    # =========================================================

    @staticmethod
    def _binary_mask(
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Convert grayscale mask to binary:

            foreground = 255
            background = 0
        """

        if mask is None:
            raise ValueError(
                "Mask is None."
            )

        if mask.ndim != 2:
            raise ValueError(
                "Mask must be single-channel."
            )

        return np.where(
            mask > 127,
            255,
            0,
        ).astype(np.uint8)

    # =========================================================
    # PERSON MASK NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_person_mask(
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize person segmentation polarity.

        Final convention:

            PERSON     = 255
            BACKGROUND = 0

        The mask may arrive inverted, so the border is
        inspected because the image border normally belongs
        to the background.
        """

        binary = TryOnPipeline._binary_mask(
            mask
        )

        h, w = binary.shape

        if h < 2 or w < 2:
            raise ValueError(
                "Person mask is too small."
            )

        border_pixels = np.concatenate(
            [
                binary[0, :],
                binary[-1, :],
                binary[:, 0],
                binary[:, -1],
            ]
        )

        white_ratio = np.mean(
            border_pixels == 255
        )

        black_ratio = np.mean(
            border_pixels == 0
        )

        # -----------------------------------------------------
        # Mostly-white border:
        #
        # white = background
        # black = person
        #
        # Therefore invert.
        # -----------------------------------------------------

        if white_ratio > 0.50:

            binary = cv2.bitwise_not(
                binary
            )

        elif black_ratio > 0.50:

            # Already:
            #
            # person = white
            # background = black
            #
            pass

        return binary

    # =========================================================
    # MASK CLEANING
    # =========================================================

    @staticmethod
    def _clean_mask(
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Remove small noise and close small gaps.
        """

        binary = TryOnPipeline._binary_mask(
            mask
        )

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        # Close small holes/gaps.
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # Remove tiny isolated noise.
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

        return binary

    # =========================================================
    # GARMENT ALPHA
    # =========================================================

    @staticmethod
    def _get_garment_alpha(
        garment: np.ndarray,
        garment_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Obtain garment transparency.

        RGBA garment:
            use alpha channel.

        BGR garment:
            use garment mask.
        """

        if garment.ndim != 3:
            raise ValueError(
                "Garment image must be a 3D array."
            )

        channels = garment.shape[2]

        if channels == 4:

            alpha = garment[:, :, 3]

        elif channels == 3:

            alpha = garment_mask

        else:

            raise ValueError(
                "Garment must have either "
                "3 or 4 channels."
            )

        return np.clip(
            alpha,
            0,
            255,
        ).astype(np.uint8)

    # =========================================================
    # TRY-ON MASK
    # =========================================================

    @staticmethod
    def _create_tryon_mask(
        garment_mask: np.ndarray,
        person_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create the valid garment placement region.

        Final mask:

            garment ∩ person
        """

        # -----------------------------------------------------
        # Garment mask
        # -----------------------------------------------------

        garment_binary = (
            TryOnPipeline._clean_mask(
                garment_mask
            )
        )

        # -----------------------------------------------------
        # Person mask
        # -----------------------------------------------------

        person_binary = (
            TryOnPipeline._normalize_person_mask(
                person_mask
            )
        )

        # -----------------------------------------------------
        # Intersection
        # -----------------------------------------------------

        tryon_mask = cv2.bitwise_and(
            garment_binary,
            person_binary,
        )

        return tryon_mask

    # =========================================================
    # MASK FEATHERING
    # =========================================================

    def _soften_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Feather the edge of the final mask.
        """

        if self.feather_radius <= 0:
            return mask.copy()

        # Gaussian kernel must be odd.
        kernel_size = (
            self.feather_radius * 2 + 1
        )

        mask_float = mask.astype(
            np.float32
        )

        mask_float = cv2.GaussianBlur(
            mask_float,
            (
                kernel_size,
                kernel_size,
            ),
            0,
        )

        return np.clip(
            mask_float,
            0,
            255,
        ).astype(np.uint8)

    # =========================================================
    # COMPOSITING
    # =========================================================

    def _composite(
        self,
        person: np.ndarray,
        garment: np.ndarray,
        tryon_mask: np.ndarray,
        garment_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Alpha-composite aligned garment over person.
        """

        # -----------------------------------------------------
        # Garment color
        # -----------------------------------------------------

        if garment.shape[2] == 4:

            garment_bgr = garment[:, :, :3]

        elif garment.shape[2] == 3:

            garment_bgr = garment

        else:

            raise ValueError(
                "Garment must have 3 or 4 channels."
            )

        # -----------------------------------------------------
        # Garment alpha
        # -----------------------------------------------------

        garment_alpha = (
            self._get_garment_alpha(
                garment,
                garment_mask,
            )
        )

        # -----------------------------------------------------
        # Feather final mask
        # -----------------------------------------------------

        soft_mask = (
            self._soften_mask(
                tryon_mask
            )
        )

        # -----------------------------------------------------
        # Normalize to [0,1]
        # -----------------------------------------------------

        mask_float = (
            soft_mask.astype(
                np.float32
            )
            / 255.0
        )

        alpha_float = (
            garment_alpha.astype(
                np.float32
            )
            / 255.0
        )

        # -----------------------------------------------------
        # Final alpha
        # -----------------------------------------------------

        alpha = (
            mask_float
            * alpha_float
            * self.garment_opacity
        )

        alpha = np.clip(
            alpha,
            0.0,
            1.0,
        )

        alpha = alpha[:, :, None]

        # -----------------------------------------------------
        # Convert images to float
        # -----------------------------------------------------

        person_float = (
            person.astype(
                np.float32
            )
        )

        garment_float = (
            garment_bgr.astype(
                np.float32
            )
        )

        # -----------------------------------------------------
        # Alpha composite
        # -----------------------------------------------------

        result = (
            garment_float * alpha
            +
            person_float * (1.0 - alpha)
        )

        # -----------------------------------------------------
        # Convert back to uint8
        # -----------------------------------------------------

        result = np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

        return result

    # =========================================================
    # MASK STATISTICS
    # =========================================================

    @staticmethod
    def _print_mask_statistics(
        name: str,
        mask: np.ndarray,
    ) -> None:
        """
        Print foreground statistics for debugging.
        """

        foreground = np.count_nonzero(
            mask > 127
        )

        total = mask.size

        percentage = (
            foreground
            / total
            * 100.0
        )

        print(
            f"{name}: "
            f"{foreground}/{total} pixels "
            f"({percentage:.2f}%) foreground"
        )

    # =========================================================
    # MAIN PIPELINE
    # =========================================================

    def generate(
        self,
        person_path: str,
        garment_path: str,
        person_mask_path: str,
        garment_mask_path: str,
        output_path: str,
    ) -> np.ndarray:
        """
        Run the complete final compositing pipeline.
        """

        print()
        print("=" * 60)
        print("Raritone VTON")
        print("Final Try-On Pipeline")
        print("=" * 60)

        # =====================================================
        # 1. PERSON
        # =====================================================

        print()
        print("Loading person...")

        person = self._load_image(
            person_path,
            cv2.IMREAD_COLOR,
        )

        print(
            "Person:",
            person.shape,
        )

        # =====================================================
        # 2. ALIGNED GARMENT
        # =====================================================

        print()
        print("Loading aligned garment...")

        garment = self._load_image(
            garment_path,
            cv2.IMREAD_UNCHANGED,
        )

        print(
            "Aligned garment:",
            garment.shape,
        )

        if garment.ndim != 3:
            raise ValueError(
                "Aligned garment must be a 3D image."
            )

        if garment.shape[2] not in (3, 4):
            raise ValueError(
                "Aligned garment must be BGR or BGRA."
            )

        # =====================================================
        # 3. PERSON MASK
        # =====================================================

        print()
        print("Loading person mask...")

        person_mask = self._load_image(
            person_mask_path,
            cv2.IMREAD_GRAYSCALE,
        )

        print(
            "Person mask:",
            person_mask.shape,
        )

        # =====================================================
        # 4. GARMENT MASK
        # =====================================================

        print()
        print("Loading aligned garment mask...")

        garment_mask = self._load_image(
            garment_mask_path,
            cv2.IMREAD_GRAYSCALE,
        )

        print(
            "Garment mask:",
            garment_mask.shape,
        )

        # =====================================================
        # 5. DIMENSIONS
        # =====================================================

        print()
        print("Validating dimensions...")

        self._validate_dimensions(
            person,
            garment,
            person_mask,
            garment_mask,
        )

        print(
            "Dimension validation: OK"
        )

        # =====================================================
        # 6. PERSON MASK
        # =====================================================

        print()
        print("Normalizing person mask...")

        normalized_person_mask = (
            self._normalize_person_mask(
                person_mask
            )
        )

        self._print_mask_statistics(
            "Person mask",
            normalized_person_mask,
        )

        # =====================================================
        # 7. GARMENT MASK
        # =====================================================

        print()
        print("Cleaning garment mask...")

        cleaned_garment_mask = (
            self._clean_mask(
                garment_mask
            )
        )

        self._print_mask_statistics(
            "Garment mask",
            cleaned_garment_mask,
        )

        # =====================================================
        # 8. TRY-ON MASK
        # =====================================================

        print()
        print("Creating try-on mask...")

        tryon_mask = (
            self._create_tryon_mask(
                cleaned_garment_mask,
                normalized_person_mask,
            )
        )

        self._print_mask_statistics(
            "Try-on mask",
            tryon_mask,
        )

        # =====================================================
        # SAFETY CHECK
        # =====================================================

        tryon_pixels = np.count_nonzero(
            tryon_mask > 127
        )

        if tryon_pixels == 0:
            raise ValueError(
                "Try-on mask contains zero foreground pixels. "
                "Check person mask and garment mask."
            )

        # =====================================================
        # 9. OUTPUT DIRECTORY
        # =====================================================

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # 10. DEBUG MASK
        # =====================================================

        debug_mask_path = (
            output.parent
            / "tryon_mask.png"
        )

        cv2.imwrite(
            str(debug_mask_path),
            tryon_mask,
        )

        print()
        print(
            "Debug mask:",
            debug_mask_path,
        )

        # =====================================================
        # 11. COMPOSITE
        # =====================================================

        print()
        print("Compositing garment...")

        result = self._composite(
            person,
            garment,
            tryon_mask,
            cleaned_garment_mask,
        )

        # =====================================================
        # 12. SAVE FINAL IMAGE
        # =====================================================

        print()
        print("Saving final result...")

        success = cv2.imwrite(
            str(output),
            result,
        )

        if not success:
            raise IOError(
                f"Unable to save final try-on image: "
                f"{output}"
            )

        # =====================================================
        # 13. COMPLETED
        # =====================================================

        print()
        print("=" * 60)
        print("TRY-ON COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(
            "Final image :",
            output,
        )

        print(
            "Debug mask  :",
            debug_mask_path,
        )

        print(
            "Output size :",
            result.shape,
        )

        print(
            "Feather     :",
            self.feather_radius,
        )

        print(
            "Opacity     :",
            self.garment_opacity,
        )

        print("=" * 60)
        print()

        return result