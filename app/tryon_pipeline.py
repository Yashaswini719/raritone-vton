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

    Expected dimensions:

        Person:
            (H, W, 3)

        Aligned garment:
            (H, W, 4)

        Person mask:
            (H, W)

        Garment mask:
            (H, W)

    """

    def __init__(self):
        pass

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
                f"File not found: {path}"
            )

        image = cv2.imread(
            str(image_path),
            flags,
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {path}"
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
    ):
        """
        Make sure every image belongs to the same
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
        Convert any grayscale mask to:

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
        Normalize the person segmentation mask.

        The final convention is:

            PERSON       = 255
            BACKGROUND   = 0

        Your current segmentation output appears to be:

            PERSON       = 0
            BACKGROUND   = 255

        Therefore we inspect the image border.

        Since the border should normally belong to the
        background, a mostly-white border indicates that
        the mask needs inversion.
        """

        binary = np.where(
            mask > 127,
            255,
            0,
        ).astype(np.uint8)

        height, width = binary.shape

        if height < 2 or width < 2:
            raise ValueError(
                "Person mask is too small."
            )

        # -----------------------------------------------------
        # Collect border pixels.
        # -----------------------------------------------------

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
        # If most border pixels are white, white is likely
        # background and therefore the mask is inverted.
        # -----------------------------------------------------

        if white_ratio > 0.50:
            binary = cv2.bitwise_not(
                binary
            )

        elif black_ratio > 0.50:
            # Already conventional:
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
        Remove small holes/noise while preserving the
        main foreground region.
        """

        binary = TryOnPipeline._binary_mask(
            mask
        )

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        # Fill small gaps.
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # Remove tiny isolated regions.
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

        return binary

    # =========================================================
    # CREATE TRY-ON MASK
    # =========================================================

    @staticmethod
    def _create_tryon_mask(
        garment_mask: np.ndarray,
        person_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create the final garment placement mask.

        Garment:

            garment = 255
            background = 0

        Person:

            person = 255
            background = 0

        Final:

            garment AND person
        """

        # -----------------------------------------------------
        # Normalize garment mask.
        # -----------------------------------------------------

        garment_binary = (
            TryOnPipeline._clean_mask(
                garment_mask
            )
        )

        # -----------------------------------------------------
        # Normalize person mask.
        #
        # IMPORTANT:
        # Do NOT call _clean_mask() before this because the
        # current person mask has inverted polarity.
        # -----------------------------------------------------

        person_binary = (
            TryOnPipeline._normalize_person_mask(
                person_mask
            )
        )

        # Clean after polarity correction.
        person_binary = (
            TryOnPipeline._clean_mask(
                person_binary
            )
        )

        # -----------------------------------------------------
        # Intersection.
        # -----------------------------------------------------

        tryon_mask = cv2.bitwise_and(
            garment_binary,
            person_binary,
        )

        return tryon_mask

    # =========================================================
    # GARMENT ALPHA
    # =========================================================

    @staticmethod
    def _get_garment_alpha(
        garment: np.ndarray,
        garment_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Get garment alpha.

        If garment is RGBA, use its alpha channel.

        If garment is BGR, use the garment mask.
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

        alpha = np.where(
            alpha > 127,
            alpha,
            0,
        ).astype(np.uint8)

        return alpha

    # =========================================================
    # COMPOSITING
    # =========================================================

    @staticmethod
    def _composite(
        person: np.ndarray,
        garment: np.ndarray,
        tryon_mask: np.ndarray,
        garment_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Composite the aligned garment onto the person.

        Uses:

            final_alpha =
                garment_alpha
                × tryon_mask
        """

        # -----------------------------------------------------
        # Get garment RGB/BGR channels.
        # -----------------------------------------------------

        if garment.shape[2] == 4:

            garment_bgr = garment[:, :, :3]

        else:

            garment_bgr = garment

        # -----------------------------------------------------
        # Get garment alpha.
        # -----------------------------------------------------

        garment_alpha = (
            TryOnPipeline._get_garment_alpha(
                garment,
                garment_mask,
            )
        )

        # -----------------------------------------------------
        # Convert masks to floating point.
        # -----------------------------------------------------

        mask_float = (
            tryon_mask.astype(
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
        # Final alpha.
        # -----------------------------------------------------

        alpha = (
            mask_float
            * alpha_float
        )

        alpha = np.clip(
            alpha,
            0.0,
            1.0,
        )

        alpha = alpha[:, :, None]

        # -----------------------------------------------------
        # Convert images to float.
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
        # Alpha composite.
        # -----------------------------------------------------

        result = (
            garment_float * alpha
            +
            person_float * (1.0 - alpha)
        )

        result = np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

        return result

    # =========================================================
    # DEBUG INFORMATION
    # =========================================================

    @staticmethod
    def _print_mask_statistics(
        name: str,
        mask: np.ndarray,
    ):
        """
        Print useful mask statistics.
        """

        foreground = np.count_nonzero(
            mask > 127
        )

        total = mask.size

        percentage = (
            foreground / total * 100.0
        )

        print(
            f"{name}: "
            f"{foreground}/{total} pixels "
            f"({percentage:.2f}%) foreground"
        )

    # =========================================================
    # MAIN GENERATE
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
        Generate final virtual try-on image.
        """

        print()
        print("=" * 60)
        print("Raritone VTON")
        print("Final Try-On Compositing")
        print("=" * 60)

        # -----------------------------------------------------
        # 1. Load person.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 2. Load aligned garment.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 3. Load person mask.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 4. Load aligned garment mask.
        # -----------------------------------------------------

        print()
        print("Loading garment mask...")

        garment_mask = self._load_image(
            garment_mask_path,
            cv2.IMREAD_GRAYSCALE,
        )

        print(
            "Aligned garment mask:",
            garment_mask.shape,
        )

        # -----------------------------------------------------
        # 5. Validate.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 6. Normalize person mask.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 7. Clean garment mask.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 8. Create final try-on mask.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Safety check.
        # -----------------------------------------------------

        tryon_pixels = np.count_nonzero(
            tryon_mask > 127
        )

        if tryon_pixels == 0:
            raise ValueError(
                "Try-on mask contains zero foreground pixels. "
                "Check person/garment masks."
            )

        # -----------------------------------------------------
        # 9. Composite.
        # -----------------------------------------------------

        print()
        print("Compositing garment...")

        result = self._composite(
            person,
            garment,
            tryon_mask,
            cleaned_garment_mask,
        )

        # -----------------------------------------------------
        # 10. Save final image.
        # -----------------------------------------------------

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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

        # -----------------------------------------------------
        # 11. Save debug mask.
        # -----------------------------------------------------

        debug_mask_path = (
            output.parent
            / "tryon_mask.png"
        )

        cv2.imwrite(
            str(debug_mask_path),
            tryon_mask,
        )

        # -----------------------------------------------------
        # 12. Print result.
        # -----------------------------------------------------

        print()
        print("=" * 60)
        print("TRY-ON COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(
            f"Final image : {output}"
        )

        print(
            f"Debug mask  : {debug_mask_path}"
        )

        print(
            f"Output size : {result.shape}"
        )

        print("=" * 60)
        print()

        return result