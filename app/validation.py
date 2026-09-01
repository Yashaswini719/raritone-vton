from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError


@dataclass
class ValidationResult:
    valid: bool
    message: str
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None


# Minimum dimensions for Raritone VTON input.
MIN_WIDTH = 256
MIN_HEIGHT = 256

# Supported input formats.
SUPPORTED_FORMATS = {"JPEG", "PNG"}


def validate_image(image_path: str) -> ValidationResult:
    """
    Validate a person or garment image.

    Checks:
    - file exists
    - file is readable
    - image can be decoded
    - supported image format
    - minimum resolution
    """

    path = Path(image_path)

    # 1. Check that the file exists.
    if not path.exists():
        return ValidationResult(
            valid=False,
            message=f"Image file does not exist: {image_path}",
        )

    # 2. Check that it is actually a file.
    if not path.is_file():
        return ValidationResult(
            valid=False,
            message=f"Path is not a file: {image_path}",
        )

    try:
        # 3. Open and inspect the image.
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size

            # 4. Check supported format.
            if image_format not in SUPPORTED_FORMATS:
                return ValidationResult(
                    valid=False,
                    message=(
                        f"Unsupported image format: {image_format}. "
                        f"Supported formats: JPG/JPEG and PNG."
                    ),
                    width=width,
                    height=height,
                    format=image_format,
                )

            # 5. Check resolution.
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return ValidationResult(
                    valid=False,
                    message=(
                        f"Image resolution is too small: "
                        f"{width}x{height}. "
                        f"Minimum required resolution is "
                        f"{MIN_WIDTH}x{MIN_HEIGHT}."
                    ),
                    width=width,
                    height=height,
                    format=image_format,
                )

            # 6. Force decoding to detect corrupted images.
            image.verify()

        return ValidationResult(
            valid=True,
            message="Image validation successful.",
            width=width,
            height=height,
            format=image_format,
        )

    except UnidentifiedImageError:
        return ValidationResult(
            valid=False,
            message="The file is not a valid or supported image.",
        )

    except OSError as exc:
        return ValidationResult(
            valid=False,
            message=f"Unable to read image: {exc}",
        )