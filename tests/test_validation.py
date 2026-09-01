from PIL import Image

from app.validation import validate_image


def test_valid_jpeg(tmp_path):
    image_path = tmp_path / "person.jpg"

    image = Image.new("RGB", (512, 512), "white")
    image.save(image_path, format="JPEG")

    result = validate_image(str(image_path))

    assert result.valid is True
    assert result.width == 512
    assert result.height == 512
    assert result.format == "JPEG"


def test_valid_png(tmp_path):
    image_path = tmp_path / "garment.png"

    image = Image.new("RGB", (512, 512), "white")
    image.save(image_path, format="PNG")

    result = validate_image(str(image_path))

    assert result.valid is True
    assert result.format == "PNG"


def test_missing_file():
    result = validate_image("does_not_exist.jpg")

    assert result.valid is False
    assert "does not exist" in result.message


def test_small_image(tmp_path):
    image_path = tmp_path / "small.jpg"

    image = Image.new("RGB", (100, 100), "white")
    image.save(image_path, format="JPEG")

    result = validate_image(str(image_path))

    assert result.valid is False
    assert "too small" in result.message