from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.tryon import TryOnPipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RUNTIME_DIR = BASE_DIR / "runtime"
PERSON_DIR = RUNTIME_DIR / "persons"
GARMENT_DIR = RUNTIME_DIR / "garments"
MASK_DIR = RUNTIME_DIR / "masks"
OUTPUT_DIR = BASE_DIR / "outputs" / "api"

for directory in [
    PERSON_DIR,
    GARMENT_DIR,
    MASK_DIR,
    OUTPUT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Raritone VTON API",
    description="Raritone 2D Virtual Try-On Service",
    version="1.0.0",
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = TryOnPipeline(
    feather_radius=2,
    garment_opacity=1.0,
)


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/ai/health")
def health():
    """
    Health check endpoint.
    """

    return {
        "success": True,
        "service": "raritone-vton",
        "status": "healthy",
        "pipeline": "available",
    }


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_upload(
    upload: UploadFile,
    field_name: str,
) -> str:
    """
    Validate uploaded image filename/type.
    """

    if upload is None:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": f"{field_name} is required.",
            },
        )

    filename = upload.filename or ""

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": (
                    f"Unsupported {field_name} format. "
                    f"Allowed formats: "
                    f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
                ),
            },
        )

    return extension


# ============================================================
# SAVE UPLOAD
# ============================================================

async def save_upload(
    upload: UploadFile,
    destination: Path,
):
    """
    Save uploaded file safely.
    """

    try:
        with destination.open("wb") as output:
            shutil.copyfileobj(
                upload.file,
                output,
            )

    finally:
        await upload.close()


# ============================================================
# PERSON PROCESSING
# ============================================================

def process_person(
    person_path: Path,
    person_mask_path: Path,
):
    """
    Generate the person mask required by the
    current try-on pipeline.

    This uses the existing Raritone person-processing
    implementation when available.
    """

    try:
        from app.person_processor import PersonProcessor

        processor = PersonProcessor()

        result = processor.process(
            str(person_path)
        )

        if isinstance(result, dict):
            mask = result.get("mask")

            if mask is not None:
                import cv2

                cv2.imwrite(
                    str(person_mask_path),
                    mask,
                )

                return

        if hasattr(result, "mask"):
            import cv2

            cv2.imwrite(
                str(person_mask_path),
                result.mask,
            )

            return

    except Exception:
        pass

    # --------------------------------------------------------
    # Fallback:
    #
    # Use the existing normalized person mask from the
    # project dataset if the processor interface differs.
    #
    # This keeps the API usable while the processing modules
    # are being finalized.
    # --------------------------------------------------------

    import cv2
    import numpy as np

    image = cv2.imread(
        str(person_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Unable to read uploaded person image."
        )

    # Simple foreground approximation.
    #
    # This is intentionally a fallback only.
    # The dedicated segmentation module should be used
    # when deployed with the complete preprocessing stack.

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    mask = np.where(
        gray < 250,
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    cv2.imwrite(
        str(person_mask_path),
        mask,
    )


# ============================================================
# GARMENT PROCESSING
# ============================================================

def process_garment(
    garment_path: Path,
    garment_processed_path: Path,
    garment_mask_path: Path,
):
    """
    Generate garment image and mask.

    Uses the existing garment processor when available.
    """

    try:
        from app.garment_processor import GarmentProcessor

        processor = GarmentProcessor()

        result = processor.process(
            str(garment_path)
        )

        if isinstance(result, dict):

            processed = result.get(
                "image",
                result.get("garment"),
            )

            mask = result.get("mask")

            if processed is not None and mask is not None:

                import cv2

                cv2.imwrite(
                    str(garment_processed_path),
                    processed,
                )

                cv2.imwrite(
                    str(garment_mask_path),
                    mask,
                )

                return

        if (
            hasattr(result, "image")
            and hasattr(result, "mask")
        ):

            import cv2

            cv2.imwrite(
                str(garment_processed_path),
                result.image,
            )

            cv2.imwrite(
                str(garment_mask_path),
                result.mask,
            )

            return

    except Exception:
        pass

    # --------------------------------------------------------
    # Fallback garment processing.
    # --------------------------------------------------------

    import cv2
    import numpy as np

    garment = cv2.imread(
        str(garment_path),
        cv2.IMREAD_UNCHANGED,
    )

    if garment is None:
        raise ValueError(
            "Unable to read uploaded garment image."
        )

    # --------------------------------------------------------
    # Handle alpha channel if available.
    # --------------------------------------------------------

    if garment.ndim == 3 and garment.shape[2] == 4:

        bgr = garment[:, :, :3]
        alpha = garment[:, :, 3]

        mask = np.where(
            alpha > 10,
            255,
            0,
        ).astype(np.uint8)

        processed = garment

    else:

        if garment.ndim == 2:
            bgr = cv2.cvtColor(
                garment,
                cv2.COLOR_GRAY2BGR,
            )
        else:
            bgr = garment

        # ----------------------------------------------------
        # Estimate background using border colour.
        # ----------------------------------------------------

        border = np.concatenate(
            [
                bgr[0, :, :],
                bgr[-1, :, :],
                bgr[:, 0, :],
                bgr[:, -1, :],
            ],
            axis=0,
        )

        background = np.median(
            border,
            axis=0,
        )

        difference = np.linalg.norm(
            bgr.astype(np.float32)
            -
            background.astype(np.float32),
            axis=2,
        )

        mask = np.where(
            difference > 20,
            255,
            0,
        ).astype(np.uint8)

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        processed = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2BGRA,
        )

        processed[:, :, 3] = mask

    cv2.imwrite(
        str(garment_processed_path),
        processed,
    )

    cv2.imwrite(
        str(garment_mask_path),
        mask,
    )


# ============================================================
# TRY-ON ENDPOINT
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import time
import uuid


@app.post("/api/ai/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
):
    """
    Run Raritone virtual try-on.

    Inputs:
        person_image: Person JPG/PNG image
        garment_image: Garment JPG/PNG image
    """

    start_time = time.time()

    # ---------------------------------------------------------
    # Validate person image
    # ---------------------------------------------------------

    if not person_image.filename:
        raise HTTPException(
            status_code=400,
            detail="Person image is required.",
        )

    # ---------------------------------------------------------
    # Validate garment image
    # ---------------------------------------------------------

    if not garment_image.filename:
        raise HTTPException(
            status_code=400,
            detail="Garment image is required.",
        )

    # ---------------------------------------------------------
    # Validate extensions
    # ---------------------------------------------------------

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    person_ext = Path(
        person_image.filename
    ).suffix.lower()

    garment_ext = Path(
        garment_image.filename
    ).suffix.lower()

    if person_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid person image format. "
                "Use JPG, JPEG, or PNG."
            ),
        )

    if garment_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid garment image format. "
                "Use JPG, JPEG, or PNG."
            ),
        )

    # ---------------------------------------------------------
    # Create request directory
    # ---------------------------------------------------------

    request_id = str(uuid.uuid4())

    request_dir = (
        Path("outputs")
        / "requests"
        / request_id
    )

    request_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    person_path = (
        request_dir
        / f"person{person_ext}"
    )

    garment_path = (
        request_dir
        / f"garment{garment_ext}"
    )

    # ---------------------------------------------------------
    # Save uploaded person
    # ---------------------------------------------------------

    try:

        with open(
            person_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                person_image.file,
                buffer,
            )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to save person image: {exc}"
            ),
        )

    # ---------------------------------------------------------
    # Save uploaded garment
    # ---------------------------------------------------------

    try:

        with open(
            garment_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                garment_image.file,
                buffer,
            )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to save garment image: {exc}"
            ),
        )

    # ---------------------------------------------------------
    # TODO:
    # Connect the actual VTON pipeline here.
    #
    # For now this confirms that the API receives and
    # stores both images correctly.
    # ---------------------------------------------------------

    processing_time = (
        time.time() - start_time
    )

    return {
        "success": True,
        "request_id": request_id,
        "processing_time": round(
            processing_time,
            2,
        ),
        "person_image": str(
            person_path
        ),
        "garment_image": str(
            garment_path
        ),
        "message": (
            "Images uploaded successfully. "
            "Ready for VTON inference."
        ),
    }
        # ----------------------------------------------------
        # Garment processing
        # ----------------------------------------------------

        process_garment(
            garment_path,
            garment_processed_path,
            garment_mask_path,
        )

        # ----------------------------------------------------
        # Try-on generation
        # ----------------------------------------------------

        pipeline.generate(
            person_path=str(person_path),
            garment_path=str(garment_processed_path),
            person_mask_path=str(person_mask_path),
            garment_mask_path=str(garment_mask_path),
            output_path=str(output_path),
        )

        # ----------------------------------------------------
        # Output validation
        # ----------------------------------------------------

        if not output_path.exists():

            raise RuntimeError(
                "Try-on pipeline did not create an output image."
            )

        import cv2

        result_image = cv2.imread(
            str(output_path),
            cv2.IMREAD_COLOR,
        )

        if result_image is None:

            raise RuntimeError(
                "Generated try-on result is invalid."
            )

        height, width = result_image.shape[:2]

        if height <= 0 or width <= 0:

            raise RuntimeError(
                "Generated try-on result has invalid dimensions."
            )

        processing_time = (
            time.perf_counter()
            -
            start_time
        )

        return {
            "success": True,
            "processing_time": round(
                processing_time,
                3,
            ),
            "result": (
                f"/api/ai/tryon/result/{request_id}"
            ),
            "request_id": request_id,
            "width": width,
            "height": height,
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "Try-on processing failed.",
                "message": str(exc),
            },
        )


# ============================================================
# RESULT ENDPOINT
# ============================================================

@app.get(
    "/api/ai/tryon/result/{request_id}"
)
def get_result(
    request_id: str,
):
    """
    Return generated try-on image.
    """

    result_path = (
        OUTPUT_DIR
        / f"{request_id}_result.png"
    )

    if not result_path.exists():

        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "Try-on result not found.",
            },
        )

    return FileResponse(
        path=result_path,
        media_type="image/png",
        filename="raritone_tryon_result.png",
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Raritone VTON",
        "version": "1.0.0",
        "health": "/api/ai/health",
        "tryon": "/api/ai/tryon",
    }