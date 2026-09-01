from pathlib import Path

from app.tryon import TryOnPipeline


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PERSON = ROOT / "dataset" / "persons" / "person_001.jpg"
GARMENT = ROOT / "outputs" / "aligned_garment.png"
PERSON_MASK = ROOT / "dataset" / "masks" / "person_mask.png"
GARMENT_MASK = ROOT / "outputs" / "aligned_garment_mask.png"

OUTPUT = ROOT / "outputs" / "tryon_final.png"

def test_tryon_pipeline():

    print()
    print("=" * 60)
    print("Raritone VTON - TRY-ON TEST")
    print("=" * 60)

    pipeline = TryOnPipeline(
        feather_radius=2,
        garment_opacity=1.0,
    )

    result = pipeline.generate(
        person_path=str(PERSON),
        garment_path=str(GARMENT),
        person_mask_path=str(PERSON_MASK),
        garment_mask_path=str(GARMENT_MASK),
        output_path=str(OUTPUT),
    )

    assert result is not None

    assert result.ndim == 3

    assert result.shape[2] == 3

    assert result.shape[:2] == (
        695,
        441,
    )

    assert OUTPUT.exists()

    print()
    print("=" * 60)
    print("TRY-ON TEST PASSED")
    print("=" * 60)