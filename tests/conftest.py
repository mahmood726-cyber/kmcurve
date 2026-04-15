from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "ipd_km_pipeline"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _ensure_sample_render():
    """Regenerate the bundled page-6 render if it's missing or empty.

    The artifacts directory is gitignored and the rendered PNG filename embeds
    a SHA-256 of the raster bytes, so the exact filename varies per PyMuPDF
    build. To keep the smoke tests runnable on a fresh clone without shipping
    a binary fixture, we regenerate the render from the bundled sample PDF
    when no valid render is present.
    """
    try:
        from project_paths import (
            artifact_path,
            ensure_dir,
            sample_pdf_path,
            sample_render_path,
        )
    except Exception:
        return

    try:
        sample_render_path()
        return
    except FileNotFoundError:
        pass

    try:
        import fitz  # noqa: F401
    except Exception:
        return

    try:
        pdf = sample_pdf_path()
    except FileNotFoundError:
        return

    try:
        from pdf_io.extract import extract_images_from_pdf, save_extracted_images
    except Exception:
        return

    render_dir = ensure_dir(artifact_path("test_extraction", "page_6"))
    # Clean any zero-byte stragglers left by prior interrupted runs.
    for stale in render_dir.glob("page6_00_render_*.png"):
        try:
            if stale.stat().st_size == 0:
                stale.unlink()
        except OSError:
            pass

    results = extract_images_from_pdf(
        pdf_path=str(pdf),
        page_num=6,
        dpi=300,
        force_render=False,
    )
    if results:
        save_extracted_images(results, str(render_dir), prefix="page6")
