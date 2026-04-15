"""
Repo-relative path helpers for the maintained KM extraction workflow.

The repo historically relied on OneDrive-specific absolute paths. Centralizing
fixture and artifact discovery here keeps the pipeline portable across machines.
"""

from pathlib import Path
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
ARTIFACTS_ROOT = PACKAGE_ROOT / "artifacts"


def repo_path(*parts: str) -> Path:
    """Build a path relative to the repository root."""
    return REPO_ROOT.joinpath(*parts)


def package_path(*parts: str) -> Path:
    """Build a path relative to the structured Python package directory."""
    return PACKAGE_ROOT.joinpath(*parts)


def artifact_path(*parts: str) -> Path:
    """Build a path under the maintained artifact directory."""
    return ARTIFACTS_ROOT.joinpath(*parts)


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def first_existing(candidates: Iterable[Path], label: str) -> Path:
    """Return the first path that exists, or raise a clear error."""
    tried = []
    for candidate in candidates:
        tried.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {label}. Tried: {', '.join(tried)}")


def sample_pdf_path(filename: str = "NEJMoa0802987.pdf") -> Path:
    """Locate the bundled sample PDF used by package demos/tests."""
    return first_existing(
        [
            repo_path("papers_to_process", filename),
            repo_path(filename),
        ],
        label=f"sample PDF '{filename}'",
    )


def sample_render_path(
    filename: str | None = None,
) -> Path:
    """Locate the bundled rendered page fixture used by layout/curve tests.

    The rendered PNG filename embeds a SHA-256 prefix of the raster bytes,
    which varies with the installed PyMuPDF version. When no explicit filename
    is given, we glob for any ``page6_00_render_*.png`` produced by
    :func:`pdf_io.extract.save_extracted_images` and return a non-empty match.
    This keeps the helper portable across environments without re-committing
    binary fixtures.
    """
    render_dir = artifact_path("test_extraction", "page_6")

    if filename is not None:
        return first_existing(
            [
                render_dir / filename,
                repo_path(filename),
            ],
            label=f"sample render '{filename}'",
        )

    if render_dir.exists():
        candidates = sorted(render_dir.glob("page6_00_render_*.png"))
        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return candidate
            except OSError:
                continue

    raise FileNotFoundError(
        "Could not find sample render. "
        f"Expected a non-empty 'page6_00_render_*.png' under {render_dir}. "
        "Run tests/test_pipeline_smoke.py::test_sample_pdf_can_be_rendered_and_saved "
        "or rebuild the fixture via pdf_io.extract.save_extracted_images."
    )
