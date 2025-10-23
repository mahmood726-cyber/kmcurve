"""
PDF→Image extraction with fallback ladder.

Strategy (per playbook):
1. Try vector: PyMuPDF get_drawings() + get_text() for text bboxes
2. Try native image: get_images() → extract_image()
3. Fallback render: High-DPI render (600-1200 DPI) as PNG

Returns panel images with metadata.
"""
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import hashlib
import io


def extract_images_from_pdf(
    pdf_path: str,
    page_num: int = 0,
    dpi: int = 600,
    force_render: bool = False
) -> List[Dict]:
    """
    Extract images from PDF page with fallback ladder.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        dpi: DPI for rendering if needed
        force_render: Skip vector/native, go straight to render

    Returns:
        List of dicts with:
            - image: PIL Image object
            - method: 'vector', 'native', or 'render'
            - hash: stable hash of image data
            - bbox: bounding box if available
            - metadata: additional info
    """
    results = []

    doc = fitz.open(pdf_path)

    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} not found (PDF has {len(doc)} pages)")

    page = doc[page_num]

    if not force_render:
        # Method 1: Try vector extraction
        vector_result = _try_vector_extraction(page)
        if vector_result:
            results.append(vector_result)
            return results

        # Method 2: Try native image extraction
        native_results = _try_native_images(page, doc)
        if native_results:
            results.extend(native_results)
            return results

    # Method 3: Fallback to high-DPI render
    render_result = _render_page(page, dpi)
    if render_result:
        results.append(render_result)

    doc.close()
    return results


def _try_vector_extraction(page: fitz.Page) -> Optional[Dict]:
    """
    Try to extract vector graphics (drawings + text).

    Per playbook: Use get_drawings() for paths and get_text("rawdict") for text bboxes.

    Note: Previous R implementation found this problematic for rasterized PDFs.
    We'll try, but expect this to fail often.
    """
    try:
        # Get vector drawings
        drawings = page.get_drawings()

        # Get text with bboxes
        text_dict = page.get_text("rawdict")

        if not drawings or len(drawings) == 0:
            return None

        # This is a simplified check - in reality, we'd need to:
        # 1. Detect if drawings are actual K-M curves vs just axis lines
        # 2. Parse the drawing paths
        # 3. Handle transforms
        #
        # Since the playbook notes this often fails (PDFs are rasterized),
        # we return None here to trigger fallback

        # TODO: Implement proper vector curve detection
        return None

    except Exception:
        return None


def _try_native_images(page: fitz.Page, doc: fitz.Document) -> List[Dict]:
    """
    Try to extract native embedded images from PDF.

    Per playbook: Use page.get_images(full=True) → doc.extract_image(xref)
    """
    results = []

    try:
        image_list = page.get_images(full=True)

        if not image_list:
            return []

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # xref is first element

            try:
                # Extract image data
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Force RGB (avoid CMYK)
                if pil_image.mode == 'CMYK':
                    pil_image = pil_image.convert('RGB')
                elif pil_image.mode not in ['RGB', 'L']:  # L is grayscale
                    pil_image = pil_image.convert('RGB')

                # Get bbox if available
                bbox = page.get_image_bbox(img_info)

                # Create stable hash
                img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

                results.append({
                    'image': pil_image,
                    'method': 'native',
                    'hash': img_hash,
                    'bbox': bbox,
                    'metadata': {
                        'xref': xref,
                        'width': pil_image.width,
                        'height': pil_image.height,
                        'mode': pil_image.mode
                    }
                })

            except Exception as e:
                # Skip this image if extraction fails
                continue

        return results

    except Exception:
        return []


def _render_page(page: fitz.Page, dpi: int) -> Dict:
    """
    Render page to high-DPI PNG.

    Per playbook: Use 600-1200 DPI, force RGB/GRAY, no alpha.
    """
    # Calculate zoom factor (72 DPI is default)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # Render page
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # Convert to PIL Image
    img_data = pix.samples
    img_mode = "RGB" if pix.n == 3 else "L"  # RGB or grayscale
    pil_image = Image.frombytes(img_mode, (pix.width, pix.height), img_data)

    # Create stable hash
    img_bytes = pil_image.tobytes()
    img_hash = hashlib.sha256(img_bytes).hexdigest()[:16]

    return {
        'image': pil_image,
        'method': 'render',
        'hash': img_hash,
        'bbox': page.rect,  # Full page bbox
        'metadata': {
            'dpi': dpi,
            'width': pil_image.width,
            'height': pil_image.height,
            'mode': pil_image.mode,
            'zoom': zoom
        }
    }


def save_extracted_images(results: List[Dict], output_dir: str, prefix: str = "page"):
    """
    Save extracted images to disk.

    Args:
        results: List of extraction results from extract_images_from_pdf()
        output_dir: Directory to save images
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        img = result['image']
        method = result['method']
        hash_val = result['hash']

        filename = f"{prefix}_{i:02d}_{method}_{hash_val}.png"
        filepath = output_path / filename

        img.save(str(filepath), "PNG")
        print(f"Saved: {filepath}")
