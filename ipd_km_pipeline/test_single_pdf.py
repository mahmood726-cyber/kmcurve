#!/usr/bin/env python3
"""Test processing a single PDF directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from batch_processor import BatchProcessor

# Create processor
processor = BatchProcessor(
    input_dir="test_pdfs/medrxiv",
    output_dir="test_single",
    dpi=300,
    n_workers=1
)

# Get one PDF
pdfs = list(Path("test_pdfs/medrxiv").glob("*.pdf"))
print(f"Found {len(pdfs)} PDFs")

if pdfs:
    test_pdf = pdfs[0]
    print(f"\nTesting: {test_pdf}\n")
    print("="*70)

    # Process directly (no multiprocessing, no tqdm)
    result = processor.process_single_pdf(test_pdf)

    print("="*70)
    print(f"\nResult:")
    print(f"  Status: {result['status']}")
    print(f"  Figures: {len(result['figures'])}")
    print(f"  Errors: {len(result['errors'])}")

    if result['errors']:
        print(f"\nErrors:")
        for err in result['errors']:
            print(f"  - {err}")
