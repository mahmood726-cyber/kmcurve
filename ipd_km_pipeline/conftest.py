"""pytest config for the ipd_km_pipeline unit tests.

Several ``test_*.py`` files here are arg-driven CLI scripts, not unit tests:
they take a ``pdf_path`` (or a real sample PDF / extra module deps) and are run
manually, e.g. ``python test_single_pdf.py some.pdf``. pytest mis-collects them
as tests, producing fixture-not-found / ImportError collection errors that
otherwise mask the real unit-test results. Exclude them from automated
collection so ``python -m pytest`` here reports only the genuine unit tests.
"""

collect_ignore = [
    "test_single_pdf.py",        # BatchProcessor on one PDF (deep ocr.* import chain)
    "test_extraction.py",        # test_page(pdf_path, page_num, output_dir) -- CLI script
    "test_pdf_text_extraction.py",  # test_pdf_text_extraction(pdf_path, ...) -- CLI script
    "test_trocr_performance.py",    # test_trocr_on_pdf(pdf_path, ...) -- CLI script
]
