#!/usr/bin/env Rscript
# Quick test script
setwd("C:/Users/user/OneDrive - NHS/Documents/KMcurve")
source("km_pdf_vector_extract_ultra.R")

# Use C:/temp for SVG processing to avoid space issues
temp_dir <- "C:/temp/km_svgs"
dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)

pdf_file <- "papers_to_process/NEJMoa0802987.pdf"

cat("Extracting KM curves from:", pdf_file, "\n")
result <- extract_km_from_pdf(pdf_file, temp_svg_dir = temp_dir)

cat("\nExtraction complete!\n")
cat("Pages processed:", length(result), "\n")

# Save results
if (length(result) > 0) {
  bound <- km_extract_bind(result)
  cat("Curves found:", nrow(bound$curves), "\n")
  cat("Panels found:", nrow(bound$panels), "\n")

  # Save CSV
  readr::write_csv(bound$curves, "test_curves.csv")
  cat("Curves saved to test_curves.csv\n")
}
