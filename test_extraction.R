#!/usr/bin/env Rscript
# Quick test script
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
} else {
  normalizePath(getwd())
}

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(script_dir)

source("km_pdf_vector_extract_ultra.R")

# Use a repo-local temp directory
temp_dir <- file.path(tempdir(), "km_svgs")
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
  output_csv <- file.path(script_dir, "test_curves.csv")
  readr::write_csv(bound$curves, output_csv)
  cat("Curves saved to", output_csv, "\n")
}
