#!/usr/bin/env Rscript
# km_pdf_vector_extract_ultra.R
# Full-featured, vector-first KM extractor (single file)
# Version: 1.1.0

.VERSION <- "1.1.0"

# Package management with cleaner approach
check_and_load_package <- function(pkg, required = TRUE) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (required) {
      stop("Missing required package: ", pkg,
           "\nInstall with: install.packages(\"", pkg, "\")", call. = FALSE)
    }
    return(FALSE)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  return(TRUE)
}

# Load required packages
req_packages <- c("xml2", "dplyr", "stringr", "purrr", "tibble", "tidyr",
                  "readr", "pdftools", "ggplot2", "tools", "scales")
for (pkg in req_packages) check_and_load_package(pkg, required = TRUE)

# Load optional packages
has_mass <- check_and_load_package("MASS", required = FALSE)
has_progress <- check_and_load_package("progress", required = FALSE)
has_tesseract <- check_and_load_package("tesseract", required = FALSE)
has_future <- check_and_load_package("future", required = FALSE)
has_furrr <- check_and_load_package("furrr", required = FALSE)
has_magick <- check_and_load_package("magick", required = FALSE)
has_imager <- check_and_load_package("imager", required = FALSE)

# ------------------------ CONFIG & LOGGING ------------------------
CONFIG <- list(
  AXIS_MIN_LENGTH       = 50,
  PANEL_MARGIN          = 6,
  PANEL_HEIGHT_ESTIMATE = 150,
  NAR_SEARCH_BELOW      = 40,
  STEP_TOL              = 0.5,
  SVG_DPI               = 144,
  MUTOOL_TIMEOUT_SEC    = 300,
  CLEANUP_SVGS          = TRUE,
  CLEAN_INTERMEDIATE    = TRUE,  # New: clean memory between PDFs

  # Insets & panel selection
  INSET_AREA_FRAC       = 0.25,     # inset if <= 25% of largest and contained
  INSET_KEEP            = FALSE,
  PANEL_MODE            = "bestscore",  # "bestscore" | "largest" | "all"

  # Unmix overlapping curves
  UNMIX_EPS_Y           = 1.5,      # px tolerance per x-slice
  UNMIX_WINDOW_X        = 30,       # px window along x
  UNMIX_K_MAX           = 4,
  UNMIX_SMOOTH_LAM      = 2.0,

  # Bezier flattening
  BEZIER_MAX_SEG        = 30,       # max segs per command
  BEZIER_TOL            = 0.25,     # adaptive: smaller => more segments

  # Legend mapping
  LEGEND_SWATCH_MAXLEN  = 40,       # max length for small legend line
  LEGEND_NEAR_TEXT_D    = 25,       # px proximity legend swatch↔text

  # Censor ticks
  CENSOR_LEN_MAX        = 8,        # small line length (px)
  CENSOR_LEN_MIN        = 2,        # minimum to ignore dots
  CENSOR_DIST_TO_CURVE  = 4,        # max px distance to assign to curve
  CENSOR_ORTHO_ANGLE    = 25,       # degrees tolerance from orthogonal to step

  # Calibration parameters (previously hardcoded)
  CALIB_Y_QUANTILE_LOW  = 0.2,      # lower quantile for Y-axis text detection
  CALIB_Y_QUANTILE_HIGH = 0.8,      # upper quantile for Y-axis text detection
  CALIB_X_MARGIN_FRAC   = 0.25,     # fraction of panel width for Y-axis label detection
  CALIB_R2_THRESHOLD    = 0.95,     # R-squared threshold for calibration quality

  # Number-at-risk extraction
  NAR_CLUSTER_HEIGHT    = 3,        # height threshold for hierarchical clustering
  NAR_OCR_CONFIDENCE    = 60,       # minimum OCR confidence threshold
  NAR_OCR_DPI           = 300,      # DPI for OCR conversion

  # Raster curve extraction (fallback when no vector curves found)
  RASTER_DPI            = 300,      # DPI for raster extraction
  RASTER_THRESHOLD      = 0.3,      # darkness threshold (0-1, lower = darker pixels only)
  RASTER_MIN_POINTS     = 10,       # minimum points to consider as a curve
  RASTER_Y_CLUSTER_TOL  = 5         # y-pixel tolerance for clustering multiple curves
)

# Validate configuration
validate_config <- function() {
  required_numeric <- c("AXIS_MIN_LENGTH", "PANEL_MARGIN", "SVG_DPI", "MUTOOL_TIMEOUT_SEC")
  for (param in required_numeric) {
    if (!is.numeric(CONFIG[[param]]) || CONFIG[[param]] <= 0) {
      stop(sprintf("CONFIG$%s must be positive numeric", param), call. = FALSE)
    }
  }

  if (!CONFIG$PANEL_MODE %in% c("bestscore", "largest", "all")) {
    stop("CONFIG$PANEL_MODE must be 'bestscore', 'largest', or 'all'", call. = FALSE)
  }

  if (!is.logical(CONFIG$CLEANUP_SVGS) || !is.logical(CONFIG$INSET_KEEP)) {
    stop("CONFIG boolean flags must be TRUE or FALSE", call. = FALSE)
  }
}

.LOG_LEVEL <- toupper(Sys.getenv("KM_LOG_LEVEL", "INFO"))
.LOG_LEVELS <- c(ERROR=1, WARN=2, INFO=3, DEBUG=4)
.level <- function() ifelse(.LOG_LEVEL %in% names(.LOG_LEVELS), .LOG_LEVELS[.LOG_LEVEL], 3)
log_debug <- function(...) if (.level() >= 4) message("[DEBUG] ", sprintf(...))
log_info  <- function(...) if (.level() >= 3) message("[INFO]  ", sprintf(...))
log_warn  <- function(...) if (.level() >= 2) message("[WARN]  ", sprintf(...))
log_err   <- function(...) if (.level() >= 1) message("[ERROR] ", sprintf(...))

# ------------------------ UTILITIES ------------------------
.which_exists <- function(cmd) nzchar(Sys.which(cmd))
.require_cmd <- function(cmd, hint) {
  if (!.which_exists(cmd))
    stop(sprintf("%s not found. %s", cmd, hint), call. = FALSE)
}

sanitize_basename <- function(path) {
  gsub("[^A-Za-z0-9_-]", "_", tools::file_path_sans_ext(basename(path)))
}

.num <- function(x, default = NA_real_, warn = FALSE) {
  res <- suppressWarnings(as.numeric(x))
  if (warn && any(is.na(res) & !is.na(x)))
    log_warn("Numeric cast failed for: %s", paste0(x[is.na(res)], collapse=","))
  res
}

# Add missing string utility functions
str_trim <- function(x) {
  gsub("^\\s+|\\s+$", "", x)
}

str_squish <- function(x) {
  gsub("\\s+", " ", str_trim(x))
}

# Custom coalesce function (avoiding dplyr:: namespace collision)
coalesce_km <- function(...) {
  args <- list(...)
  for (arg in args) {
    if (!is.null(arg) && !is.na(arg) && nzchar(as.character(arg)))
      return(arg)
  }
  return(args[[length(args)]])
}

# -------------------- SVG TRANSFORM HANDLING -----------------
# Parse SVG transform attribute (supports matrix() for now)
parse_transform <- function(transform_str) {
  if (is.na(transform_str) || !nzchar(transform_str)) {
    return(list(type = "identity", a = 1, b = 0, c = 0, d = 1, e = 0, f = 0))
  }

  # Match matrix(a,b,c,d,e,f) or matrix(a b c d e f)
  matrix_match <- regexpr("matrix\\(([^)]+)\\)", transform_str, perl = TRUE)
  if (matrix_match > 0) {
    matrix_content <- regmatches(transform_str, matrix_match)
    # Extract numbers from inside parentheses
    nums_str <- sub("matrix\\(", "", sub("\\)", "", matrix_content))
    # Split by comma or space
    nums <- as.numeric(unlist(strsplit(nums_str, "[,\\s]+", perl = TRUE)))

    if (length(nums) == 6) {
      return(list(type = "matrix", a = nums[1], b = nums[2], c = nums[3],
                 d = nums[4], e = nums[5], f = nums[6]))
    }
  }

  # Default to identity if can't parse
  list(type = "identity", a = 1, b = 0, c = 0, d = 1, e = 0, f = 0)
}

# Apply transform matrix to a point (x, y)
apply_transform <- function(x, y, transform) {
  if (is.null(transform) || transform$type == "identity") {
    return(list(x = x, y = y))
  }

  # Matrix transform: [x'] = [a c e] [x]
  #                   [y']   [b d f] [y]
  #                              [1]
  # x' = a*x + c*y + e
  # y' = b*x + d*y + f
  list(
    x = transform$a * x + transform$c * y + transform$e,
    y = transform$b * x + transform$d * y + transform$f
  )
}

# Apply transform to a data frame of points
apply_transform_df <- function(df, transform) {
  if (is.null(df) || !nrow(df)) return(df)
  if (is.null(transform) || transform$type == "identity") return(df)

  if ("x" %in% names(df) && "y" %in% names(df)) {
    transformed <- mapply(apply_transform, df$x, df$y,
                          MoreArgs = list(transform = transform),
                          SIMPLIFY = FALSE)
    df$x <- sapply(transformed, `[[`, "x")
    df$y <- sapply(transformed, `[[`, "y")
  }

  if ("x1" %in% names(df) && "y1" %in% names(df)) {
    transformed1 <- mapply(apply_transform, df$x1, df$y1,
                           MoreArgs = list(transform = transform),
                           SIMPLIFY = FALSE)
    df$x1 <- sapply(transformed1, `[[`, "x")
    df$y1 <- sapply(transformed1, `[[`, "y")
  }

  if ("x2" %in% names(df) && "y2" %in% names(df)) {
    transformed2 <- mapply(apply_transform, df$x2, df$y2,
                           MoreArgs = list(transform = transform),
                           SIMPLIFY = FALSE)
    df$x2 <- sapply(transformed2, `[[`, "x")
    df$y2 <- sapply(transformed2, `[[`, "y")
  }

  if ("svg_x" %in% names(df) && "svg_y" %in% names(df)) {
    transformed_svg <- mapply(apply_transform, df$svg_x, df$svg_y,
                              MoreArgs = list(transform = transform),
                              SIMPLIFY = FALSE)
    df$svg_x <- sapply(transformed_svg, `[[`, "x")
    df$svg_y <- sapply(transformed_svg, `[[`, "y")
  }

  df
}

# -------------------- PDF -> SVG (vector) -----------------
pdf_to_svg <- function(pdf, pages = NULL, outdir = tempdir(), dpi = CONFIG$SVG_DPI) {
  .require_cmd("mutool",
               "Install MuPDF tools (winget install ArtifexSoftware.mutool | brew install mupdf-tools | apt-get install mupdf-tools).")

  if (!is.character(pdf) || length(pdf) != 1)
    stop("'pdf' must be a single file path", call. = FALSE)
  if (!file.exists(pdf))
    stop("PDF not found: ", pdf, call. = FALSE)
  if (!grepl("\\.pdf$", pdf, ignore.case = TRUE))
    stop("File must have .pdf extension: ", pdf, call. = FALSE)

  pdf <- normalizePath(pdf, mustWork = TRUE)
  outdir <- normalizePath(outdir, mustWork = FALSE)

  info <- pdftools::pdf_info(pdf)
  if (is.null(pages)) pages <- seq_len(info$pages)
  if (!is.numeric(pages) || any(pages < 1) || any(pages != floor(pages)))
    stop("'pages' must be positive integers", call. = FALSE)

  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  safe <- sanitize_basename(pdf)
  svg_files <- character(length(pages))

  for (i in seq_along(pages)) {
    p <- as.integer(pages[i])
    # Note: mutool automatically appends page number to output filename
    # So if we specify "file.svg" and page 1, it creates "file1.svg"
    out_base <- file.path(outdir, sprintf("%s_page", safe))
    out_expected <- sprintf("%s%d.svg", out_base, p)
    args <- c("draw", "-F", "svg", "-o", paste0(out_base, ".svg"), "-r", as.character(dpi), pdf, sprintf("%d", p))

    status <- tryCatch(
      suppressWarnings(system2("mutool", args, stdout = TRUE, stderr = TRUE,
                               timeout = CONFIG$MUTOOL_TIMEOUT_SEC)),
      error = function(e) {
        log_warn("system2 timeout not supported; running without timeout (page %d).", p)
        suppressWarnings(system2("mutool", args, stdout = TRUE, stderr = TRUE))
      }
    )

    if (!file.exists(out_expected))
      stop("Failed SVG conversion for page ", p, " (", basename(pdf), "):\n",
           paste(status, collapse="\n"), call. = FALSE)

    svg_files[i] <- out_expected
  }
  svg_files
}

# ----------------------- SVG PRIMITIVES --------------------
svg_text_df <- function(doc, pdf_file = NULL, page_num = NULL, dpi = CONFIG$SVG_DPI) {
  texts <- xml2::xml_find_all(doc, ".//*[local-name()='text'][not(ancestor::*[local-name()='defs'])]")
  if (length(texts)) {
    return(tibble::tibble(
      text = xml2::xml_text(texts),
      x = .num(xml2::xml_attr(texts, "x")),
      y = .num(xml2::xml_attr(texts, "y"))
    ) %>% dplyr::filter(!is.na(x), !is.na(y)))
  }

  if (!is.null(pdf_file) && !is.null(page_num)) {
    log_debug("SVG has no text; using PDF text for calibration (page %s)", page_num)

    # Get SVG dimensions for coordinate transformation
    svg_root <- xml2::xml_find_first(doc, "//*[local-name()='svg']")
    svg_width <- .num(xml2::xml_attr(svg_root, "width"))
    svg_height <- .num(xml2::xml_attr(svg_root, "height"))

    # Get PDF page dimensions (in points, 72 DPI)
    pdf_info <- tryCatch(pdftools::pdf_info(pdf_file), error = function(e) NULL)
    if (is.null(pdf_info)) {
      log_warn("Failed to get PDF info for coordinate scaling")
      return(tibble::tibble(text = character(), x = numeric(), y = numeric()))
    }

    # PDF dimensions are in points (1/72 inch)
    # Calculate scaling factor: SVG pixels / PDF points
    # For standard letter size PDF: 612 x 792 pts @ 72 DPI
    # SVG at 144 DPI would be: 1224 x 1584 pts
    # But actual dimensions may vary

    tb <- tryCatch(pdftools::pdf_data(pdf_file)[[page_num]] %>% tibble::as_tibble(),
                   error = function(e) NULL)
    if (!is.null(tb) && nrow(tb)) {
      # pdf_data returns coordinates in PDF points (72 DPI)
      # We need to scale to SVG coordinate space
      # The scaling factor is (SVG DPI / 72)
      scale_factor <- dpi / 72

      return(tb %>% dplyr::transmute(
        text = .data$text,
        x = .data$x * scale_factor,
        y = .data$y * scale_factor
      ))
    }
  }
  tibble::tibble(text = character(), x = numeric(), y = numeric())
}

svg_line_df <- function(doc) {
  ln <- xml2::xml_find_all(doc, ".//*[local-name()='line'][not(ancestor::*[local-name()='defs'])]")
  if (!length(ln)) return(tibble())

  df <- tibble(x1 = .num(xml2::xml_attr(ln, "x1")),
               y1 = .num(xml2::xml_attr(ln, "y1")),
               x2 = .num(xml2::xml_attr(ln, "x2")),
               y2 = .num(xml2::xml_attr(ln, "y2")),
               stroke = xml2::xml_attr(ln, "stroke"),
               sw = .num(xml2::xml_attr(ln, "stroke-width")),
               dash = xml2::xml_attr(ln, "stroke-dasharray"),
               transform = xml2::xml_attr(ln, "transform"))

  # Apply transforms to each row
  for (i in seq_len(nrow(df))) {
    trans <- parse_transform(df$transform[i])
    df[i, ] <- apply_transform_df(df[i, ], trans)
  }

  df %>% dplyr::select(-transform)
}

svg_polyline_df <- function(doc) {
  pl <- xml2::xml_find_all(doc, ".//*[local-name()='polyline'][not(ancestor::*[local-name()='defs'])] | .//*[local-name()='polygon'][not(ancestor::*[local-name()='defs'])]")
  if (!length(pl)) return(tibble())

  tibble(points = xml2::xml_attr(pl, "points"),
         stroke = xml2::xml_attr(pl, "stroke"),
         fill = xml2::xml_attr(pl, "fill"),
         sw = .num(xml2::xml_attr(pl, "stroke-width")),
         dash = xml2::xml_attr(pl, "stroke-dasharray")) %>%
    dplyr::filter(!is.na(.data$points),
                  !is.na(.data$stroke),
                  is.na(.data$fill) | .data$fill %in% c("none","#00000000","transparent"))
}

svg_path_df <- function(doc) {
  p <- xml2::xml_find_all(doc, ".//*[local-name()='path'][not(ancestor::*[local-name()='defs'])]")
  if (!length(p)) return(tibble())

  tibble(d = xml2::xml_attr(p, "d"),
         stroke = xml2::xml_attr(p, "stroke"),
         fill = xml2::xml_attr(p, "fill"),
         sw = .num(xml2::xml_attr(p, "stroke-width")),
         dash = xml2::xml_attr(p, "stroke-dasharray"),
         transform = xml2::xml_attr(p, "transform")) %>%
    dplyr::filter(!is.na(.data$stroke),
                  is.na(.data$fill) | .data$fill %in% c("none","#00000000","transparent"))
}

# -------- Bezier helpers (flatten C/Q to polyline points) ----------
bez_q <- function(t, p0, p1, p2) (1-t)^2*p0 + 2*(1-t)*t*p1 + t^2*p2
bez_c <- function(t, p0, p1, p2, p3) (1-t)^3*p0 + 3*(1-t)^2*t*p1 + 3*(1-t)*t^2*p2 + t^3*p3

seg_count <- function(dx, dy) {
  L <- sqrt(dx*dx + dy*dy)
  n <- ceiling(max(3, min(CONFIG$BEZIER_MAX_SEG, L / CONFIG$BEZIER_TOL)))
  as.integer(n)
}

path_to_points <- function(d) {
  if (is.na(d) || !nzchar(d)) return(NULL)

  # Split command letters from numbers: "H298" -> "H 298"
  # Insert space after command letter if followed by digit/sign/decimal
  s <- gsub(",", " ", d)
  s <- gsub("([MLHVCSQZmlhvcsqz])([0-9.+-])", "\\1 \\2", s, perl = TRUE)
  toks <- strsplit(str_trim(s), "\\s+", perl = TRUE)[[1]]
  i <- 1; x <- 0; y <- 0; sx <- NA; sy <- NA; cmd <- NULL
  pts <- list(); n <- 0L

  take_num <- function() {
    v <- suppressWarnings(as.numeric(toks[i]));
    i <<- i + 1;
    v
  }
  add_pt <- function(xx, yy) {
    n <<- n + 1L;
    pts[[n]] <<- c(xx, yy)
  }

  while (i <= length(toks)) {
    tk <- toks[i]; i <- i + 1

    if (grepl("^[MLHVCSQmlhvcsqZz]$", tk)) {
      cmd <- tk
      if (cmd %in% c("Z","z")) {
        if (!is.na(sx)) add_pt(sx, sy)
        next
      }
    }

    if (cmd %in% c("M","L")) {
      i <- i - 1
      repeat {
        if (i + 1 > length(toks)) break
        x <- take_num(); y <- take_num()
        if (is.na(x) || is.na(y)) break
        if (cmd == "M") { sx <- x; sy <- y; cmd <- "L" }
        add_pt(x, y)
      }

    } else if (cmd %in% c("m","l")) {
      i <- i - 1
      repeat {
        if (i + 1 > length(toks)) break
        dx <- take_num(); dy <- take_num()
        if (is.na(dx) || is.na(dy)) break
        x <- x + dx; y <- y + dy
        if (cmd == "m") { sx <- x; sy <- y; cmd <- "l" }
        add_pt(x, y)
      }

    } else if (cmd %in% c("H","h")) {
      if (cmd == "H") { x <- take_num() } else { x <- x + take_num() }
      add_pt(x, y)

    } else if (cmd %in% c("V","v")) {
      if (cmd == "V") { y <- take_num() } else { y <- y + take_num() }
      add_pt(x, y)

    } else if (cmd %in% c("Q","q")) {
      if (cmd == "Q") {
        x1 <- take_num(); y1 <- take_num(); x2 <- take_num(); y2 <- take_num()
      } else {
        x1 <- x + take_num(); y1 <- y + take_num()
        x2 <- x + take_num(); y2 <- y + take_num()
      }
      nseg <- seg_count(x2 - x, y2 - y)
      for (t in seq(0, 1, length.out = nseg)) {
        xx <- bez_q(t, x, x1, x2); yy <- bez_q(t, y, y1, y2)
        add_pt(xx, yy)
      }
      x <- x2; y <- y2

    } else if (cmd %in% c("C","c")) {
      if (cmd == "C") {
        x1 <- take_num(); y1 <- take_num()
        x2 <- take_num(); y2 <- take_num()
        x3 <- take_num(); y3 <- take_num()
      } else {
        x1 <- x + take_num(); y1 <- y + take_num()
        x2 <- x + take_num(); y2 <- y + take_num()
        x3 <- x + take_num(); y3 <- y + take_num()
      }
      nseg <- seg_count(x3 - x, y3 - y)
      for (t in seq(0, 1, length.out = nseg)) {
        xx <- bez_c(t, x, x1, x2, x3); yy <- bez_c(t, y, y1, y2, y3)
        add_pt(xx, yy)
      }
      x <- x3; y <- y3
    }
  }

  if (!length(pts)) return(NULL)
  mat <- do.call(rbind, pts)
  tibble(svg_x = mat[,1], svg_y = mat[,2])
}

poly_points_df <- function(points_str) {
  chunks <- strsplit(str_squish(points_str), "\\s+", perl = TRUE)[[1]]
  xy <- strsplit(chunks, ",", fixed = TRUE)
  mat <- do.call(rbind, lapply(xy, function(p) c(as.numeric(p[1]), as.numeric(p[2]))))
  tibble(svg_x = mat[,1], svg_y = mat[,2])
}

step_regularize <- function(df, tol = CONFIG$STEP_TOL) {
  if (is.null(df) || !is.data.frame(df)) {
    log_warn("step_regularize: non-data.frame")
    return(tibble())
  }
  if (nrow(df) < 2) return(df)
  if (!all(c("svg_x","svg_y") %in% names(df)))
    stop("step_regularize: missing svg_x/svg_y", call. = FALSE)

  out <- df[1, , drop = FALSE]
  for (i in 2:nrow(df)) {
    prev <- out[nrow(out),]; cur <- df[i,]
    dx <- abs(cur$svg_x - prev$svg_x); dy <- abs(cur$svg_y - prev$svg_y)
    if (dx < tol && dy < tol) next
    if (dx < dy) cur$svg_x <- prev$svg_x else if (dy < dx) cur$svg_y <- prev$svg_y
    out <- dplyr::bind_rows(out, cur)
  }
  out
}

# ------------------ PANELS & CALIBRATION -------------------
# Extract straight horizontal/vertical paths that could be axes
extract_axis_paths <- function(doc, len_min = CONFIG$AXIS_MIN_LENGTH) {
  paths <- svg_path_df(doc)
  if (!nrow(paths)) return(tibble())

  # Parse simple paths that are straight lines (M x1 y1 L x2 y2 or M x1 y1 H x2 or M x1 y1 V y2)
  parse_simple_line <- function(d) {
    if (is.na(d) || !nzchar(d)) return(NULL)

    # Clean up path string - insert spaces before command letters
    s <- gsub(",", " ", d)
    s <- gsub("([MLHVCSQZmlhvcsqz])", " \\1 ", s, perl = TRUE)  # Add spaces around commands
    s <- str_trim(s)
    s <- gsub("\\s+", " ", s)  # Collapse multiple spaces
    toks <- strsplit(s, "\\s+", perl = TRUE)[[1]]

    # Only parse very simple paths: M x y L x y or M x y H x or M x y V y
    if (length(toks) < 4) return(NULL)

    # Must start with M (moveto)
    if (toks[1] != "M") return(NULL)

    x1 <- suppressWarnings(as.numeric(toks[2]))
    y1 <- suppressWarnings(as.numeric(toks[3]))

    if (is.na(x1) || is.na(y1)) return(NULL)

    # Check for L (lineto), H (horizontal), or V (vertical) command
    if (length(toks) >= 6 && toks[4] == "L") {
      x2 <- suppressWarnings(as.numeric(toks[5]))
      y2 <- suppressWarnings(as.numeric(toks[6]))
      if (!is.na(x2) && !is.na(y2)) {
        return(list(x1=x1, y1=y1, x2=x2, y2=y2))
      }
    } else if (length(toks) >= 5 && toks[4] == "H") {
      x2 <- suppressWarnings(as.numeric(toks[5]))
      if (!is.na(x2)) {
        return(list(x1=x1, y1=y1, x2=x2, y2=y1))
      }
    } else if (length(toks) >= 5 && toks[4] == "V") {
      y2 <- suppressWarnings(as.numeric(toks[5]))
      if (!is.na(y2)) {
        return(list(x1=x1, y1=y1, x2=x1, y2=y2))
      }
    }

    NULL
  }

  # Extract coordinates from simple paths
  path_lines <- purrr::map(paths$d, parse_simple_line)
  path_lines <- purrr::compact(path_lines)

  if (!length(path_lines)) return(tibble())

  # Convert to tibble format matching svg_line_df
  coords <- purrr::map_dfr(path_lines, ~ tibble(
    x1 = .x$x1, y1 = .x$y1, x2 = .x$x2, y2 = .x$y2
  ))

  # Add stroke info and transform from original paths
  coords <- dplyr::bind_cols(
    coords,
    paths[seq_len(nrow(coords)), c("stroke", "sw", "dash", "transform")]
  )

  # Apply transforms to coordinates
  if ("transform" %in% names(coords)) {
    for (i in seq_len(nrow(coords))) {
      trans <- parse_transform(coords$transform[i])
      coords[i, ] <- apply_transform_df(coords[i, ], trans)
    }
    coords <- coords %>% dplyr::select(-transform)
  }

  # Calculate length and filter
  coords <- coords %>%
    dplyr::mutate(len = sqrt((x2-x1)^2 + (y2-y1)^2)) %>%
    dplyr::filter(len >= len_min)

  # Only keep horizontal or vertical paths
  coords %>%
    dplyr::filter(abs(y2 - y1) < 1 | abs(x2 - x1) < 1) %>%
    dplyr::select(-len)
}

find_axis_lines <- function(lines, len_min = CONFIG$AXIS_MIN_LENGTH) {
  if (!nrow(lines)) return(list(h=tibble(), v=tibble()))

  lines <- lines %>% dplyr::mutate(len = sqrt((x2-x1)^2 + (y2-y1)^2))
  long  <- lines %>% dplyr::filter(len >= len_min)

  list(
    h = long %>% dplyr::filter(abs(y2 - y1) < 1) %>% dplyr::arrange(dplyr::desc(len)),
    v = long %>% dplyr::filter(abs(x2 - x1) < 1) %>% dplyr::arrange(dplyr::desc(len))
  )
}

# Detect X-axis labels: look for horizontally aligned, monotonically increasing sequences
detect_xaxis_labels <- function(texts, h_y, h_xmin, h_xmax) {
  if (is.null(texts) || !nrow(texts)) return(NULL)

  # Find numeric texts in the search region around horizontal axis
  # X-axis labels can be either above or below the bottom axis border
  candidates <- texts %>%
    dplyr::mutate(val = suppressWarnings(as.numeric(text))) %>%
    dplyr::filter(!is.na(val),
                  y >= h_y - 100,    # Search above the axis
                  y <= h_y + 20,     # Allow margin below
                  x >= h_xmin - 50,  # Allow margin left
                  x <= h_xmax + 500) # Look far beyond h_xmax

  if (nrow(candidates) < 3) return(NULL)

  # Group by y-coordinate (texts at similar heights are likely on same axis)
  # Use hierarchical clustering to find y-bands
  y_dist <- stats::dist(candidates$y)
  if (length(y_dist) == 0) return(NULL)

  y_clusters <- stats::cutree(stats::hclust(y_dist), h = 10)  # 10px tolerance
  candidates$y_group <- y_clusters

  # For each y-group, check if it's a valid X-axis label sequence
  # Suppress warnings from min/max when no valid groups exist (expected behavior)
  valid_groups <- suppressWarnings(
    candidates %>%
      dplyr::group_by(y_group) %>%
      dplyr::filter(dplyr::n() >= 3) %>%  # Need at least 3 labels
      dplyr::arrange(x, .by_group = TRUE) %>%
      dplyr::summarise(
        n = dplyr::n(),
        x_min = min(x),
        x_max = max(x),
        val_min = min(val),
        val_max = max(val),
        # Check if monotonically increasing
        is_increasing = all(diff(val) > 0),
        # Check if reasonable X-axis range (typically 0-120 for KM plots in months)
        is_reasonable = val_min >= 0 && val_max <= 150,
        # Check if x positions are increasing with values
        x_val_corr = stats::cor(x, val, method = "spearman"),
        .groups = "drop"
      ) %>%
      dplyr::filter(is_increasing,
                    is_reasonable,
                    x_val_corr > 0.9)  # Strong positive correlation
  )

  if (nrow(valid_groups) == 0) return(NULL)

  # Return the group with the most labels and rightmost extent
  best_group <- valid_groups %>%
    dplyr::arrange(dplyr::desc(n), dplyr::desc(x_max)) %>%
    dplyr::slice(1)

  # Get the actual labels from this group
  x_labels <- candidates %>%
    dplyr::filter(y_group == best_group$y_group[[1]]) %>%
    dplyr::arrange(x)

  list(
    labels = x_labels,
    x_extent = as.numeric(best_group$x_max[[1]])
  )
}

build_panels <- function(axes, margin = CONFIG$PANEL_MARGIN, texts = NULL) {
  if (!nrow(axes$h) || !nrow(axes$v)) return(tibble())

  # Only use the longest horizontal axes (likely the actual X-axes)
  # Take top 6 to ensure we capture all axes with maximum length (academic journals
  # often have multiple axes of same length as decorative borders)
  h_axes <- axes$h %>%
    dplyr::arrange(dplyr::desc(len)) %>%
    dplyr::slice(1:min(6, dplyr::n()))

  # Only use vertical axes that span a reasonable height
  # Filter for axes that are at least 30px tall
  v_axes <- axes$v %>%
    dplyr::filter(len >= 30) %>%
    dplyr::arrange(dplyr::desc(len))

  if (!nrow(h_axes) || !nrow(v_axes)) return(tibble())

  # Get x positions of vertical axes (they're vertical so x1 ≈ x2)
  v_x <- v_axes$x1

  # For academic journal layouts, the horizontal axis lines are often just
  # decorative borders and don't extend to the full plotting region.
  # The actual curves may extend much further - we need to find X-axis labels
  # to determine the true extent.

  rects_list <- list()
  rect_count <- 0

  for (h in seq_len(nrow(h_axes))) {
    hl <- h_axes[h,]
    h_y <- mean(c(hl$y1, hl$y2))
    h_xmin <- min(hl$x1, hl$x2)
    h_xmax <- max(hl$x1, hl$x2)

    # Find X-axis labels to determine true panel extent
    x_extent <- h_xmax  # Default fallback
    x_label_result <- detect_xaxis_labels(texts, h_y, h_xmin, h_xmax)

    if (!is.null(x_label_result)) {
      x_extent <- x_label_result$x_extent + 50  # Add margin
      log_debug("Panel h=%d: X-labels extend to %.1f (axis ends at %.1f)",
                h, x_label_result$x_extent, h_xmax)
    }

    # Method 1: Try pairing vertical axes (standard KM plots)
    if (nrow(v_axes) >= 2) {
      for (i in 1:(nrow(v_axes)-1)) {
        for (j in (i+1):nrow(v_axes)) {
          x_left <- min(v_x[i], v_x[j])
          x_right <- max(v_x[i], v_x[j])

          # Extend right edge to X-label extent if available
          if (x_extent > x_right) {
            x_right <- x_extent
          }

          width <- x_right - x_left

          # Only keep panels that are wide enough (at least 50px)
          if (width >= 50) {
            rect_count <- rect_count + 1
            rects_list[[rect_count]] <- tibble(
              xmin = x_left - margin,
              xmax = x_right + margin,
              ymin = h_y - CONFIG$PANEL_HEIGHT_ESTIMATE,
              ymax = h_y + 5
            )
          }
        }
      }
    }

    # Method 2: Single vertical axis case
    if (nrow(v_axes) >= 1) {
      x_left <- min(v_x)
      x_right <- x_extent

      width <- x_right - x_left
      if (width >= 100) {
        rect_count <- rect_count + 1
        rects_list[[rect_count]] <- tibble(
          xmin = x_left - margin,
          xmax = x_right + margin,
          ymin = h_y - CONFIG$PANEL_HEIGHT_ESTIMATE,
          ymax = h_y + 5
        )
      }
    }
  }

  if (rect_count == 0) return(tibble())

  rects <- dplyr::bind_rows(rects_list) %>%
    dplyr::distinct() %>%
    dplyr::mutate(panel_id = dplyr::row_number())

  rects
}

calibrate_panel <- function(panel_bbox, texts) {
  tx <- texts %>% dplyr::filter(
    x >= panel_bbox$xmin - 10, x <= panel_bbox$xmax + 10,
    y >= panel_bbox$ymin - 40, y <= panel_bbox$ymax + 80
  )

  tx_num <- tx %>%
    dplyr::mutate(val = suppressWarnings(readr::parse_number(text))) %>%
    dplyr::filter(!is.na(val))

  if (nrow(tx_num) < 3) return(NULL)

  y_quant <- stats::quantile(tx_num$y, probs = c(CONFIG$CALIB_Y_QUANTILE_LOW, CONFIG$CALIB_Y_QUANTILE_HIGH), na.rm = TRUE)
  x_ticks <- tx_num %>%
    dplyr::filter(y >= y_quant[2]) %>%
    dplyr::arrange(x)
  y_ticks <- tx_num %>%
    dplyr::filter(x <= (panel_bbox$xmin + CONFIG$CALIB_X_MARGIN_FRAC*(panel_bbox$xmax - panel_bbox$xmin))) %>%
    dplyr::arrange(dplyr::desc(y))

  if (nrow(x_ticks) < 2 || nrow(y_ticks) < 2) return(NULL)

  fit_fun <- if (has_mass) MASS::rlm else lm
  fit_x <- fit_fun(val ~ x, data = x_ticks)
  fit_y <- fit_fun(val ~ y, data = y_ticks)

  r2 <- function(f) if (inherits(f, "rlm")) NA_real_ else summary(f)$r.squared
  quality <- list(x_r2 = r2(fit_x), y_r2 = r2(fit_y),
                  x_ticks_n = nrow(x_ticks), y_ticks_n = nrow(y_ticks))

  if ((!is.na(quality$x_r2) && quality$x_r2 < CONFIG$CALIB_R2_THRESHOLD) || (!is.na(quality$y_r2) && quality$y_r2 < CONFIG$CALIB_R2_THRESHOLD)) {
    log_warn("Weak calibration fit: x_r2=%.3f, y_r2=%.3f", quality$x_r2, quality$y_r2)
  }

  list(
    x_map = function(x) as.numeric(coef(fit_x)[1] + coef(fit_x)[2]*x),
    y_map = function(y) as.numeric(coef(fit_y)[1] + coef(fit_y)[2]*y),
    ticks = list(x = x_ticks, y = y_ticks),
    quality = quality
  )
}

# --------------------- STYLE KEYS & LEGEND -----------------
dash_key <- function(stroke, dash, sw) {
  sig <- ifelse(is.na(dash) || dash == "" || dash == "none", "solid",
                gsub("[^0-9.]+", "-", dash))
  paste0(tolower(coalesce_km(stroke, "none")), "|", sig, "|",
         ifelse(is.na(sw), "NA", format(sw, trim = TRUE)))
}

find_legend_map <- function(doc, panels) {
  texts <- svg_text_df(doc)
  lines <- svg_line_df(doc)
  polys <- svg_polyline_df(doc)

  cand_lines <- tibble(
    x = (lines$x1 + lines$x2)/2,
    y = (lines$y1 + lines$y2)/2,
    len = sqrt((lines$x2 - lines$x1)^2 + (lines$y2 - lines$y1)^2),
    stroke = lines$stroke, dash = lines$dash, sw = lines$sw
  ) %>% dplyr::filter(len > 0, len <= CONFIG$LEGEND_SWATCH_MAXLEN)

  if (nrow(panels)) {
    inside <- purrr::pmap_lgl(list(cand_lines$x, cand_lines$y),
                              function(xx, yy) any(xx >= panels$xmin & xx <= panels$xmax &
                                                     yy >= panels$ymin & yy <= panels$ymax))
    cand_lines <- cand_lines[!inside, , drop = FALSE]
  }

  if (!nrow(cand_lines) || !nrow(texts))
    return(tibble(style_key = character(), label = character()))

  cand_lines %>%
    dplyr::mutate(style_key = dash_key(stroke, dash, sw)) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(label = {
      d <- sqrt((texts$x - x)^2 + (texts$y - y)^2)
      j <- which.min(d)
      if (length(j) && d[j] <= CONFIG$LEGEND_NEAR_TEXT_D) texts$text[j] else NA_character_
    }) %>% dplyr::ungroup() %>%
    dplyr::filter(!is.na(label)) %>%
    dplyr::group_by(style_key) %>%
    dplyr::summarise(label = label[which.min(nchar(label))], .groups = "drop")
}

# --------------------- CURVES & CENSOR TICKS ----------------
svg_path_poly_df <- function(doc) {
  safe_pts <- purrr::possibly(function(s) {
    if (is.na(s) || !nzchar(s)) return(NULL)
    path_to_points(s)
  }, otherwise = NULL, quiet = TRUE)

  paths <- svg_path_df(doc) %>%
    dplyr::mutate(pts = purrr::map(.data$d, safe_pts)) %>%
    dplyr::filter(purrr::map_int(.data$pts, ~ if (is.null(.x)) 0L else nrow(.x)) >= 2)

  # Apply transforms to path points
  if (nrow(paths) > 0 && "transform" %in% names(paths)) {
    for (i in seq_len(nrow(paths))) {
      trans <- parse_transform(paths$transform[i])
      if (!is.null(paths$pts[[i]])) {
        paths$pts[[i]] <- apply_transform_df(paths$pts[[i]], trans)
      }
    }
  }

  paths <- paths %>% dplyr::transmute(stroke, fill, sw, dash, pts)

  polyl_raw <- svg_polyline_df(doc)
  if (nrow(polyl_raw) > 0) {
    polyl <- polyl_raw %>%
      dplyr::mutate(pts = purrr::map(.data$points, poly_points_df)) %>%
      dplyr::filter(purrr::map_int(.data$pts, nrow) >= 2) %>%
      dplyr::transmute(stroke, fill, sw, dash, pts)
  } else {
    polyl <- tibble::tibble()
  }

  dplyr::bind_rows(paths, polyl)
}

harvest_curves <- function(doc, panel_bbox) {
  all <- svg_path_poly_df(doc)
  if (!nrow(all)) return(tibble())

  clip_ok <- function(df) df %>%
    dplyr::filter(svg_x >= panel_bbox$xmin, svg_x <= panel_bbox$xmax,
                  svg_y >= panel_bbox$ymin, svg_y <= panel_bbox$ymax)

  curves <- all %>%
    dplyr::mutate(pts = purrr::map(pts, clip_ok),
                  pts = purrr::map(pts, step_regularize)) %>%
    dplyr::filter(purrr::map_int(pts, nrow) >= 2)

  if (!nrow(curves)) return(tibble())

  curves %>%
    dplyr::mutate(style_key = dash_key(stroke, dash, sw),
                  data = purrr::map(pts, ~ .x %>% dplyr::mutate(idx = dplyr::row_number()))) %>%
    dplyr::select(style_key, stroke, dash, sw, data) %>%
    tidyr::unnest(data) %>%
    dplyr::group_by(style_key) %>%
    dplyr::mutate(curve_id = dplyr::cur_group_id()) %>%
    dplyr::ungroup()
}

# Raster-based curve extraction (fallback for non-vector PDFs)
harvest_curves_from_raster <- function(pdf_path, page_num, panel_bbox,
                                       dpi = CONFIG$RASTER_DPI,
                                       threshold = CONFIG$RASTER_THRESHOLD,
                                       min_points = CONFIG$RASTER_MIN_POINTS,
                                       y_cluster_tol = CONFIG$RASTER_Y_CLUSTER_TOL) {

  if (!has_magick) {
    log_warn("Raster extraction requires magick package - skipping")
    return(tibble())
  }

  log_debug("Attempting raster-based curve extraction (page %d)", page_num)

  # Step 1: Convert PDF page to high-res PNG
  png_file <- tempfile(fileext = ".png")
  on.exit(unlink(png_file), add = TRUE)

  tryCatch({
    pdftools::pdf_convert(pdf_path, format = "png", pages = page_num,
                         filenames = png_file, dpi = dpi, verbose = FALSE)
  }, error = function(e) {
    log_warn("Failed to convert PDF to PNG: %s", e$message)
    return(tibble())
  })

  if (!file.exists(png_file)) {
    log_warn("PNG conversion failed - file not created")
    return(tibble())
  }

  # Step 2: Load image and crop to panel boundaries
  img <- tryCatch({
    magick::image_read(png_file)
  }, error = function(e) {
    log_warn("Failed to load PNG: %s", e$message)
    return(NULL)
  })

  if (is.null(img)) return(tibble())

  # Calculate scaling factor from SVG space to raster space
  # SVG uses CONFIG$SVG_DPI (typically 144), raster uses RASTER_DPI (typically 300)
  scale_factor <- dpi / CONFIG$SVG_DPI

  # Convert panel bbox to raster coordinates
  crop_x <- max(0, floor(panel_bbox$xmin * scale_factor))
  crop_y <- max(0, floor(panel_bbox$ymin * scale_factor))
  crop_w <- ceiling((panel_bbox$xmax - panel_bbox$xmin) * scale_factor)
  crop_h <- ceiling((panel_bbox$ymax - panel_bbox$ymin) * scale_factor)

  # Crop to panel region
  img_info <- magick::image_info(img)
  crop_x <- min(crop_x, img_info$width - 1)
  crop_y <- min(crop_y, img_info$height - 1)
  crop_w <- min(crop_w, img_info$width - crop_x)
  crop_h <- min(crop_h, img_info$height - crop_y)

  img_cropped <- tryCatch({
    magick::image_crop(img, sprintf("%dx%d+%d+%d", crop_w, crop_h, crop_x, crop_y))
  }, error = function(e) {
    log_warn("Failed to crop image: %s", e$message)
    return(NULL)
  })

  if (is.null(img_cropped)) return(tibble())

  # Step 3: Convert to grayscale and get pixel matrix
  img_gray <- magick::image_convert(img_cropped, colorspace = "gray")
  img_matrix <- tryCatch({
    # Get pixel values as array (normalized 0-1)
    as.integer(magick::image_data(img_gray, channels = "gray"))
  }, error = function(e) {
    log_warn("Failed to extract pixel data: %s", e$message)
    return(NULL)
  })

  if (is.null(img_matrix)) return(tibble())

  # Reshape to 2D matrix (height x width)
  img_dim <- dim(img_matrix)
  if (length(img_dim) == 3) {
    # Format: [channel, width, height] - need to transpose
    img_matrix <- img_matrix[1,,]
    img_matrix <- t(img_matrix)  # Now [height, width]
  }

  # Normalize to 0-1 range (255 = white, 0 = black)
  img_matrix <- as.numeric(img_matrix) / 255

  # Step 4: Threshold to find dark pixels (curves)
  # Lower threshold = darker pixels only
  dark_pixels <- which(img_matrix < threshold, arr.ind = TRUE)

  if (nrow(dark_pixels) < min_points) {
    log_debug("Insufficient dark pixels found (%d < %d)", nrow(dark_pixels), min_points)
    return(tibble())
  }

  # Convert to data frame with raster coordinates
  pixels_df <- tibble(
    raster_x = dark_pixels[, 2],  # column index
    raster_y = dark_pixels[, 1]   # row index
  )

  # Step 5: Column-by-column extraction
  # For each x position, find all y values (potential multiple curves)
  curves_list <- pixels_df %>%
    dplyr::group_by(raster_x) %>%
    dplyr::summarise(y_vals = list(raster_y), .groups = "drop")

  # Step 6: Cluster y-values to separate multiple curves
  # For each x, cluster the y-values by proximity
  all_points <- list()

  for (i in seq_len(nrow(curves_list))) {
    x_pos <- curves_list$raster_x[i]
    y_vals <- curves_list$y_vals[[i]]

    if (length(y_vals) == 0) next

    # Simple clustering: group y-values within tolerance
    y_sorted <- sort(y_vals)
    clusters <- rep(1, length(y_sorted))
    current_cluster <- 1

    for (j in 2:length(y_sorted)) {
      if (y_sorted[j] - y_sorted[j-1] > y_cluster_tol) {
        current_cluster <- current_cluster + 1
      }
      clusters[j] <- current_cluster
    }

    # Take mean y for each cluster at this x position
    for (clust_id in unique(clusters)) {
      y_mean <- mean(y_sorted[clusters == clust_id])
      all_points[[length(all_points) + 1]] <- tibble(
        raster_x = x_pos,
        raster_y = y_mean,
        curve_id = clust_id
      )
    }
  }

  if (length(all_points) == 0) {
    log_debug("No curve points extracted after clustering")
    return(tibble())
  }

  # Combine all points
  curves_raster <- dplyr::bind_rows(all_points)

  # Filter out curves with too few points
  curves_filtered <- curves_raster %>%
    dplyr::group_by(curve_id) %>%
    dplyr::filter(dplyr::n() >= min_points) %>%
    dplyr::ungroup()

  if (nrow(curves_filtered) == 0) {
    log_debug("No curves meet minimum point threshold")
    return(tibble())
  }

  # Step 7: Convert back to SVG coordinate space
  # Raster coords are relative to cropped panel, need to add panel offset
  curves_svg <- curves_filtered %>%
    dplyr::mutate(
      svg_x = panel_bbox$xmin + (raster_x / scale_factor),
      svg_y = panel_bbox$ymin + (raster_y / scale_factor)
    ) %>%
    dplyr::arrange(curve_id, svg_x) %>%
    dplyr::mutate(
      idx = dplyr::row_number(),
      style_key = sprintf("raster_curve_%d", curve_id),
      stroke = "#000000",  # Black (default)
      dash = NA_character_,
      sw = 1
    ) %>%
    dplyr::select(style_key, stroke, dash, sw, svg_x, svg_y, idx, curve_id)

  log_info("Raster extraction: found %d curves with %d total points",
           dplyr::n_distinct(curves_svg$curve_id), nrow(curves_svg))

  curves_svg
}

detect_censor_ticks <- function(doc, panel_bbox, curve_tbl) {
  if (!nrow(curve_tbl)) return(tibble())

  ln <- svg_line_df(doc)
  if (!nrow(ln)) return(tibble())

  len <- sqrt((ln$x2 - ln$x1)^2 + (ln$y2 - ln$y1)^2)
  midx <- (ln$x1 + ln$x2)/2; midy <- (ln$y1 + ln$y2)/2

  small <- tibble(x = midx, y = midy, len = len,
                  stroke = ln$stroke, dash = ln$dash, sw = ln$sw,
                  x1=ln$x1, y1=ln$y1, x2=ln$x2, y2=ln$y2) %>%
    dplyr::filter(len >= CONFIG$CENSOR_LEN_MIN, len <= CONFIG$CENSOR_LEN_MAX,
                  x >= panel_bbox$xmin, x <= panel_bbox$xmax,
                  y >= panel_bbox$ymin, y <= panel_bbox$ymax)

  if (!nrow(small)) return(tibble())

  ct <- small %>% dplyr::rowwise() %>% dplyr::mutate(
    nearest = {
      d <- sqrt((curve_tbl$svg_x - x)^2 + (curve_tbl$svg_y - y)^2)
      j <- which.min(d)
      if (length(j) && d[j] <= CONFIG$CENSOR_DIST_TO_CURVE) j else NA_integer_
    }
  ) %>% dplyr::ungroup() %>% dplyr::filter(!is.na(nearest))

  if (!nrow(ct)) return(tibble())

  tibble(
    svg_x = ct$x, svg_y = ct$y, curve_id = curve_tbl$curve_id[ct$nearest],
    stroke = ct$stroke, dash = ct$dash, sw = ct$sw
  )
}

# ----------------- UNMIX NEAR-OVERLAPPING CURVES -----------
unmix_close_curves <- function(df) {
  if (!nrow(df)) return(df)

  n_per_style <- df %>% dplyr::count(style_key)
  if (all(n_per_style$n < 2)) return(df)

  split_style <- split(df, df$style_key)
  out <- vector("list", length(split_style)); k <- 0L

  for (st in names(split_style)) {
    g <- split_style[[st]] %>% dplyr::arrange(svg_x, svg_y)
    x0 <- min(g$svg_x); x1 <- max(g$svg_x)
    brks <- seq(x0, x1 + CONFIG$UNMIX_WINDOW_X, by = CONFIG$UNMIX_WINDOW_X)
    g$bin <- cut(g$svg_x, brks, labels = FALSE, include.lowest = TRUE)

    est_k <- g %>% dplyr::group_by(bin) %>%
      dplyr::summarise(kbin = {
        yy <- svg_y
        if (length(yy) < 2) 1L else {
          labs <- integer(length(yy)); cid <- 0L
          for (i in seq_along(yy)) if (labs[i]==0L) {
            cid <- cid + 1L
            labs[abs(yy - yy[i]) <= CONFIG$UNMIX_EPS_Y] <- cid
          }
          max(labs)
        }
      }, .groups = "drop") %>% dplyr::pull(kbin) %>% max(na.rm = TRUE)

    K <- max(1L, min(CONFIG$UNMIX_K_MAX, est_k))
    if (K == 1L) { g$curve_id2 <- 1L; out[[k<-k+1L]] <- g; next }

    g <- g %>% dplyr::group_by(bin) %>%
      dplyr::arrange(svg_y, .by_group = TRUE) %>%
      dplyr::mutate(cls0 = rep(seq_len(min(K, dplyr::n())), length.out = dplyr::n())) %>%
      dplyr::ungroup()

    bins <- sort(unique(g$bin)); g$cls <- g$cls0
    if (length(bins) > 1) {
      prev <- NULL
      for (b in bins) {
        sub <- g[g$bin == b, , drop = FALSE]
        cent <- sub %>% dplyr::group_by(cls0) %>%
          dplyr::summarise(yc = mean(svg_y), .groups="drop")
        if (!is.null(prev)) {
          cm <- outer(cent$yc, prev$yc, function(a,b) abs(a-b))
          match <- apply(cm, 1, which.min)
          map <- setNames(prev$cls0[match], cent$cls0)
          sub$cls <- map[as.character(sub$cls0)]
          sub$cls[is.na(sub$cls)] <- sub$cls0[is.na(sub$cls)]
        } else sub$cls <- sub$cls0
        g$cls[g$bin == b] <- sub$cls
        prev <- sub %>% dplyr::group_by(cls) %>%
          dplyr::summarise(yc = mean(svg_y), cls0 = unique(cls), .groups="drop") %>%
          dplyr::rename(cls0 = cls)
      }
    }
    g$curve_id2 <- g$cls
    out[[k<-k+1L]] <- g
  }

  out <- dplyr::bind_rows(out)
  map_tbl <- out %>%
    dplyr::distinct(style_key, curve_id2) %>%
    dplyr::arrange(style_key, curve_id2) %>%
    dplyr::mutate(curve_id2_norm = dplyr::row_number())

  out %>%
    dplyr::left_join(map_tbl, by = c("style_key","curve_id2")) %>%
    dplyr::mutate(curve_id2 = curve_id2_norm) %>%
    dplyr::select(-curve_id2_norm)
}

# ------------------ NUMBER-AT-RISK (NAR) -------------------
extract_nar_pdf <- function(pdf, page, panel_bbox, y_search = CONFIG$NAR_SEARCH_BELOW) {
  txt <- tryCatch(pdftools::pdf_data(pdf)[[page]] %>% tibble::as_tibble(),
                  error = function(e) NULL)
  if (!is.null(txt) && nrow(txt)) {
    txt <- txt %>% dplyr::mutate(x1 = x, y1 = y, x2 = x + width, y2 = y + height)
    band <- txt %>%
      dplyr::filter(y1 >= panel_bbox$ymax, y1 <= panel_bbox$ymax + y_search) %>%
      dplyr::mutate(num = suppressWarnings(readr::parse_number(text))) %>%
      dplyr::filter(!is.na(num))

    if (nrow(band) >= 3) {
      band <- band %>% dplyr::arrange(y1, x1)
      y_cut <- stats::cutree(stats::hclust(stats::dist(band$y1)), h = CONFIG$NAR_CLUSTER_HEIGHT)
      band$y_grp <- y_cut
      return(
        band %>% dplyr::group_by(y_grp) %>% dplyr::arrange(x1, .by_group = TRUE) %>%
          dplyr::mutate(col = dplyr::row_number()) %>% dplyr::ungroup() %>%
          dplyr::select(y_grp, col, num) %>% dplyr::arrange(y_grp, col)
      )
    }
  }

  if (has_tesseract) {
    log_debug("NAR OCR fallback p%d", page)
    png_file <- tempfile(fileext = ".png")
    pdftools::pdf_convert(pdf, format = "png", pages = page, filenames = png_file, dpi = CONFIG$NAR_OCR_DPI)
    on.exit(unlink(png_file), add = TRUE)

    txt_ocr <- tryCatch(tesseract::ocr_data(png_file), error = function(e) NULL)
    if (!is.null(txt_ocr) && nrow(txt_ocr)) {
      cand <- txt_ocr %>%
        dplyr::filter(confidence > CONFIG$NAR_OCR_CONFIDENCE, grepl("^[0-9]+$", word)) %>%
        dplyr::transmute(num = as.integer(word))
      if (nrow(cand) >= 3) {
        cand$y_grp <- 1L; cand$col <- seq_len(nrow(cand))
        return(cand %>% dplyr::select(y_grp,col,num))
      }
    }
  }
  NULL
}

# ----------------------- INSETS & SCORING ------------------
flag_insets <- function(panels_df) {
  if (nrow(panels_df) < 2) return(dplyr::mutate(panels_df, is_inset = FALSE))

  pd <- dplyr::mutate(panels_df, area = (xmax - xmin) * (ymax - ymin))
  contains <- function(a,b) (a$xmin >= b$xmin && a$xmax <= b$xmax &&
                               a$ymin >= b$ymin && a$ymax <= b$ymax)
  idx <- rep(FALSE, nrow(pd))
  main_area <- max(pd$area, na.rm = TRUE)

  for (i in seq_len(nrow(pd))) for (j in seq_len(nrow(pd))) if (i!=j) {
    if (contains(pd[i,], pd[j,]) && pd$area[i] <= CONFIG$INSET_AREA_FRAC * main_area) {
      idx[i] <- TRUE; break
    }
  }
  dplyr::mutate(pd, is_inset = idx)
}

score_panels <- function(calibrations, pdf, page) {
  if (!nrow(calibrations)) return(calibrations)

  y_max <- purrr::map_dbl(calibrations$cal,
                          ~ suppressWarnings(max(.x$ticks$y$val, na.rm = TRUE)))
  area  <- (calibrations$xmax - calibrations$xmin) * (calibrations$ymax - calibrations$ymin)
  nar_hit <- purrr::pmap_lgl(
    list(calibrations$xmin, calibrations$xmax, calibrations$ymin, calibrations$ymax),
    ~ !is.null(extract_nar_pdf(pdf, page, c(xmin=..1,xmax=..2,ymin=..3,ymax=..4)))
  )

  yscore <- -abs(y_max - 1)
  s <- scales::rescale(area) + scales::rescale(yscore) + ifelse(nar_hit, 0.25, 0)
  dplyr::mutate(calibrations, panel_score = s, y_max = y_max, nar_hit = nar_hit)
}

# -------------------- VALIDATION FUNCTIONS -------------------
validate_km_curves <- function(curves_df) {
  if (!nrow(curves_df)) return(curves_df)

  validation <- curves_df %>%
    dplyr::group_by(panel_id, curve_id) %>%
    dplyr::arrange(t) %>%
    dplyr::mutate(
      S_diff = c(0, diff(S)),
      is_valid = all(S_diff <= 0.01),  # Small tolerance for noise
      n_points = dplyr::n()
    ) %>%
    dplyr::ungroup()

  invalid <- validation %>%
    dplyr::filter(!is_valid) %>%
    dplyr::distinct(panel_id, curve_id)

  if (nrow(invalid) > 0) {
    log_warn("Non-monotonic curves detected: %s",
             paste(sprintf("Panel%d-Curve%d", invalid$panel_id, invalid$curve_id),
                   collapse = ", "))
  }

  # Add validation status to the dataframe
  validation %>% dplyr::select(-S_diff)
}

# ----------------------- PUBLIC API ------------------------
#' Extract Kaplan-Meier Curves from PDF
#'
#' @description
#' Extracts Kaplan-Meier survival curves from PDF files by converting to SVG,
#' detecting panels and axes, calibrating coordinates, and extracting curve data.
#'
#' @param pdf Character string. Path to the PDF file to process.
#' @param pages Integer vector. Specific pages to process. If NULL (default), all pages are processed.
#' @param temp_svg_dir Character string. Directory for temporary SVG files. Defaults to tempdir().
#' @param dpi Numeric. DPI resolution for SVG conversion. Defaults to CONFIG$SVG_DPI (144).
#'
#' @return An object of class "km_pdf_extract" containing a list for each processed page with:
#'   \item{panels}{Data frame of detected panels with bounding boxes and scores}
#'   \item{curves}{Data frame of extracted curve coordinates in both SVG and calibrated space}
#'   \item{ticks}{Data frame of detected censor tick marks}
#'   \item{nar}{List of number-at-risk tables (if detected)}
#'   \item{legend}{Data frame mapping curve styles to labels}
#'
#' @details
#' This function performs the following steps:
#' 1. Converts PDF pages to SVG format using mutool
#' 2. Detects panel boundaries from axis lines
#' 3. Calibrates coordinates using axis tick labels
#' 4. Extracts curve paths and flattens Bezier curves
#' 5. Detects censor tick marks
#' 6. Extracts number-at-risk tables
#' 7. Maps legend styles to curve labels
#'
#' @examples
#' \dontrun{
#' # Extract from all pages
#' result <- extract_km_from_pdf("survival_plot.pdf")
#'
#' # Extract specific pages only
#' result <- extract_km_from_pdf("paper.pdf", pages = c(1, 3, 5))
#'
#' # Use custom DPI
#' result <- extract_km_from_pdf("figure.pdf", dpi = 300)
#' }
#'
#' @export
extract_km_from_pdf <- function(pdf, pages = NULL, temp_svg_dir = tempdir(),
                                dpi = CONFIG$SVG_DPI) {
  if (!is.character(pdf) || length(pdf) != 1)
    stop("'pdf' must be a single file path", call. = FALSE)
  if (!file.exists(pdf))
    stop("PDF file not found: ", pdf, call. = FALSE)
  if (!dir.exists(temp_svg_dir)) {
    dir.create(temp_svg_dir, recursive = TRUE, showWarnings = FALSE)
  }
  if (file.access(temp_svg_dir, 2) != 0)
    stop("Temp dir not writable: ", temp_svg_dir, call. = FALSE)

  svgs <- pdf_to_svg(pdf, pages = pages, outdir = temp_svg_dir, dpi = dpi)
  if (CONFIG$CLEANUP_SVGS) {
    on.exit({ unlink(svgs); log_debug("Cleaned %d temp SVGs", length(svgs)) }, add = TRUE)
  }

  out <- vector("list", length(svgs))
  for (i in seq_along(svgs)) {
    svg_file <- svgs[i]
    pg <- if (is.null(pages)) i else pages[i]

    doc <- tryCatch(xml2::read_xml(svg_file), error = function(e) NULL)
    if (is.null(doc)) {
      log_warn("Failed to parse SVG p%d (%s) - Check if file is corrupted or mutool version is compatible",
               pg, svg_file)
      next
    }

    texts <- svg_text_df(doc, pdf_file = pdf, page_num = pg)
    lines <- svg_line_df(doc)
    path_lines <- extract_axis_paths(doc)

    # Combine regular lines and path-based lines for axis detection
    all_lines <- dplyr::bind_rows(lines, path_lines)

    axes  <- find_axis_lines(all_lines)
    panels <- build_panels(axes, texts = texts)

    if (!nrow(panels)) {
      out[[i]] <- list(
        panels = tibble(page=pg, panel_id=integer(), xmin=numeric(),
                        xmax=numeric(), ymin=numeric(), ymax=numeric()),
        curves = tibble(), nar = tibble(), ticks = tibble(), legend = tibble()
      )
      next
    }

    # Wrap calibrate_panel to handle errors gracefully
    safe_calibrate <- purrr::possibly(calibrate_panel, otherwise = NULL, quiet = TRUE)

    calibrations <- panels %>%
      dplyr::mutate(cal = purrr::map(seq_len(dplyr::n()),
                                     ~ safe_calibrate(panels[.x,], texts))) %>%
      dplyr::mutate(ok = purrr::map_lgl(cal, ~ !is.null(.x))) %>%
      dplyr::filter(ok) %>%
      dplyr::mutate(panel_id = dplyr::row_number())

    if (!nrow(calibrations)) {
      out[[i]] <- list(
        panels = panels %>% dplyr::mutate(page = pg, panel_id = dplyr::row_number()),
        curves = tibble(), nar = tibble(), ticks = tibble(), legend = tibble()
      )
      next
    }

    calibrations <- flag_insets(calibrations)
    if (!CONFIG$INSET_KEEP) calibrations <- calibrations %>% dplyr::filter(!is_inset)
    calibrations <- score_panels(calibrations, pdf, pg)

    if (CONFIG$PANEL_MODE == "largest") {
      calibrations <- calibrations %>%
        dplyr::arrange(dplyr::desc((xmax-xmin)*(ymax-ymin))) %>%
        dplyr::slice(1)
    } else if (CONFIG$PANEL_MODE == "bestscore") {
      calibrations <- calibrations %>%
        dplyr::arrange(dplyr::desc(panel_score)) %>%
        dplyr::slice(1)
    }

    legend_map <- find_legend_map(doc, calibrations %>% dplyr::select(xmin,xmax,ymin,ymax))

    curve_tbls <- purrr::pmap_dfr(list(seq_len(nrow(calibrations))), function(k) {
      pb <- calibrations[k, c("xmin","xmax","ymin","ymax")] %>% as.numeric()
      names(pb) <- c("xmin","xmax","ymin","ymax")

      # Try vector extraction with timeout to avoid hanging
      hc <- tryCatch({
        R.utils::withTimeout({
          harvest_curves(doc, pb)
        }, timeout = 10, onTimeout = "error")
      }, error = function(e) {
        log_debug("Vector extraction failed/timeout for panel %d: %s", k, e$message)
        tibble()
      })

      # Fallback to raster extraction if vector extraction fails
      if (!nrow(hc)) {
        log_debug("Vector extraction returned 0 curves for panel %d, trying raster fallback", k)
        hc <- harvest_curves_from_raster(pdf, pg, pb)
        if (!nrow(hc)) return(tibble())
      }

      hc <- unmix_close_curves(hc)
      if ("curve_id2" %in% names(hc)) {
        hc$curve_id <- hc$curve_id2
        hc$curve_id2 <- NULL
      }

      xmap <- calibrations$cal[[k]]$x_map
      ymap <- calibrations$cal[[k]]$y_map

      hc %>%
        dplyr::mutate(panel_id = calibrations$panel_id[k],
                      t = xmap(svg_x), S = ymap(svg_y)) %>%
        dplyr::mutate(S = pmin(pmax(S, -0.05), 1.05)) %>%
        dplyr::left_join(legend_map %>% dplyr::rename(curve_label = label),
                         by = c("style_key" = "style_key")) %>%
        dplyr::select(panel_id, curve_id, svg_x, svg_y, t, S,
                      stroke, dash, sw, style_key, curve_label)
    })

    # Validate curves
    if (nrow(curve_tbls)) {
      curve_tbls <- validate_km_curves(curve_tbls)
    }

    ticks_tbls <- purrr::pmap_dfr(list(seq_len(nrow(calibrations))), function(k) {
      pb <- calibrations[k, c("xmin","xmax","ymin","ymax")] %>% as.numeric()
      names(pb) <- c("xmin","xmax","ymin","ymax")

      cur <- curve_tbls %>% dplyr::filter(panel_id == calibrations$panel_id[k])
      if (!nrow(cur)) return(tibble())

      ct <- detect_censor_ticks(doc, pb, cur %>% dplyr::select(svg_x, svg_y, curve_id))
      if (!nrow(ct)) return(tibble())

      dplyr::mutate(ct, panel_id = calibrations$panel_id[k]) %>%
        dplyr::select(panel_id, curve_id, svg_x, svg_y)
    })

    nar_list <- purrr::pmap_dfr(list(seq_len(nrow(calibrations))), function(k) {
      pb <- calibrations[k, c("xmin","xmax","ymin","ymax")] %>% as.numeric()
      names(pb) <- c("xmin","xmax","ymin","ymax")

      nar <- extract_nar_pdf(pdf, page = pg, panel_bbox = pb)
      if (is.null(nar)) nar <- NA
      tibble(page = pg, panel_id = calibrations$panel_id[k], nar = list(nar))
    })

    out[[i]] <- list(
      panels = calibrations %>%
        dplyr::mutate(page = pg) %>%
        dplyr::select(page, panel_id, xmin, xmax, ymin, ymax,
                      is_inset, panel_score, y_max, nar_hit),
      curves = curve_tbls,
      ticks  = ticks_tbls,
      nar    = nar_list,
      legend = legend_map
    )
  }
  structure(out, class = "km_pdf_extract")
}

#' Combine Extraction Results into Single Data Frames
#'
#' @description
#' Consolidates multi-page extraction results into single data frames for easier analysis.
#'
#' @param x An object of class "km_pdf_extract" returned by extract_km_from_pdf()
#'
#' @return A list containing:
#'   \item{panels}{Combined data frame of all detected panels}
#'   \item{curves}{Combined data frame of all extracted curves}
#'   \item{ticks}{Combined data frame of all censor ticks}
#'   \item{nar}{Combined data frame of all number-at-risk tables}
#'   \item{legend}{Combined data frame of all legend mappings}
#'
#' @examples
#' \dontrun{
#' result <- extract_km_from_pdf("paper.pdf")
#' combined <- km_extract_bind(result)
#' head(combined$curves)
#' }
#'
#' @export
km_extract_bind <- function(x) {
  stopifnot(inherits(x, "km_pdf_extract"))
  list(
    panels = purrr::map_dfr(x, "panels"),
    curves = purrr::map_dfr(x, "curves"),
    ticks  = purrr::map_dfr(x, "ticks"),
    nar    = purrr::map_dfr(x, "nar"),
    legend = purrr::map_dfr(x, "legend")
  )
}

#' Plot Extracted Curves in SVG Coordinate Space
#'
#' @description
#' Creates a ggplot visualization of extracted curves in SVG coordinate space for QA purposes.
#'
#' @param x An object of class "km_pdf_extract" returned by extract_km_from_pdf()
#' @param page Integer. Which page to plot (default: 1)
#'
#' @return A ggplot2 object
#'
#' @examples
#' \dontrun{
#' result <- extract_km_from_pdf("paper.pdf")
#' plot <- km_autoplot_svg(result, page = 1)
#' print(plot)
#' }
#'
#' @export
km_autoplot_svg <- function(x, page = 1) {
  stopifnot(inherits(x, "km_pdf_extract"))
  cur <- x[[page]]$curves

  if (is.null(cur) || !nrow(cur)) {
    return(ggplot2::ggplot() +
             ggplot2::theme_minimal() +
             ggplot2::ggtitle(sprintf("No curves on page %d", page)))
  }

  ggplot2::ggplot(cur, ggplot2::aes(svg_x, svg_y,
                                    group = interaction(panel_id, curve_id),
                                    linetype = as.factor(curve_id),
                                    color = as.factor(curve_id))) +
    ggplot2::geom_path(alpha = 0.95) +
    ggplot2::scale_y_reverse() +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = sprintf("KM curves (SVG space) - page %d", page),
                  x = "svg_x", y = "svg_y",
                  linetype = "curve_id", color = "curve_id")
}

#' Export Extracted Curves to CSV Files
#'
#' @description
#' Exports each extracted curve to a separate CSV file for further analysis.
#'
#' @param x An object of class "km_pdf_extract" returned by extract_km_from_pdf()
#' @param prefix Character string. File prefix for exported CSVs (default: "km_extract")
#'
#' @return Invisibly returns TRUE on success
#'
#' @details
#' Creates one CSV file per curve with the naming pattern:
#' prefix_panelXX_curveXX.csv
#'
#' @examples
#' \dontrun{
#' result <- extract_km_from_pdf("paper.pdf")
#' km_export_curves(result, prefix = "my_study")
#' }
#'
#' @export
km_export_curves <- function(x, prefix = "km_extract") {
  stopifnot(inherits(x, "km_pdf_extract"))
  allc <- purrr::map_dfr(x, "curves")

  if (!nrow(allc)) stop("No curves to export.", call. = FALSE)

  allc %>%
    dplyr::group_by(panel_id, curve_id) %>%
    dplyr::arrange(t, .by_group = TRUE) %>%
    dplyr::group_walk(~ readr::write_csv(.x,
                                         sprintf("%s_panel%02d_curve%02d.csv", prefix, .y$panel_id[1], .y$curve_id[1])))

  invisible(TRUE)
}

# --------------------- SINGLE-PDF PROCESSOR -----------------
process_single_pdf <- function(pdf_path, output_dir, pages = NULL,
                               dpi = CONFIG$SVG_DPI,
                               clean_intermediate = CONFIG$CLEAN_INTERMEDIATE) {
  base_name <- sanitize_basename(pdf_path)
  pdf_outdir <- file.path(output_dir, base_name)
  dir.create(pdf_outdir, showWarnings = FALSE, recursive = TRUE)
  log_info("Processing %s", basename(pdf_path))

  res <- tryCatch({
    extract_km_from_pdf(pdf_path, pages = pages, dpi = dpi)
  }, error = function(e) {
    log_err("Process failed %s: %s", basename(pdf_path), e$message)
    NULL
  })

  if (is.null(res) || !length(res)) {
    log_warn("No KM data found: %s", basename(pdf_path))
    return(invisible(FALSE))
  }

  log_info("Saving QA plots ...")
  for (pg_idx in seq_along(res)) {
    # Extract actual page number from the result (handles --pages parameter correctly)
    actual_page <- if (!is.null(res[[pg_idx]]$panels) && nrow(res[[pg_idx]]$panels) > 0) {
      res[[pg_idx]]$panels$page[1]
    } else {
      # Fallback: if pages was specified, use that; otherwise use index
      if (!is.null(pages)) pages[pg_idx] else pg_idx
    }

    p <- km_autoplot_svg(res, page = pg_idx)
    ggplot2::ggsave(filename = file.path(pdf_outdir, sprintf("QA_page_%03d.png", actual_page)),
                    plot = p, width = 10, height = 8, bg = "white")
  }

  log_info("Exporting curve CSVs ...")
  csv_prefix <- file.path(pdf_outdir, base_name)

  ok <- tryCatch({
    km_export_curves(res, prefix = csv_prefix)
    TRUE
  }, error = function(e) {
    log_err("CSV export failed for %s: %s", basename(pdf_path), e$message)
    FALSE
  })

  if (ok) {
    bound <- km_extract_bind(res)
    readr::write_csv(bound$panels, file.path(pdf_outdir, "panels.csv"))
    if (nrow(bound$curves)) readr::write_csv(bound$curves, file.path(pdf_outdir, "curves_all.csv"))
    if (nrow(bound$ticks))  readr::write_csv(bound$ticks,  file.path(pdf_outdir, "censor_ticks.csv"))
    if (nrow(bound$legend)) readr::write_csv(bound$legend, file.path(pdf_outdir, "legend_map.csv"))
    if (nrow(bound$nar)) {
      purrr::pwalk(bound$nar, function(page, panel_id, nar) {
        if (is.list(nar) && !is.null(nar[[1]]) && !all(is.na(nar[[1]]))) {
          readr::write_csv(nar[[1]], file.path(pdf_outdir,
                                               sprintf("nar_page%03d_panel%02d.csv", page, panel_id)))
        }
      })
    }
  }

  # Clean up memory if requested
  if (clean_intermediate) {
    rm(res, bound)
    gc(verbose = FALSE)
  }

  log_info("Done: %s", basename(pdf_path))
  invisible(TRUE)
}

# ----------------------- BATCH RUNNER -----------------------
#' Process Multiple PDFs in Batch Mode
#'
#' @description
#' Processes all PDF files in a directory, extracting KM curves from each and
#' creating a consolidated output with QA plots and CSV exports.
#'
#' @param input_dir Character string. Directory containing PDF files to process
#' @param output_dir Character string. Directory for output files
#' @param pages Integer vector. Specific pages to process (NULL = all pages)
#' @param n_cores Integer. Number of parallel workers (requires future/furrr packages)
#' @param dpi Numeric. DPI for SVG conversion
#'
#' @return Invisibly returns NULL. Creates output files as side effect.
#'
#' @details
#' Creates a subdirectory for each PDF with:
#' - Individual curve CSVs
#' - QA plots showing detected curves
#' - Combined data files (panels.csv, curves_all.csv, etc.)
#' - A master _MASTER_CURVES.csv consolidating all PDFs
#'
#' @examples
#' \dontrun{
#' # Process all PDFs in default directory
#' run_batch()
#'
#' # Use parallel processing with 4 cores
#' run_batch(n_cores = 4)
#'
#' # Process specific pages only
#' run_batch(pages = c(1, 2, 3))
#' }
#'
#' @export
run_batch <- function(input_dir = "papers_to_process",
                      output_dir = "extraction_results",
                      pages = NULL,
                      n_cores = 1,
                      dpi = CONFIG$SVG_DPI) {
  if (!dir.exists(input_dir)) stop("Input dir not found: ", input_dir, call. = FALSE)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  pdf_files <- list.files(path = input_dir, pattern = "\\.pdf$", full.names = TRUE)
  if (!length(pdf_files)) stop("No PDF files found in: ", input_dir, call. = FALSE)

  log_info("Found %d PDFs to process in '%s'.", length(pdf_files), input_dir)

  # Track results for summary
  results <- rep(FALSE, length(pdf_files))
  names(results) <- basename(pdf_files)

  pb <- NULL
  if (has_progress) {
    pb <- progress::progress_bar$new(
      format = "[:bar] :current/:total (:percent) ETA: :eta",
      total = length(pdf_files)
    )
  }

  use_parallel <- n_cores > 1 && has_future && has_furrr

  if (use_parallel) {
    future::plan(future::multisession, workers = n_cores)
    on.exit({ try(future::plan(future::sequential), silent = TRUE) }, add = TRUE)
    log_info("Parallel: %d workers.", future::nbrOfWorkers())

    results <- furrr::future_map_lgl(pdf_files, function(pdf_path) {
      ok <- process_single_pdf(pdf_path, output_dir, pages = pages, dpi = dpi)
      if (!is.null(pb)) pb$tick()
      ok
    }, .options = furrr::furrr_options(seed = TRUE))
  } else {
    for (i in seq_along(pdf_files)) {
      results[i] <- process_single_pdf(pdf_files[i], output_dir, pages = pages, dpi = dpi)
      if (!is.null(pb)) pb$tick()
    }
  }

  log_info("Consolidating curves_all.csv ...")
  all_curve_files <- list.files(output_dir, "curves_all.csv", recursive = TRUE, full.names = TRUE)

  n_curves_total <- 0
  if (length(all_curve_files)) {
    all_curves <- purrr::map_dfr(all_curve_files, function(f) {
      source_pdf <- basename(dirname(f))
      tryCatch({
        readr::read_csv(f, col_types = readr::cols(), show_col_types = FALSE) %>%
          dplyr::mutate(source_pdf = source_pdf)
      }, error = function(e) {
        log_warn("Failed to read %s: %s", f, e$message)
        tibble::tibble()
      })
    })
    n_curves_total <- nrow(all_curves)
    if (n_curves_total > 0) {
      readr::write_csv(all_curves, file.path(output_dir, "_MASTER_CURVES.csv"))
      log_info("Wrote %s", file.path(output_dir, "_MASTER_CURVES.csv"))
    } else {
      log_warn("No valid curves to consolidate.")
    }
  } else {
    log_warn("No curves_all.csv to consolidate.")
  }

  # Summary statistics
  n_success <- sum(results)
  n_failed <- length(results) - n_success

  log_info("Summary: %d PDFs processed, %d successful, %d failed, %d total curves extracted",
           length(pdf_files), n_success, n_failed, n_curves_total)

  if (n_failed > 0) {
    failed_files <- names(results)[!results]
    log_warn("Failed PDFs: %s", paste(failed_files, collapse = ", "))
  }

  log_info("All files processed.")
}

# ----------------------- SELF-TEST ------------------------
run_self_test <- function() {
  log_info("Running self-test...")

  # Test 1: Check required commands
  if (!.which_exists("mutool")) {
    log_err("✗ mutool not found")
    return(FALSE)
  }
  log_info("✓ mutool found")

  # Test 2: Check required packages
  required_pkgs <- c("xml2", "dplyr", "pdftools", "ggplot2")
  for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      log_err("✗ Package %s not installed", pkg)
      return(FALSE)
    }
  }
  log_info("✓ All required packages available")

  # Test 3: Create minimal test SVG
  test_svg <- '<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="500" height="400">
  <line x1="50" y1="350" x2="450" y2="350" stroke="black" stroke-width="1"/>
  <line x1="50" y1="50" x2="50" y2="350" stroke="black" stroke-width="1"/>
  <text x="25" y="350">0</text>
  <text x="25" y="250">0.5</text>
  <text x="25" y="150">1.0</text>
</svg>'

  temp_file <- tempfile(fileext = ".svg")
  writeLines(test_svg, temp_file)

  tryCatch({
    doc <- xml2::read_xml(temp_file)
    lines <- svg_line_df(doc)
    if (nrow(lines) != 2) stop("Line extraction failed")
    log_info("✓ SVG parsing works")

    texts <- svg_text_df(doc)
    if (nrow(texts) != 3) stop("Text extraction failed")
    log_info("✓ Text extraction works")

    # Test configuration validation
    validate_config()
    log_info("✓ Configuration valid")

    log_info("✓ All tests passed")
    return(TRUE)
  }, error = function(e) {
    log_err("✗ Self-test failed: %s", e$message)
    return(FALSE)
  }, finally = {
    unlink(temp_file)
  })
}

# ------------------------ CLI ENTRYPOINT --------------------
print_help <- function() {
  cat("Usage: Rscript km_pdf_vector_extract_ultra.R [options]\n",
      "Options:\n",
      "  --input_dir=PATH     Input directory (default: papers_to_process)\n",
      "  --output_dir=PATH    Output directory (default: extraction_results)\n",
      "  --pages=1,2,3        Specific pages to process (default: all)\n",
      "  --n_cores=N          Parallel workers (default: 1)\n",
      "  --dpi=144            SVG DPI (default: 144)\n",
      "  --panel_mode=bestscore|largest|all  (default: bestscore)\n",
      "  --keep_insets=true|false            (default: false)\n",
      "  --test               Run self-test and exit\n",
      "  --version            Show version and exit\n",
      "  --help, -h           Show this help and exit\n",
      "Env:\n",
      "  KM_LOG_LEVEL=DEBUG|INFO|WARN|ERROR  (default: INFO)\n", sep = "")
}

if (identical(environment(), globalenv())) {
  args <- commandArgs(trailingOnly = TRUE)

  if ("--help" %in% args || "-h" %in% args) {
    print_help()
    quit(save="no", status=0)
  }

  if ("--version" %in% args) {
    cat("km_pdf_vector_extract_ultra.R version", .VERSION, "\n")
    quit(save="no", status=0)
  }

  if ("--test" %in% args) {
    if (run_self_test()) quit(save="no", status=0)
    else quit(save="no", status=1)
  }

  get_arg <- function(flag, default=NULL) {
    hit <- grep(paste0("^", flag, "="), args, value = TRUE)
    if (length(hit)) sub(paste0("^", flag, "="), "", hit[1]) else default
  }

  input_dir  <- get_arg("--input_dir", "papers_to_process")
  output_dir <- get_arg("--output_dir", "extraction_results")
  pages_arg  <- get_arg("--pages", NULL)
  pages      <- if (!is.null(pages_arg)) {
    pages_vec <- suppressWarnings(as.integer(unlist(strsplit(pages_arg, ",", fixed = TRUE))))
    if (any(is.na(pages_vec))) {
      cat("ERROR: Invalid page specification: ", pages_arg, "\n", sep="")
      quit(save="no", status=1)
    }
    pages_vec
  } else NULL
  ncores_arg <- get_arg("--n_cores", "1")
  dpi_arg    <- get_arg("--dpi", as.character(CONFIG$SVG_DPI))
  panel_mode <- get_arg("--panel_mode", CONFIG$PANEL_MODE)
  keep_ins   <- tolower(get_arg("--keep_insets", ifelse(CONFIG$INSET_KEEP,"true","false"))) == "true"

  if (!dir.exists(input_dir)) {
    cat("ERROR: Input dir not found: ", input_dir, "\n", sep="")
    quit(save="no", status=1)
  }

  n_cores <- suppressWarnings(as.integer(ncores_arg))
  if (is.na(n_cores) || n_cores < 1) n_cores <- 1

  dpi <- suppressWarnings(as.integer(dpi_arg))
  if (is.na(dpi) || dpi < 72) dpi <- CONFIG$SVG_DPI

  # Create modified config instead of modifying global state
  run_config <- CONFIG
  run_config$PANEL_MODE <- panel_mode
  run_config$INSET_KEEP <- keep_ins

  # Temporarily assign to CONFIG for run (not ideal but maintains compatibility)
  old_config <- CONFIG
  CONFIG <<- run_config

  # Validate configuration before running
  validate_config()

  if (!interactive()) {
    tryCatch({
      run_batch(input_dir, output_dir, pages, n_cores, dpi)
    }, finally = {
      CONFIG <<- old_config  # Restore original config
    })
  }
}
