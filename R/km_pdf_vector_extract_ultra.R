#!/usr/bin/env Rscript
# km_pdf_vector_extract_ultra.R
# Full-featured, vector-first KM extractor (single file)

suppressPackageStartupMessages({
  req <- c("xml2","dplyr","stringr","purrr","tibble","tidyr","readr",
           "pdftools","ggplot2","tools","scales")
  miss <- setdiff(req, rownames(installed.packages()))
  if (length(miss)) stop("Missing packages: ", paste(miss, collapse=", "),
                         "\nInstall with: install.packages(c('", paste(miss, collapse="','"), "'))",
                         call. = FALSE)
  lapply(req, library, character.only = TRUE)
  # Optional
  if (!"MASS"     %in% rownames(installed.packages())) assign(".NO_MASS", TRUE, envir = .GlobalEnv)
  if (!"progress" %in% rownames(installed.packages())) assign(".NO_PROGRESS", TRUE, envir = .GlobalEnv)
  if (!"tesseract"%in% rownames(installed.packages())) assign(".NO_TESS", TRUE, envir = .GlobalEnv)
  if (!"future"   %in% rownames(installed.packages())) assign(".NO_FUTURE", TRUE, envir = .GlobalEnv)
  if (!"furrr"    %in% rownames(installed.packages())) assign(".NO_FURRR", TRUE, envir = .GlobalEnv)
})

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
  CENSOR_ORTHO_ANGLE    = 25        # degrees tolerance from orthogonal to step
)

.LOG_LEVEL <- toupper(Sys.getenv("KM_LOG_LEVEL", "INFO"))
.LOG_LEVELS <- c(ERROR=1, WARN=2, INFO=3, DEBUG=4)
.level <- function() ifelse(.LOG_LEVEL %in% names(.LOG_LEVELS), .LOG_LEVELS[.LOG_LEVEL], 3)
log_debug <- function(...) if (.level() >= 4) message("[DEBUG] ", sprintf(...))
log_info  <- function(...) if (.level() >= 3) message("[INFO]  ", sprintf(...))
log_warn  <- function(...) if (.level() >= 2) message("[WARN]  ", sprintf(...))
log_err   <- function(...) if (.level() >= 1) message("[ERROR] ", sprintf(...))

# ------------------------ UTILITIES ------------------------
.which_exists <- function(cmd) nzchar(Sys.which(cmd))
.require_cmd <- function(cmd, hint) { if (!.which_exists(cmd)) stop(sprintf("%s not found. %s", cmd, hint), call. = FALSE) }
sanitize_basename <- function(path) gsub("[^A-Za-z0-9_-]", "_", tools::file_path_sans_ext(basename(path)))
.num <- function(x, default = NA_real_, warn = FALSE) {
  res <- suppressWarnings(as.numeric(x))
  if (warn && any(is.na(res) & !is.na(x))) log_warn("Numeric cast failed for: %s", paste0(x[is.na(res)], collapse=","))
  res
}

# -------------------- PDF -> SVG (vector) -----------------
pdf_to_svg <- function(pdf, pages = NULL, outdir = tempdir(), dpi = CONFIG$SVG_DPI) {
  .require_cmd("mutool", "Install MuPDF tools (brew install mupdf-tools | apt-get install mupdf-tools).")

  if (!is.character(pdf) || length(pdf) != 1) stop("'pdf' must be a single file path", call. = FALSE)
  if (!file.exists(pdf)) stop("PDF not found: ", pdf, call. = FALSE)
  if (!grepl("\\.pdf$", pdf, ignore.case = TRUE)) stop("File must have .pdf extension: ", pdf, call. = FALSE)
  pdf <- normalizePath(pdf, mustWork = TRUE)
  outdir <- normalizePath(outdir, mustWork = FALSE)

  info <- pdftools::pdf_info(pdf)
  if (is.null(pages)) pages <- seq_len(info$pages)
  if (!is.numeric(pages) || any(pages < 1) || any(pages != floor(pages))) stop("'pages' must be positive integers", call. = FALSE)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  safe <- sanitize_basename(pdf)
  svg_files <- character(length(pages))
  for (i in seq_along(pages)) {
    p <- as.integer(pages[i])
    out <- file.path(outdir, sprintf("%s_page-%03d.svg", safe, p))
    args <- c("draw", "-F", "svg", "-o", out, "-r", as.character(dpi), pdf, sprintf("%d", p))
    status <- tryCatch(
      suppressWarnings(system2("mutool", args, stdout = TRUE, stderr = TRUE, timeout = CONFIG$MUTOOL_TIMEOUT_SEC)),
      error = function(e) { log_warn("system2 timeout not supported; running without timeout (page %d).", p)
        suppressWarnings(system2("mutool", args, stdout = TRUE, stderr = TRUE)) }
    )
    if (!file.exists(out)) stop("Failed SVG conversion for page ", p, " (", basename(pdf), "):\n", paste(status, collapse="\n"))
    svg_files[i] <- out
  }
  svg_files
}

# ----------------------- SVG PRIMITIVES --------------------
svg_text_df <- function(doc) {
  texts <- xml_find_all(doc, ".//text")
  if (!length(texts)) return(tibble(node = character(), text = character(), x = numeric(), y = numeric()))
  tibble(node = texts, text = xml_text(texts),
         x = .num(xml_attr(texts, "x")), y = .num(xml_attr(texts, "y"))) %>%
    filter(!is.na(x), !is.na(y))
}
svg_line_df <- function(doc) {
  ln <- xml_find_all(doc, ".//line")
  if (!length(ln)) return(tibble())
  tibble(node = ln,
         x1 = .num(xml_attr(ln, "x1")), y1 = .num(xml_attr(ln, "y1")),
         x2 = .num(xml_attr(ln, "x2")), y2 = .num(xml_attr(ln, "y2")),
         stroke = xml_attr(ln, "stroke"), sw = .num(xml_attr(ln, "stroke-width")),
         dash = xml_attr(ln, "stroke-dasharray"))
}
svg_polyline_df <- function(doc) {
  pl <- xml_find_all(doc, ".//polyline|.//polygon")
  if (!length(pl)) return(tibble())
  tibble(node = pl,
         points = xml_attr(pl, "points"),
         stroke = xml_attr(pl, "stroke"),
         fill = xml_attr(pl, "fill"),
         sw = .num(xml_attr(pl, "stroke-width")),
         dash = xml_attr(pl, "stroke-dasharray")) %>%
    filter(!is.na(points))
}
svg_path_df <- function(doc) {
  p <- xml_find_all(doc, ".//path")
  if (!length(p)) return(tibble())
  tibble(node = p,
         d = xml_attr(p, "d"),
         stroke = xml_attr(p, "stroke"),
         fill = xml_attr(p, "fill"),
         sw = .num(xml_attr(p, "stroke-width")),
         dash = xml_attr(p, "stroke-dasharray"))
}

# -------- Bezier helpers (flatten C/Q to polyline points) ----------
# Evaluate quadratic Bezier
bez_q <- function(t, p0, p1, p2) (1-t)^2*p0 + 2*(1-t)*t*p1 + t^2*p2
# Evaluate cubic Bezier
bez_c <- function(t, p0, p1, p2, p3) (1-t)^3*p0 + 3*(1-t)^2*t*p1 + 3*(1-t)*t^2*p2 + t^3*p3

# Adaptive segment count by total span
seg_count <- function(dx, dy) {
  L <- sqrt(dx*dx + dy*dy)
  n <- ceiling(max(3, min(CONFIG$BEZIER_MAX_SEG, L / CONFIG$BEZIER_TOL)))
  as.integer(n)
}

path_to_points <- function(d) {
  if (is.na(d) || !nzchar(d)) return(NULL)
  s <- gsub(",", " ", d)
  toks <- strsplit(str_trim(s), "\\s+", perl = TRUE)[[1]]
  i <- 1; x <- 0; y <- 0; sx <- NA; sy <- NA; cmd <- NULL
  pts <- list(); n <- 0L
  take_num <- function() { v <- suppressWarnings(as.numeric(toks[i])); i <<- i + 1; v }

  add_pt <- function(xx, yy) { n <<- n + 1L; pts[[n]] <<- c(xx, yy) }

  while (i <= length(toks)) {
    tk <- toks[i]; i <- i + 1
    if (grepl("^[MLHVCSQmlhvcsqZz]$", tk)) { cmd <- tk; if (cmd %in% c("Z","z")) { if (!is.na(sx)) add_pt(sx, sy); next } }

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

    } else if (cmd %in% c("Q","q")) { # quadratic
      if (cmd == "Q") { x1 <- take_num(); y1 <- take_num(); x2 <- take_num(); y2 <- take_num() }
      else { x1 <- x + take_num(); y1 <- y + take_num(); x2 <- x + take_num(); y2 <- y + take_num() }
      nseg <- seg_count(x2 - x, y2 - y)
      for (t in seq(0, 1, length.out = nseg)) {
        xx <- bez_q(t, x, x1, x2); yy <- bez_q(t, y, y1, y2); add_pt(xx, yy)
      }
      x <- x2; y <- y2

    } else if (cmd %in% c("C","c")) { # cubic
      if (cmd == "C") { x1 <- take_num(); y1 <- take_num(); x2 <- take_num(); y2 <- take_num(); x3 <- take_num(); y3 <- take_num() }
      else { x1 <- x + take_num(); y1 <- y + take_num(); x2 <- x + take_num(); y2 <- y + take_num(); x3 <- x + take_num(); y3 <- y + take_num() }
      nseg <- seg_count(x3 - x, y3 - y)
      for (t in seq(0, 1, length.out = nseg)) {
        xx <- bez_c(t, x, x1, x2, x3); yy <- bez_c(t, y, y1, y2, y3); add_pt(xx, yy)
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
  if (is.null(df) || !is.data.frame(df)) { log_warn("step_regularize: non-data.frame"); return(tibble()) }
  if (nrow(df) < 2) return(df)
  if (!all(c("svg_x","svg_y") %in% names(df))) stop("step_regularize: missing svg_x/svg_y", call. = FALSE)
  out <- df[1, , drop = FALSE]
  for (i in 2:nrow(df)) {
    prev <- out[nrow(out),]; cur <- df[i,]
    dx <- abs(cur$svg_x - prev$svg_x); dy <- abs(cur$svg_y - prev$svg_y)
    if (dx < tol && dy < tol) next
    if (dx < dy) cur$svg_x <- prev$svg_x else if (dy < dx) cur$svg_y <- prev$svg_y
    out <- bind_rows(out, cur)
  }
  out
}

# ------------------ PANELS & CALIBRATION -------------------
find_axis_lines <- function(lines, len_min = CONFIG$AXIS_MIN_LENGTH) {
  if (!nrow(lines)) return(list(h=tibble(), v=tibble()))
  lines <- lines %>% mutate(len = sqrt((x2-x1)^2 + (y2-y1)^2))
  long  <- lines %>% filter(len >= len_min)
  list(
    h = long %>% filter(abs(y2 - y1) < 1) %>% arrange(desc(len)),
    v = long %>% filter(abs(x2 - x1) < 1) %>% arrange(desc(len))
  )
}
build_panels <- function(axes, margin = CONFIG$PANEL_MARGIN) {
  if (!nrow(axes$h) || !nrow(axes$v)) return(tibble())
  combos <- tidyr::crossing(h = seq_len(nrow(axes$h)), v = seq_len(nrow(axes$v)))
  rects <- pmap_dfr(combos, function(h, v) {
    hl <- axes$h[h,]; vl <- axes$v[v,]
    xmin <- min(vl$x1, vl$x2); xmax <- max(vl$x1, vl$x2)
    y    <- min(hl$y1, hl$y2)
    tibble(xmin = xmin - margin, xmax = xmax + margin,
           ymin = y - CONFIG$PANEL_HEIGHT_ESTIMATE, ymax = y + 5)
  }) %>% distinct() %>% mutate(panel_id = row_number())
  rects
}
calibrate_panel <- function(panel_bbox, texts) {
  tx <- texts %>% filter(
    x >= panel_bbox$xmin - 10, x <= panel_bbox$xmax + 10,
    y >= panel_bbox$ymin - 40, y <= panel_bbox$ymax + 80
  )
  tx_num <- tx %>% mutate(val = suppressWarnings(readr::parse_number(text))) %>% filter(!is.na(val))
  if (nrow(tx_num) < 3) return(NULL)

  y_quant <- quantile(tx_num$y, probs = c(0.2, 0.8), na.rm = TRUE)
  x_ticks <- tx_num %>% filter(y >= y_quant[2]) %>% arrange(x)
  y_ticks <- tx_num %>% filter(x <= (panel_bbox$xmin + 0.25*(panel_bbox$xmax - panel_bbox$xmin))) %>% arrange(desc(y))
  if (nrow(x_ticks) < 2 || nrow(y_ticks) < 2) return(NULL)

  fit_fun <- if (!exists(".NO_MASS", inherits = TRUE)) MASS::rlm else lm
  fit_x <- fit_fun(val ~ x, data = x_ticks)
  fit_y <- fit_fun(val ~ y, data = y_ticks)

  r2 <- function(f) if (inherits(f, "rlm")) NA_real_ else summary(f)$r.squared
  quality <- list(x_r2 = r2(fit_x), y_r2 = r2(fit_y),
                  x_ticks_n = nrow(x_ticks), y_ticks_n = nrow(y_ticks))
  if (!is.na(quality$x_r2) && quality$x_r2 < 0.95 || !is.na(quality$y_r2) && quality$y_r2 < 0.95) {
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
  sig <- ifelse(is.na(dash) || dash == "" || dash == "none", "solid", gsub("[^0-9.]+", "-", dash))
  paste0(tolower(coalesce(stroke, "none")), "|", sig, "|", ifelse(is.na(sw), "NA", format(sw, trim = TRUE)))
}

# Heuristic: legend swatches = short lines/polyline segments near text outside panels
find_legend_map <- function(doc, panels) {
  texts <- svg_text_df(doc)
  lines <- svg_line_df(doc)
  polys <- svg_polyline_df(doc)

  # Legend candidate lines: short-ish
  cand_lines <- tibble(
    x = (lines$x1 + lines$x2)/2,
    y = (lines$y1 + lines$y2)/2,
    len = sqrt((lines$x2 - lines$x1)^2 + (lines$y2 - lines$y1)^2),
    stroke = lines$stroke, dash = lines$dash, sw = lines$sw
  ) %>% filter(len > 0, len <= CONFIG$LEGEND_SWATCH_MAXLEN)

  # Exclude inside any panel
  if (nrow(panels)) {
    inside <- purrr::pmap_lgl(list(cand_lines$x, cand_lines$y),
                              function(xx, yy) any(xx >= panels$xmin & xx <= panels$xmax & yy >= panels$ymin & yy <= panels$ymax))
    cand_lines <- cand_lines[!inside, , drop = FALSE]
  }
  if (!nrow(cand_lines) || !nrow(texts)) return(tibble(style_key = character(), label = character()))

  # Nearest text within D px
  map_tbl <- cand_lines %>%
    mutate(style_key = dash_key(stroke, dash, sw)) %>%
    rowwise() %>%
    mutate(label = {
      d <- sqrt((texts$x - x)^2 + (texts$y - y)^2)
      j <- which.min(d)
      if (length(j) && d[j] <= CONFIG$LEGEND_NEAR_TEXT_D) texts$text[j] else NA_character_
    }) %>% ungroup() %>%
    filter(!is.na(label)) %>%
    group_by(style_key) %>%
    summarise(label = label[which.min(nchar(label))], .groups = "drop")

  map_tbl
}

# --------------------- CURVES & CENSOR TICKS ----------------
svg_path_poly_df <- function(doc) {
  paths <- svg_path_df(doc) %>%
    mutate(pts = map(d, path_to_points)) %>%
    filter(map_lgl(pts, ~ !is.null(.x) && nrow(.x) >= 2)) %>%
    transmute(stroke, fill, sw, dash, pts)

  polyl <- svg_polyline_df(doc) %>%
    mutate(pts = map(points, poly_points_df)) %>%
    transmute(stroke, fill, sw, dash, pts)

  bind_rows(paths, polyl)
}

harvest_curves <- function(doc, panel_bbox) {
  all <- svg_path_poly_df(doc)
  if (!nrow(all)) return(tibble())

  clip_ok <- function(df) df %>% filter(svg_x >= panel_bbox$xmin, svg_x <= panel_bbox$xmax,
                                        svg_y >= panel_bbox$ymin, svg_y <= panel_bbox$ymax)

  curves <- all %>%
    mutate(pts = map(pts, clip_ok),
           pts = map(pts, step_regularize)) %>%
    filter(map_int(pts, nrow) >= 2)
  if (!nrow(curves)) return(tibble())

  curves %>%
    mutate(style_key = dash_key(stroke, dash, sw),
           data = map(pts, ~ .x %>% mutate(idx = row_number()))) %>%
    select(style_key, stroke, dash, sw, data) %>%
    unnest(data) %>%
    group_by(style_key) %>% mutate(curve_id = cur_group_id()) %>% ungroup()
}

# Censor ticks: tiny lines inside panel assigned to nearest curve by distance
detect_censor_ticks <- function(doc, panel_bbox, curve_tbl) {
  if (!nrow(curve_tbl)) return(tibble())
  ln <- svg_line_df(doc)
  if (!nrow(ln)) return(tibble())

  len <- sqrt((ln$x2 - ln$x1)^2 + (ln$y2 - ln$y1)^2)
  midx <- (ln$x1 + ln$x2)/2; midy <- (ln$y1 + ln$y2)/2

  small <- tibble(x = midx, y = midy, len = len, stroke = ln$stroke, dash = ln$dash, sw = ln$sw,
                  x1=ln$x1, y1=ln$y1, x2=ln$x2, y2=ln$y2) %>%
    filter(len >= CONFIG$CENSOR_LEN_MIN, len <= CONFIG$CENSOR_LEN_MAX,
           x >= panel_bbox$xmin, x <= panel_bbox$xmax, y >= panel_bbox$ymin, y <= panel_bbox$ymax)

  if (!nrow(small)) return(tibble())

  # Assign to nearest curve point if within distance threshold
  ct <- small %>% rowwise() %>% mutate(
    nearest = {
      d <- sqrt((curve_tbl$svg_x - x)^2 + (curve_tbl$svg_y - y)^2)
      j <- which.min(d); if (length(j) && d[j] <= CONFIG$CENSOR_DIST_TO_CURVE) j else NA_integer_
    }
  ) %>% ungroup() %>% filter(!is.na(nearest))

  if (!nrow(ct)) return(tibble())

  # Return ticks with inherited curve_id
  tibble(
    svg_x = ct$x, svg_y = ct$y, curve_id = curve_tbl$curve_id[ct$nearest],
    stroke = ct$stroke, dash = ct$dash, sw = ct$sw
  )
}

# ----------------- UNMIX NEAR-OVERLAPPING CURVES -----------
unmix_close_curves <- function(df) {
  if (!nrow(df)) return(df)
  n_per_style <- df %>% count(style_key)
  if (all(n_per_style$n < 2)) return(df)

  split_style <- split(df, df$style_key)
  out <- vector("list", length(split_style)); k <- 0L

  for (st in names(split_style)) {
    g <- split_style[[st]] %>% arrange(svg_x, svg_y)
    x0 <- min(g$svg_x); x1 <- max(g$svg_x)
    brks <- seq(x0, x1 + CONFIG$UNMIX_WINDOW_X, by = CONFIG$UNMIX_WINDOW_X)
    g$bin <- cut(g$svg_x, brks, labels = FALSE, include.lowest = TRUE)

    est_k <- g %>% group_by(bin) %>%
      summarise(kbin = {
        yy <- svg_y
        if (length(yy) < 2) 1L else {
          labs <- integer(length(yy)); cid <- 0L
          for (i in seq_along(yy)) if (labs[i]==0L) {
            cid <- cid + 1L; labs[abs(yy - yy[i]) <= CONFIG$UNMIX_EPS_Y] <- cid
          }
          max(labs)
        }
      }, .groups = "drop") %>% pull(kbin) %>% max(na.rm = TRUE)

    K <- max(1L, min(CONFIG$UNMIX_K_MAX, est_k))
    if (K == 1L) { g$curve_id2 <- 1L; out[[k<-k+1L]] <- g; next }

    g <- g %>% group_by(bin) %>%
      arrange(svg_y, .by_group = TRUE) %>%
      mutate(cls0 = rep(seq_len(min(K, n())), length.out = n())) %>%
      ungroup()

    bins <- sort(unique(g$bin)); g$cls <- g$cls0
    if (length(bins) > 1) {
      prev <- NULL
      for (b in bins) {
        sub <- g[g$bin == b, , drop = FALSE]
        cent <- sub %>% group_by(cls0) %>% summarise(yc = mean(svg_y), .groups="drop")
        if (!is.null(prev)) {
          cm <- outer(cent$yc, prev$yc, function(a,b) abs(a-b))
          match <- apply(cm, 1, which.min)
          map <- setNames(prev$cls0[match], cent$cls0)
          sub$cls <- map[as.character(sub$cls0)]
          sub$cls[is.na(sub$cls)] <- sub$cls0[is.na(sub$cls)]
        } else sub$cls <- sub$cls0
        g$cls[g$bin == b] <- sub$cls
        prev <- sub %>% group_by(cls) %>% summarise(yc = mean(svg_y), cls0 = unique(cls), .groups="drop") %>% rename(cls0 = cls)
      }
    }
    g$curve_id2 <- g$cls
    out[[k<-k+1L]] <- g
  }
  out <- bind_rows(out)
  map_tbl <- out %>% distinct(style_key, curve_id2) %>% arrange(style_key, curve_id2) %>%
    mutate(curve_id2_norm = row_number())
  out %>% left_join(map_tbl, by = c("style_key","curve_id2")) %>%
    mutate(curve_id2 = curve_id2_norm) %>% select(-curve_id2_norm)
}

# ------------------ NUMBER-AT-RISK (NAR) -------------------
extract_nar_pdf <- function(pdf, page, panel_bbox, y_search = CONFIG$NAR_SEARCH_BELOW) {
  txt <- tryCatch(pdftools::pdf_data(pdf)[[page]] %>% as_tibble(), error = function(e) NULL)
  if (!is.null(txt) && nrow(txt)) {
    txt <- txt %>% mutate(x1 = x, y1 = y, x2 = x + width, y2 = y + height)
    band <- txt %>% filter(y1 >= panel_bbox$ymax, y1 <= panel_bbox$ymax + y_search) %>%
      mutate(num = suppressWarnings(readr::parse_number(text))) %>% filter(!is.na(num))
    if (nrow(band) >= 3) {
      band <- band %>% arrange(y1, x1)
      y_cut <- stats::cutree(hclust(dist(band$y1)), h = 3)
      band$y_grp <- y_cut
      return(
        band %>% group_by(y_grp) %>% arrange(x1, .by_group = TRUE) %>%
          mutate(col = row_number()) %>% ungroup() %>% select(y_grp, col, num) %>% arrange(y_grp, col)
      )
    }
  }
  if (!exists(".NO_TESS", inherits = TRUE)) {
    log_debug("NAR OCR fallback p%d", page)
    png_file <- tempfile(fileext = ".png")
    pdftools::pdf_convert(pdf, format = "png", pages = page, filenames = png_file, dpi = 300)
    on.exit(unlink(png_file), add = TRUE)
    txt_ocr <- tryCatch(tesseract::ocr_data(png_file), error = function(e) NULL)
    if (!is.null(txt_ocr) && nrow(txt_ocr)) {
      cand <- txt_ocr %>% filter(confidence > 60, grepl("^[0-9]+$", word)) %>%
        transmute(num = as.integer(word))
      if (nrow(cand) >= 3) { cand$y_grp <- 1L; cand$col <- seq_len(nrow(cand)); return(cand %>% select(y_grp,col,num)) }
    }
  }
  NULL
}

# ----------------------- INSETS & SCORING ------------------
flag_insets <- function(panels_df) {
  if (nrow(panels_df) < 2) return(mutate(panels_df, is_inset = FALSE))
  pd <- mutate(panels_df, area = (xmax - xmin) * (ymax - ymin))
  contains <- function(a,b) (a$xmin >= b$xmin && a$xmax <= b$xmax && a$ymin >= b$ymin && a$ymax <= b$ymax)
  idx <- rep(FALSE, nrow(pd)); main_area <- max(pd$area, na.rm = TRUE)
  for (i in seq_len(nrow(pd))) for (j in seq_len(nrow(pd))) if (i!=j) {
    if (contains(pd[i,], pd[j,]) && pd$area[i] <= CONFIG$INSET_AREA_FRAC * main_area) { idx[i] <- TRUE; break }
  }
  mutate(pd, is_inset = idx)
}

score_panels <- function(calibrations, pdf, page) {
  if (!nrow(calibrations)) return(calibrations)
  y_max <- map_dbl(calibrations$cal, ~ suppressWarnings(max(.x$ticks$y$val, na.rm = TRUE)))
  area  <- (calibrations$xmax - calibrations$xmin) * (calibrations$ymax - calibrations$ymin)
  nar_hit <- pmap_lgl(list(calibrations$xmin, calibrations$xmax, calibrations$ymin, calibrations$ymax),
                      ~ !is.null(extract_nar_pdf(pdf, page, c(xmin=..1,xmax=..2,ymin=..3,ymax=..4))))
  yscore <- -abs(y_max - 1)
  s <- scales::rescale(area) + scales::rescale(yscore) + ifelse(nar_hit, 0.25, 0)
  mutate(calibrations, panel_score = s, y_max = y_max, nar_hit = nar_hit)
}

# ----------------------- PUBLIC API ------------------------
#' Extract Kaplan–Meier curves from a PDF
#' @param pdf path to a PDF
#' @param pages integer vector of pages (NULL = all)
#' @param temp_svg_dir directory for intermediate SVGs
#' @param dpi SVG DPI
#' @return S3 object 'km_pdf_extract'
extract_km_from_pdf <- function(pdf, pages = NULL, temp_svg_dir = tempdir(), dpi = CONFIG$SVG_DPI) {
  if (!is.character(pdf) || length(pdf) != 1) stop("'pdf' must be a single file path", call. = FALSE)
  if (!file.exists(pdf)) stop("PDF file not found: ", pdf, call. = FALSE)
  if (!dir.exists(temp_svg_dir)) { dir.create(temp_svg_dir, recursive = TRUE, showWarnings = FALSE) }
  if (file.access(temp_svg_dir, 2) != 0) stop("Temp dir not writable: ", temp_svg_dir, call. = FALSE)

  svgs <- pdf_to_svg(pdf, pages = pages, outdir = temp_svg_dir, dpi = dpi)
  if (CONFIG$CLEANUP_SVGS) on.exit({ unlink(svgs); log_debug("Cleaned %d temp SVGs", length(svgs)) }, add = TRUE)

  out <- vector("list", length(svgs))
  for (i in seq_along(svgs)) {
    svg_file <- svgs[i]
    pg <- if (is.null(pages)) i else pages[i]
    doc <- tryCatch(read_xml(svg_file), error = function(e) NULL)
    if (is.null(doc)) { log_warn("Failed to parse SVG p%d (%s)", pg, svg_file); next }

    texts <- svg_text_df(doc)
    lines <- svg_line_df(doc)
    axes  <- find_axis_lines(lines)
    panels <- build_panels(axes)

    if (!nrow(panels)) {
      out[[i]] <- list(panels = tibble(page=pg, panel_id=integer(), xmin=numeric(), xmax=numeric(), ymin=numeric(), ymax=numeric()),
                       curves = tibble(), nar = tibble(), ticks = tibble(), legend = tibble())
      next
    }

    calibrations <- panels %>%
      mutate(cal = map(seq_len(n()), ~ calibrate_panel(panels[.x,], texts))) %>%
      mutate(ok = map_lgl(cal, ~ !is.null(.x))) %>% filter(ok) %>% mutate(panel_id = row_number())

    if (!nrow(calibrations)) {
      out[[i]] <- list(panels = panels %>% mutate(page = pg, panel_id = row_number()),
                       curves = tibble(), nar = tibble(), ticks = tibble(), legend = tibble())
      next
    }

    # Inset handling + scoring
    calibrations <- flag_insets(calibrations)
    if (!CONFIG$INSET_KEEP) calibrations <- calibrations %>% filter(!is_inset)
    calibrations <- score_panels(calibrations, pdf, pg)
    if (CONFIG$PANEL_MODE == "largest") {
      calibrations <- calibrations %>% arrange(desc((xmax-xmin)*(ymax-ymin))) %>% slice(1)
    } else if (CONFIG$PANEL_MODE == "bestscore") {
      calibrations <- calibrations %>% arrange(desc(panel_score)) %>% slice(1)
    } # "all" keeps all

    # Legend mapping for the whole page
    legend_map <- find_legend_map(doc, calibrations %>% select(xmin,xmax,ymin,ymax))

    curve_tbls <- pmap_dfr(list(seq_len(nrow(calibrations))), function(k) {
      pb <- calibrations[k, c("xmin","xmax","ymin","ymax")] %>% as.numeric(); names(pb) <- c("xmin","xmax","ymin","ymax")
      hc <- harvest_curves(doc, pb)
      if (!nrow(hc)) return(tibble())

      # Unmix when styles collide
      hc <- unmix_close_curves(hc)
      if ("curve_id2" %in% names(hc)) { hc$curve_id <- hc$curve_id2; hc$curve_id2 <- NULL }

      # map to data space
      xmap <- calibrations$cal[[k]]$x_map; ymap <- calibrations$cal[[k]]$y_map
      hc %>% mutate(panel_id = calibrations$panel_id[k], t = xmap(svg_x), S = ymap(svg_y)) %>%
        mutate(S = pmin(pmax(S, -0.05), 1.05)) %>%
        # attach legend label if style key matches
        left_join(legend_map %>% rename(curve_label = label), by = c("style_key" = "style_key")) %>%
        select(panel_id, curve_id, svg_x, svg_y, t, S, stroke, dash, sw, style_key, curve_label)
    })

    # Censor tick detection per panel
    ticks_tbls <- pmap_dfr(list(seq_len(nrow(calibrations))), function(k) {
      pb <- calibrations[k, c("xmin","xmax","ymin","ymax")] %>% as.numeric(); names(pb) <- c("xmin","xmax","ymin","ymax")
      cur <- curve_tbls %>% filter(panel_id == calibrations$panel_id[k])
      if (!nrow(cur)) return(tibble())
      ct <- detect_censor_ticks(doc, pb, cur %>% select(svg_x, svg_y, curve_id))
      if (!nrow(ct)) return(tibble())
      mutate(ct, panel_id = calibrations$panel_id[k]) %>% select(panel_id, curve_id, svg_x, svg_y)
    })

    # NAR per panel
    nar_list <- pmap_dfr(list(seq_len(nrow(calibrations))), function(k) {
      pb <- calibrations[k, c("xmin","xmax","ymin","ymax")] %>% as.numeric(); names(pb) <- c("xmin","xmax","ymin","ymax")
      nar <- extract_nar_pdf(pdf, page = pg, panel_bbox = pb)
      if (is.null(nar)) nar <- NA
      tibble(page = pg, panel_id = calibrations$panel_id[k], nar = list(nar))
    })

    out[[i]] <- list(
      panels = calibrations %>% mutate(page = pg) %>% select(page, panel_id, xmin, xmax, ymin, ymax, is_inset, panel_score, y_max, nar_hit),
      curves = curve_tbls,
      ticks  = ticks_tbls,
      nar    = nar_list,
      legend = legend_map
    )
  }
  structure(out, class = "km_pdf_extract")
}

km_extract_bind <- function(x) {
  stopifnot(inherits(x, "km_pdf_extract"))
  list(
    panels = map_dfr(x, "panels"),
    curves = map_dfr(x, "curves"),
    ticks  = map_dfr(x, "ticks"),
    nar    = map_dfr(x, "nar"),
    legend = map_dfr(x, "legend")
  )
}

km_autoplot_svg <- function(x, page = 1) {
  stopifnot(inherits(x, "km_pdf_extract"))
  cur <- x[[page]]$curves
  if (is.null(cur) || !nrow(cur)) {
    return(ggplot() + theme_minimal() + ggtitle(sprintf("No curves on page %d", page)))
  }
  ggplot(cur, aes(svg_x, svg_y, group = interaction(panel_id, curve_id), linetype = as.factor(curve_id), color = as.factor(curve_id))) +
    geom_path(alpha = 0.95) +
    scale_y_reverse() +
    theme_minimal() +
    labs(title = sprintf("KM curves (SVG space) - page %d", page), x = "svg_x", y = "svg_y",
         linetype = "curve_id", color = "curve_id")
}

km_export_curves <- function(x, prefix = "km_extract") {
  stopifnot(inherits(x, "km_pdf_extract"))
  allc <- map_dfr(x, "curves")
  if (!nrow(allc)) stop("No curves to export.")
  allc %>%
    group_by(panel_id, curve_id) %>%
    arrange(t, .by_group = TRUE) %>%
    group_walk(~ readr::write_csv(.x, sprintf("%s_panel%02d_curve%02d.csv", prefix, .y$panel_id[1], .y$curve_id[1])))
  invisible(TRUE)
}

# --------------------- SINGLE-PDF PROCESSOR -----------------
process_single_pdf <- function(pdf_path, output_dir, pages = NULL, dpi = CONFIG$SVG_DPI) {
  base_name <- sanitize_basename(pdf_path)
  pdf_outdir <- file.path(output_dir, base_name)
  dir.create(pdf_outdir, showWarnings = FALSE, recursive = TRUE)
  log_info("Processing %s", basename(pdf_path))

  res <- tryCatch({ extract_km_from_pdf(pdf_path, pages = pages, dpi = dpi) },
                  error = function(e) { log_err("Process failed %s: %s", basename(pdf_path), e$message); NULL })
  if (is.null(res) || !length(res)) { log_warn("No KM data found: %s", basename(pdf_path)); return(invisible(FALSE)) }

  # QA plots
  log_info("Saving QA plots ...")
  for (pg in seq_along(res)) {
    p <- km_autoplot_svg(res, page = pg)
    ggsave(filename = file.path(pdf_outdir, sprintf("QA_page_%03d.png", pg)), plot = p, width = 10, height = 8, bg = "white")
  }

  # Export curve CSVs
  log_info("Exporting curve CSVs ...")
  csv_prefix <- file.path(pdf_outdir, base_name)
  ok <- tryCatch({ km_export_curves(res, prefix = csv_prefix); TRUE },
                 error = function(e) { log_err("CSV export failed for %s: %s", basename(pdf_path), e$message); FALSE })

  # Write bound summaries
  if (ok) {
    bound <- km_extract_bind(res)
    write_csv(bound$panels, file.path(pdf_outdir, "panels.csv"))
    if (nrow(bound$curves)) write_csv(bound$curves, file.path(pdf_outdir, "curves_all.csv"))
    if (nrow(bound$ticks))  write_csv(bound$ticks,  file.path(pdf_outdir, "censor_ticks.csv"))
    if (nrow(bound$legend)) write_csv(bound$legend, file.path(pdf_outdir, "legend_map.csv"))
    if (nrow(bound$nar)) {
      purrr::pwalk(bound$nar, function(page, panel_id, nar) {
        if (is.list(nar) && !is.null(nar[[1]]) && !all(is.na(nar[[1]]))) {
          write_csv(nar[[1]], file.path(pdf_outdir, sprintf("nar_page%03d_panel%02d.csv", page, panel_id)))
        }
      })
    }
  }

  log_info("Done: %s", basename(pdf_path))
  invisible(TRUE)
}

# ----------------------- BATCH RUNNER -----------------------
run_batch <- function(input_dir = "papers_to_process",
                      output_dir = "extraction_results",
                      pages = NULL,
                      n_cores = 1,
                      dpi = CONFIG$SVG_DPI) {
  if (!dir.exists(input_dir)) stop("Input dir not found: ", input_dir)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  pdf_files <- list.files(path = input_dir, pattern = "\\.pdf$", full.names = TRUE)
  if (!length(pdf_files)) stop("No PDF files found in: ", input_dir)

  log_info("Found %d PDFs to process in '%s'.", length(pdf_files), input_dir)

  pb <- NULL
  if (!exists(".NO_PROGRESS", inherits = TRUE)) {
    pb <- progress::progress_bar$new(format = "[:bar] :current/:total (:percent) ETA: :eta",
                                     total = length(pdf_files))
  }

  use_parallel <- n_cores > 1 && !exists(".NO_FUTURE", inherits = TRUE) && !exists(".NO_FURRR", inherits = TRUE)
  if (use_parallel) {
    future::plan(future::multisession, workers = n_cores)
    on.exit({ try(future::plan(future::sequential), silent = TRUE) }, add = TRUE)
    log_info("Parallel: %d workers.", future::nbrOfWorkers())
    furrr::future_walk(pdf_files, function(pdf_path) {
      ok <- process_single_pdf(pdf_path, output_dir, pages = pages, dpi = dpi)
      if (!is.null(pb)) pb$tick()
      ok
    }, .options = furrr::furrr_options(seed = TRUE))
  } else {
    for (pdf_path in pdf_files) {
      process_single_pdf(pdf_path, output_dir, pages = pages, dpi = dpi)
      if (!is.null(pb)) pb$tick()
    }
  }

  # Consolidate
  log_info("Consolidating curves_all.csv ...")
  all_curve_files <- list.files(output_dir, "curves_all.csv", recursive = TRUE, full.names = TRUE)
  if (length(all_curve_files)) {
    all_curves <- map_dfr(all_curve_files, function(f) {
      source_pdf <- basename(dirname(f))
      read_csv(f, col_types = cols()) %>% mutate(source_pdf = source_pdf)
    })
    write_csv(all_curves, file.path(output_dir, "_MASTER_CURVES.csv"))
    log_info("Wrote %s", file.path(output_dir, "_MASTER_CURVES.csv"))
  } else log_warn("No curves_all.csv to consolidate.")

  log_info("All files processed.")
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
      "Env:\n",
      "  KM_LOG_LEVEL=DEBUG|INFO|WARN|ERROR  (default: INFO)\n", sep = "")
}

if (identical(environment(), globalenv())) {
  args <- commandArgs(trailingOnly = TRUE)
  if ("--help" %in% args || "-h" %in% args) { print_help(); quit(save="no", status=0) }

  get_arg <- function(flag, default=NULL) { hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^", flag, "="), "", hit[1]) else default }

  input_dir  <- get_arg("--input_dir", "papers_to_process")
  output_dir <- get_arg("--output_dir", "extraction_results")
  pages_arg  <- get_arg("--pages", NULL)
  pages      <- if (!is.null(pages_arg)) as.integer(unlist(strsplit(pages_arg, ","))) else NULL
  ncores_arg <- get_arg("--n_cores", "1")
  dpi_arg    <- get_arg("--dpi", as.character(CONFIG$SVG_DPI))
  panel_mode <- get_arg("--panel_mode", CONFIG$PANEL_MODE)
  keep_ins   <- tolower(get_arg("--keep_insets", ifelse(CONFIG$INSET_KEEP,"true","false"))) == "true"

  if (!dir.exists(input_dir)) { cat("ERROR: Input dir not found: ", input_dir, "\n", sep=""); quit(save="no", status=1) }
  n_cores <- suppressWarnings(as.integer(ncores_arg)); if (is.na(n_cores) || n_cores < 1) n_cores <- 1
  dpi <- suppressWarnings(as.integer(dpi_arg)); if (is.na(dpi) || dpi < 72) dpi <- CONFIG$SVG_DPI

  CONFIG$PANEL_MODE <- panel_mode
  CONFIG$INSET_KEEP <- keep_ins

  if (!interactive()) run_batch(input_dir, output_dir, pages, n_cores, dpi)
}
