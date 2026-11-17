# Export the final CHARLS XGBoost model to a portable, dependency-free website artifact.
# This script does not refit the model and never exports IDs or respondent records.

required_packages <- c("xgboost", "jsonlite")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Missing R packages: ", paste(missing_packages, collapse = ", "))
}

project_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source_dir <- getOption(
  "trajectory_export_source_dir",
  Sys.getenv("TRAJECTORY_EXPORT_SOURCE_DIR", file.path(project_dir, "..", "model-export-source"))
)
source_dir <- normalizePath(source_dir, winslash = "/", mustWork = TRUE)

# Copy these four unchanged analysis outputs to an ASCII-only path when the
# local R runtime cannot read Chinese paths. They are read-only export inputs.
model_file <- file.path(source_dir, "models_results.rds")
preprocess_file <- file.path(source_dir, "preprocessing.rds")
train_file <- file.path(source_dir, "train_baseline.csv")
metrics_file <- file.path(source_dir, "model_metrics.csv")

for (path in c(model_file, preprocess_file, train_file, metrics_file)) {
  if (!file.exists(path)) stop("Required source file is missing: ", path)
}

model_bundle <- readRDS(model_file)
preprocess <- readRDS(preprocess_file)$full
fit <- model_bundle$models$XGBoost
if (!inherits(fit, "xgb.Booster")) stop("The final XGBoost model was not found.")

selected <- preprocess$selected
feature_names <- colnames(preprocess$original)
classes <- paste0("Class", 1:4)
continuous <- intersect(c("hchild", "bmi", "sleep", "age", "total_cognition"), selected)
categorical <- setdiff(selected, continuous)

train <- read.csv(train_file, check.names = FALSE, stringsAsFactors = FALSE)
if (!all(selected %in% names(train))) stop("The baseline training file lacks selected inputs.")

fill <- unlist(preprocess$fill[selected], use.names = TRUE)
levels_map <- lapply(train[categorical], function(x) sort(unique(x[!is.na(x)])))
for (name in selected) train[[name]][is.na(train[[name]])] <- fill[[name]]

encode <- function(data) {
  data <- data[selected]
  for (name in categorical) {
    data[[name]] <- factor(data[[name]], levels = levels_map[[name]])
  }
  matrix_data <- model.matrix(~ ., data = data)[, -1, drop = FALSE]
  if (!identical(colnames(matrix_data), feature_names)) {
    stop("Website encoding columns do not match the fitted model.")
  }
  storage.mode(matrix_data) <- "double"
  matrix_data
}

labels_zh <- c(
  gender = "性别", rural = "居住地", srh = "自评健康", adlab_c = "ADL受损",
  arthre = "关节炎", digeste = "胃病", retire = "是否退休",
  satlife = "生活满意度", pain = "疼痛", sleep = "睡眠时间",
  disability = "是否残疾", fall_down = "跌倒", eyesight_distance = "远视情况",
  eyesight_close = "近视情况", hear = "听力", total_cognition = "认知能力",
  pension = "是否有养老保险", iadl = "IADL受损"
)
labels_en <- c(
  gender = "Gender", rural = "Residence", srh = "Self-rated health",
  adlab_c = "ADL impairment", arthre = "Arthritis",
  digeste = "Stomach or digestive disease", retire = "Retirement status",
  satlife = "Life satisfaction", pain = "Pain", sleep = "Sleep duration",
  disability = "Disability", fall_down = "History of falls",
  eyesight_distance = "Distance vision", eyesight_close = "Near vision",
  hear = "Hearing", total_cognition = "Total cognition score",
  pension = "Pension insurance", iadl = "IADL impairment"
)
units <- c(sleep = "hours/night", total_cognition = "points")
helps_en <- c(
  adlab_c = "Impaired if any difficulty is reported in dressing, bathing, eating, getting in or out of bed, toileting, or continence.",
  iadl = "Impaired if any difficulty is reported in housework, cooking, shopping, taking medication, or managing money. The 2011 baseline definition excludes telephone use.",
  total_cognition = "Baseline total cognition score used in the final prediction model."
)

binary_labels <- list(
  gender = c("0" = "Female", "1" = "Male"),
  rural = c("0" = "Urban", "1" = "Rural"),
  adlab_c = c("0" = "No impairment", "1" = "Impairment"),
  arthre = c("0" = "No", "1" = "Yes"),
  digeste = c("0" = "No", "1" = "Yes"),
  retire = c("0" = "Not retired", "1" = "Retired"),
  pain = c("0" = "No", "1" = "Yes"),
  disability = c("0" = "No", "1" = "Yes"),
  fall_down = c("0" = "No", "1" = "Yes"),
  pension = c("0" = "No", "1" = "Yes"),
  iadl = c("0" = "No impairment", "1" = "Impairment")
)
ordinal_labels <- list(
  srh = c("1" = "Very poor", "2" = "Poor", "3" = "Fair", "4" = "Good", "5" = "Very good"),
  satlife = c("1" = "Not at all satisfied", "2" = "Not very satisfied",
              "3" = "Somewhat satisfied", "4" = "Very satisfied", "5" = "Completely satisfied"),
  eyesight_distance = c("1" = "Poor", "2" = "Fair", "3" = "Good", "4" = "Very good", "5" = "Excellent"),
  eyesight_close = c("1" = "Poor", "2" = "Fair", "3" = "Good", "4" = "Very good", "5" = "Excellent"),
  hear = c("1" = "Poor", "2" = "Fair", "3" = "Good", "4" = "Very good", "5" = "Excellent")
)
category_labels <- c(binary_labels, ordinal_labels)

reference <- as.data.frame(as.list(fill[selected]), check.names = FALSE)
for (name in selected) reference[[name]] <- as.numeric(reference[[name]])
reference_x <- encode(reference)

input_specs <- lapply(selected, function(name) {
  spec <- list(
    name = name,
    label = unname(labels_zh[[name]]),
    label_en = unname(labels_en[[name]]),
    default = as.numeric(fill[[name]])
  )
  if (name %in% continuous) {
    values <- train[[name]]
    spec$type <- "number"
    spec$unit <- unname(units[[name]])
    spec$observed_min <- min(values, na.rm = TRUE)
    spec$observed_max <- max(values, na.rm = TRUE)
  } else {
    spec$type <- "category"
    mapping <- category_labels[[name]]
    codes <- as.character(levels_map[[name]])
    if (is.null(mapping) || !all(codes %in% names(mapping))) {
      stop("Missing category labels for ", name)
    }
    spec$default <- as.character(fill[[name]])
    spec$options <- lapply(codes, function(code) {
      list(value = code, label_en = unname(mapping[[code]]))
    })
  }
  if (name %in% names(helps_en)) spec$help_en <- unname(helps_en[[name]])
  spec
})

origin_of <- function(feature) {
  hits <- selected[feature == selected | startsWith(feature, selected)]
  if (!length(hits)) stop("Cannot map encoded feature: ", feature)
  hits[[which.max(nchar(hits))]]
}
origins <- vapply(feature_names, origin_of, character(1))
feature_specs <- lapply(seq_along(feature_names), function(index) {
  feature <- feature_names[[index]]
  original <- origins[[index]]
  spec <- list(name = feature, original = original)
  if (original %in% continuous) {
    spec$type <- "number"
  } else {
    codes <- as.character(levels_map[[original]])
    values <- vapply(codes, function(code) {
      trial <- reference
      trial[[original]] <- as.numeric(code)
      as.numeric(encode(trial)[1, index])
    }, numeric(1))
    spec$type <- "category"
    spec$values <- as.list(setNames(values, codes))
  }
  spec
})

# Parse the text dump so Python inference does not need an R or XGBoost runtime.
tree_text <- xgboost::xgb.dump(fit, dump_format = "json")
trees <- if (length(tree_text) == 1L) {
  jsonlite::fromJSON(tree_text[[1]], simplifyVector = FALSE)
} else {
  lapply(tree_text, function(text) jsonlite::fromJSON(text, simplifyVector = FALSE)[[1]])
}
if (length(trees) != 800L) stop("Expected 800 trees (200 rounds x 4 classes).")

tree_leaf <- function(node, row) {
  if (!is.null(node$leaf)) return(as.numeric(node$leaf))
  value <- row[[match(node$split, feature_names)]]
  target <- if (value < as.numeric(node$split_condition)) node$yes else node$no
  child_ids <- vapply(node$children, function(child) as.integer(child$nodeid), integer(1))
  child <- node$children[[match(as.integer(target), child_ids)]]
  tree_leaf(child, row)
}

tree_margins <- function(x) {
  margins <- matrix(0, nrow(x), 4L)
  for (tree_index in seq_along(trees)) {
    class_index <- ((tree_index - 1L) %% 4L) + 1L
    margins[, class_index] <- margins[, class_index] +
      apply(x, 1, function(row) tree_leaf(trees[[tree_index]], row))
  }
  margins
}

# Synthetic verification profiles cover every category and both numeric ranges.
case_rows <- list(reference)
for (spec in input_specs) {
  name <- spec$name
  values <- if (spec$type == "category") {
    vapply(spec$options, function(option) as.numeric(option$value), numeric(1))
  } else {
    unique(c(spec$observed_min, spec$default, spec$observed_max))
  }
  for (value in values) {
    trial <- reference
    trial[[name]] <- value
    case_rows[[length(case_rows) + 1L]] <- trial
  }
}
cases <- do.call(rbind, case_rows)
case_x <- encode(cases)
dmatrix <- xgboost::xgb.DMatrix(case_x)
native_probability <- predict(fit, dmatrix)
if (!is.matrix(native_probability)) native_probability <- matrix(native_probability, ncol = 4L, byrow = TRUE)
native_margin <- predict(fit, dmatrix, outputmargin = TRUE)
if (!is.matrix(native_margin)) native_margin <- matrix(native_margin, ncol = 4L, byrow = TRUE)

leaf_margin <- tree_margins(case_x)
base_matrix <- native_margin - leaf_margin
base_margin <- colMeans(base_matrix)
base_error <- max(abs(sweep(base_matrix, 2, base_margin, "-")))
if (!is.finite(base_error) || base_error > 1e-6) stop("XGBoost base-margin recovery failed.")

portable_margin <- sweep(leaf_margin, 2, base_margin, "+")
portable_probability <- t(apply(portable_margin, 1, function(row) {
  values <- exp(row - max(row))
  values / sum(values)
}))
max_probability_error <- max(abs(native_probability - portable_probability))
if (!is.finite(max_probability_error) || max_probability_error > 1e-6) {
  stop("Portable XGBoost probabilities do not match R. Max error: ", max_probability_error)
}

verification_cases <- lapply(seq_len(nrow(cases)), function(index) {
  raw <- lapply(selected, function(name) {
    if (name %in% continuous) as.numeric(cases[[name]][index]) else as.character(cases[[name]][index])
  })
  names(raw) <- selected
  list(
    inputs = raw,
    encoded = as.list(unname(case_x[index, ])),
    probabilities = as.list(unname(native_probability[index, ]))
  )
})

metrics <- read.csv(metrics_file, check.names = FALSE, stringsAsFactors = FALSE)
metrics <- metrics[metrics$model == "XGBoost" & metrics$dataset %in% c("train", "test"), ]

artifact <- list(
  format = "charls-xgboost-v1",
  model_type = "XGBoost multiclass classifier",
  created_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  classes = as.list(classes),
  class_labels = as.list(setNames(c(
    "G1: Persistent low depressive symptoms",
    "G2: Increasing depressive symptoms",
    "G3: Decreasing-then-increasing depressive symptoms",
    "G4: Persistent high depressive symptoms"
  ), classes)),
  inputs = input_specs,
  features = feature_specs,
  base_margin = as.list(unname(base_margin)),
  trees = trees,
  n_trees = length(trees),
  trees_per_class = length(trees) / 4L,
  training = list(
    max_depth = 3L, min_child_weight = 20, lambda = 10, alpha = 1,
    eta = 0.03, nrounds = 200L, subsample = 0.7,
    colsample_bytree = 0.7, gamma = 0.1, class_weight_power = 0.5
  ),
  development_n = nrow(preprocess$original),
  validation_n = nrow(preprocess$validation),
  metrics = lapply(seq_len(nrow(metrics)), function(index) as.list(metrics[index, , drop = FALSE])),
  verification_cases = verification_cases,
  parity = list(
    case_count = nrow(cases),
    base_margin_error = base_error,
    max_probability_error = max_probability_error,
    full_development_encoding_rows = nrow(preprocess$original)
  ),
  notes = list(
    predictor_count = length(selected),
    missing = "Website inputs are complete; training-set medians and modes are shown as defaults.",
    adl = "Binary: any difficulty in one of six ADL activities is impairment.",
    iadl = "Binary: any difficulty in one of five baseline IADL activities is impairment; telephone use is excluded.",
    validation = "Random 70:30 internal validation; research use only."
  )
)

dir.create(file.path(project_dir, "artifacts"), recursive = TRUE, showWarnings = FALSE)
artifact_path <- file.path(project_dir, "artifacts", "model.json.gz")
connection <- gzfile(artifact_path, open = "wb")
tryCatch(
  jsonlite::write_json(
    artifact, connection, auto_unbox = TRUE, digits = NA,
    na = "null", null = "null", matrix = "rowmajor"
  ),
  finally = close(connection)
)

summary <- list(
  format = artifact$format,
  created_at = artifact$created_at,
  model_type = artifact$model_type,
  inputs = as.list(selected),
  encoded_feature_count = length(feature_names),
  n_trees = length(trees),
  development_n = artifact$development_n,
  validation_n = artifact$validation_n,
  training = artifact$training,
  parity = artifact$parity,
  class_labels = artifact$class_labels,
  metrics = artifact$metrics,
  definitions = list(adl = artifact$notes$adl, iadl = artifact$notes$iadl)
)
jsonlite::write_json(
  summary, file.path(project_dir, "model_info.json"),
  auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null"
)

cat("Exported final XGBoost website model:\n", normalizePath(artifact_path, winslash = "/"), "\n", sep = "")
cat("Raw inputs:", length(selected), " Encoded features:", length(feature_names),
    " Trees:", length(trees), "\n")
cat("Verification profiles:", nrow(cases),
    " Max probability error:", format(max_probability_error, scientific = TRUE), "\n")
