required <- c("arules", "randomForest", "pROC", "jsonlite", "ggplot2",
              "testthat", "rmarkdown", "knitr")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  message("All analysis, testing, and document packages are already installed.")
}
