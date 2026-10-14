# Execute from the repository root. Only GitHub-flavored Markdown is produced.
if (!requireNamespace("knitr", quietly = TRUE)) stop("knitr is required.")
if (!file.exists("outputs/r/metrics.csv")) source("R/run_analysis.R")
root <- normalizePath(".", winslash = "/")
knitr::opts_knit$set(root.dir = root)
knitr::knit("R/analysis.Rmd", output = "outputs/r/analysis.md",
            envir = new.env(parent = globalenv()), quiet = TRUE)
if (requireNamespace("rmarkdown", quietly = TRUE) && rmarkdown::pandoc_available()) {
  rmarkdown::render("R/analysis.Rmd", output_format = "github_document",
                     output_file = "analysis.md",
                     output_dir = file.path(root, "outputs", "r"),
                     knit_root_dir = root, quiet = TRUE,
                     envir = new.env(parent = globalenv()))
} else {
  message("Pandoc unavailable: all Rmd chunks executed with knitr; Markdown saved.")
}
