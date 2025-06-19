list.files(
  "an_llm/data/handcoding/",
  pattern = "data_filtered_sample",
  full.names = TRUE
) |>
  lapply(function(file) {
    print(file)
    readxl::read_xlsx(file) |> mutate(file = file)
  }) |>
  bind_rows() |>
  write.csv("an_llm/data/handcoding/data_filtered_sample.csv")

library(dplyr)
library(stringr)


data_eng <- readxl::read_xlsx(
  "an_llm/data/handcoding/data_sample_results_combined_v7_eng.xlsx"
)
data_ger <- readxl::read_xlsx(
  "an_llm/data/handcoding/data_sample_results_combined_v7_ger.xlsx"
)

data <- bind_rows(data_eng, data_ger, .id = "language") |>
  mutate(
    language = case_match(
      language,
      "1" ~ "eng",
      "2" ~ "ger"
    ),
    text = str_to_lower(text)
  ) |>
  select(c(language, text, LLM, SG, YS)) |>
  tidyr::pivot_longer(
    cols = c("LLM", "SG", "YS"),
    names_to = "coder",
    values_to = "concept"
  )

saveRDS(
  data,
  "an_llm/data/handcoding/shiny_data_sample_results_combined_v7.rds"
)
