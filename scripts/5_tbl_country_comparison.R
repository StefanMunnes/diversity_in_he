library(dplyr)
library(stringr)
library(arsenal)

data <- read.csv("data/uni_classified_infos.csv") |>
  mutate(
    country_name = case_match(
      country,
      "uk" ~ "UK",
      "usa" ~ "USA",
      "india" ~ "India",
      "germany" ~ "Germany",
      .default = country
    ),
    across(c(name, url, elite, private, old), ~ na_if(., "")),
    across(c(urls_filtered_count, texts_filtered_total), ~ na_if(., 0))
  ) |>
  filter(!is.na(name)) # remove Universities without meta information


ctrl <- arsenal::tableby.control(
  test = FALSE,
  digits = 2,
  digits.count = 0,
  cat.simplify = TRUE,
  numeric.stats = c("Nmiss", "meansd", "median", "range"),
  stats.labels = list(myTRUE = "Count (TRUE)")
)

descriptives <- arsenal::tableby(
  country_name ~
    urls_all_count +
      texts_all_total +
      urls_filtered_count +
      texts_filtered_total +
      elite +
      private +
      old +
      student_size_n +
      international_per +
      Collective +
      Individual +
      ind_coll_prop,
  data = data,
  control = ctrl
) |>
  arsenal::set_labels(
    c(
      "urls_all_count" = "All URLs (start)",
      "texts_all_total" = "All Texts (start)",
      "urls_filtered_count" = "Filtered URLs",
      "texts_filtered_total" = "Filtered Texts",
      "elite" = "Elite",
      "private" = "Private",
      "old" = "Before 1960",
      "student_size_n" = "# Students",
      "international_per" = "% Internat. Students",
      "Collective" = "Collective (# Texts)",
      "Individual" = "Individual (# Texts)",
      "ind_coll_prop" = "Proportion Individual"
    )
  ) |>
  arsenal::write2word(
    "tlb_country_comparison.docx",
    quiet = TRUE
  )

# move file to output folder
file.rename("tlb_country_comparison.docx", "output/tlb_country_comparison.docx")
