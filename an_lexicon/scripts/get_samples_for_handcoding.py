import polars as pl


# load filtered data to draw sample for goldstandard
data_filtered = pl.read_csv("an_lexicon/data/data_filtered.csv")

# for-loop to get sample for each country
for cntry in ["ger", "usa", "uk"]:
  print(f"Country: {cntry}")

  (
  data_filtered
    .filter(pl.col("country") == cntry)
    .sample(n=200, seed = 161)
    .write_excel(f"an_lexicon/data/handcoding/data_filtered_sample_{cntry}.xlsx")
  )
