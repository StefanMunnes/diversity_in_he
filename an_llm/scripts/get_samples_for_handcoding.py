import polars as pl


# load filtered data to draw sample for goldstandard
data_filtered = pl.read_csv("an_lexicon/data/data_filtered.csv")

# for-loop to get sample for each country
for cntry in ["ger", "usa", "uk", "ind"]:
  print(f"Country: {cntry}")

  (
  data_filtered
    .filter(pl.col("country") == cntry)
    .sample(n=200, seed = 161161)
    # add new empty column for handcoding
    .with_columns(concept = pl.lit(""))
    .select(["concept", "text", "keywords", "domain", "url"])
    .write_excel(f"an_llm/data/handcoding/data_filtered_sample_{cntry}.xlsx")
  )
