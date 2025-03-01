import polars as pl


# load and prepare data (combine person and country and label categories)


concept_lab = {1: "Individual", 2: "Collective", 3: "Neutral", 4: "Irrelevant"}


df_prsn_ls = []

for prsn in ["SG", "YS"]:

  df_cntry_ls = []

  for cntry in ["usa", "uk"]:
      temp_pl = (
        pl
        .read_excel(f"an_llm/data/handcoding/done/data_filtered_sample_{cntry}_{prsn}.xlsx")
        .select(["concept", "text", "url"])
      )

      df_cntry_ls.append(temp_pl)

  df_cntry = (
    pl
    .concat(df_cntry_ls, how="vertical")
    .with_columns(concept = pl.col("concept").replace_strict(concept_lab))
    .rename({"concept":f"concept_{prsn}"})
  )

  df_prsn_ls.append(df_cntry)

# join list of dataframes by text and url
df_hc = (
  df_prsn_ls[0]
  .join(df_prsn_ls[1], on=["text", "url"], how="left")
)


# load openAI classification and add handcoding
df_ai = pl.read_excel("an_llm/data/data_results_v5_eng.xlsx")

df_combined = (
  pl
  .concat([df_ai, df_hc], how = "horizontal")
  .with_columns(
      match_LLM_SG = pl.col("token_1") == pl.col("concept_SG"),
      match_LLM_YS = pl.col("token_1") == pl.col("concept_YS"),
      match_SG_YS = pl.col("concept_YS") == pl.col("concept_SG")
  )
#   .with_columns(
#       match_all = pl.when(
#           pl.col("token_1").is_not_null() & pl.col("concept_SG").is_not_null() & pl.col("concept_YS").is_not_null()
#       )
#   )
  .select(
    ["token_1", "concept_SG", "concept_YS", "match_SG_YS", "match_LLM_SG", "match_LLM_YS", "text", "url"]
  )
)


df_combined.write_excel("an_llm/data/handcoding/results/data_sample_results_combined_v5_eng.xlsx")


# Define a function to compute match percentage
def compute_match_percentage(df, col1, col2):
    valid_rows = df.filter(df[col1].is_not_null() & df[col2].is_not_null())
    matches = (valid_rows[col1] == valid_rows[col2]).sum()
    return (matches / len(valid_rows)) * 100

# Compute pairwise match percentages
match_token1_conceptSG = compute_match_percentage(df_combined, "token_1", "concept_SG")
match_token1_conceptYS = compute_match_percentage(df_combined, "token_1", "concept_YS")
match_conceptSG_conceptYS = compute_match_percentage(df_combined, "concept_SG", "concept_YS")

# Print results
print(f"token_1 vs concept_SG: {match_token1_conceptSG:.2f}% match")
print(f"token_1 vs concept_YS: {match_token1_conceptYS:.2f}% match")
print(f"concept_SG vs concept_YS: {match_conceptSG_conceptYS:.2f}% match")



# draw 3 scatter plots for each inter-coder pair (dis)agreement of categories
import matplotlib.pyplot as plt

pairs = [("token_1", "concept_SG"), ("token_1", "concept_YS"), ("concept_SG", "concept_YS")]

df_dropped = df_combined.drop_nulls(["token_1", "concept_SG", "concept_YS"])

# TODO color by country

fig, axes = plt.subplots(3, 1, figsize=(15, 5))
for i, (x_rater, y_rater) in enumerate(pairs):
    counts = (
        df_dropped
        .group_by([x_rater, y_rater])
        .agg(pl.len().alias("count"))
        .sort([x_rater, y_rater])
    )

    x_vals = counts[x_rater].to_list()
    y_vals = counts[y_rater].to_list()
    sizes = counts["count"].to_list()

    axes[i].scatter(x_vals, y_vals, s=[s * 10 for s in sizes], alpha=0.6)

    for dot in range(0,counts.shape[0]):
      axes[i].text(
            counts[x_rater][dot],
            counts[y_rater][dot],
            counts["count"][dot],
            va='center', ha='center',
            fontsize=7
      )

    axes[i].set_xlabel(x_rater)
    axes[i].set_ylabel(y_rater)

    match_rate = compute_match_percentage(df_combined, x_rater, y_rater)
    axes[i].set_title(f"{x_rater} vs. {y_rater} {match_rate:.2f}% match")

plt.tight_layout()
plt.show()

plt.savefig("an_llm/data/handcoding/results/plot_scatter_rater_v5_eng.png", dpi=300, bbox_inches='tight')



# frequency table for concept_category variable over country variable
frequency_table = (
    data_results_combined
    .group_by(["country", "token_1"])
    .agg(freq = pl.len())
    # Calculate total counts per country
    .with_columns(
        freq_country = 
        pl.col("freq")
        .sum()
        .over("country")
    )
    # Calculate relative frequency as a proportion
    .with_columns(
        relative_freq = pl.col("freq") / pl.col("freq_country")
    )
    .sort(["country", "token_1"])
    # .select(["country", "freq_country", "concept_category", "freq", "relative_freq"])
)
