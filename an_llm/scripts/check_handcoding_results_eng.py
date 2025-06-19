import polars as pl


# load and prepare data (combine person and country and label categories)

concept_lab = {1: "Individual", 2: "Collective", 3: "Neutral/Irrelevant"}


df_cntry_ls = []

for cntry in ["usa", "uk", "ind"]:
    temp_pl = (
        pl.read_excel(f"an_llm/data/handcoding/done/data_filtered_sample_{cntry}.xlsx")
        .select(["SG concept", "YS concept", "text", "url"])
        .with_columns(country=pl.lit(cntry))
    )

    df_cntry_ls.append(temp_pl)

df_cntry = (
    pl.concat(df_cntry_ls, how="vertical")
    .with_columns(pl.col(["SG concept", "YS concept"]).replace({4: 3}))
    .with_columns(
        YS=pl.col("YS concept").replace_strict(concept_lab),
        SG=pl.col("SG concept").replace_strict(concept_lab),
    )
)


# load openAI classification and add handcoding
df_ai = pl.read_excel("an_llm/data/data_results_v7_eng.xlsx")

df_combined = (
    pl.concat([df_ai, df_cntry], how="horizontal")
    .with_columns(pl.col("token_1").alias("LLM"))
    .with_columns(
        match_LLM_SG=pl.col("LLM") == pl.col("SG"),
        match_LLM_YS=pl.col("LLM") == pl.col("YS"),
        match_SG_YS=pl.col("YS") == pl.col("SG"),
    )
    .select(
        [
            "LLM",
            "SG",
            "YS",
            "match_SG_YS",
            "match_LLM_SG",
            "match_LLM_YS",
            "text",
            "url",
            "country",
        ]
    )
)

# write combined result data to excel
df_combined.write_excel(
    "an_llm/data/handcoding/data_sample_results_combined_v7_eng.xlsx"
)


# Define a function to compute match percentage
def compute_match_percentage(df, col1, col2):
    valid_rows = df.filter(df[col1].is_not_null() & df[col2].is_not_null())
    matches = (valid_rows[col1] == valid_rows[col2]).sum()
    return (matches / len(valid_rows)) * 100


# Compute pairwise match percentages
match_token1_conceptSG = compute_match_percentage(df_combined, "LLM", "SG")
match_token1_conceptYS = compute_match_percentage(df_combined, "LLM", "YS")
match_conceptSG_conceptYS = compute_match_percentage(df_combined, "SG", "YS")

# Print results
print(f"LLM vs SG: {match_token1_conceptSG:.2f}% match")
print(f"LLM vs YS: {match_token1_conceptYS:.2f}% match")
print(f"SG vs YS: {match_conceptSG_conceptYS:.2f}% match")


# draw 3 scatter plots for each inter-coder pair (dis)agreement of categories
import matplotlib.pyplot as plt

pairs = [
    ("LLM", "SG"),
    ("LLM", "YS"),
    ("SG", "YS"),
]

df_dropped = df_combined.drop_nulls(["LLM", "SG", "YS"])

# TODO color by country

fig, axes = plt.subplots(3, 1, figsize=(15, 5))
for i, (x_rater, y_rater) in enumerate(pairs):
    counts = (
        df_dropped.group_by([x_rater, y_rater])
        .agg(pl.len().alias("count"))
        .sort([x_rater, y_rater])
    )

    x_vals = counts[x_rater].to_list()
    y_vals = counts[y_rater].to_list()
    sizes = counts["count"].to_list()

    axes[i].scatter(x_vals, y_vals, s=[s * 10 for s in sizes], alpha=0.6)

    for dot in range(0, counts.shape[0]):
        axes[i].text(
            counts[x_rater][dot],
            counts[y_rater][dot],
            counts["count"][dot],
            va="center",
            ha="center",
            fontsize=7,
        )

    axes[i].set_xlabel(x_rater)
    axes[i].set_ylabel(y_rater)

    match_rate = compute_match_percentage(df_combined, x_rater, y_rater)
    axes[i].set_title(f"{x_rater} vs. {y_rater} -  {match_rate:.2f}% match")

plt.tight_layout()
plt.show()

plt.savefig(
    "an_llm/data/handcoding/plot_scatter_rater_v7_eng.png", dpi=300, bbox_inches="tight"
)


# frequency table for concept_category variable over country variable
frequency_table = (
    df_combined.group_by(["country", "LLM"])
    .agg(freq=pl.len())
    # Calculate total counts per country
    .with_columns(freq_country=pl.col("freq").sum().over("country"))
    # Calculate relative frequency as a proportion
    .with_columns(relative_freq=pl.col("freq") / pl.col("freq_country"))
    .sort(["country", "LLM"])
    # .select(["country", "freq_country", "concept_category", "freq", "relative_freq"])
)
