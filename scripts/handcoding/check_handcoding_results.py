import polars as pl
import itertools
import matplotlib.pyplot as plt

# Define language and version variables
LANGUAGES = ["eng", "ger"]  # ["ger", "eng"]
VERSION = "v8"

# Label mapping for concept categories
concept_lab = {1: "Individual", 2: "Collective", 3: "Neutral/Irrelevant"}


# Utility function to compute match percentage between two columns
def compute_match_percentage(df, col1, col2):
    """
    Computes the matching percentage between two columns in a Polars DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.

    Returns:
        float: The matching percentage.
    """
    valid_rows = df.filter(pl.col(col1).is_not_null() & pl.col(col2).is_not_null())
    if len(valid_rows) == 0:
        return 0.0
    matches = (valid_rows[col1] == valid_rows[col2]).sum()
    return (matches / len(valid_rows)) * 100


# Loop over defined languages to process each dataset
for lang in LANGUAGES:
    print(f"Processing language: {lang}")
    if lang == "ger":

        coders = ["Laura", "Stefan"]

        # For German, load the handcoding sample directly and add language as country column
        df_handcoding = (
            pl.read_excel(
                f"an_llm/testing/handcoding/data_filtered_sample_{lang}.xlsx"
            ).with_columns(pl.col(coders).replace({4: 3}))
            # .with_columns(
            #     Laura=pl.col("Laura").replace_strict(concept_lab),
            #     Stefan=pl.col("Stefan").replace_strict(concept_lab),
            # )
            .with_columns(pl.lit(lang).alias("country"))
        )
        # Load the corresponding OpenAI classification results
        df_ai = pl.read_excel(
            f"an_llm/testing/openai/data_results_{VERSION}_{lang}.xlsx"
        )
    elif lang == "eng":

        coders = ["YS", "SG"]

        # For English, loop over a list of country codes and combine their handcoding samples
        countries = ["usa", "uk", "ind"]
        df_list = []
        for cntry in countries:
            temp_df = (
                pl.read_excel(
                    f"an_llm/testing/handcoding/data_filtered_sample_{cntry}.xlsx"
                )
                .select(coders + ["text", "url"])
                .with_columns(pl.lit(cntry).alias("country"))
            )
            df_list.append(temp_df)

        df_handcoding = (
            pl.concat(df_list, how="vertical")
            .with_columns(pl.col(coders).replace({4: 3}))
            .with_columns(
                [pl.col(var).replace_strict(concept_lab).alias(var) for var in coders]
            )
        )
        # Load the corresponding OpenAI classification results
        df_ai = pl.read_excel(
            f"an_llm/testing/openai/data_results_{VERSION}_{lang}.xlsx"
        )
    else:
        # Skip undefined language block
        continue

    # Combine the AI classification data and handcoding sample horizontally.
    # Create a new column "LLM" as an alias for token_1 and compute match flags.
    df_combined = pl.concat([df_ai, df_handcoding], how="horizontal").with_columns(
        pl.col("token_1").alias("LLM")
    )

    all_raters = ["LLM"] + coders

    # Dynamically create match columns

    match_expressions = []
    for rater1, rater2 in itertools.combinations(all_raters, 2):
        match_col_name = f"match_{rater1}_{rater2}"
        match_expressions.append(
            (pl.col(rater1) == pl.col(rater2)).alias(match_col_name)
        )

    df_combined = df_combined.with_columns(match_expressions).select(
        all_raters
        + [f"match_{r1}_{r2}" for r1, r2 in itertools.combinations(all_raters, 2)]
        + ["text", "url", "country"]
    )

    # Write the combined result to an Excel file
    output_file = (
        f"an_llm/testing/combined/data_sample_results_combined_{VERSION}_{lang}.xlsx"
    )
    df_combined.write_excel(output_file)
    print(f"Combined data written to {output_file}")

    # Compute pairwise match percentages and print the results dynamically
    match_percentages = {}
    for rater1, rater2 in itertools.combinations(all_raters, 2):
        match_percentage = compute_match_percentage(df_combined, rater1, rater2)
        match_percentages[f"{rater1}_vs_{rater2}"] = match_percentage
        print(f"{rater1} vs {rater2}: {match_percentage:.2f}% match")

    # Generate scatter plots for each pair of raters
    pairs = list(itertools.combinations(all_raters, 2))
    df_dropped = df_combined.drop_nulls(all_raters)

    fig, axes = plt.subplots(len(pairs), 1, figsize=(5, 2 * len(pairs)))
    axes = axes.flatten()

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
        for dot in range(counts.shape[0]):
            axes[i].text(
                x_vals[dot],
                y_vals[dot],
                counts["count"][dot],
                va="center",
                ha="center",
                fontsize=7,
            )
        axes[i].set_xlabel(x_rater)
        axes[i].set_ylabel(y_rater)
        match_rate = compute_match_percentage(df_combined, x_rater, y_rater)
        axes[i].set_title(f"{x_rater} vs. {y_rater}: {match_rate:.2f}% match")

    fig.suptitle(f"Scatter plots for Inter-Rater Comparison: {lang}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the scatter plot figure to a PNG file
    plot_file = f"an_llm/testing/combined/plot_scatter_rater_{VERSION}_{lang}.png"

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")

    plt.close()
    print(f"Scatter plot saved to {plot_file}")
