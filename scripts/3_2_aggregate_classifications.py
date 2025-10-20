import polars as pl

# 1. combine language specific classified data
data_results_eng = pl.read_csv("data/classification/data_results_v9_eng.csv")
data_results_ger = pl.read_csv("data/classification/data_results_v9_ger.csv")

data_results = pl.concat([data_results_eng, data_results_ger], how="vertical")

data_results.write_csv("data/data_classified.csv")


# 2. combine classified and filtered data
data_filtered = pl.read_csv("data/data_filtered.csv")

data_aggregated = (
    pl.concat([data_filtered, data_results], how="horizontal")
    .group_by("country", "domain", "token_1")
    .len()
    .pivot(values="len", index=["country", "domain"], on="token_1")
    .fill_null(0)
    .with_columns(
        total=pl.col("Neutral/Irrelevant")
        + pl.col("Collective")
        + pl.col("Individual"),
        total_ind_coll=pl.col("Collective") + pl.col("Individual"),
        ind_coll_prop=pl.when((pl.col("Individual") == 0) & (pl.col("Collective") == 0))
        .then(0)
        .when((pl.col("Individual") == 0) | (pl.col("Collective") == 0))
        .then(1)
        .otherwise(
            pl.col("Individual") / (pl.col("Collective") + pl.col("Individual"))
        ),
    )
    .with_columns(
        ind_coll_prop=pl.when(pl.col("total_ind_coll") <= 1)
        .then(None)
        .otherwise(pl.col("ind_coll_prop"))
    )
)

data_aggregated.write_excel("data/classifications_aggregated.xlsx")
data_aggregated.write_csv("data/classifications_aggregated.csv")
