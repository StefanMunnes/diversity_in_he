import polars as pl
import tldextract


def extract_domain(url):
    extracted = tldextract.extract(url)
    return f"{extracted.domain}.{extracted.suffix}"


# define names
col_names_ger = [
    "name",
    "url",
    "elite",
    "founding_year",
    "student_size_n",
    "student_size_correct",
    "region",
    "type",
    "international_per",
    "notes",
]
col_names_usa = [
    "name",
    "elite",
    "type",
    "student_size_n",
    "founding_year",
    "region",
    "url",
    "international_per",
    "notes",
]
col_names_uk = [
    "name",
    "url",
    "elite",
    "founding_year",
    "student_size_n",
    "region",
    "type",
    "international_per",
    "notes",
]
col_names_ind = [
    "name",
    "founding_year",
    "student_size_n",
    "url",
    "elite",
    "type",
    "region",
]

# map country code to its column names
col_map = {
    "ger": col_names_ger,
    "usa": col_names_usa,
    "uk": col_names_uk,
    "ind": col_names_ind,
}


df_list = []

for country in ["ger", "usa", "uk", "ind"]:
    print(f"Country: {country}")
    cols = col_map[country]

    df_tmp = (
        pl.read_excel(f"data/uni_infos/university_infos_{country}.xlsx")
        .pipe(lambda d: d.rename({old: new for old, new in zip(d.columns, cols)}))
        .with_columns(
            domain=pl.col("url").map_elements(extract_domain, return_dtype=pl.Utf8),
            country=pl.lit(country),
            elite=(
                pl.col("elite")
                .cast(str)
                .fill_null("None")
                .str.strip_chars()
                .str.to_lowercase()
                .is_in(("elite", "1", "true", "yes"))
            ),
            private=(
                pl.col("type")
                .cast(str)
                .str.strip_chars()
                .str.to_lowercase()
                .is_in(
                    (
                        "private",
                        "true",
                        "kirchlich",
                        "kirchlich / private",
                        "private / kirchlich",
                    )
                )
            ),
            old=pl.col("founding_year") < 1960,
            international_per=(
                pl.col("international_per")
                if "international_per" in cols
                else pl.lit(None, dtype=pl.Float64)
            ),
            student_size_n=pl.col("student_size_n").cast(pl.Int64),
        )
        .with_columns(
            international_per_above_50=(
                pl.col("international_per") > pl.col("international_per").median()
            ),
            student_size_above_50=(
                pl.col("student_size_n") > pl.col("student_size_n").median()
            ),
        )
        .select(
            [
                "country",
                "name",
                "url",
                "domain",
                "elite",
                "private",
                "old",
                "student_size_n",
                "international_per",
                "student_size_above_50",
                "international_per_above_50",
            ]
        )
    )

    df_list.append(df_tmp)

data = pl.concat(df_list, how="vertical")


# Step 4) Add global rank
# data_rank = pl.read_excel("data/uni_infos/university_rank.xlsx").select(
#     ["domain", "global_rank"]
# )

# data = data.join(data_rank, on=["country", "domain"], how="left").with_columns(
#     rank=pl.when(pl.col("global_rank").str.to_integer(strict=False) <= 100)
#     .then(pl.lit("competitors"))
#     .when(
#         (pl.col("global_rank").str.to_integer(strict=False) >= 101)
#         & (pl.col("global_rank").str.to_integer(strict=False) <= 200)
#     )
#     .then(pl.lit("competitors"))
#     .when(
#         pl.col("global_rank").is_in(
#             [
#                 "201–250",
#                 "251–300",
#                 "301–350",
#                 "351–400",
#                 "401–500",
#                 "501–600",
#             ]
#         )
#     )
#     .then(pl.lit("aspirants"))
#     .otherwise(pl.lit("non competitors")),
# )


# Step 6) Export
data.write_csv("data/uni_infos/university_infos.csv")
