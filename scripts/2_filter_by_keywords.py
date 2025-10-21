import polars as pl
import xlsxwriter

# 1. filter all text elements by diviersity keywords -> get unique urls
# 2. keep all text elements from filteres urls
# 3. keep just paragraphs (just for BERT relevant?)


def print_data_info(df: pl.DataFrame) -> None:
    """
    Prints the number of rows, unique urls, and unique domains in the given DataFrame.
    """
    num_rows = df.height
    num_unique_urls = df["url"].n_unique()
    num_unique_domains = df["domain"].n_unique()

    # mean of rows per url (round to 2 decimal places)
    mean_rows_per_url = num_rows / num_unique_urls
    mean_rows_per_url = round(mean_rows_per_url, 2)

    print(f"Number of rows: {num_rows}")
    print(f"Number of unique urls: {num_unique_urls}")
    print(f"Number of unique domains: {num_unique_domains}")
    print(f"Mean rows per url: {mean_rows_per_url}")


data_clean = pl.read_csv("data/data_scraped_all_clean.csv")

print_data_info(data_clean)
# Number of rows: 994223
# Number of unique urls: 100998
# Number of unique domains: 1172
# Mean rows per url: 9.84


# 1. filter polars dataframe text elements by diversity keywords

# keyword list
with open("data/keywords.txt", "r", encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]

pattern = r"\b(" + "|".join(keywords) + r")\b"


data_prepare_filtering = (
    data_clean
    # remove page title rows
    .filter(~pl.col("tag").str.contains(r"title"))
    # keep just unique text elements per url
    .unique(subset=["url", "text"]).sort(["country", "domain", "url", "order"])
    # extract and mark all keywords included in the text elements
    .with_columns(keywords=pl.col("text").str.to_lowercase().str.extract_all(pattern))
    # add indicators for heading and correct order
    .with_columns(
        is_heading=pl.col("tag").str.contains("h[1-6]"),
        is_following=(pl.col("order") - pl.col("order").shift(-1) == -1),
    )
    # add indicator to merge following text when heading with keywords & correct order
    .with_columns(
        merge_following=pl.when(
            pl.col("keywords").list.len() > 0,
            pl.col("is_following"),
            pl.col("is_heading"),
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False)),
        # add following keywords to add later if merged
        keywords_following=pl.col("keywords").shift(-1),
    )
    # merge following text to heading if indicated
    .with_columns(
        text=pl.when(pl.col("merge_following"))
        .then(pl.col("text") + " " + pl.col("text").shift(-1))
        .otherwise(pl.col("text")),
        # add keywords list of added following text
        keywords=pl.when(pl.col("merge_following"))
        .then(pl.concat_list("keywords", "keywords_following"))
        .otherwise(pl.col("keywords")),
        # change heading to false if merged, bc headings beeing removed later
        is_heading=pl.when(pl.col("merge_following"))
        .then(pl.lit(False))
        .otherwise(pl.col("is_heading")),
        # add indicator for merged followed paragraph to be removed
        remove_following=pl.when(pl.col("merge_following").shift(+1))
        .then(pl.lit(True))
        .otherwise(pl.lit(False)),
    )
    # keep just unique keywords (also after merge with followup)
    .with_columns(keywords=pl.col("keywords").list.unique())
)

# write prepared data to excel
countries = {
    "ger": "Germany",
    "usa": "USA",
    "uk": "UK",
    "ind": "India",
}

with xlsxwriter.Workbook("data/data_prepare_filtering.xlsx") as workbook:
    for code, sheet in countries.items():
        (
            data_prepare_filtering.filter(pl.col("country") == code)
            .select(["keywords", "text", "domain"])
            .with_columns(keywords=pl.col("keywords").list.join(", "))
            .write_excel(workbook=workbook, worksheet=sheet)
        )


data_filtered = (
    data_prepare_filtering
    # keep just text elements with keywords, that aren't headings and merged paragraphs
    .filter(
        pl.col("keywords").list.len() >= 2,
        ~pl.col("is_heading"),
        ~pl.col("remove_following"),
    ).select(["keywords", "text", "country", "domain", "url"])
    # cast list of keywords to string
    .with_columns(keywords=pl.col("keywords").list.join(", "))
)

print_data_info(data_filtered)
# Number of rows: 6610 (5840)
# Number of unique urls: 3758 (3367)
# Number of unique domains: 764 (721)
# Mean rows per url: 1.76 (1.73)

data_filtered.write_csv("data/data_filtered.csv")


# export all text with excatlcy one keyword or at least two different for check

with xlsxwriter.Workbook("data/data_filtered.xlsx") as workbook:
    (
        data_filtered.filter(
            pl.col("keywords").str.split(", ").list.len() == 1
        ).write_excel(workbook=workbook, worksheet="keyword_1")
    )
    (
        data_filtered.filter(
            pl.col("keywords").str.split(", ").list.unique().list.len() >= 2
        ).write_excel(workbook=workbook, worksheet="keyword_2+")
    )
