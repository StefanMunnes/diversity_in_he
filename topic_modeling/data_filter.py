import polars as pl
import pandas as pd

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


data_clean = pl.read_csv("scraping/data_clean.csv")

print_data_info(data_clean)
# Number of rows: 720546
# Number of unique urls: 68603
# Number of unique domains: 761
# Mean rows per url: 10.5


# 1. filter polars dataframe text elements by diversity keywords

# keywords from Sonia via Mail (2024.10.23 16:42)
keywords = [
    "diversity", "inkluson", "inclusion", "discrimination", "equality", "equal opportunity", "gleichstellung"
    # "divers",
    # "gleichstellung",
    # "equality",
    # "equal opportunity",
    # "equity",
    # "inclus",
    # "inklusion",
    # "chancengleichheit",
    # "frauen",
    # "women",
    # "gender",
    # "geschlecht",
    # "family",
    # "familien"
]

pattern = "|".join([f"(?i){keyword}" for keyword in keywords])

data_keywords = (
    data_clean
    .filter(pl.col("text").str.contains(pattern))
    .filter(~pl.col("text").str.contains("(?i)biodivers"))
    .filter(~pl.col("text").str.contains("(?i)private equity"))
)

print_data_info(data_keywords)
# Number of rows: 30778
# Number of unique urls: 13227
# Number of unique domains: 589
# Mean rows per url: 2.33

data_keywords.write_csv("scraping/data_keywords.csv")


urls_filtered = data_keywords["url"].unique()


# 2. keep all text elements from filteres urls (one part of page with diversity)

urls_filtered = data_keywords["url"].unique()
data_filtered = data_clean.filter(pl.col("url").is_in(urls_filtered))

print_data_info(data_filtered)
# Number of rows: 290808
# Number of unique urls: 13227
# Number of unique domains: 589
# Mean rows per url: 21.99

data_filtered.write_csv("scraping/data_filtered.csv")


# EXIT




data_keywords = data_clean.to_pandas()
data_keywords = data_keywords[
    data_keywords["text"].str.contains("(?![bB]io)[dD]iversit")
]



# filter data by keywords (keep just text elements that contain Diversity)
data_keywords = data_clean.filter(pl.col("text").str.contains(keywords))
data_keywords = data_clean.filter(pl.col("text").str.contains(f"[dD]iversit"))

data_keywords = data_clean.to_pandas()
data_keywords = data_keywords[
    data_keywords["text"].str.contains("(?![bB]io)[dD]iversit")
]

# get unique urls from filtered dataframe and filter dataframe "data" by unique urls
urls_filtered = data_keywords["url"].unique()
data_filtered = data_clean.filter(pl.col("url").is_in(urls_filtered))

# keep just paragraphs
data_filtered = data_filtered.filter(pl.col("tag") == "p")

data_filtered.write_csv("scraping/data_filtered.csv")


data_filtered.group_by("country").count()



keywords = [
    "gleichstellung",
    "equality",
    "inclusion",
    "inklusion",
    "equity",
    "",
]

keywords = [
    "divers",
    "vielfalt",
    "geschlecht",
    "gender",
    "race",
    "rasse",
    "integration",
    "diskriminierung",
    "discrimination",
    "migration",
    "minderheit",
    "minority",
]



keywords = seed_tpcs_list = [
    ["diversity", "diversität", "vielfalt", ""],
    [
        "gender",
        "geschlecht",
        "frau",
        "women",
        "familie",
        "female",
        "leadership",
        "parity",
        "kaskade",
        "mutter",
        "family",
        "families",
        "sexual",
        "parental",
    ],
    [
        "race",
        "ethnic",
        "rassismus",
        "racism",
        "migration",
        "black",
        "bame",
        "minority",
        "color",
        "bipoc",
        "poc",
        "ausland",
    ],
    ["behinde", "ability", "disabled", "neurodivergent", "accommodation"],
    ["international", "global", "south", "cultural", "kultur", "welt", "world"],
    [
        "discrimination",
        "complaint",
        "grievance",
        "sexual",
        "harassment",
        "assault",
        "gewalt",
        "diskriminierung",
    ],
]


# create simple list from nested list
keywords = [item for sublist in keywords for item in sublist]
