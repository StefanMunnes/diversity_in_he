import polars as pl
import re

# 1. filter all text elements by diviersity keywords -> get unique urls
# 2. keep all text elements from filteres urls
# 3. keep just paragraphs (just for BERT relevant?)

# TODO Check for Umlaute encoding, e.g. Diversity/Diversit√§t
# https://www.bht-berlin.de/4422

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

# keywords from meeting with Sonia and Yasemin (see mails from 2024.12.02 and 2024.12.03)
keywords = [
    "diversity",
    "diversität",
    "inclusion",
    "inklusion",
    "discrimination",
    "diskriminierung",
    "equality",
    "gleichstellung",
    "equal opportunity",
    "chancengerechtigkeit",
    "chancengleichheit"
]

pattern = r'\b(' + '|'.join(re.escape(word) for word in keywords) + r')\b'

# data_keywords = (
#     data_clean
#     .filter(pl.col("text").str.contains(pattern))
#     .filter(~pl.col("text").str.contains("(?i)biodivers"))
#     # .filter(~pl.col("text").str.contains("(?i)private equity"))
# )

print_data_info(data_keywords)
# Number of rows: 12166
# Number of unique urls: 5415
# Number of unique domains: 547
# Mean rows per url: 2.25

data_keywords.write_csv("scraping/data_keywords.csv")


# add new column, containing the keywords, that are contained in the text column

data_keywords = (
    data_clean
    # extract and mark all keywords included in the text elements 
    .with_columns(
        pl.col('text')
        .str.to_lowercase()
        .str.extract_all(pattern)
        .alias('keywords')
    )
    # keep only rows (text elements) containing at least one keyword
    # for controll: also previous and next text elements (add extra columns)
    .with_columns(
        pl.col('text').shift(1).over('url').alias('before_text'),      # Previous text
        pl.col('text').shift(-1).over('url').alias('following_text'),   # Next text
    )
    .filter(pl.col('keywords').list.len() > 0)
    .select(["country", "url", "keywords", "before_text", "text", "following_text"])
)

# write to excel file
data_keywords.write_excel("topic_modeling/filtered_text_keywords.xlsx")

# TODO: keep unique text elements per url 
# (e.g. dislaimer text repeats per page, https://www.akkon-hochschule.de/stellenangebote)
# Die Akkon Hochschule strebt an, die Diversität an der Hochschule zu erhöhen. Menschen mit Migrationsgeschichte und Rassismuserfahrungen sowie Frauen*, Trans* und nicht-binäre Personen und andere von intersektionaler Diskriminierung bedrohte oder be-troffene Personen werden ausdrücklich ermutigt, sich zu bewerben. Bewerbungen von Menschen mit Behinderung und ihnen Gleichgestellten werden bei vergleichbarer fachlicher und persönlicher Eignung bevorzugt berücksichtigt. Die Akkon Hochschule ist eine familienfreundliche Hochschule.


df_keywords_per_country = (
    data_keywords
    .group_by('country')
    .agg(
        pl.concat_list('keywords').flatten().alias('keywords'),
    )
    .explode('keywords')
    .group_by('country', 'keywords')
    .agg(
        pl.len().alias('count')
    )
    .sort('country', 'count', descending=True)
)


import matplotlib.pyplot as plt

df_keyword_counts_pd = df_keywords_per_country.to_pandas()

def plot_keyword_counts(df, cntry):
    df_cntry = df[df['country'] == cntry].sort_values('count', ascending=True)

    plt.figure(figsize=(6, 4))
    plt.barh(y = df_cntry['keywords'], width = df_cntry['count'], color='skyblue')

    plt.xlabel('Keywords')
    plt.ylabel('Count')
    plt.title(f'Keyword Counts for {cntry}')

    plt.tight_layout()

    plt.show()

# Plot keyword counts for each URL
for cntry in df_keyword_counts_pd['country'].unique():
    plot_keyword_counts(df_keyword_counts_pd, cntry)


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
