import polars as pl
import json
import re

# TODO check for to_be

data_preprocessed = pl.read_parquet("an_lexicon/data/data_preprocessed.parquet")

data_eng = data_preprocessed.filter(pl.col("language") == "english")

# frequencie table of country variable
data_eng["country"].value_counts()

# load lexicon (prepared json)
with open("an_lexicon/data/lexicon.json", 'r') as handle:
    lexicon = json.load(handle)

# combine compound words
for key, value in lexicon.items():
    for n, phrase in enumerate(value):
        lexicon[key][n] =  re.sub(r" ", "_", phrase)


# function: lookup lexicon
def lookup_lexicon(tokens, lexicon_words):
    return [word for word in tokens if word in lexicon_words]


concept_lab = {1: "Individual", 2: "Hybrid", 3: "Collective", 4: "Neither"}


data_lexicon_lookedup = (
    data_eng
    # lookup lexicon concepts
    .with_columns(
        individual = pl.col("tokens").map_elements(lambda x: lookup_lexicon(x, lexicon["individual"]), return_dtype=pl.List(pl.Utf8)),
        collective = pl.col("tokens").map_elements(lambda x: lookup_lexicon(x, lexicon["collective"]), return_dtype=pl.List(pl.Utf8))
    )
    # count occurence of concept tokens
    .with_columns(
        individual_count = pl.col("individual").list.len(),
        collective_count = pl.col("collective").list.len()
    )
    .with_columns(
        lexicon_count = pl.col("individual_count") + pl.col("collective_count")
    )
    # calculate proportion of occurence of concepts and normalize between 0 and 1
    .with_columns(
        individual_prop = pl.col("individual_count") / pl.col("lexicon_count"),
        collective_prop = pl.col("collective_count") / pl.col("lexicon_count")
    )
    # define categories of concepts
    .with_columns(
        concept_category = 
        pl.when(pl.col("lexicon_count") < 2).then(4)
        .when(pl.col("individual_prop") >= 0.6).then(1)
        .when(pl.col("individual_prop").is_between(0.4, 0.6, closed="none")).then(2)
        .when(pl.col("individual_prop") <= 0.4).then(3)
        .otherwise(None)
        .replace_strict(concept_lab)
    )
)


data_lexicon_lookedup.write_parquet("an_lexicon/data/data_lexicon_lookedup.parquet")


# frequency table for concept_category variable over country variable
frequency_table = (
    data_lexicon_lookedup
    .group_by(["country", "concept_category"])
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
    .sort(["country", "concept_category"])
    .select(["country", "freq_country", "concept_category", "freq", "relative_freq"])
)

pl.Config.set_tbl_rows(100) 
print(frequency_table)

frequency_table.write_excel("an_lexicon/output/concept_category_freq_table.xlsx")


# create flat (unnested) dataframe for export and inspection and creation of network plot
data_lexicon_lookedup_flat = data_lexicon_lookedup.with_columns(
    tokens = pl.col("tokens").map_elements(lambda x: " ".join(x), return_dtype=pl.Utf8),
    individual = pl.col("individual").map_elements(lambda x: ", ".join(x), return_dtype=pl.Utf8),
    collective = pl.col("collective").map_elements(lambda x: ", ".join(x), return_dtype=pl.Utf8)
).select(
    ["text", "tokens", "keywords", "concept_category", "individual", "collective",
    "individual_count", "collective_count", "lexicon_count",
    "country", "url"]
)

data_lexicon_lookedup_flat.write_excel("an_lexicon/data/data_lexicon_lookedup_flat.xlsx")

data_lexicon_lookedup_flat.write_csv("an_lexicon/data/data_lexicon_lookedup_flat.csv")
