import polars as pl
import tldextract

# TODO list of URLs: remove ?.*
# TODO list of URLs: keep ^http

# TODO texts: remove Cookies
# TODO texts: remove No Title

# helper function: extract the domain from a given url
def extract_domain(url):
    extracted = tldextract.extract(url)
    return f"{extracted.domain}.{extracted.suffix}"


# 1. load and prepare data

# 1.1 load data & add country column
data_ger = pl.scan_csv("scraping/Germany/scraped_data.csv").with_columns(
    pl.lit("ger").alias("country")
)
data_usa = pl.scan_csv("scraping/USA/scraped_data.csv").with_columns(
    pl.lit("usa").alias("country")
)

querry = (
    pl
    # 1. combine the dataframes (append rows)
    .concat([data_ger, data_usa])
    # 2. filter all text elements (keep just valid urls; remove cookies & no title)
    .filter(pl.col("url").str.contains(r"^http"))
    .filter(~pl.col("text").str.contains(r"(?i)cookies"))
    .filter(~pl.col("text").str.contains(r"No title"))
    # 3. add column: # of words
    .with_columns(
        pl.col("text")
        .str.split(" ")
        .map_elements(lambda lst: len(lst), return_dtype=pl.Int64)
        .alias("text_words")
    )
    # 4. add column: domain
    .with_columns(
        pl.when(pl.col("url_redirect").is_not_null())
        .then(pl.col("url_redirect").map_elements(extract_domain, return_dtype=pl.Utf8))
        .otherwise(pl.col("url").map_elements(extract_domain, return_dtype=pl.Utf8))
        .alias("domain")
    )
    # 5. add order of text elements per url
    .with_columns(
        (pl.cum_count('text').over('url') + 1).alias('order')
    )
    # 6. keep just unique text elements per domain
    .unique(subset=["domain", "text"])
    # 7. restore order of text elements per url
    .sort(["country", "domain", "url", "order"])
    # 8. remove urls containing "datenschutz"
    .filter(~pl.col("url").str.contains(r"(?i)datenschutz"))
    # 9. remove all text elements with text_length greater than 4000 characters
    .filter(pl.col('text_length') < 4000)
)

data_clean = querry.collect()

data_clean.write_csv("scraping/data_clean.csv")
