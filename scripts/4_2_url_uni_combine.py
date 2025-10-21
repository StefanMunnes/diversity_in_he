import pandas as pd
import pickle
import tldextract


# helper function: extract the domain from a given url
def extract_domain(url):
    extracted = tldextract.extract(url)
    return f"{extracted.domain}.{extracted.suffix}"


# 1. get list of all start urls from pickle files and number of texts scraped

# loop to open each pickle file and add to dataframe
data_urls_scraped = pd.DataFrame(columns=["url", "country"])

for country in ["USA", "UK", "India", "Germany"]:

    print(f"Country: {country}")

    path_urls = f"data/scraping/{country}/homepage_urls_{country.lower()}.pkl"
    path_scraped = f"data/scraping/{country}/scraped_data.csv"
    path_errors = f"data/scraping/{country}/scraped_data_errors.csv"

    # Load the pickle file (assuming it's a list of URLs)
    with open(path_urls, "rb") as file:
        urls_all = pickle.load(file)

    # Load csv file with scraped data
    with open(path_scraped, "rb") as file:
        urls_scraped = (
            pd.read_csv(file)
            .groupby("url", as_index=False)
            .agg(texts_all=("text", "count"))
        )

    # Load csv file with errors
    with open(path_errors, "rb") as file:
        urls_errors = pd.read_csv(file)

    # Convert list to DataFrame and add a 'country' column
    df_temp = pd.DataFrame(urls_all, columns=["url"])
    df_temp["country"] = country.lower()
    df_temp["domain"] = df_temp["url"].map(extract_domain)

    # merge scraped by url
    df_temp = pd.merge(df_temp, urls_scraped, how="left", on="url")

    # merge error by url
    df_temp = pd.merge(df_temp, urls_errors, how="left", on="url")
    df_temp["error"] = df_temp["error"].notna()

    # Append to the main DataFrame
    data_urls_scraped = pd.concat([data_urls_scraped, df_temp], ignore_index=True)

data_urls_scraped["country"] = data_urls_scraped["country"].replace(
    {"germany": "ger", "india": "ind"}
)

# 2. load filtered data and add number of texts
data_filtered = pd.read_csv("data/data_filtered.csv")

data_urls_filtered = (
    data_filtered.groupby(["country", "domain", "url"])["text"]
    .count()
    .reset_index(name="texts_filtered")
)

data_urls = pd.merge(
    data_urls_scraped, data_urls_filtered, how="left", on=["country", "domain", "url"]
)


# 3. aggregate data for number of (filtered) texts by university domain
data_uni_aggregated = (
    data_urls.groupby(["country", "domain"])
    .agg(
        urls_all_count=("url", "nunique"),
        urls_filtered_count=("texts_filtered", "count"),
        texts_all_total=("texts_all", "sum"),
        texts_all_mean_per_url=("texts_all", "mean"),
        texts_filtered_total=("texts_filtered", "sum"),
        texts_filtered_mean_per_url=("texts_filtered", "mean"),
    )
    .reset_index()
)


# 4. load and add aggregated classification data
data_aggregated = pd.read_csv("data/classifications_aggregated.csv")

data_uni_classified = data_uni_aggregated.merge(
    data_aggregated, how="left", on=["country", "domain"]
)

data_uni_classified.to_csv("data/uni_classified.csv", index=False)


# 5. load and add university information data
data_uni_infos = pd.read_csv("data/uni_infos/university_infos.csv")

data_uni_classified_infos = data_uni_classified.merge(
    data_uni_infos, how="left", on=["country", "domain"]
).dropna(subset=["name"])

data_uni_classified_infos.to_csv("data/uni_classified_infos.csv", sep = ";", index=False)
data_uni_classified_infos.to_excel("data/uni_classified_infos.xlsx", index=False)
