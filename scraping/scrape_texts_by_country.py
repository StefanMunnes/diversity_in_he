import pandas as pd
import pickle
import os
import csv
import requests
from bs4 import BeautifulSoup
import re
from random import sample


# define variables used in the functions
variables = [
    "text",
    "text_length",
    "tag",
    "url",
    "url_redirect",
]  # names for the extracted text dataframe
min_length = 150  # minimum text length of paragraphs
tags_to_keep = ["p", "h1", "h2", "h3"]  # keep just the text elements in this list

sample_num = 10


def write_error_to_csv(url, error, file):
    """
    Writes an error to a csv file.

    Parameters
    ----------
    url : str
        The url that caused the error.
    error : str
        The error message to write.
    file : str
        The path to the csv file to write the url and error to.
    """
    print(error)
    pd.DataFrame({"url": [url], "error": [error]}).to_csv(
        file, mode="a", index=False, header=False
    )


def extract_html_text(url, file_good, file_bad):
    """
    Extracts the main text from a given url, keep just useful text elements and writes to csv.

    Parameters
    ----------
    url : str
        The url to extract the text from.
    file_good : str
        The path to the csv file to write extracted text to.
    file_bad : str
        The path to the csv file to write errors to.

    Returns
    -------
    texts_df : pd.DataFrame
        A DataFrame containing the extracted text.
    error : str or None
        None if no error occurred during extraction.
    """
    try:
        response = requests.get(url)

        response.raise_for_status()

        # Exit with error if redirected to a PDF
        if response.headers["Content-Type"] == "application/pdf":
            write_error_to_csv(url, "Redirected, file_bad to PDF")
            return pd.DataFrame(), "Redirect to PDF"

        # Parse the content of the response with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Get the title and add as first row for this url
        title = soup.title.string if soup.title else "No title"
        title_list = [title, len(title), "title", url, None]
        text_elements = [dict(zip(variables, title_list))]

        # Loop over valid tags, get the text and meta data, and add to the list
        for tag in soup.find_all(tags_to_keep, recursive=True):
            text = tag.get_text(separator=" ", strip=True)
            text_length = len(text)
            if (tag.name == "p" and text_length >= min_length) or (
                re.match(r"^h", tag.name) and text_length > 0
            ):
                # Store the text with its tag, and length as meta information
                results = [text, text_length, tag.name, url, None]
                text_elements.append(dict(zip(variables, results)))

        # Convert the list of dictionaries to a pandas DataFrame
        texts_df = pd.DataFrame(text_elements)

        # Drop duplicates (because of nested structure)
        texts_df = texts_df.drop_duplicates(subset=["text"])

        # Drop rows with headlines that follow other headlines
        texts_df = texts_df[
            ~(
                texts_df["tag"].isin(["h1", "h2", "h3"])
                & texts_df["tag"].shift(+1).isin(["h1", "h2", "h3"])
            )
        ]

        # Add a value for the new url if redirected
        if response.history:
            texts_df["url_redirect"] = response.url

        # Append the DataFrame to the CSV file
        texts_df.to_csv(file_good, mode="a", index=False, header=False)

        return texts_df, None  # No error

    # If error for request, write to csv and return empty dataframe
    except requests.HTTPError as http_err:
        http_err_code = http_err.response.status_code
        write_error_to_csv(url, http_err_code, file_bad)
        return pd.DataFrame(), http_err_code
    except Exception as err:
        write_error_to_csv(url, err, file_bad)
        return pd.DataFrame(), err


def extract_texts_from_urls(country="Germany"):
    """
    Extracts html text from a list of urls, filters out short texts and writes to csv.

    Parameters
    ----------
    country : str, optional
        The country to extract urls from. Defaults to "Germany".

    Returns
    -------
    combined_df : pd.DataFrame
        A DataFrame containing the extracted text.
    errors_df : pd.DataFrame
        A DataFrame containing errors that occurred during extraction.
    """
    # load country specific pkl file from scraping country folder
    file_urls = os.listdir(f"scraping/{country}")
    file_urls = [f for f in file_urls if f.endswith(".pkl")]

    with open(f"scraping/{country}/{file_urls[0]}", "rb") as file:
        urls_raw = pickle.load(file)  # [22021:]

    # Remove hashtags (link to subsections) from urls and remove duplicates
    urls_clean = [re.sub(r"#.*$", "", url) for url in urls_raw]
    urls_clean = list(set(urls_clean))

    urls_test = sample(urls_clean, sample_num)

    # Define and write initial csv files (if not existing yet)
    outfile_good = f"scraping/{country}/scraped_data.csv"

    if not os.path.isfile(outfile_good):
        with open(outfile_good, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(variables)

    outfile_bad = f"scraping/{country}/scraped_data_errors.csv"

    if not os.path.isfile(outfile_bad):
        with open(outfile_bad, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["url", "error"])

    # Check for already done urls (in csv files) and skip them
    urls_good = (
        pd.read_csv(outfile_good).drop_duplicates(subset=["url"])["url"].tolist()
    )
    urls_bad = pd.read_csv(outfile_bad)["url"].tolist()

    urls_done = urls_good + urls_bad

    # Initialize lists to store extracted texts and errors
    all_dfs = []
    errors = []

    # Loop over unique and unscrape urls and write results (texts or error) to csv
    for url in [url for url in urls_test if url not in urls_done]:
        print(url)

        df, error = extract_html_text(url, outfile_good, outfile_bad)

        if not df.empty:
            all_dfs.append(df)
        if error:
            errors.append({"url": url, "error": error})

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    errors_df = pd.DataFrame(errors)

    return combined_df, errors_df


test, test_errors = extract_texts_from_urls()


# TODO indexing urls to save storage space ? necessary ?
# TODO check redirected urls for patterns: just "full" redirects

# IDEA count total length for page, # of headlines, # of paragraphs
# IDEA filter whole page by topic related keywords
