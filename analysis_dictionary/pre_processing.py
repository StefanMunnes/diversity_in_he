
import polars as pl
import re
import langdetect
langdetect.DetectorFactory.seed = 0 # set for deterministic results

from pprint import pprint

import nltk
nltk.download('punkt')

from HanTa import HanoverTagger as ht
tagger_en = ht.HanoverTagger('morphmodel_en.pgz')
tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')


data_clean = pl.read_csv("scraping/data_clean.csv")

# function: detect language (also english pages on german URLs) for lemmatization
def detect_language(text):
    try:
        lang = langdetect.detect(text)
        if lang == "en":
            return "english"
        elif lang == "de":
            return "german"
        else:
            return 'Other'
    except:
        return 'Error'

# function: replace uppercase sequences with first-letter capitalized words
def capitalize_all_uppercase(text):
    # Regular expression pattern: matches sequences of 4 or more uppercase words
    pattern = r"((\b[A-ZÄÖÜ]+\b[ ,]*){4,})"

    def capitalize(match):
        capitalized = ' '.join(word.capitalize() for word in match.group().split())
        return ' ' + capitalized + ' '

    # Substitute matched pattern with capitalized versions of words
    return re.sub(pattern, capitalize, text)

# function: tokenize and lemmatize based on language
def lemmatize(text, language="english"):

    tokens = nltk.word_tokenize(text, language = language)

    # use german tagger and lemmatizer or english for english an other languages
    tagger = tagger_de if language == 'german' else tagger_en

    lemmata = [
        (token, "") if token.isupper() else tagger.analyze(token) for token in tokens
    ]

    lemmas = [lemma[0] for lemma in lemmata]

    return lemmas


# sample of polars dataframe
test = (
  data_clean
  .sample(n=30)
  .with_columns(
    language = pl.col("text").map_elements(detect_language, return_dtype=pl.Utf8)
  )
  # TODO if Other/Error, fall back to most common language for URL
  .with_columns(
    text = pl.col("text").map_elements(capitalize_all_uppercase, return_dtype=pl.Utf8)
  )
  .with_columns(
    tokens = pl
    .struct(["text", "language"])
    .map_elements(lambda x: lemmatize(x["text"], x["language"]), return_dtype=pl.List(pl.Utf8))
  )
)
