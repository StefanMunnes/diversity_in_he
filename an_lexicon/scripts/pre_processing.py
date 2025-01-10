import polars as pl
import json
import re
import langdetect
langdetect.DetectorFactory.seed = 0 # set for deterministic results

from pprint import pprint

import nltk
# from nltk.tokenize import MWETokenizer
# nltk.download('punkt')
# nltk.download('stopwords')

from HanTa import HanoverTagger as ht
tagger_en = ht.HanoverTagger('morphmodel_en.pgz')
tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')


data_filtered = pl.read_csv("an_lexicon/data/data_filtered.csv")

# load lexicon (prepared json)
with open("an_lexicon/data/lexicon.json", 'r') as handle:
    lexicon = json.load(handle)

def get_compound_words(dictionary):
    compound_words = []
    for value_list in dictionary.values():
        for phrase in value_list:
            if re.search(r"\b \b", phrase):
                compound_words.append(phrase)
                print(phrase)
    return compound_words


compound_words = get_compound_words(lexicon)


pattern = r'\b(' + '|'.join(compound_words) + r')\b'
regex = re.compile(pattern, re.IGNORECASE)

# Define the replacement function
def combine_compound_tokens(text):
    def replacer(match):
        matched_text = match.group(0)
        # Replace spaces with underscores in the matched text
        return matched_text.replace(' ', '_')
    return regex.sub(replacer, text).lower()


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
    # pattern = r"((\b[A-ZÄÖÜ]+\b[ ]*){4,})"
    # pattern = r"\b[A-ZÄÖÜ]{6,}\b"
    pattern = r'((?:\b[A-Z]+\b\s+){3,})|\b[A-Z]{6,}\b'

    # Capitalize each word of sequence and add enough spaces
    def capitalize(match):
        capitalized = ' '.join(word.capitalize() for word in match.group().split())
        return ' ' + capitalized + ' '

    # Substitute matched pattern with capitalized versions of words
    return re.sub(pattern, capitalize, text)



# function: tokenize and lemmatize based on language
def lemmatize_n_clean(text, language="english"):
    
    tokens = nltk.word_tokenize(text, language = language)

    # use german lemmatizer & stopwords or english for english an other languages
    if language == 'german':
        tagger = tagger_de
        # stopwords = nltk.corpus.stopwords.words('german')
    else:
        tagger = tagger_en
        # stopwords = nltk.corpus.stopwords.words('english')

    lemmata = [
        (token, "") if token.isupper() else tagger.analyze(token) for token in tokens
    ]

    lemmas = [lemma[0] for lemma in lemmata]

    # remove stopwords
    # lemmas_filtered = [lemma for lemma in lemmas if lemma not in stopwords]

    # TODO remove non-valid words (e.g. 12, :), .,!?, (1))
    # TODO remove stop words

    return lemmas

# tokenizer = MWETokenizer([("artificial","intelligence"), ("data","science")], separator=' ')

# preprocess data (detect language, tokenize, lemmatize, compound words)
data_preprocessed = (
  data_filtered
  # .sample(n=60)
  .with_columns(
    language = pl.col("text").map_elements(detect_language, return_dtype=pl.Utf8),
    text = pl.col("text").map_elements(combine_compound_tokens, return_dtype=pl.Utf8)
  )
  # TODO if Other/Error, fall back to most common language for URL
#   .group_by("url", maintain_order=True)
#   .agg([
#     pl.col("language").mode().first().alias("language_pref"),
#     pl.col("language").fill_null(pl.col("language").mode().first())
#   ])
# )
  .with_columns(
    text = pl.col("text").map_elements(capitalize_all_uppercase, return_dtype=pl.Utf8)
  )
  .with_columns(
    tokens = pl
    .struct(["text", "language"])
    .map_elements(lambda x: lemmatize_n_clean(x["text"], x["language"]), return_dtype=pl.List(pl.Utf8))
  )
)

data_preprocessed.write_parquet("an_lexicon/data/data_preprocessed.parquet")

a = data_preprocessed.filter(pl.col("text").str.contains(r"regardless_of"))
