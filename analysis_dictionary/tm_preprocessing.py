import pandas as pd

pd.options.mode.chained_assignment = None

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

from gensim.utils import simple_preprocess




# Load data
data = pd.read_csv("analysis_dictionary/data_keywords.csv")


# Keep just necessary columns
data = data[['text', 'country']]


##################


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from gensim.utils import simple_preprocess

# Ensure necessary NLTK data packages are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Define the mapping of countries to their languages
country_language_map = {
    'usa': 'english',
    'india': 'english',  # Replace with 'hindi' if processing Hindi text
    'uk': 'english',
    'germany': 'german',
    # Add more countries and their languages here
}

# Function to get the language-specific tools
def get_language_tools(language):
    # Initialize stemmer
    try:
        stemmer = SnowballStemmer(language=language)
    except ValueError:
        stemmer = None
        print(f"No stemmer available for language: {language}")
    # Initialize stopwords
    try:
        stop_words = set(stopwords.words(language))
    except OSError:
        stop_words = set()
        print(f"No stopwords available for language: {language}")
    # Initialize lemmatizer (only for English)
    if language == 'english':
        lemmatizer = WordNetLemmatizer()
    else:
        lemmatizer = None
    return stemmer, lemmatizer, stop_words

# Function to preprocess text
def preprocess_text(text, stop_words, stemmer, lemmatizer=None):
    # Tokenize and lower
    tokens = simple_preprocess(text, deacc=True)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Stem tokens if stemmer is available
    if stemmer is not None:
        tokens = [stemmer.stem(token) for token in tokens]
    # Lemmatize tokens if lemmatizer is available
    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Assuming 'data' is your DataFrame
# Keep just necessary columns
data = data[['text', 'country']]

# Combine the text per country
data = data.groupby('country').agg({'text': ' '.join}).reset_index()

# Loop over each country and process its text
for country, language in country_language_map.items():
    # Select data for the current country
    data_country = data[data['country'] == country].copy()
    
    # Check if there is data for the country
    if data_country.empty:
        print(f"No data for country: {country}")
        continue
    
    # Get language-specific tools
    stemmer, lemmatizer, stop_words = get_language_tools(language)
    
    # Apply the preprocessing function
    data_country['tokens'] = data_country['text'].apply(
        preprocess_text, args=(stop_words, stemmer, lemmatizer)
    )
    
    # Explode the tokens into individual rows
    tokens = data_country['tokens'].explode()
    
    # Count unique tokens
    unique_counts = tokens.value_counts().reset_index()
    unique_counts.columns = ['token', 'count']
    
    # Save the token counts to an Excel file
    output_filename = f'analysis_dictionary/tokens_count_{country}.xlsx'
    unique_counts.to_excel(output_filename, index=False)

###############












# combine the text per URL to one text (keep just unique urls)
aggregation_functions = {
  'text': ' '.join
}

data = data.groupby('country').agg(aggregation_functions).reset_index()

# Select correct country
data_usa = data[data['country'] == 'usa']
data_ger = data[data['country'] == 'ger']

# Prepare preprocessing function (define stemmer and lemmatizer)
snowball_usa = SnowballStemmer(language='english')
snowball_ger = SnowballStemmer(language='german')
lemmatizer = WordNetLemmatizer()

stopwords_eng = set(stopwords.words('english'))
stopwords_ger = set(stopwords.words('german'))

# Function to preprocess text
def preprocess_text(text, stopwords=stopwords_eng):
    # Tokenize and lower
    tokens = simple_preprocess(text, deacc=True)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords_ger]
    # Stem
    stemmed_tokens = [snowball_ger.stem(token) for token in tokens]
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(stemmed_tokens) for stemmed_tokens in tokens]

    return lemmatized_tokens

# Apply the preprocessing function
data_usa['tokens'] = data_usa['text'].apply(preprocess_text)
data_ger['tokens'] = data_ger['text'].apply(preprocess_text)

tokens = data_usa['tokens'].explode()
tokens = data_ger['tokens'].explode()
unique_counts = tokens.value_counts().reset_index()

unique_counts.to_excel('analysis_dictionary/tokens_count_ger.xlsx', index=False)


# write dataframe data_usa to file csv
data_usa.to_csv("topic_modeling/data_lda_preprocessed_usa.csv", index=False, encoding='utf-8')



https://realpython.com/python-nltk-sentiment-analysis/

text = nltk.Text(nltk.corpus.state_union.words())
text.concordance("america", lines=5)
