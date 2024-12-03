import pandas as pd

pd.options.mode.chained_assignment = None

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

from gensim.utils import simple_preprocess




# Load data
data = pd.read_csv('scraping/data_filtered.csv')

# Remove homepage titles
data = data[data['tag'] != 'title']

# Keep just necessary columns
data = data[['url', 'text', 'country']]

# combine the text per URL to one text (keep just unique urls)
aggregation_functions = {
  'text': ' '.join,
  'country': 'first'
}

data = data.groupby('url').agg(aggregation_functions).reset_index()

# Select correct country
data_usa = data[data['country'] == 'usa']


# Prepare preprocessing function (define stemmer and lemmatizer)
snowball = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()

stopwords_eng = set(stopwords.words('english'))
stopwords_ger = set(stopwords.words('german'))

# Function to preprocess text
def preprocess_text(text):
    # Tokenize and lower
    tokens = simple_preprocess(text, deacc=True)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords_eng]
    # Stem
    stemmed_tokens = [snowball.stem(token) for token in tokens]
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(stemmed_tokens) for stemmed_tokens in tokens]

    return lemmatized_tokens

# Apply the preprocessing function
data_usa['tokens'] = data_usa['text'].apply(preprocess_text)


# write dataframe data_usa to file csv
data_usa.to_csv("topic_modeling/data_lda_preprocessed_usa.csv", index=False, encoding='utf-8')
