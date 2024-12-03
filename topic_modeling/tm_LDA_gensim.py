import pandas as pd
import ast

# !conda install gensim==3.8.3 # downgrade to 3.8.3 for Mallet wrapper
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import little_mallet_wrapper

mallet_path = "C:/Users/munnes/Nextcloud/Projekte/diversity_in_he/mallet-2.0.8/bin/mallet"

import os
os.environ['MALLET_HOME'] = "C:/Users/munnes/Nextcloud/Projekte/diversity_in_he/mallet-2.0.8"

import pprint

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Run in terminal if necessary to get mallet:
# !wget "http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip"
# !unzip mallet-2.0.8.zip



data = pd.read_csv("topic_modeling/data_lda_preprocessed_usa.csv", encoding='utf-8')

# list as string to list of strings
data['tokens'] = data['tokens'].apply(ast.literal_eval)


### 2. Topic Modelling using gensim for Latent Dirichlet Allocation (LDA) ###

# Create Dictionary
dictionary = Dictionary(data['tokens'])

print('Total Vocabulary Size:', len(dictionary))

# Filter out extremes to limit the number of features
dictionary.filter_extremes(no_below=10, no_above=0.6, keep_n=100000)

print('Total Vocabulary Size:', len(dictionary))


# Create Corpus
corpus = [dictionary.doc2bow(text) for text in data['tokens']]


### get best number of topics using Coherence Score

def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """

    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):

        print(f'\nNum Topics: {num_topics}')

        # model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            chunksize=100,
            update_every=1,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


limit=150; start=20; step=5;

model_list, coherence_values = compute_coherence_values(
    dictionary=dictionary,
    corpus=corpus,
    texts=data['tokens'],
    start=start,
    limit=limit,
    step=step
)

# Show graph
import matplotlib.pyplot as plt

x = range(start, limit, step)

plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


best_result_index = coherence_values.index(max(coherence_values))
optimal_model = model_list[best_result_index]
# Select the model and print the topics
model_topics = optimal_model.show_topics(formatted=False)
print(f'''The {x[best_result_index]} topics gives the highest coherence score 
of {coherence_values[best_result_index]}''')


# Prepare the visualization
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(optimal_model, corpus, dictionary)

# Display the visualization in a Jupyter Notebook
vis

# To save the visualization to an HTML file
pyLDAvis.save_html(vis, 'topic_modeling/topics_lda_usa_40.html')



### 3. Assign top 3 relevant topics to each document (webpage)
top_topics = []
for doc_bow in corpus:
    # Get the topic probabilities for the document
    doc_topics = optimal_model.get_document_topics(doc_bow, minimum_probability=0.0)
    # Sort the topics by probability in descending order
    sorted_doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
    # Get the top 3 topics
    top_3 = sorted_doc_topics[:3]
    top_topics.append(top_3)


# Create a DataFrame from data and add the top 3 topics
data_topics_top3 = data
data_topics_top3.drop('text', axis=1, inplace=True)

data_topics_top3['topic_1'] = [t[0] for t in top_topics]
data_topics_top3['topic_2'] = [t[1] for t in top_topics]
data_topics_top3['topic_3'] = [t[2] for t in top_topics]


# Write the DataFrame to an Excel file
data_topics_top3.to_excel('topic_modeling/topics_top3_lda_usa_40.xlsx', index=False)


### 4. Get top 30 most relevant words per topic
top_words = optimal_model.show_topics(num_topics = 40, num_words = 30)

data_topics_words = pd.DataFrame()
data_topics_words["topic"] = [t[0] for t in top_words]
data_topics_words["words"] = [t[1] for t in top_words]

data_topics_words.to_excel('topic_modeling/topics_words_lda_usa_40.xlsx', index=False)








# Set number of topics
num_topics = 90


# Train the LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=100,
    chunksize=100,
    update_every=1,
    passes=10,
    alpha='auto',
    per_word_topics=True
)


# Prepare the visualization
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, dictionary)

# Display the visualization in a Jupyter Notebook
vis

# To save the visualization to an HTML file
pyLDAvis.save_html(vis, 'topic_modeling/topics_lda_usa_90.html')



### 3. Assign top 3 relevant topics to each document (webpage)
top_topics = []
for doc_bow in corpus:
    # Get the topic probabilities for the document
    doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
    # Sort the topics by probability in descending order
    sorted_doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
    # Get the top 3 topics
    top_3 = sorted_doc_topics[:3]
    top_topics.append(top_3)


# Create a DataFrame from data and add the top 3 topics
data_topics_top3 = data
data_topics_top3.drop('text', axis=1, inplace=True)

data_topics_top3['topic_1'] = [t[0] for t in top_topics]
data_topics_top3['topic_2'] = [t[1] for t in top_topics]
data_topics_top3['topic_3'] = [t[2] for t in top_topics]


# Write the DataFrame to an Excel file
data_topics_top3.to_excel('topic_modeling/topics_top3_lda_usa_90.xlsx', index=False)


### 4. Get top 30 most relevant words per topic
top_words = lda_model.show_topics(num_topics = 90, num_words = 30)

data_topics_words = pd.DataFrame()
data_topics_words["topic"] = [t[0] for t in top_words]
data_topics_words["words"] = [t[1] for t in top_words]

data_topics_words.to_excel('topic_modeling/topics_words_lda_usa_90.xlsx', index=False)










# List of topics and related words
lda_model.print_topics()

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]


### Evaluate the model

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Perplexity:  -7.614774216851785

# Compute Coherence Score
coherence_model_lda = CoherenceModel(
    model=lda_model,
    texts=data['tokens'],
    dictionary=dictionary,
    coherence='c_v'
)
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)
# Coherence Score:  0.44696027675676464


# Prepare the visualization
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, dictionary)

# Display the visualization in a Jupyter Notebook
vis

# To save the visualization to an HTML file
pyLDAvis.save_html(vis, 'topic_model/lda_visualization.html')





### Use Mallet Model
data_preprocessed = data['tokens'].apply(lambda tokens: ' '.join(tokens))
data_preprocessed = data_preprocessed.str.replace('\u03bf', 'o', regex=False)
data_preprocessed = data_preprocessed.str.replace('\ufb01', 'fi', regex=False)
data_preprocessed = data_preprocessed.str.replace('\u02bb', ' ', regex=False)
data_preprocessed = data_preprocessed.str.replace('\u207f', ' ', regex=False)
data_preprocessed = data_preprocessed.str.replace('\u2082', ' ', regex=False)

data_preprocessed[3202]

num_topics = 50

topic_keys, topic_distributions = little_mallet_wrapper.quick_train_topic_model(
    mallet_path,
    ".",
    num_topics,
    data_preprocessed
)

topic_keys[0]

ldamallet = LdaMallet(
    mallet_path,
    corpus=corpus,
    num_topics=num_topics,
    id2word=dictionary
)











### 1. Topic Modelling using sci-kit for Latent Dirichlet Allocation (LDA) ###
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Join the tokens into a single string per document
data_usa["document"] = data_usa['tokens'].apply(lambda tokens: ' '.join(tokens))

# Create a list of documents
corpus = data_usa['document'].tolist()

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the corpus
dtm = vectorizer.fit_transform(corpus)


n_topics = 100  # You can adjust this number

# Initialize LDA model
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=10,
    learning_method='online',
    random_state=42,
    batch_size=128,
    evaluate_every = -1,
    n_jobs = -1,
)

# Fit the LDA model on the document-term matrix
lda_model.fit(dtm)


# Step 5: Display the topics
def display_topics(model, feature_names, no_top_words):
    for idx, topic in enumerate(model.components_):
        print(f"\nTopic {idx +1}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

feature_names = vectorizer.get_feature_names_out()
no_top_words = 10
display_topics(lda_model, feature_names, no_top_words)


pyLDAvis.enable_notebook()

# Prepare the visualization
lda_visualization = pyLDAvis.sklearn.prepare(lda_model, doc_term_matrix, vectorizer, sort_topics=True)

# Display the visualization
pyLDAvis.display(lda_visualization)


no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()













lda_model.show_topic(2)


# 4. Assign Dominant Topic to Each Document
# Initialize a list to store the results
topics_data = []

# For each document
for i, corp in enumerate(corpus):
    # Get the topic probability distribution for the document
    topic_probs = lda_model.get_document_topics(corp)

    # Sort the topics by probability
    topic_probs_sorted = sorted(topic_probs, key=lambda x: x[1], reverse=True)

    # Assign the dominant topic
    dominant_topic = topic_probs_sorted[0][0]
    topic_perc_contrib = topic_probs_sorted[0][1]

    # Get the topic keywords
    topic_keywords = lda_model.show_topic(dominant_topic)
    topic_keywords = ", ".join([word for word, prop in topic_keywords])

    # Append the results for this document to the list
    topics_data.append([i, dominant_topic, round(topic_perc_contrib, 4), topic_keywords])

# Create a DataFrame from the list
dominant_topics_df = pd.DataFrame(topics_data, columns=['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords'])

# 5. (Optional) Add Documents to the DataFrame
dominant_topics_df['Document'] = data_usa['url']

# 6. View the Results
print(dominant_topics_df)