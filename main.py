import xml.etree.ElementTree as ET
import io
import re
import nltk
import string
import matplotlib

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from nltk.tokenize.treebank import TreebankWordDetokenizer

from collections import Counter


data_path = "C:/Users/nacho/OneDrive/Documentos/TELECO/TFG/CODALAB/DATASETS/public_data_development/es/"
parser_dev = ET.XMLParser(encoding='utf-8')
parser_train = ET.XMLParser(encoding='utf-8')

tree_dev = ET.parse(data_path + "intertass_es_dev.xml", parser=parser_dev)
tree_train = ET.parse(data_path + "intertass_es_train.xml", parser=parser_train)


def get_dataframe_from_xml(data):
    print("Preparing data...")
    tweet_id, user, content, day_of_week, month, hour, lang, sentiment = [], [], [], [], [], [], [], []
    for tweet in data.iter('tweet'):
        for element in tweet.iter():
            if element.tag == 'tweetid':
                tweet_id.append(element.text)
            elif element.tag == 'user':
                user.append(element.text)
            elif element.tag == 'content':
                content.append(element.text)
            elif element.tag == 'date':
                day_of_week.append(element.text[:3])
                month.append(element.text[4:7])
                hour.append(element.text[11:13])
            elif element.tag == 'lang':
                lang.append(element.text)
            elif element.tag == 'value':
                sentiment.append(element.text)

    result_df = pandas.DataFrame()
    result_df['tweet_id'] = tweet_id
    # result_df['user'] = user
    result_df['content'] = content
    # result_df['lang'] = lang
    result_df['day_of_week'] = day_of_week
    result_df['month'] = month
    result_df['hour'] = hour

    encoder = preprocessing.LabelEncoder()
    result_df['sentiment'] = encoder.fit_transform(sentiment)

    return result_df


def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_label, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_label)


def perform_count_vectors(data, train_set_x, valid_set_x):
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(data)

    # transform the training and validation data using count vectorizer object
    return count_vect.transform(train_set_x), count_vect.transform(valid_set_x)


def perform_tf_idf_vectors(data, train_set_x, valid_set_x):
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(data)
    tr_tfidf = tfidf_vect.transform(train_set_x)
    val_tfidf = tfidf_vect.transform(valid_set_x)

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_ngram.fit(data)
    tr_ngram = tfidf_vect_ngram.transform(train_set_x)
    val_ngram = tfidf_vect_ngram.transform(valid_set_x)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                             max_features=5000)
    tfidf_vect_ngram_chars.fit(data)
    tr_chars = tfidf_vect_ngram_chars.transform(train_set_x)
    val_chars = tfidf_vect_ngram_chars.transform(valid_set_x)

    return tr_tfidf, tr_ngram, tr_chars, val_tfidf, val_ngram, val_chars


def text_preprocessing(data):
    result = data
    result = [tweet.replace('\n', '').strip() for tweet in result]  # Newline and leading/trailing spaces
    result = [tweet.replace(u'\u2018', "'").replace(u'\u2019', "'") for tweet in result]  # Quotes replace by general
    result = [tweet.lower() for tweet in result]
    result = [re.sub(r"^.*http.*$", 'http', tweet) for tweet in result]  # Remove all http contents
    result = [re.sub(r"\B#\w+", 'hashtag', tweet) for tweet in result]  # Remove all usernames
    result = [re.sub(r"\B@\w+", 'username', tweet) for tweet in result]  # Remove all hashtags
    # result = [re.sub(r"^.*jaj.*$", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"\d+", '', tweet) for tweet in result]  # Remove all numbers
    result = [tweet.translate(str.maketrans('', '', string.punctuation)) for tweet in result]  # Remove punctuation

    return result


def tokenize_stopwords_list(datalist):
    result = []
    for row in datalist:
        result.append(word for word in nltk.word_tokenize(row) if word not in nltk.corpus.stopwords.words("spanish"))
    return result


def stem_list(datalist):
    stemmer = nltk.stem.SnowballStemmer('spanish')
    result = []
    for row in datalist:
        stemmed_words = [stemmer.stem(word) for word in row]
        result.append(stemmed_words)
    return result


# GET THE DATA
train_data = get_dataframe_from_xml(tree_train)
valid_data = get_dataframe_from_xml(tree_dev)

# PRE-PROCESSING
train_data['content'] = text_preprocessing(train_data['content'])
valid_data['content'] = text_preprocessing(valid_data['content'])
print(train_data['content'])

# TOKENIZE AND REMOVE STOPWORDS
train_data['content'] = tokenize_stopwords_list(train_data['content'])
valid_data['content'] = tokenize_stopwords_list(valid_data['content'])

# STEMMING
train_data['content'] = stem_list(train_data['content'])
valid_data['content'] = stem_list(valid_data['content'])

train_data['content'] = [TreebankWordDetokenizer().detokenize(row) for row in train_data['content']]
valid_data['content'] = [TreebankWordDetokenizer().detokenize(row) for row in valid_data['content']]
print(train_data['content'])


all_words = [item for sublist in train_data['content'] for item in sublist]

word_counter = Counter(all_words)
most_common_words = word_counter.most_common(50)
print(most_common_words)

print(train_data.groupby(['hour', 'sentiment']).size())


# SPLIT DATASET INTO TRAIN AND VALIDATION
train_x = train_data['content']
train_y = train_data['sentiment']
valid_x = valid_data['content']
valid_y = valid_data['sentiment']
# train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_data['content'], train_data['sentiment'])

# COUNT VECTORS
x_train_count_vectors, x_valid_count_vectors = perform_count_vectors(train_data['content'], train_x, valid_x)

# TF-IDF VECTORS
xtrain_tfidf, xtrain_tfidf_ngram, xtrain_tfidf_ngram_chars, xvalid_tfidf, xvalid_tfidf_ngram, xvalid_tfidf_ngram_chars = perform_tf_idf_vectors(train_data['content'], train_x, valid_x)


# WORD EMBEDDINGS
'''print("Messing with Word Embeddings...")
# load the pre-trained word-embedding vectors
embeddings_index = {}
for i, line in enumerate(open('C:/Users/nacho/Downloads/cc.es.300.vec/cc.es.300.vec', encoding='utf-8')):
    if i % 1000 == 0:
        print(i)
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer
token = text.Tokenizer()
token.fit_on_texts(dataframe['content'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


''''''
# NAIVE BAYES
# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), x_train_count_vectors, train_y, x_valid_count_vectors, valid_y)
print("NB, Count Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print ("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print ("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
print ("NB, CharLevel Vectors: ", accuracy)

# LINEAR CLASSIFIER
# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), x_train_count_vectors, train_y, x_valid_count_vectors, valid_y)
print ("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print ("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
print ("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
print ("LR, CharLevel Vectors: ", accuracy)

# SVM
# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), x_train_count_vectors, train_y, x_valid_count_vectors, valid_y)
print("SVM, Count Vectors: ", accuracy)
'''