import xml.etree.ElementTree as ET
import io
import re
import nltk
import unidecode
import spacy
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

from gensim.models import Word2Vec

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers, models, optimizers
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

from nltk.tokenize.treebank import TreebankWordDetokenizer

from collections import Counter


LANGUAGE_CODE = 'es'
data_path = "C:/Users/nacho/OneDrive/Documentos/TELECO/TFG/CODALAB/DATASETS/public_data_development/"
parser_dev = ET.XMLParser(encoding='utf-8')
parser_train = ET.XMLParser(encoding='utf-8')

tree_dev = ET.parse(data_path + LANGUAGE_CODE + "/intertass_" + LANGUAGE_CODE + "_dev.xml", parser=parser_dev)
tree_train = ET.parse(data_path + LANGUAGE_CODE + "/intertass_" + LANGUAGE_CODE + "_train.xml", parser=parser_train)


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


def extract_length_feature(dataframe):
    tokenized_dataframe = tokenize_list(dataframe)
    return [len(tweet) for tweet in tokenized_dataframe]


def extract_uppercase_feature(dataframe):
    regex = r"\b[A-Z][A-Z]+\b"
    tokenized_dataframe = tokenize_list(dataframe)
    result = []
    for tweet in tokenized_dataframe:
        result.append(re.finditer(regex, tweet))
    return result


def text_preprocessing(data):
    result = data
    result = [tweet.replace('\n', '').strip() for tweet in result]  # Newline and leading/trailing spaces
    result = [tweet.replace(u'\u2018', "'").replace(u'\u2019', "'") for tweet in result]  # Quotes replace by general
    result = [tweet.lower() for tweet in result]
    result = [re.sub(r"^.*http.*$", 'http', tweet) for tweet in result]  # Remove all http contents
    result = [re.sub(r"\B#\w+", 'hashtag', tweet) for tweet in result]  # Remove all usernames
    result = [re.sub(r"\B@\w+", 'username', tweet) for tweet in result]  # Remove all hashtags
    result = [re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"\d+", '', tweet) for tweet in result]  # Remove all numbers
    result = [unidecode.unidecode(tweet) for tweet in result]
    result = [tweet.translate(str.maketrans('', '', string.punctuation)) for tweet in result]  # Remove punctuation

    return result


def tokenize_list(datalist):
    result = []
    for row in datalist:
        result.append(nltk.word_tokenize(row))
    return result


def remove_stopwords(tokenized_data):
    result = []
    for row in tokenized_data:
        result.append([word for word in row if word not in nltk.corpus.stopwords.words('spanish')])
    return result


def stem_list(datalist):
    stemmer = nltk.stem.SnowballStemmer('spanish')
    result = []
    for row in datalist:
        stemmed_words = [stemmer.stem(word) for word in row]
        result.append(stemmed_words)
    return result


def lemmatize_list(datalist):
    lemmatizer = spacy.load("es_core_news_sm")
    result = []
    i = 0
    for row in datalist:
        if i % 10 == 0:
            print(i)
        lemmatized_words = [lemmatizer(token)[0].lemma_ for token in row]
        result.append(lemmatized_words)
        i += 1
    return result


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


def train_model(classifier, feature_vector_train, label):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    return classifier


def get_model_accuracy(trained_classifier, feature_test_vector, validation_labels):
    # predict the labels on validation dataset
    predictions = trained_classifier.predict(feature_test_vector)
    return metrics.accuracy_score(predictions, validation_labels)


# GET THE DATA
train_data = get_dataframe_from_xml(tree_train)
dev_data = get_dataframe_from_xml(tree_dev)

# PRE-PROCESSING
preprocessed_train_content = text_preprocessing(train_data['content'])
preprocessed_dev_content = text_preprocessing(dev_data['content'])

# TOKENIZE AND REMOVE STOPWORDS
tokenized_train_content = tokenize_list(preprocessed_train_content)
tokenized_dev_data = tokenize_list(preprocessed_dev_content)
clean_train_content = tokenized_train_content  # remove_stopwords(tokenized_train_content)
clean_dev_content = tokenized_dev_data  # remove_stopwords(tokenized_dev_data)

# STEMMING
lemmatized_train_tweets = lemmatize_list(clean_train_content)
lemmatized_dev_tweets = lemmatize_list(clean_dev_content)

final_train_content = [TreebankWordDetokenizer().detokenize(row) for row in clean_train_content]
final_dev_content = [TreebankWordDetokenizer().detokenize(row) for row in clean_dev_content]

# COUNT VECTORS
x_train_count_vectors, x_dev_count_vectors = perform_count_vectors(final_train_content, final_train_content, final_dev_content)

# TF-IDF VECTORS
xtrain_tfidf, xtrain_tfidf_ngram, xtrain_tfidf_ngram_chars, xdev_tfidf, xdev_tfidf_ngram, xdev_tfidf_ngram_chars = perform_tf_idf_vectors(final_train_content, final_train_content, final_dev_content)


# WORD EMBEDDINGS
print("Messing with Word Embeddings...")
# load the pre-trained word-embedding vectors

'''
embeddings_index = {}
for i, line in enumerate(open('C:/Users/nacho/Downloads/cc.es.300.vec/cc.es.300.vec', encoding='utf-8')):
    if i % 100000 == 0:
        print(i)
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(final_train_content)
sequences = tokenizer_obj.texts_to_sequences(final_train_content)

word_index = tokenizer_obj.word_index
print("Found %s unique tokens." % len(word_index))

max_length = max([len(s.split()) for s in final_train_content])
tweet_pad = pad_sequences(sequences, maxlen=max_length)

sentiment = train_data['sentiment'].values
num_words = len(word_index) + 1

embedding_matrix = numpy.zeros((num_words, 300))
for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential()
embedding_layer = Embedding(num_words, 300, embeddings_initializer=Constant(embedding_matrix), input_length=max_length, trainable=False)
model.add(embedding_layer)
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(tweet_pad, train_data['sentiment'], batch_size=128, epochs=100, validation_data=(tweet_pad, train_data['sentiment']), verbose=2)
'''

# NAIVE BAYES
# Naive Bayes on Count Vectors

training_labels = train_data['sentiment']
test_labels = dev_data['sentiment']

nb_cv_classifier = train_model(naive_bayes.MultinomialNB(), x_train_count_vectors, training_labels)
print("NB, Count Vectors: ", get_model_accuracy(nb_cv_classifier, x_dev_count_vectors, test_labels))

# Naive Bayes on Word Level TF IDF Vectors
nb_word_classifier = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, training_labels)
print("NB, WordLevel TF-IDF: ", get_model_accuracy(nb_word_classifier, xdev_tfidf, test_labels))
'''
# Naive Bayes on Ngram Level TF IDF Vectors
nb_ngram_classifier = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, training_labels)
print("NB, N-Gram Vectors: ", get_model_accuracy(nb_ngram_classifier, xdev_tfidf_ngram, test_labels))

# Naive Bayes on Character Level TF IDF Vectors
nb_chars_classifier = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, training_labels)
print("NB, CharLevel Vectors: ", get_model_accuracy(nb_chars_classifier, xdev_tfidf_ngram_chars, test_labels))
'''
# LINEAR CLASSIFIER
# Linear Regression on Count Vectors
lr_cv_classifier = train_model(linear_model.LogisticRegression(), x_train_count_vectors, training_labels)
print("LR, Count Vectors: ", get_model_accuracy(lr_cv_classifier, x_dev_count_vectors, test_labels))

# Linear Regression on Word Level TF IDF Vectors
lr_word_classifier = train_model(linear_model.LogisticRegression(), xtrain_tfidf, training_labels)
print("LR, WordLevel TF-IDF: ", get_model_accuracy(lr_word_classifier, xdev_tfidf, test_labels))
'''
# Linear Regression on Ngram Level TF IDF Vectors
lr_ngram_classifier = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, training_labels)
print("LR, N-Gram Vectors: ", get_model_accuracy(lr_ngram_classifier, xdev_tfidf_ngram, test_labels))

# Linear Regression on Character Level TF IDF Vectors
lr_chars_classifier = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, training_labels)
print("LR, CharLevel Vectors: ", get_model_accuracy(lr_chars_classifier, xdev_tfidf_ngram_chars, test_labels))
'''
# SVM
svm_chars_classifier = train_model(linear_model.LogisticRegression(), x_train_count_vectors, training_labels)
print("SVM: ", get_model_accuracy(svm_chars_classifier, x_dev_count_vectors, test_labels))

# RF on Count Vectors
rf_cv_classifier = train_model(ensemble.RandomForestClassifier(), x_train_count_vectors, training_labels)
print("RF, Count Vectors: ", get_model_accuracy(rf_cv_classifier, x_dev_count_vectors, test_labels))

# RF on TF IDF WordLevel
rf_word_classifier = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, training_labels)
print("RF, WordLevel TF_IDF: ", get_model_accuracy(rf_word_classifier, xdev_tfidf, test_labels))
