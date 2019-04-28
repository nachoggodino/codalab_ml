import xml.etree.ElementTree as ET
import io
import re
import nltk
import unidecode
import spacy
import textacy

from scipy.sparse import coo_matrix, hstack

from sklearn.ensemble import VotingClassifier
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, tree
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn import decomposition, ensemble
from sklearn.metrics import confusion_matrix

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
data_path_mint = "/home/nacho/DATASETS/public_data_development/"
parser_dev = ET.XMLParser(encoding='utf-8')
parser_train = ET.XMLParser(encoding='utf-8')

tree_dev = ET.parse(data_path + LANGUAGE_CODE + "/intertass_" + LANGUAGE_CODE + "_dev.xml", parser=parser_dev)
tree_train = ET.parse(data_path + LANGUAGE_CODE + "/intertass_" + LANGUAGE_CODE + "_train.xml", parser=parser_train)


def get_dataframe_from_xml(data):
    print("Preparing data...")
    tweet_id, user, content, day_of_week, month, hour, lang, sentiment, ternary_sentiment = [], [], [], [], [], [], [], [], []
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
                if element.text == 'NONE' or element.text == 'NEU':
                    ternary_sentiment.append('N-N')
                else:
                    ternary_sentiment.append(element.text)

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
    result_df['ternary_sentiment'] = encoder.fit_transform(ternary_sentiment)
    return result_df


def extract_length_feature(tokenized_dataframe):
    return [len(tweet) for tweet in tokenized_dataframe]


def extract_uppercase_feature(dataframe):
    regex = re.compile(r"\b[A-Z][A-Z]+\b")
    result = []
    for tweet in dataframe:
        result.append(len(regex.findall(tweet)))
    return result


def extract_question_mark_feature(dataframe):
    result = []
    for tweet in dataframe:
        if re.search(r"[/?/]", tweet):
            result.append(1)
        else:
            result.append(0)
    return result


def extract_exclamation_mark_feature(dataframe):
    result = []
    for tweet in dataframe:
        if re.search(r"[/!/]", tweet):
            result.append(1)
        else:
            result.append(0)
    return result


def extract_letter_repetition_feature(dataframe):
    result = []
    for tweet in dataframe:
        if re.search(r"(\w)(\1{2,})", tweet):
            result.append(1)
        else:
            result.append(0)
    return result


def text_preprocessing(data):
    result = data
    result = [tweet.replace('\n', '').strip() for tweet in result]  # Newline and leading/trailing spaces
    result = [tweet.replace(u'\u2018', "'").replace(u'\u2019', "'") for tweet in result]  # Quotes replace by general
    result = [tweet.lower() for tweet in result]
    result = [re.sub(r"^.*http.*$", 'http', tweet) for tweet in result]  # Remove all http contents
    result = [re.sub(r"\B#\w+", 'hashtag', tweet) for tweet in result]  # Remove all hashtags
    result = [re.sub(r"\B@\w+", 'username', tweet) for tweet in result]  # Remove all usernames
    result = [re.sub(r"(\w)(\1{2,})", r"\1", tweet) for tweet in result]  # Remove all letter repetitions
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


def remove_accents(tokenized_data):
    result = []
    for tweet in tokenized_data:
        result.append([unidecode.unidecode(word) for word in tweet])
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
    print("Lemmatizing data...")
    result = []
    i = 0
    for row in datalist:
        mini_result = []
        mini_result = [token.lemma_ for token in lemmatizer(row)]
        result.append(mini_result)
        i += 1
    return result


def perform_count_vectors(data, train_set_x, valid_set_x):
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(data)

    # transform the training and validation data using count vectorizer object
    return count_vect.transform(train_set_x), count_vect.transform(valid_set_x)


def perform_tf_idf_vectors(train_set_x, valid_set_x):
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(train_set_x)
    tr_tfidf = tfidf_vect.transform(train_set_x)
    val_tfidf = tfidf_vect.transform(valid_set_x)



    return tr_tfidf, val_tfidf


def add_feature(matrix, new_features):
    result = matrix
    for feature_vector in new_features:
        result = pandas.concat([result, feature_vector], axis=1)
    return result


def train_model(classifier, feature_vector_train, label):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    return classifier


def get_predictions(trained_classifier, feature_test_vector):
    return trained_classifier.predict(feature_test_vector)


def get_model_accuracy(predictions, validation_labels):
    return metrics.accuracy_score(predictions, validation_labels)


def print_confusion_matrix(predictions, labels):
    preds = pandas.Series(predictions, name='Predicted')
    labs = pandas.Series(labels, name='Actual')
    df_confusion = pandas.crosstab(labs, preds)
    print(df_confusion)
    return


# GET THE DATA
train_data = get_dataframe_from_xml(tree_train)
dev_data = get_dataframe_from_xml(tree_dev)

# PRE-PROCESSING
preprocessed_train_content = text_preprocessing(train_data['content'])
preprocessed_dev_content = text_preprocessing(dev_data['content'])

# TOKENIZE
tokenized_train_content = tokenize_list(preprocessed_train_content)
tokenized_dev_data = tokenize_list(preprocessed_dev_content)

# FEATURE EXTRACTION
train_data['tweet_length'] = extract_length_feature(tokenized_train_content)
dev_data['tweet_length'] = extract_length_feature(tokenized_dev_data)

train_data['has_uppercase'] = extract_uppercase_feature(train_data['content'])
dev_data['has_uppercase'] = extract_uppercase_feature(dev_data['content'])

train_data['question_mark'] = extract_question_mark_feature(train_data['content'])
dev_data['question_mark'] = extract_question_mark_feature(dev_data['content'])

train_data['exclamation_mark'] = extract_exclamation_mark_feature(train_data['content'])
dev_data['exclamation_mark'] = extract_exclamation_mark_feature(dev_data['content'])

train_data['letter_repetition'] = extract_letter_repetition_feature(train_data['content'])
dev_data['letter_repetition'] = extract_letter_repetition_feature(dev_data['content'])


'''
clean_train_content = tokenized_train_content  # remove_stopwords(tokenized_train_content)
clean_dev_content = tokenized_dev_data  # remove_stopwords(tokenized_dev_data)
'''

# LEMMATIZING
lemmatized_train_tweets = lemmatize_list(preprocessed_train_content)
lemmatized_dev_tweets = lemmatize_list(preprocessed_dev_content)

# REMOVING ACCENTS
without_accents_train = remove_accents(lemmatized_train_tweets)
without_accents_dev = remove_accents(lemmatized_dev_tweets)

final_train_content = [TreebankWordDetokenizer().detokenize(row) for row in lemmatized_train_tweets]
final_dev_content = [TreebankWordDetokenizer().detokenize(row) for row in lemmatized_dev_tweets]

# COUNT VECTORS
x_train_count_vectors, x_dev_count_vectors = perform_count_vectors(final_train_content, final_train_content, final_dev_content)

# TF-IDF VECTORS
xtrain_tfidf, xdev_tfidf = perform_tf_idf_vectors(final_train_content, final_dev_content)

train_features = [
    train_data['tweet_length'],
    train_data['has_uppercase'],
    train_data['exclamation_mark'],
    train_data['question_mark'],
    train_data['letter_repetition']
]

dev_features = [
    dev_data['tweet_length'],
    dev_data['has_uppercase'],
    dev_data['exclamation_mark'],
    dev_data['question_mark'],
    dev_data['letter_repetition']
]

x_train_count_vectors = add_feature(pandas.DataFrame(x_train_count_vectors.todense()), train_features)
x_dev_count_vectors = add_feature(pandas.DataFrame(x_dev_count_vectors.todense()), dev_features)

xtrain_tfidf = add_feature(pandas.DataFrame(xtrain_tfidf.todense()), train_features)
xdev_tfidf = add_feature(pandas.DataFrame(xdev_tfidf.todense()), dev_features)

# WORD EMBEDDINGS
print("Messing with Word Embeddings...")
# load the pre-trained word-embedding vectors

'''
embeddings_index = {}
for i, line in enumerate(open('/home/nacho/cc.es.300.vec', encoding='utf-8')):
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
nb_cv_predictions = get_predictions(nb_cv_classifier, x_dev_count_vectors)
print("NB, Count Vectors: ", get_model_accuracy(nb_cv_predictions, test_labels))
print_confusion_matrix(nb_cv_predictions, test_labels)

# Naive Bayes on Word Level TF IDF Vectors
nb_word_classifier = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, training_labels)
nb_word_predictions = get_predictions(nb_word_classifier, xdev_tfidf)
print("NB, WordLevel TF-IDF: ", get_model_accuracy(nb_word_predictions, test_labels))
print_confusion_matrix(nb_word_predictions, test_labels)

# LINEAR CLASSIFIER
# Linear Regression on Count Vectors
lr_cv_classifier = train_model(linear_model.LogisticRegression(), x_train_count_vectors, training_labels)
lr_cv_predictions = get_predictions(lr_cv_classifier, x_dev_count_vectors)
print("LR, Count Vectors: ", get_model_accuracy(lr_cv_predictions, test_labels))
print_confusion_matrix(lr_cv_predictions, test_labels)

# Linear Regression on Word Level TF IDF Vectors
lr_word_classifier = train_model(linear_model.LogisticRegression(), xtrain_tfidf, training_labels)
lr_word_predictions = get_predictions(lr_word_classifier, xdev_tfidf)
print("LR, WordLevel TF-IDF: ", get_model_accuracy(lr_word_predictions, test_labels))
print_confusion_matrix(lr_word_predictions, test_labels)

# DECISION TREE
# Decision Tree on Count Vectors
dt_cv_classifier = train_model(tree.DecisionTreeClassifier(), x_train_count_vectors, training_labels)
dt_cv_predictions = get_predictions(dt_cv_classifier, x_dev_count_vectors)
print("DT, Count Vectors: ", get_model_accuracy(dt_cv_predictions, test_labels))
print_confusion_matrix(dt_cv_predictions, test_labels)

# Decision Tree on Word Level TF IDF Vectors
dt_word_classifier = train_model(tree.DecisionTreeClassifier(), xtrain_tfidf, training_labels)
dt_word_predictions = get_predictions(dt_word_classifier, xdev_tfidf)
print("DT, WordLevel TF-IDF: ", get_model_accuracy(dt_word_predictions, test_labels))
print_confusion_matrix(dt_word_predictions, test_labels)

# SVM on Count Vectors
svm_cv_classifier = train_model(svm.LinearSVC(), x_train_count_vectors, training_labels)
svm_cv_predictions = get_predictions(svm_cv_classifier, x_dev_count_vectors)
print("SVM, CountVectors: ", get_model_accuracy(svm_cv_predictions, test_labels))
print_confusion_matrix(svm_cv_predictions, test_labels)

# SVM on Word Level TF IDF Vectors
svm_word_classifier = train_model(svm.LinearSVC(), xtrain_tfidf, training_labels)
svm_word_predictions = get_predictions(svm_word_classifier, xdev_tfidf)
print("SVM, WordLevel TF-IDF: ", get_model_accuracy(svm_word_predictions, test_labels))
print_confusion_matrix(svm_word_predictions, test_labels)

# ENSEMBLINGS
voting_model = VotingClassifier(estimators=[
    ('lr', linear_model.LogisticRegression()),
    ('svm', svm.LinearSVC())], voting='hard')

voting_classifier = train_model(voting_model, x_train_count_vectors, training_labels)
voting_predictions = get_predictions(voting_classifier, x_dev_count_vectors)
print("VOTING CLASSIFIER: ", get_model_accuracy(voting_predictions, test_labels))
print_confusion_matrix(voting_predictions, test_labels)

xgboost_model = xgboost.XGBClassifier(random_state=1, learning_rate=0.01)
xgboost_classifier = train_model(xgboost_model, x_train_count_vectors, training_labels)
xgboost_predictions = get_predictions(xgboost_classifier, x_dev_count_vectors)
print("XGBOOST CLASSIFIER: ", get_model_accuracy(xgboost_predictions, test_labels))
print_confusion_matrix(xgboost_predictions, test_labels)


