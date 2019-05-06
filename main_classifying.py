import xml.etree.ElementTree as ET
import io
import re
import csv
import nltk
import unidecode
import spacy
import hunspell

from scipy.sparse import coo_matrix, hstack

import warnings

from re import finditer

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing, linear_model, naive_bayes, metrics, tree, svm
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn import decomposition, ensemble
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
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

from textacy import keyterms

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)

LANGUAGE_CODE = 'es'
dictionary = hunspell.Hunspell('es_ANY', hunspell_data_dir="C:/Users/nacho/Downloads/")
CROSS_LINGUAL = False
LABEL_ENCODER = preprocessing.LabelEncoder()
TERNARY_LABEL_ENCODER = preprocessing.LabelEncoder()
data_path = "C:/Users/nacho/OneDrive/Documentos/TELECO/TFG/CODALAB/DATASETS/public_data_development/"
data_path_mint = "/home/nacho/DATASETS/public_data_development/"

emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

print()
print("Language: " + LANGUAGE_CODE)
print()


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

    LABEL_ENCODER.fit(sentiment)
    TERNARY_LABEL_ENCODER.fit(ternary_sentiment)
    result_df['sentiment'] = LABEL_ENCODER.transform(sentiment)
    result_df['ternary_sentiment'] = TERNARY_LABEL_ENCODER.transform(ternary_sentiment)
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


def extract_sent_words_feature(tokenized_data, data_feed):
    positive_voc, negative_voc = get_sentiment_vocabulary(data_feed, 3, 0)
    pos_result = []
    neg_result = []
    neutral_result = []
    none_result = []
    for tokenized_tweet in tokenized_data:
        pos_count = 0
        neg_count = 0
        for word in tokenized_tweet:
            if word in positive_voc:
                pos_count += 1
            if word in negative_voc:
                neg_count += 1
        pos_result.append(pos_count)
        neg_result.append(neg_count)
        neutral_result.append(pos_count-neg_count)
        none_result.append(pos_count+neg_count)
    return pos_result, neg_result, neutral_result, none_result


def get_sentiment_vocabulary(data, positive, negative):
    pos_neg_tweets = []
    pos_neg_bool_labels = []
    for index, tweet in enumerate(data):
        sentiment = train_data['sentiment'][index]
        if sentiment == positive:
            pos_neg_tweets.append(tweet)
            pos_neg_bool_labels.append(True)
        elif sentiment == negative:
            pos_neg_tweets.append(tweet)
            pos_neg_bool_labels.append(False)

    positive_vocabulary, negative_vocabulary = keyterms.most_discriminating_terms(pos_neg_tweets, pos_neg_bool_labels)

    return positive_vocabulary, negative_vocabulary


def text_preprocessing(data):
    result = data
    result = [tweet.replace('\n', '').strip() for tweet in result]  # Newline and leading/trailing spaces
    result = [emoji_pattern.sub(r'', tweet) for tweet in result]
    result = [tweet.replace(u'\u2018', "'").replace(u'\u2019', "'") for tweet in result]  # Quotes replace by general
    result = [re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), tweet) for tweet in result]  # Hashtag
    result = [tweet.lower() for tweet in result]
    result = [re.sub(r"^.*http.*$", 'http', tweet) for tweet in result]  # Remove all http contents
    result = [re.sub(r"\B@\w+", 'username', tweet) for tweet in result]  # Remove all usernames
    result = [re.sub(r"(\w)(\1{2,})", r"\1", tweet) for tweet in result]  # Remove all letter repetitions
    result = [re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"\d+", '', tweet) for tweet in result]  # Remove all numbers
    result = [tweet.translate(str.maketrans('', '', string.punctuation)) for tweet in result]  # Remove punctuation

    return result


def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])


def libreoffice_processing(tokenized_data):
    result = []
    for tweet in tokenized_data:
        mini_result = []
        for word in tweet:
            print("From " + word + " to")
            print(next(iter(dictionary.suggest(word)), word))
            if not dictionary.spell(word):
                mini_result.append(next(iter(dictionary.suggest(word)), word))
            else:
                mini_result.append(word)
        result.append(mini_result)
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


def perform_count_vectors(train_set_x, valid_set_x):
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(train_set_x)

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
    print()
    print(f1_score(labs, preds, average='macro'))
    print()
    print(recall_score(labs, preds, average='macro'))
    print()
    print(precision_score(labs, preds, average='macro'))
    print()
    return


def decode_label(predictions_array):
    labels = ['N', 'NONE', 'NEU', 'P']
    result = [labels[one_prediction] for one_prediction in predictions_array]
    return result


def get_oof(clf, x_train, y_train, x_test):
    oof_train = numpy.zeros((x_train.shape[0],))
    oof_test = numpy.zeros((x_test.shape[0],))
    oof_test_skf = numpy.empty((5, x_test.shape[0]))
    kf = KFold(n_splits=5, random_state=0)

    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def get_pad_sequences(data, embeddings_index):
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(data)
    sequences = tokenizer_obj.texts_to_sequences(data)

    word_index = tokenizer_obj.word_index
    print("Found %s unique tokens." % len(word_index))

    max_length = max([len(s.split()) for s in final_train_content])
    tweet_pad = pad_sequences(sequences, maxlen=max_length)

    num_words = len(word_index) + 1

    embedding_matrix = numpy.zeros((num_words, 300))
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return num_words, embedding_matrix, max_length, tweet_pad


# GET THE DATA
if CROSS_LINGUAL:
    parser_es = ET.XMLParser(encoding='utf-8')
    parser_uy = ET.XMLParser(encoding='utf-8')
    parser_pe = ET.XMLParser(encoding='utf-8')
    parser_cr = ET.XMLParser(encoding='utf-8')
    tree_es = ET.parse(data_path + "es/intertass_es_train.xml", parser=parser_es)
    tree_uy = ET.parse(data_path + "uy/intertass_uy_train.xml", parser=parser_uy)
    tree_pe = ET.parse(data_path + "pe/intertass_pe_train.xml", parser=parser_pe)
    tree_cr = ET.parse(data_path + "cr/intertass_cr_train.xml", parser=parser_cr)
    es_df = get_dataframe_from_xml(tree_es)
    uy_df = get_dataframe_from_xml(tree_uy)
    pe_df = get_dataframe_from_xml(tree_pe)
    cr_df = get_dataframe_from_xml(tree_cr)
    if LANGUAGE_CODE == 'es':
        dev_data = es_df
        train_data = pandas.concat([uy_df, pe_df, cr_df], ignore_index=True)
    elif LANGUAGE_CODE == 'uy':
        dev_data = uy_df
        train_data = pandas.concat([es_df, pe_df, cr_df], ignore_index=True)
    elif LANGUAGE_CODE == 'pe':
        dev_data = pe_df
        train_data = pandas.concat([uy_df, es_df, cr_df], ignore_index=True)
    else :
        dev_data = cr_df
        train_data = pandas.concat([uy_df, pe_df, es_df], ignore_index=True)
else:
    parser_dev = ET.XMLParser(encoding='utf-8')
    parser_train = ET.XMLParser(encoding='utf-8')
    tree_dev = ET.parse(data_path + LANGUAGE_CODE + "/intertass_" + LANGUAGE_CODE + "_dev.xml", parser=parser_dev)
    tree_train = ET.parse(data_path + LANGUAGE_CODE + "/intertass_" + LANGUAGE_CODE + "_train.xml", parser=parser_train)
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

train_data['pos_voc'], train_data['neg_voc'], train_data['neu_voc'], train_data['none_voc'] = extract_sent_words_feature(tokenized_train_content, tokenized_train_content)
dev_data['pos_voc'], dev_data['neg_voc'], dev_data['neu_voc'], dev_data['none_voc'] = extract_sent_words_feature(tokenized_dev_data, tokenized_train_content)

'''
clean_train_content = tokenized_train_content  # remove_stopwords(tokenized_train_content)
clean_dev_content = tokenized_dev_data  # remove_stopwords(tokenized_dev_data)
'''

# LEMMATIZING
lemmatized_train_tweets = lemmatize_list(libreoffice_processing(tokenized_train_content))
lemmatized_dev_tweets = lemmatize_list(libreoffice_processing(tokenized_dev_data))

# REMOVING ACCENTS
without_accents_train = remove_accents(lemmatized_train_tweets)
without_accents_dev = remove_accents(lemmatized_dev_tweets)

final_train_content = [TreebankWordDetokenizer().detokenize(row) for row in lemmatized_train_tweets]
final_dev_content = [TreebankWordDetokenizer().detokenize(row) for row in lemmatized_dev_tweets]

# COUNT VECTORS
x_train_count_vectors, x_dev_count_vectors = perform_count_vectors(final_train_content, final_dev_content)

# TF-IDF VECTORS
xtrain_tfidf, xdev_tfidf = perform_tf_idf_vectors(final_train_content, final_dev_content)

train_features = [
    #train_data['tweet_length'],
    #train_data['has_uppercase'],
    #train_data['exclamation_mark'],
    #train_data['question_mark'],
    #train_data['pos_voc'],
    #train_data['neg_voc'],
    #train_data['neu_voc'],
    #train_data['none_voc'],
    train_data['letter_repetition']
]

dev_features = [
    #dev_data['tweet_length'],
    #dev_data['has_uppercase'],
    #dev_data['exclamation_mark'],
    #dev_data['question_mark'],
    #dev_data['pos_voc'],
    #dev_data['neg_voc'],
    #dev_data['neu_voc'],
    #dev_data['none_voc'],
    dev_data['letter_repetition']
]

x_train_count_vectors = add_feature(pandas.DataFrame(x_train_count_vectors.todense()), train_features)
x_dev_count_vectors = add_feature(pandas.DataFrame(x_dev_count_vectors.todense()), dev_features)

xtrain_tfidf = add_feature(pandas.DataFrame(xtrain_tfidf.todense()), train_features)
xdev_tfidf = add_feature(pandas.DataFrame(xdev_tfidf.todense()), dev_features)

print(x_train_count_vectors)

'''
cv_scaler = MinMaxScaler()
cv_scaler.fit(x_train_count_vectors)
x_train_count_vectors = cv_scaler.transform(x_train_count_vectors)
x_dev_count_vectors = cv_scaler.transform(x_dev_count_vectors)

tf_scaler = MinMaxScaler()
tf_scaler.fit(xtrain_tfidf)
xtrain_tfidf = tf_scaler.transform(xtrain_tfidf)
xdev_tfidf = tf_scaler.transform(xdev_tfidf)
'''

training_labels = train_data['sentiment']
test_labels = dev_data['sentiment']

# WORD EMBEDDINGS
# load the pre-trained word-embedding vectors

'''
embeddings_index = {}
for i, line in enumerate(open('C:/Users/nacho/Downloads/cc.es.300.vec/cc.es.300.vec', encoding='utf-8')):
    if i % 100000 == 0:
        print(i)
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

train_num_words, train_embedding_matrix, train_max_length, train_pad = get_pad_sequences(final_train_content, embeddings_index)
dev_num_words, dev_embedding_matrix, dev_max_length, dev_pad = get_pad_sequences(final_dev_content, embeddings_index)

model = Sequential()
embedding_layer = Embedding(train_num_words, 300, embeddings_initializer=Constant(train_embedding_matrix),
                            input_length=train_max_length, trainable=False)
model.add(embedding_layer)
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_pad, training_labels, batch_size=16, epochs=10, validation_data=(dev_pad, test_labels), verbose=2)
nn_predictions = model.predict_classes(dev_pad, batch_size=16, verbose=2)
print("NN, Word Embeddings: ", get_model_accuracy(nn_predictions, test_labels))
print_confusion_matrix(nn_predictions, test_labels)
'''

# NAIVE BAYES
# Naive Bayes on Count Vectors

print()
'''
nb_cv_classifier = train_model(naive_bayes.MultinomialNB(), x_train_count_vectors, training_labels)
nb_cv_predictions = get_predictions(nb_cv_classifier, x_dev_count_vectors)
print("NB, Count Vectors: ", get_model_accuracy(nb_cv_predictions, test_labels))
print_confusion_matrix(nb_cv_predictions, test_labels)

# Naive Bayes on Word Level TF IDF Vectors
nb_word_classifier = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, training_labels)
nb_word_predictions = get_predictions(nb_word_classifier, xdev_tfidf)
print("NB, WordLevel TF-IDF: ", get_model_accuracy(nb_word_predictions, test_labels))
print_confusion_matrix(nb_word_predictions, test_labels)
'''
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
'''
dt_cv_classifier = train_model(tree.DecisionTreeClassifier(), x_train_count_vectors, training_labels)
dt_cv_predictions = get_predictions(dt_cv_classifier, x_dev_count_vectors)
print("DT, Count Vectors: ", get_model_accuracy(dt_cv_predictions, test_labels))
print_confusion_matrix(dt_cv_predictions, test_labels)

# Decision Tree on Word Level TF IDF Vectors
dt_word_classifier = train_model(tree.DecisionTreeClassifier(), xtrain_tfidf, training_labels)
dt_word_predictions = get_predictions(dt_word_classifier, xdev_tfidf)
print("DT, WordLevel TF-IDF: ", get_model_accuracy(dt_word_predictions, test_labels))
print_confusion_matrix(dt_word_predictions, test_labels)
'''

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
    ('svm', svm.LinearSVC()),
    ('nb', naive_bayes.MultinomialNB())], voting='hard')

voting_classifier = train_model(voting_model, x_train_count_vectors, training_labels)
voting_predictions = get_predictions(voting_classifier, x_dev_count_vectors)
print("VOTING CLASSIFIER: ", get_model_accuracy(voting_predictions, test_labels))
print_confusion_matrix(voting_predictions, test_labels)


'''
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

rf = RandomForestClassifier(**rf_params)
et = ExtraTreesClassifier(**et_params)
ada = AdaBoostClassifier(**ada_params)
gb = GradientBoostingClassifier(**gb_params)
svc = svm.SVC(**svc_params)

training_set = xtrain_tfidf
test_set = xdev_tfidf

rf_oof_train, rf_oof_test = get_oof(rf, training_set, training_labels, test_set)
print("1")
et_oof_train, et_oof_test = get_oof(et, training_set, training_labels, test_set)
print("1")
ada_oof_train, ada_oof_test = get_oof(ada, training_set, training_labels, test_set)
print("1")
gb_oof_train, gb_oof_test = get_oof(gb, training_set, training_labels, test_set)
print("1")
svc_oof_train, svc_oof_test = get_oof(svc, training_set, training_labels, test_set)
print("1")
'''

'''
nb_oof_train, nb_oof_test = get_oof(naive_bayes.MultinomialNB(), xtrain_tfidf, training_labels, xdev_tfidf)
lr_oof_train, lr_oof_test = get_oof(linear_model.LogisticRegression(), xtrain_tfidf, training_labels, xdev_tfidf)
svm_oof_train, svm_oof_test = get_oof(svm.LinearSVC(), xtrain_tfidf, training_labels, xdev_tfidf)


xgb_train = numpy.concatenate((nb_oof_train, lr_oof_train, svm_oof_train), axis=1)
xgb_dev = numpy.concatenate((nb_oof_test, lr_oof_test, svm_oof_test), axis=1)

xgboost_model = xgboost.XGBClassifier(random_state=1, learning_rate=0.01)
xgboost_classifier = train_model(xgboost_model, xgb_train, training_labels)
xgboost_predictions = get_predictions(xgboost_classifier, xgb_dev)
print("XGBOOST CLASSIFIER: ", get_model_accuracy(xgboost_predictions, test_labels))
print_confusion_matrix(xgboost_predictions, test_labels)
'''

if CROSS_LINGUAL:
    cross = 'cross_'
else:
    cross = ''
with open(data_path + "final_outputs/" + cross + LANGUAGE_CODE + ".tsv", 'w', newline='') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i, prediction in enumerate(LABEL_ENCODER.inverse_transform(lr_word_predictions)):
        tsv_writer.writerow([dev_data['tweet_id'][i], prediction])
