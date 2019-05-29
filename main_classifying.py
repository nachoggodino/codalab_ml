import xml.etree.ElementTree as ET
import io
import re
import csv
import nltk
import unidecode
import spacy
import hunspell

from scipy.sparse import coo_matrix, hstack
import sys
import os
import zipfile

import warnings

# Append script path to import path to locate tass_eval
sys.path.append(os.path.realpath(__file__))

# Import evalTask1 function fro tass_eval module
# from tass_eval import evalTask1

from re import finditer

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing, linear_model, naive_bayes, metrics, tree, svm, model_selection
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn import decomposition, ensemble
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from gensim.models import Word2Vec

import pandas as pd, numpy, textblob, string
import xgboost
# from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras import layers, models, optimizers
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM, GRU
# from keras.layers.embeddings import Embedding
# from keras.initializers import Constant

from imblearn.over_sampling import RandomOverSampler

from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter

from textacy import keyterms

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore')

LANGUAGE_CODE = ['es', 'cr', 'mx', 'pe', 'uy']
CROSS_LINGUAL = [True, False]
# CROSS_LINGUAL = [True]

# PARAMETERS
bTestPhase = False  # If we are doing test, then concatenate train + dev, if not use dev as test
bEvalPhase = True
bUpsampling = True

# dictionary = hunspell.Hunspell('es_ANY', hunspell_data_dir="./dictionaries")  # In case you're using CyHunspell
print("Loading Hunspell directory")
dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")  # In case you're using Hunspell

LABEL_ENCODER = preprocessing.LabelEncoder()
TERNARY_LABEL_ENCODER = preprocessing.LabelEncoder()
data_test_path = "./previous_years/ft_processed/csv/"  # TODO
data_path = "./codalab/DATASETS/public_data_development/"
data_eval_path = "./codalab/DATASETS/public_data_task1/"
# data_path = "../TASS2019/DATASETS/public_data/"

print("Loading Spacy Model")
lemmatizer = spacy.load("es_core_news_sm")  # GLOBAL to avoid loading the model several times

print("Loading NLTK stuff")
stemmer = nltk.stem.SnowballStemmer('spanish')
regex_uppercase = re.compile(r"\b[A-Z][A-Z]+\b")
stopwords = nltk.corpus.stopwords.words('spanish')

emoji_pattern = re.compile("[" 
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)


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

    result_df = pd.DataFrame()
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
    print("Extracting length feature")
    return [len(tweet) for tweet in tokenized_dataframe]


def extract_uppercase_feature(dataframe):
    print("Extracting uppercase feature")
    return [len(regex_uppercase.findall(tweet)) for tweet in dataframe]


def extract_hashtag_number_feature(dataframe):
    print("Extracting hashtag numbers feature")
    return[tweet.count('#') for tweet in dataframe]


def extract_question_mark_feature(dataframe):
    print("Extracting question mark")
    return [1 if re.search(r"[/?/]", tweet) is True else 0 for tweet in dataframe]


def extract_exclamation_mark_feature(dataframe):
    print("Extracting exclamation mark")
    return [1 if re.search(r"[/!/]", tweet) is True else 0 for tweet in dataframe]


def extract_letter_repetition_feature(dataframe):
    print("Extracting letter repetition")
    return [1 if re.search(r"(\w)(\1{2,})", tweet) is True else 0 for tweet in dataframe]


def extract_sent_words_feature(tokenized_data, data_feed):
    positive_voc, negative_voc = get_sentiment_vocabulary(data_feed, 3, 0)
    pos_result = []
    neg_result = []
    neutral_result = []
    none_result = []
    for index, tokenized_tweet in enumerate(tokenized_data):
        pos_count = sum(word in tokenized_tweet for word in positive_voc)
        neg_count = sum(word in tokenized_tweet for word in negative_voc)
        length = len(tokenized_tweet)

        pos_result.append(pos_count/length)
        neg_result.append(neg_count/length)
        neutral_result.append(0 if (pos_count + neg_count) == 0 else 1-(pos_count-neg_count)/(pos_count+neg_count))
        none_result.append(1-(max(neg_count, pos_count)/length))
    return pos_result, neg_result, neutral_result, none_result


def get_sentiment_vocabulary(data, positive, negative):
    print("Sentiment Vocabulary Extraction")
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
    # positive_vocabulary, negative_vocabulary = keyterms.most_discriminating_terms(pos_neg_tweets, pos_neg_bool_labels)

    pos_df = pd.read_csv('./lexicons/isol/positivas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
    neg_df = pd.read_csv('./lexicons/isol/negativas_mejorada.csv', encoding='latin-1', header=None, names=['words'])

    return pos_df['words'].array, neg_df['words'].array


def text_preprocessing(data):
    result = data
    result = [tweet.replace('\n', '').strip() for tweet in result]  # Newline and leading/trailing spaces
    result = [emoji_pattern.sub(r'', tweet) for tweet in result]
    result = [tweet.replace(u'\u2018', "'").replace(u'\u2019', "'") for tweet in result]  # Quotes replace by general
    result = [re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), tweet) for tweet in result]  # Hashtag
    result = [tweet.lower() for tweet in result]
    result = [re.sub(r"^.*http.*$", 'http', tweet) for tweet in result]  # Remove all http contents
    result = [re.sub(r"\B@\w+", '', tweet) for tweet in result]  # Remove all usernames
    result = [re.sub(r"(\w)(\1{2,})", r"\1", tweet) for tweet in result]  # Remove all letter repetitions
    result = [re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"\d+", '', tweet) for tweet in result]  # Remove all numbers
    result = [tweet.translate(str.maketrans('', '', string.punctuation + '¡')) for tweet in result]  # Remove punctuation

    return result


def perform_upsampling(dataframe):
    ros = RandomOverSampler()
    x_resampled, y_resampled = ros.fit_resample(dataframe[['tweet_id', 'content']], dataframe['sentiment'])
    df = pd.DataFrame(data=x_resampled[0:, 0:], columns=['tweet_id', 'content'])
    df['sentiment'] = y_resampled
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])


def libreoffice_processing(tokenized_data):
    print("Libreoffice processing")
    return [[word if dictionary.spell(word) is True else next(iter(dictionary.suggest(word)), word) for word in tweet] for tweet in tokenized_data]


def tokenize_list(datalist):
    print("Tokenizing")
    return [nltk.word_tokenize(row) for row in datalist]


def remove_accents(tokenized_data):
    print("Removing accents")
    return [[unidecode.unidecode(word) for word in tweet] for tweet in tokenized_data]


def remove_stopwords(tokenized_data):
    print('Removing stopwords')
    return [[word for word in row if word not in stopwords] for row in tokenized_data]


def stem_list(datalist):
    print("Applying stemming")
    return [[stemmer.stem(word) for word in row] for row in datalist]


def lemmatize_list(datalist):
    print("Lemmatizing data...")
    return [[token.lemma_ for token in lemmatizer(row)] for row in datalist]


def perform_count_vectors(train_set_x, valid_set_x):
    # create a count vectorizer object
    print("Count word level")
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(train_set_x)

    # transform the training and validation data using count vectorizer object
    return count_vect.transform(train_set_x), count_vect.transform(valid_set_x)


def perform_tf_idf_vectors(train_set_x, valid_set_x):
    print("word level tf-idf")
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(train_set_x)
    return tfidf_vect.transform(train_set_x), tfidf_vect.transform(valid_set_x)


def add_feature(matrix, new_features):
    matrix_df = pd.DataFrame(matrix.todense())
    return pd.concat([matrix_df, new_features], axis=1)


def train_model(classifier, x_train, y_train, x_test, y_test, clf_name, ternary_input=False, is_vso=False, description=''):
    if ternary_input:
        threshold = 1 if y_train.value_counts()[1] > y_train.value_counts()[2] else 0
        print("Ternary mode selected. NEU and NONE will be both treated as NEU" if threshold is 1 else
              "Ternary mode selected. NEU and NONE will be both treated as NONE")
        y_train = [label - 1 if label > 1 else label for label in y_train]

    classifier.fit(x_train, y_train)
    predictions, probabilities = get_predictions(classifier, x_test, is_vso)

    if ternary_input:
        predictions = [pred+1 if pred > threshold else pred for pred in predictions]

    if y_test is not None:
        print(clf_name + ", " + description + ":")
        print_confusion_matrix(predictions, y_test)
    return classifier, predictions, probabilities


def get_predictions(trained_classifier, feature_test_vector, is_vso=False):
    if is_vso:
        return trained_classifier.predict(feature_test_vector), trained_classifier.decision_function(feature_test_vector)
    else:
        return trained_classifier.predict(feature_test_vector), trained_classifier.predict_proba(feature_test_vector)


def get_model_accuracy(predictions, validation_labels):
    return metrics.accuracy_score(predictions, validation_labels)


def print_confusion_matrix(predictions, labels):
    preds = pd.Series(predictions, name='Predicted')
    labs = pd.Series(labels, name='Actual')
    df_confusion = pd.crosstab(labs, preds)
    # print(df_confusion)
    prec = precision_score(labs, preds, average='macro')
    rec = recall_score(labs, preds, average='macro')
    score = 2*(prec*rec)/(prec+rec)
    print("F1-SCORE: " + str(score))
    # print("Recall: " + str(rec))
    # print("Precision: " + str(prec))
    return


def get_averaged_predictions(predictions_array):
    averaged_predictions = numpy.zeros(predictions_array[0].shape)
    for i, predictions in enumerate(predictions_array):
        averaged_predictions += predictions
    averaged_predictions = numpy.divide(averaged_predictions, len(predictions_array)).argmax(1)
    return averaged_predictions


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
        x_tr = x_train.loc[train_index, :]
        y_tr = y_train.loc[train_index]
        x_te = x_train.loc[test_index, :]
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
        else:
            print("Embedding Not found" + str(word))

    return num_words, embedding_matrix, max_length, tweet_pad


# Create a zip file for a folder
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def read_files(sLang, bCross):
    train_data = pd.DataFrame()
    dev_data = pd.DataFrame()
    tst_data = None
    eval_data = None

    if bCross is True:
        dev_cross_data = pd.DataFrame()
        for sLangCross in [x for x in LANGUAGE_CODE if x != sLang]:
            tree_train_cross = ET.parse(data_path + sLangCross + "/intertass_" + sLangCross + "_train.xml")
            df_train_cross = get_dataframe_from_xml(tree_train_cross)

            tree_dev_cross = ET.parse(data_path + sLangCross + "/intertass_" + sLangCross + "_dev.xml")
            df_dev_cross = get_dataframe_from_xml(tree_dev_cross)

            train_data = pd.concat([train_data, df_train_cross], ignore_index=True)
            dev_cross_data = pd.concat([dev_cross_data, df_dev_cross], ignore_index=True)

        # We add train_cross + dev_cross (but never we can add dev_lang)
        train_data = pd.concat([train_data, dev_cross_data], ignore_index=True)

    else:
        tree_train = ET.parse(data_path + sLang + "/intertass_" + sLang + "_train.xml")
        train_data = get_dataframe_from_xml(tree_train)

    if bEvalPhase:
        tree_eval = ET.parse(data_eval_path + "intertass_" + sLang + "_test.xml")
        eval_data = get_dataframe_from_xml(tree_eval)

    if bTestPhase:
        tst_data = pd.read_csv(data_test_path + sLang + '_general_test.csv', delimiter='\t')

    # When working with cross or mono, we can only use dev_lang
    tree_dev = ET.parse(data_path + sLang + "/intertass_" + sLang + "_dev.xml")
    dev_data = get_dataframe_from_xml(tree_dev)

    return train_data, dev_data, tst_data, eval_data


def fasttext_to_df(fasttext_format_path):
    with open(fasttext_format_path, 'r') as ft_file:
        negs, poss, neus, nones = [], [], [], []
        for i, line in enumerate(ft_file):
            words = line.split()
            for j, word in enumerate(words):
                if j % 2 is 1:
                    continue
                else:
                    label = word.replace('__label__', '')
                    if label == 'N':
                        negs.append(float(words[j + 1]))
                    if label == 'P':
                        poss.append(float(words[j + 1]))
                    if label == 'NEU':
                        neus.append(float(words[j + 1]))
                    if label == 'NONE':
                        nones.append(float(words[j + 1]))

        final_df = pd.DataFrame({
            'N': negs,
            'NEU': neus,
            'NONE': nones,
            'P': poss,
        })
        return final_df


if __name__ == '__main__':

    # GET THE DATA
    output_dir = "./final_outputs/"
    for bCross in CROSS_LINGUAL:

        if bCross:
            continue

        print('----------------- CROSS : ' + str(bCross) + ' ----------------')

        bTern = False  # Set to True for training with NEU and NONE as one category
        prefix = 'LR'

        if bCross is True:
            output_dir = output_dir + '/cross/'
        else:
            output_dir = output_dir + '/mono/'

        for sLang in LANGUAGE_CODE:

            print('** LANG: ' + sLang)

            train_data, dev_data, tst_data, eval_data = read_files(sLang, bCross)
            if bUpsampling:
                train_data = perform_upsampling(train_data)

            # PRE-PROCESSING
            preprocessed_train_content = text_preprocessing(train_data['content'])
            preprocessed_dev_content = text_preprocessing(dev_data['content'])
            if bTestPhase is True:
                preprocessed_tst_content = text_preprocessing(tst_data['content'])
            if bEvalPhase is True:
                preprocessed_eval_content = text_preprocessing(eval_data['content'])

            # TOKENIZE
            tokenized_train_content = tokenize_list(preprocessed_train_content)
            tokenized_dev_content = tokenize_list(preprocessed_dev_content)
            if bTestPhase is True:
                tokenized_tst_data = tokenize_list(preprocessed_tst_content)
            if bEvalPhase is True:
                tokenized_eval_data = tokenize_list(preprocessed_eval_content)

            # FEATURE EXTRACTION
            train_data['tweet_length'] = extract_length_feature(tokenized_train_content)
            train_data['has_uppercase'] = extract_uppercase_feature(train_data['content'])
            train_data['question_mark'] = extract_question_mark_feature(train_data['content'])
            train_data['exclamation_mark'] = extract_exclamation_mark_feature(train_data['content'])
            train_data['letter_repetition'] = extract_letter_repetition_feature(train_data['content'])
            train_data['hashtag_number'] = extract_hashtag_number_feature(train_data['content'])
            train_data['pos_voc'], train_data['neg_voc'], train_data['neu_voc'], train_data['none_voc'] = \
                extract_sent_words_feature(tokenized_train_content, tokenized_train_content)

            dev_data['tweet_length'] = extract_length_feature(tokenized_dev_content)
            dev_data['has_uppercase'] = extract_uppercase_feature(dev_data['content'])
            dev_data['question_mark'] = extract_question_mark_feature(dev_data['content'])
            dev_data['exclamation_mark'] = extract_exclamation_mark_feature(dev_data['content'])
            dev_data['letter_repetition'] = extract_letter_repetition_feature(dev_data['content'])
            dev_data['hashtag_number'] = extract_hashtag_number_feature(dev_data['content'])
            dev_data['pos_voc'], dev_data['neg_voc'], dev_data['neu_voc'], dev_data['none_voc'] = \
                extract_sent_words_feature(tokenized_dev_content, tokenized_train_content)

            if bTestPhase:
                tst_data['tweet_length'] = extract_length_feature(tokenized_tst_data)
                tst_data['has_uppercase'] = extract_uppercase_feature(tst_data['content'])
                tst_data['question_mark'] = extract_question_mark_feature(tst_data['content'])
                tst_data['exclamation_mark'] = extract_exclamation_mark_feature(tst_data['content'])
                tst_data['letter_repetition'] = extract_letter_repetition_feature(tst_data['content'])
                tst_data['hashtag_number'] = extract_hashtag_number_feature(tst_data['content'])
                tst_data['pos_voc'], tst_data['neg_voc'], tst_data['neu_voc'], tst_data['none_voc'] = \
                    extract_sent_words_feature(tokenized_tst_data, tokenized_train_content)

            if bEvalPhase:
                eval_data['tweet_length'] = extract_length_feature(tokenized_eval_data)
                eval_data['has_uppercase'] = extract_uppercase_feature(eval_data['content'])
                eval_data['question_mark'] = extract_question_mark_feature(eval_data['content'])
                eval_data['exclamation_mark'] = extract_exclamation_mark_feature(eval_data['content'])
                eval_data['letter_repetition'] = extract_letter_repetition_feature(eval_data['content'])
                eval_data['hashtag_number'] = extract_hashtag_number_feature(eval_data['content'])
                eval_data['pos_voc'], eval_data['neg_voc'], eval_data['neu_voc'], eval_data['none_voc'] = \
                    extract_sent_words_feature(tokenized_eval_data, tokenized_train_content)

            # clean_train_content = tokenized_train_content  # remove_stopwords(tokenized_train_content)
            # clean_dev_content = tokenized_dev_data  # remove_stopwords(tokenized_dev_data)
            # if bTestPhase is True:
            #     clean_tst_content = tokenized_tst_data  # remove_stopwords(tokenized_tst_data)

            # LIBRE OFFICE PROCESSING
            # libreoffice_train_tweets = [row for row in libreoffice_processing(tokenized_train_content)]
            # libreoffice_dev_tweets = [row for row in libreoffice_processing(tokenized_dev_content)]
            # if bTestPhase:
            #     libreoffice_tst_tweets = [row for row in libreoffice_processing(tokenized_tst_data)]
            # if bEvalPhase:
            #     libreoffice_eval_tweets = [row for row in libreoffice_processing(tokenized_eval_data)]

            # LEMMATIZING
            # lemmatized_train_tweets = lemmatize_list(preprocessed_train_content)
            # lemmatized_dev_tweets = lemmatize_list(preprocessed_dev_content)
            # if bTestPhase is True:
            #     lemmatized_tst_tweets = lemmatize_list(preprocessed_tst_content)

            # # REMOVING ACCENTS
            # without_accents_train = remove_accents(lemmatized_train_tweets)
            # without_accents_dev = remove_accents(lemmatized_dev_tweets)
            # if bTestPhase is True:
            #   without_accents_tst = remove_accents(lemmatized_tst_tweets)

            final_train_content = [TreebankWordDetokenizer().detokenize(row) for row in tokenized_train_content]
            final_dev_content = [TreebankWordDetokenizer().detokenize(row) for row in tokenized_dev_content]
            if bTestPhase is True:
                final_tst_content = [TreebankWordDetokenizer().detokenize(row) for row in tokenized_tst_data]
            if bEvalPhase is True:
                final_eval_content = [TreebankWordDetokenizer().detokenize(row) for row in tokenized_eval_data]

            # COUNT VECTORS
            # if bTestPhase is True and bCross is True:  # Add train + dev to have more data
            #     x_train_count_vectors, x_tst_count_vectors = perform_count_vectors(final_train_content, final_tst_content)
            # elif bTestPhase is True:
            #     x_train_count_vectors, x_tst_count_vectors = perform_count_vectors(final_train_content + final_dev_content,
            #                                                                                             final_tst_content)
            # else:
            #     x_train_count_vectors, x_dev_count_vectors = perform_count_vectors(final_train_content, final_dev_content)


            # # TF-IDF VECTORS
            # if bTestPhase is True and bCross is True:
            #     xtrain_tfidf, xtst_tfidf = perform_tf_idf_vectors(final_train_content, final_tst_content)
            # elif bTestPhase is True:
            #     xtrain_tfidf, xtst_tfidf = perform_tf_idf_vectors(final_train_content + final_dev_content, final_tst_content)
            # else:
            #     xtrain_tfidf, xdev_tfidf = perform_tf_idf_vectors(final_train_content, final_dev_content)

            train_features = pd.DataFrame({
                'tweet_length': train_data['tweet_length'],
                'has_uppercase': train_data['has_uppercase'],
                'exclamation_mark': train_data['exclamation_mark'],
                'question_mark': train_data['question_mark'],
                'hashtag_number': train_data['hashtag_number'],
                'pos_voc': train_data['pos_voc'],
                'neg_voc': train_data['neg_voc'],
                'neu_voc': train_data['neu_voc'],
                'none_voc': train_data['none_voc'],
                'letter_repetition': train_data['letter_repetition']
            })

            dev_features = pd.DataFrame({
                'tweet_length': dev_data['tweet_length'],
                'has_uppercase': dev_data['has_uppercase'],
                'exclamation_mark': dev_data['exclamation_mark'],
                'question_mark': dev_data['question_mark'],
                'hashtag_number': dev_data['hashtag_number'],
                'pos_voc': dev_data['pos_voc'],
                'neg_voc': dev_data['neg_voc'],
                'neu_voc': dev_data['neu_voc'],
                'none_voc': dev_data['none_voc'],
                'letter_repetition': dev_data['letter_repetition']
            })

            if bTestPhase is True:
                tst_features = pd.DataFrame({
                    'tweet_length': tst_data['tweet_length'],
                    'has_uppercase': tst_data['has_uppercase'],
                    'exclamation_mark': tst_data['exclamation_mark'],
                    'question_mark': tst_data['question_mark'],
                    'hashtag_number': tst_data['hashtag_number'],
                    'pos_voc': tst_data['pos_voc'],
                    'neg_voc': tst_data['neg_voc'],
                    'neu_voc': tst_data['neu_voc'],
                    'none_voc': tst_data['none_voc'],
                    'letter_repetition': tst_data['letter_repetition']
                })

            if bEvalPhase is True:
                eval_features = pd.DataFrame({
                    'tweet_length': eval_data['tweet_length'],
                    'has_uppercase': eval_data['has_uppercase'],
                    'exclamation_mark': eval_data['exclamation_mark'],
                    'question_mark': eval_data['question_mark'],
                    'hashtag_number': eval_data['hashtag_number'],
                    'pos_voc': eval_data['pos_voc'],
                    'neg_voc': eval_data['neg_voc'],
                    'neu_voc': eval_data['neu_voc'],
                    'none_voc': eval_data['none_voc'],
                    'letter_repetition': eval_data['letter_repetition']
                })

            # CONCATENATE VECTORS AND FEATURES

            # if bTestPhase is True and bCross is True:  # Use train_dev_cross + dev
            #     x_train_count_vectors = add_feature(x_train_count_vectors, train_features)
            #     x_tst_count_vectors = add_feature(x_tst_count_vectors, tst_features)
            # elif bTestPhase is True:  # Combine train + dev
            #     x_train_count_vectors = add_feature(x_train_count_vectors + x_dev_count_vectors,
            #                                         pd.concat([train_features, dev_features]))
            #     x_tst_count_vectors = add_feature(x_tst_count_vectors, tst_features)
            # else:
            #     x_train_count_vectors = add_feature(x_train_count_vectors, train_features)
            #     x_dev_count_vectors = add_feature(x_dev_count_vectors, dev_features)

            # if bTestPhase is True and bCross is True:  # Use train_dev_cross + dev
            #     xtrain_tfidf = add_feature(xtrain_tfidf, train_features)
            #     xtst_tfidf = add_feature(xtst_tfidf, tst_features)
            # elif bTestPhase is True:  # Combine train + dev
            #     xtrain_tfidf = add_feature(xtrain_tfidf + xdev_tfidf,
            #                                pd.concat([train_features, dev_features]))
            #     xtst_tfidf = add_feature(xtst_tfidf, tst_features)
            # else:
            #     xtrain_tfidf = add_feature(xtrain_tfidf, train_features)
            #     xdev_tfidf = add_feature(xdev_tfidf, dev_features)

            # TRAINING
            # Choose the training, development and test sets
            training_set = train_features
            training_labels = train_data['sentiment']
            dev_set = dev_features
            dev_labels = dev_data['sentiment']
            if bTestPhase is True:
                tst_set = tst_features
                tst_labels = tst_data['sentiment']
            if bEvalPhase:
                eval_set = eval_features
                eval_labels = None  # We don't have the eval labels yet


            # WORD EMBEDDINGS
            # load the pre-trained word-embedding vectors

            # embeddings_index = {}
            # for i, line in enumerate(open('./VECTOR_EMBEDDINGS/cc.es.300.vec', encoding='utf-8')):
            #     if i % 100000 == 0:
            #         print(i)
            #     values = line.split()
            #     embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
            #
            # train_num_words, train_embedding_matrix, train_max_length, train_pad = get_pad_sequences(final_train_content, embeddings_index)
            # if bTestPhase is True:
                # tst_num_words, tst_embedding_matrix, tst_max_length, tst_pad = get_pad_sequences(final_tst_content, embeddings_index)
            # else:
                # dev_num_words, dev_embedding_matrix, dev_max_length, dev_pad = get_pad_sequences(final_dev_content, embeddings_index)
            #
            # model = Sequential()
            # embedding_layer = Embedding(train_num_words, 300, embeddings_initializer=Constant(train_embedding_matrix),
            #                             input_length=train_max_length, trainable=False)
            # model.add(embedding_layer)
            # model.add(GRU(units=16, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
            # model.add(Dense(4, activation='softmax'))
            #
            # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #
            # if bTestPhase is True:
                # model.fit(train_pad, training_labels, batch_size=16, epochs=10, verbose=2)
                # nn_predictions_tst = model.predict_classes(tst_pad, batch_size=16, verbose=2)
            # else:
                # model.fit(train_pad, training_labels, batch_size=16, epochs=10, validation_data=(dev_pad, dev_labels), verbose=2)
                # nn_predictions_dev = model.predict_classes(dev_pad, batch_size=16, verbose=2)
            #
            # from keras.models import Model
            #
            # embedding_layer = Embedding(train_num_words, 300, embeddings_initializer=Constant(train_embedding_matrix),
            #                             input_length=train_max_length, trainable=False)
            # gru = GRU(units=16, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(embedding_layer)
            # d = Dense(4, activation='softmax')(gru)
            # model = Model(inputs=embedding_layer, outputs=d)
            # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #
            # if bTestPhase is True:
                # model.fit(train_pad, training_labels, batch_size=16, epochs=10, validation_data=(tst_pad, tst_labels), verbose=2)
                # nn_predictions_tst = model.predict_classes(tst_pad, batch_size=16, verbose=2)
            # else:
                # model.fit(train_pad, training_labels, batch_size=16, epochs=10, validation_data=(dev_pad, dev_labels), verbose=2)
                # nn_predictions_dev = model.predict_classes(dev_pad, batch_size=16, verbose=2)
                # print("NN, Word Embeddings DEV: ", get_model_accuracy(nn_predictions, dev_labels))
                # print_confusion_matrix(nn_predictions, dev_labels)

            # CLASSIFIERS
            lr = linear_model.LogisticRegression()
            nb = naive_bayes.MultinomialNB()
            dt = tree.DecisionTreeClassifier()
            svm = SVC(probability=True)
            rf = RandomForestClassifier()
            et = ExtraTreesClassifier()
            ada = AdaBoostClassifier()
            gb = GradientBoostingClassifier()
            sgd = SGDClassifier()

            all_classifiers = [lr, nb, ada, gb]
            all_classif_names = ['LR', 'NB', 'ADA', 'GB']
            all_predictions, all_vsr_predictions, all_vso_predictions = [], [], []
            all_probabilities, all_vsr_probabilities, all_vso_probabilities = [], [], []
            all_final_predictions, all_vsr_final_predictions, all_vso_final_predictions = [], [], []
            all_tst_probabilities, all_tst_vsr_probabilities, all_tst_vso_probabilities = [], [], []

            for i, clf in enumerate(all_classifiers):
                print("TRAINING " + all_classif_names[i])
                print()
                vsr_clf = OneVsRestClassifier(clf)
                vso_clf = OneVsOneClassifier(clf)

                classif, preds, probs = train_model(clf, training_set, training_labels, dev_set, dev_labels,
                                                    'NORMAL', bTern)
                vsr_classif, vsr_preds, vsr_probs = train_model(vsr_clf, training_set, training_labels, dev_set,
                                                                dev_labels, 'ONE VS ALL', bTern)
                vso_classif, vso_preds, vso_probs = train_model(vso_clf, training_set, training_labels, dev_set,
                                                                dev_labels, 'ONE VS ONE', bTern, True)
                if bTestPhase:
                    a, tst_probs = get_predictions(classif, tst_set)
                    b, tst_vsr_probs = get_predictions(vsr_classif, tst_set)
                    c, tst_vso_probs = get_predictions(vso_classif, tst_set, is_vso=True)

                if bEvalPhase:
                    classif, preds, probs = train_model(clf, training_set, training_labels, eval_set, eval_labels,
                                                        'NORMAL', bTern)
                    vsr_classif, vsr_preds, vsr_probs = train_model(vsr_clf, training_set, training_labels, eval_set,
                                                                    eval_labels, 'ONE VS ALL', bTern)
                    vso_classif, vso_preds, vso_probs = train_model(vso_clf, training_set, training_labels, eval_set,
                                                                    eval_labels, 'ONE VS ONE', bTern, True)

                all_predictions.append(preds)
                all_probabilities.append(probs)
                all_vsr_predictions.append(vsr_preds)
                all_vsr_probabilities.append(vsr_probs)
                all_vso_predictions.append(vso_preds)
                all_vso_probabilities.append(vso_probs)
                print()

            if not bEvalPhase:
                print("VOTING ENSEMBLE")
                print_confusion_matrix(get_averaged_predictions(all_probabilities), dev_labels)
                print()

            str_mode = 'cross' if bCross else 'mono'
            str_phase = 'test' if bTestPhase else 'dev'
            str_full_reduced = 'reduced' if bTern else 'full'

            # FASTTEXT
            fasttext_df = fasttext_to_df('./new_fasttext/{0}/{1}/{2}_{1}_{0}.out'.format(str_mode, str_phase, sLang))

            # Use only if you want to use the fasttext from fasttext folder, not from new_fasttext
            # fasttext_path = './fasttext/{}_fasttext_outputs_dev-test_full-reduced/'.format(str_mode)
            # fasttext_file = '{}_{}_fasttext_{}.csv'.format(sLang, str_phase, str_full_reduced)
            # fasttext_df = pd.read_csv(fasttext_path + fasttext_file)
            # fasttext_df = fasttext_df.drop(columns='ID')

            print("FASTTEXT MODEL")
            fasttext_probabilities = fasttext_df.to_numpy()
            fasttext_predictions = fasttext_probabilities.argmax(1)
            if not bEvalPhase:
                print_confusion_matrix(fasttext_predictions, dev_labels)
            print()

            # BERT
            print("BERT MODEL")
            if not (bTern and bCross):  # The only mode without predictions
                bert_path = './bert/{}_bert_dev-test_full-reduced/'.format(str_mode)
                bert_file = '{}_{}_bert_{}.csv'.format(sLang, str_phase, str_full_reduced)

                bert_df = pd.read_csv(bert_path + bert_file)
                bert_df = bert_df.drop(bert_df.columns[0], axis=1)
                bert_df = bert_df.drop(columns='id')

                bert_probabilities = bert_df.to_numpy()
                bert_predictions = bert_probabilities.argmax(1)
                if not bEvalPhase:
                    print_confusion_matrix(bert_predictions, dev_labels)
                print()

            # GLOVE + BPE + W2V
            print("G+B+W MODEL")
            gbw_path = './outputs_flair_bpe_glove_w2v_clean_May29/{}/{}/'.format(sLang, str_mode)
            gbw_file = '{}_test_glove_word2vec_bpe_forTest_full.csv'.format(sLang, str_phase, str_mode)

            gbw_df = pd.read_csv(gbw_path + gbw_file, sep=',')
            gbw_df = gbw_df.drop(gbw_df.columns[0], axis=1)
            # gbw_df = gbw_df.drop(columns='ID')

            gbw_probabilities = gbw_df.to_numpy()
            gbw_predictions = gbw_probabilities.argmax(1)
            if not bEvalPhase:
                print_confusion_matrix(gbw_predictions, dev_labels)
            print()

            print('-------------- FINAL ENSEMBLE --------------')
            print()

            print('From Normal Models:')
            print()

            submitting_predictions = []

            for i, prob_matrix in enumerate(all_probabilities):
                print('With ' + all_classif_names[i])
                probabilities_for_voting_ensemble = [
                    # bert_probabilities,
                    gbw_probabilities,
                    # fasttext_probabilities,
                    prob_matrix
                ]
                final_predictions = get_averaged_predictions(probabilities_for_voting_ensemble)
                all_final_predictions.append(final_predictions)

                if not bEvalPhase:
                    print_confusion_matrix(final_predictions, dev_labels)
                print()

            print('From OneVsRest Models:')
            print()

            for i, prob_matrix in enumerate(all_vsr_probabilities):
                print('With ' + all_classif_names[i])
                probabilities_for_voting_ensemble = [
                    # bert_probabilities,
                    gbw_probabilities,
                    # fasttext_probabilities,
                    prob_matrix
                ]
                final_predictions = get_averaged_predictions(probabilities_for_voting_ensemble)
                all_vsr_final_predictions.append(final_predictions)

                if not bEvalPhase:
                    print_confusion_matrix(final_predictions, dev_labels)
                print()

            print('From OneVsOne Models:')
            print()
            for i, prob_matrix in enumerate(all_vso_probabilities):
                print('With ' + all_classif_names[i])
                probabilities_for_voting_ensemble = [
                    # bert_probabilities,
                    gbw_probabilities,
                    # fasttext_probabilities,
                    prob_matrix
                ]
                final_predictions = get_averaged_predictions(probabilities_for_voting_ensemble)
                all_vso_final_predictions.append(final_predictions)

                if not bEvalPhase:
                    print_confusion_matrix(final_predictions, dev_labels)
                print()

            print()
            print()
            print('--------------- NEXT LANGUAGE ------------------')

            #XGBOOST SECOND-LEVEL CLASSIFIER

            # fb_oof_train, fb_oof_dev = get_oof(naive_bayes.MultinomialNB(), xtrain_tfidf, training_labels, xdev_tfidf)
            # lr_oof_train, lr_oof_dev = get_oof(linear_model.LogisticRegression(), xtrain_tfidf, training_labels, xdev_tfidf)
            # svm_oof_train, svm_oof_dev = get_oof(svm.LinearSVC(), xtrain_tfidf, training_labels, xdev_tfidf)


            # xgb_train = numpy.concatenate((nb_oof_train, lr_oof_train, svm_oof_train), axis=1)
            # xgb_dev = numpy.concatenate((nb_oof_test, lr_oof_test, svm_oof_dev), axis=1)
            #
            # xgboost_model = xgboost.XGBClassifier(random_state=1, learning_rate=0.01)
            # xgboost_classifier = train_model(xgboost_model, xgb_train, training_labels)
            # xgboost_predictions = get_predictions(xgboost_classifier, xgb_dev)
            # print("XGBOOST CLASSIFIER: ", get_model_accuracy(xgboost_predictions, dev_labels))
            # print_confusion_matrix(xgboost_predictions, dev_labels)

            # Decide which outputs to export: ['LR', 'NB', 'SVM' 'DT', 'RF', 'ET', 'ADA', 'GB', 'SGD']
            if bCross:
                if sLang == 'es':
                    submitting_predictions = all_final_predictions[0]
                elif sLang == 'cr':
                    submitting_predictions = all_final_predictions[1]
                elif sLang == 'mx':
                    submitting_predictions = all_final_predictions[0]
                elif sLang == 'pe':
                    submitting_predictions = all_vsr_final_predictions[3]
                elif sLang == 'uy':
                    submitting_predictions = all_vsr_final_predictions[3]
            else:
                if sLang == 'es':
                    submitting_predictions = all_final_predictions[0]
                elif sLang == 'cr':
                    submitting_predictions = all_vsr_final_predictions[1]
                elif sLang == 'mx':
                    submitting_predictions = all_final_predictions[2]
                elif sLang == 'pe':
                    submitting_predictions = all_final_predictions[0]
                elif sLang == 'uy':
                    submitting_predictions = all_final_predictions[2]

            if os.path.exists(output_dir) is False:
                print("Creating output directory " + output_dir)
                os.makedirs(output_dir)

            # with open(data_path + "final_outputs/" + cross + LANGUAGE_CODE + ".tsv", 'w', newline='') as out_file:
            #     tsv_writer = csv.writer(out_file, delimiter='\t')
            #     for i, prediction in enumerate(LABEL_ENCODER.inverse_transform(lr_word_predictions)):
            #         tsv_writer.writerow([dev_data['tweet_id'][i], prediction])

            with open(output_dir + sLang + ".tsv", 'w', newline='') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                for i, prediction in enumerate(LABEL_ENCODER.inverse_transform(submitting_predictions)):
                    if bEvalPhase:
                        tsv_writer.writerow([eval_data['tweet_id'][i], prediction])
                    else:
                        tsv_writer.writerow([dev_data['tweet_id'][i], prediction])

            # with zipfile.ZipFile(output_dir + prefix + '_' + sLang + ".zip", 'w') as file_zip:
            #     print("Creating zip file: " + output_dir + prefix + '_' + sLang + ".zip")
            #     file_zip.write(output_dir + sLang + ".tsv")
            #
            # run_file = output_dir + sLang + ".tsv"
            # gold_file = data_path + sLang + "/intertass_" + sLang + "_dev_gold.tsv"
            #
            # scores = evalTask1(gold_file, run_file)
            # with open(output_dir + prefix + '_' + sLang + ".res", 'w', newline='') as out_file:
            #     print("f1_score: %f\n" % scores['maf1'])
            #     out_file.write("f1_score: %f\n" % scores['maf1'])
            #     print("precision: %f\n" % scores['map'])
            #     out_file.write("precision: %f\n" % scores['map'])
            #     print("recall: %f\n" % scores['mar'])
            #     out_file.write("recall: %f\n" % scores['mar'])

        zipf = zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir('output_dir', zipf)
        zipf.close()
