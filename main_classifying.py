import xml.etree.ElementTree as ET
import io
import re
import csv
import nltk
import unidecode
import spacy
import hunspell
import swifter
from googletrans import Translator
import utils
from utils import print_confusion_matrix
import tweet_preprocessing

from scipy.sparse import coo_matrix, hstack
import sys
import os
import zipfile

import warnings

# Append script path to import path to locate tass_eval
# sys.path.append(os.path.realpath(__file__))

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

import flair
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, BertEmbeddings, ELMoEmbeddings, CharacterEmbeddings

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

import warnings
warnings.filterwarnings('ignore')

LANGUAGE_CODE = ['es', 'cr', 'mx', 'pe', 'uy']
oovs_avg = 0

# TODO PARAMETERS (Decide the Pipeline) ###############################################################################
bTestPhase = True  # If true, a second result is given for validation.
bEvalPhase = False  # If true, the test set is used.
bUpsampling = True  # If true, upsampling is performed.
bTwoWayTranslation = False  # If true, data is augmented using the two-way translation strategy. API Call problems
bLexicons = True  # If true, the sentiment vocabulary uses external lexicons.
bLemmatize = False  # If true, the data is lemmatized.
bRemoveAccents = False  # If true, the accents are removed from the data
bRemoveStopwords = False  # If true, stopwords are removed from the data
bLibreOffice = False  # If true, words not in the libreoffice dictionary are corrected
bReduced = False  # If true, NEU and NONE are treated as one category
bOneVsRest = False  # If true, the classifier uses a One vs All strategy
bClassifVotingEnsemble = False  # If true, shows a result of an ensemble of all the selected classifiers (lr, nb, gb...)
bCountVectors = False  # If true, count vectors are performed
bTfidf = False  # If true, tf-idf vectors are performed
bClassicBow = False  # If true, bCountVectors must also be true
# TODO ################################################################################################################

print("Loading Hunspell directory")
dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")  # In case you're using Hunspell

print("Loading Spacy Model")
lemmatizer = spacy.load("es_core_news_md")  # GLOBAL to avoid loading the model several times

print("Loading NLTK stuff")
stemmer = nltk.stem.SnowballStemmer('spanish')
regex_uppercase = re.compile(r"\b[A-Z][A-Z]+\b")  # TODO
stopwords = nltk.corpus.stopwords.words('spanish')

translator = Translator()





def extract_length_feature(sentences_list):
    print("Extracting length feature")
    return [len(tweet.split(' ')) for tweet in sentences_list]


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
    positive_vocabulary, negative_vocabulary = keyterms.most_discriminating_terms(pos_neg_tweets, pos_neg_bool_labels)

    pos_df = pd.read_csv('./lexicons/isol/positivas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
    neg_df = pd.read_csv('./lexicons/isol/negativas_mejorada.csv', encoding='latin-1', header=None, names=['words'])

    if bLexicons:
        return pos_df['words'].array, neg_df['words'].array
    else:
        return positive_vocabulary, negative_vocabulary


def two_way_translation(data, sLang):
    print("Data Augmentation: Performing Two-Way Translation")
    result = []
    returned = []
    for tweet in data:
        result.append(translator.translate(tweet, src='es', dest=sLang).text)
    for tweet in result:
        print('second-way')
        returned.append(translator.translate(tweet, src=sLang, dest='es').text)
    return returned
    # return [translator.translate(translator.translate(tweet, src='es', dest=sLang).text, src=sLang, dest='es').text for tweet in data]


def remove_accents(tokenized_sentence):
    return [unidecode.unidecode(word) for word in tokenized_sentence]


def remove_stopwords(tokenized_data):
    return [word for word in tokenized_data if word not in stopwords]


def stem_list(datalist):
    print("Applying stemming")
    return [[stemmer.stem(word) for word in row] for row in datalist]


def lemmatize_sentence(sentence):
    data = utils.untokenize_sentence(sentence)
    return [token.lemma_ for token in lemmatizer(data)]


def perform_count_vectors(train_set, dev_set, print_oovs=False, classic_bow=False):
    train = [utils.untokenize_sentence(sentence) for sentence in train_set]
    dev = [utils.untokenize_sentence(sentence) for sentence in dev_set]
    print("Performing CountVectors")
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', binary=classic_bow, min_df=1, lowercase=False)
    count_vect.fit(train)
    if print_oovs:
        dev_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', binary=classic_bow)
        dev_vect.fit(dev)
        oovs = [word for word in dev_vect.vocabulary_ if word not in count_vect.vocabulary_]
        print(oovs)
        print("Length of the vocabulary: {}".format(len(count_vect.vocabulary_)))
        print("OOVS: {} ({}% of the tested vocabulary)".format(len(oovs), len(oovs)*100/len(dev_vect.vocabulary_)))
        print()

    return count_vect.transform(train), count_vect.transform(dev)


def perform_tf_idf_vectors(train_set, dev_set):
    train = [utils.untokenize_sentence(sentence) for sentence in train_set]
    dev = [utils.untokenize_sentence(sentence) for sentence in dev_set]
    print("word level tf-idf")
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(train)

    return tfidf_vect.transform(train), tfidf_vect.transform(dev)


def add_feature(matrix, new_features):
    matrix_df = pd.DataFrame(matrix.todense())
    return pd.concat([matrix_df, new_features], axis=1)


def train_model(classifier, x_train, y_train, x_test, y_test, reduced=False, description=''):
    if reduced:
        threshold = 1 if y_train.value_counts()[1] > y_train.value_counts()[2] else 0
        print("Ternary mode selected. NEU and NONE will be both treated as NEU" if threshold is 1 else
              "Ternary mode selected. NEU and NONE will be both treated as NONE")
        y_train = [label - 1 if label > 1 else label for label in y_train]

    classifier.fit(x_train, y_train)
    predictions, probabilities = get_predictions(classifier, x_test)

    if reduced:
        predictions = [pred+1 if pred > threshold else pred for pred in predictions]

    if y_test is not None:
        print_confusion_matrix(predictions, y_test)
    return classifier, predictions, probabilities


def get_predictions(trained_classifier, feature_test_vector, is_vso=False):
    if is_vso:
        return trained_classifier.predict(feature_test_vector), trained_classifier.decision_function(feature_test_vector)
    else:
        return trained_classifier.predict(feature_test_vector), trained_classifier.predict_proba(feature_test_vector)


def get_model_accuracy(predictions, validation_labels):
    return metrics.accuracy_score(predictions, validation_labels)


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


def read_files(sLang, bStoreFiles=False):
    train_data = pd.DataFrame()
    dev_data = pd.DataFrame()
    test_data = pd.DataFrame()
    valid_data = pd.DataFrame()

    if bStoreFiles:
        train_data = utils.get_dataframe_from_xml(utils.parse_xml('./dataset/xml/intertass_{}_train.xml'.format(sLang)))
        dev_data = utils.get_dataframe_from_xml(utils.parse_xml('./dataset/xml/intertass_{}_dev.xml'.format(sLang)))

        train_data.to_csv('./dataset/csv/intertass_{}_train.csv'.format(sLang), encoding='utf-8', sep='\t')
        dev_data.to_csv('./dataset/csv/intertass_{}_dev.csv'.format(sLang), encoding='utf-8', sep='\t')

    else:

        train_data = pd.read_csv('./dataset/csv/intertass_{}_train.csv'.format(sLang), encoding='utf-8', sep='\t')
        dev_data = pd.read_csv('./dataset/csv/intertass_{}_dev.csv'.format(sLang), encoding='utf-8', sep='\t')

    valid_data = pd.read_csv('./dataset/csv/intertass_{}_valid.csv'.format(sLang), encoding='utf-8', sep='\t')
    test_data = pd.read_csv('./dataset/csv/intertass_{}_test.csv'.format(sLang), encoding='utf-8', sep='\t')

    encoder = preprocessing.LabelEncoder()
    valid_data['sentiment'] = encoder.fit_transform(valid_data['sentiment'])
    test_data['sentiment'] = encoder.transform(test_data['sentiment'])

    return train_data, dev_data, test_data, valid_data


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

    all_train_data, all_dev_data, all_test_data, all_valid_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for sLang in LANGUAGE_CODE:

        print('** LANG: ' + sLang)

        if False and sLang == 'all':  # Not accesible by the moment
            train_data = all_train_data
            dev_data = all_dev_data
            test_data = all_test_data
            valid_data = all_valid_data
            train_data.to_csv('./dataset/csv/intertass_all_train.csv', encoding='utf-8', sep='\t')
            dev_data.to_csv('./dataset/csv/intertass_all_dev.csv', encoding='utf-8', sep='\t')
            test_data.to_csv('./dataset/csv/intertass_all_test.csv', encoding='utf-8', sep='\t')
            valid_data.to_csv('./dataset/csv/intertass_all_valid.csv', encoding='utf-8', sep='\t')

        else:
            train_data, dev_data, test_data, valid_data = read_files(sLang, bStoreFiles=False)
            # all_train_data = pd.concat([all_train_data, train_data], ignore_index=True).reset_index(drop=True)
            # all_dev_data = pd.concat([all_dev_data, dev_data], ignore_index=True).reset_index(drop=True)
            # all_test_data = pd.concat([all_test_data, test_data], ignore_index=True).reset_index(drop=True)
            # all_valid_data = pd.concat([all_valid_data, valid_data], ignore_index=True).reset_index(drop=True)

        if bUpsampling:
            train_data = utils.perform_upsampling(train_data)

        if bTwoWayTranslation:
            translated_data = train_data
            translated_data['content'] = two_way_translation(translated_data.content, 'en')
            train_data = pd.concat([train_data, translated_data], ignore_index=True).reset_index(drop=True)

        # PRE-PROCESSING
        train_data['preprocessed'] = tweet_preprocessing.preprocess(train_data['content'], bAll=True)
        dev_data['preprocessed'] = tweet_preprocessing.preprocess(dev_data['content'], bAll=True)
        if bTestPhase is True:
            test_data['preprocessed'] = tweet_preprocessing.preprocess(test_data['content'], bAll=True)
        if bEvalPhase is True:
            valid_data['preprocessed'] = tweet_preprocessing.preprocess(valid_data['content'], bAll=True)

        # TOKENIZE
        print("Tokenizing...")
        train_data['tokenized'] = train_data.swifter.progress_bar(False).apply(lambda row: utils.tokenize_sentence(row.preprocessed), axis=1)
        dev_data['tokenized'] = dev_data.swifter.progress_bar(False).apply(lambda row: utils.tokenize_sentence(row.preprocessed), axis=1)
        if bTestPhase is True:
            test_data['tokenized'] = test_data.swifter.progress_bar(False).apply(lambda row: utils.tokenize_sentence(row.preprocessed), axis=1)
        if bEvalPhase is True:
            valid_data['tokenized'] = valid_data.swifter.progress_bar(False).apply(lambda row: utils.tokenize_sentence(row.preprocessed), axis=1)

        train_data['final_data'] = train_data['tokenized']
        dev_data['final_data'] = dev_data['tokenized']
        if bTestPhase is True:
            test_data['final_data'] = test_data['tokenized']
        if bEvalPhase is True:
            valid_data['final_data'] = valid_data['tokenized']

        if bLibreOffice:
            print("LibreOffice Processing... ")
            train_data['final_data'] = train_data.swifter.progress_bar(True).apply(lambda row: utils.libreoffice_processing(row.final_data, dictionary), axis=1)
            dev_data['final_data'] = dev_data.swifter.apply(lambda row: utils.libreoffice_processing(row.final_data, dictionary), axis=1)
            if bTestPhase is True:
                test_data['final_data'] = test_data.swifter.apply(lambda row: utils.libreoffice_processing(row.final_data, dictionary), axis=1)
            if bEvalPhase is True:
                valid_data['final_data'] = valid_data.swifter.apply(lambda row: utils.libreoffice_processing(row.final_data, dictionary), axis=1)

        if bLemmatize:
            print("Lemmatizing data...")
            train_data['final_data'] = train_data.swifter.apply(lambda row: lemmatize_sentence(row.final_data), axis=1)
            dev_data['final_data'] = dev_data.swifter.apply(lambda row: lemmatize_sentence(row.final_data), axis=1)
            if bTestPhase is True:
                test_data['final_data'] = test_data.swifter.apply(lambda row: lemmatize_sentence(row.final_data), axis=1)
            if bEvalPhase is True:
                valid_data['final_data'] = valid_data.swifter.apply(lambda row: lemmatize_sentence(row.final_data), axis=1)

        if bRemoveAccents:
            print("Removing accents...")
            train_data['final_data'] = train_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)
            dev_data['final_data'] = dev_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)
            if bTestPhase is True:
                test_data['final_data'] = test_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)
            if bEvalPhase is True:
                valid_data['final_data'] = valid_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)

        if bRemoveStopwords:
            train_data['final_data'] = train_data.swifter.progress_bar(False).apply(lambda row: remove_stopwords(row.final_data), axis=1)
            dev_data['final_data'] = dev_data.swifter.progress_bar(False).apply(lambda row: remove_stopwords(row.final_data), axis=1)
            if bTestPhase is True:
                test_data['final_data'] = test_data.swifter.progress_bar(False).apply(lambda row: remove_stopwords(row.final_data), axis=1)
            if bEvalPhase is True:
                valid_data['final_data'] = valid_data.swifter.progress_bar(False).apply(lambda row: remove_stopwords(row.final_data), axis=1)

        # FEATURE EXTRACTION
        train_data['tweet_length'] = extract_length_feature(train_data['content'])
        train_data['has_uppercase'] = extract_uppercase_feature(train_data['content'])
        train_data['question_mark'] = extract_question_mark_feature(train_data['content'])
        train_data['exclamation_mark'] = extract_exclamation_mark_feature(train_data['content'])
        train_data['letter_repetition'] = extract_letter_repetition_feature(train_data['content'])
        train_data['hashtag_number'] = extract_hashtag_number_feature(train_data['content'])
        train_data['pos_voc'], train_data['neg_voc'], train_data['neu_voc'], train_data['none_voc'] = \
            extract_sent_words_feature(train_data['tokenized'], train_data['tokenized'])

        dev_data['tweet_length'] = extract_length_feature(dev_data['content'])
        dev_data['has_uppercase'] = extract_uppercase_feature(dev_data['content'])
        dev_data['question_mark'] = extract_question_mark_feature(dev_data['content'])
        dev_data['exclamation_mark'] = extract_exclamation_mark_feature(dev_data['content'])
        dev_data['letter_repetition'] = extract_letter_repetition_feature(dev_data['content'])
        dev_data['hashtag_number'] = extract_hashtag_number_feature(dev_data['content'])
        dev_data['pos_voc'], dev_data['neg_voc'], dev_data['neu_voc'], dev_data['none_voc'] = \
            extract_sent_words_feature(dev_data['tokenized'], train_data['tokenized'])

        if bTestPhase:
            test_data['tweet_length'] = extract_length_feature(test_data['content'])
            test_data['has_uppercase'] = extract_uppercase_feature(test_data['content'])
            test_data['question_mark'] = extract_question_mark_feature(test_data['content'])
            test_data['exclamation_mark'] = extract_exclamation_mark_feature(test_data['content'])
            test_data['letter_repetition'] = extract_letter_repetition_feature(test_data['content'])
            test_data['hashtag_number'] = extract_hashtag_number_feature(test_data['content'])
            test_data['pos_voc'], test_data['neg_voc'], test_data['neu_voc'], test_data['none_voc'] = \
                extract_sent_words_feature(test_data['tokenized'], train_data['tokenized'])

        if bEvalPhase:
            valid_data['tweet_length'] = extract_length_feature(valid_data['content'])
            valid_data['has_uppercase'] = extract_uppercase_feature(valid_data['content'])
            valid_data['question_mark'] = extract_question_mark_feature(valid_data['content'])
            valid_data['exclamation_mark'] = extract_exclamation_mark_feature(valid_data['content'])
            valid_data['letter_repetition'] = extract_letter_repetition_feature(valid_data['content'])
            valid_data['hashtag_number'] = extract_hashtag_number_feature(valid_data['content'])
            valid_data['pos_voc'], valid_data['neg_voc'], valid_data['neu_voc'], valid_data['none_voc'] = \
                extract_sent_words_feature(valid_data['tokenized'], train_data['tokenized'])

        # COUNT VECTORS
        if bCountVectors:
            train_count_vectors, dev_count_vectors = perform_count_vectors(train_data['final_data'], dev_data['final_data'], print_oovs=True, classic_bow=bClassicBow)
            if bTestPhase:
                _, test_count_vectors = perform_count_vectors(train_data['final_data'], test_data['final_data'], print_oovs=True, classic_bow=bClassicBow)
            if bEvalPhase:
                _, valid_count_vectors = perform_count_vectors(train_data['final_data'], valid_data['final_data'], print_oovs=True, classic_bow=bClassicBow)

        # TF-IDF VECTORS
        if bTfidf:
            train_tfidf, dev_tfidf = perform_tf_idf_vectors(train_data['final_data'],
                                                                                            dev_data['final_data'])
            if bTestPhase:
                _, test_tfidf = perform_tf_idf_vectors(train_data['final_data'],
                                                                       test_data['final_data'])
            if bEvalPhase:
                _, valid_tfidf = perform_tf_idf_vectors(train_data['final_data'],
                                                                        valid_data['final_data'])

        train_features = pd.DataFrame({
            # 'tweet_length': train_data['tweet_length'],
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
            # 'tweet_length': dev_data['tweet_length'],
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
            test_features = pd.DataFrame({
                # 'tweet_length': test_data['tweet_length'],
                'has_uppercase': test_data['has_uppercase'],
                'exclamation_mark': test_data['exclamation_mark'],
                'question_mark': test_data['question_mark'],
                'hashtag_number': test_data['hashtag_number'],
                'pos_voc': test_data['pos_voc'],
                'neg_voc': test_data['neg_voc'],
                'neu_voc': test_data['neu_voc'],
                'none_voc': test_data['none_voc'],
                'letter_repetition': test_data['letter_repetition']
            })

        if bEvalPhase is True:
            valid_features = pd.DataFrame({
                'tweet_length': valid_data['tweet_length'],
                'has_uppercase': valid_data['has_uppercase'],
                'exclamation_mark': valid_data['exclamation_mark'],
                'question_mark': valid_data['question_mark'],
                'hashtag_number': valid_data['hashtag_number'],
                'pos_voc': valid_data['pos_voc'],
                'neg_voc': valid_data['neg_voc'],
                'neu_voc': valid_data['neu_voc'],
                'none_voc': valid_data['none_voc'],
                'letter_repetition': valid_data['letter_repetition']
            })


        # TRAINING
        # TODO Choose the training, development and test sets #########################################################
        set_names_array = ['Features']
        training_set_array = [train_features]
        training_labels = train_data['sentiment']

        dev_set_array = [dev_features]
        dev_labels = dev_data['sentiment']

        if bTestPhase is True:
            test_set_array = [test_features]
            test_labels = test_data['sentiment']
        if bEvalPhase is True:
            valid_set_array = [valid_count_vectors]
            valid_labels = valid_data['sentiment']
        # TODO #########################################################################################################

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

        all_classifiers = [lr]  # , nb, ada, gb]
        all_classif_names = ['LR']  # , 'NB', 'ADA', 'GB']

        all_probabilities = []
        all_valid_probabilities = []
        all_test_probabilities = []

        for i, (set_name, training_set, dev_set) in enumerate(zip(set_names_array, training_set_array, dev_set_array)):
            print("------------------ Training {} ------------------".format(set_name))
            mini, mini_test, mini_valid = [], [], []
            for j, clf in enumerate(all_classifiers):

                if bOneVsRest:
                    clf = OneVsRestClassifier(clf)
                print("Classifier: " + all_classif_names[j])
                print()

                print("Development set:")
                classif, preds, probs = train_model(clf, training_set, training_labels, dev_set, dev_labels,
                                                    reduced=bReduced)
                print()

                if bTestPhase:
                    print("Test set:")
                    test_preds, test_probs = get_predictions(classif, test_set_array[i])
                    print_confusion_matrix(test_preds, test_labels)
                    mini_test.append(test_probs)
                    print()

                if bEvalPhase:
                    print("Validation set:")
                    eval_preds, valid_probs = get_predictions(classif, valid_set_array[i])
                    print_confusion_matrix(eval_preds, valid_labels)
                    mini_valid.append(valid_probs)
                    print()

                mini.append(probs)
                print()

            if bClassifVotingEnsemble:
                print("DEVELOPMENT VOTING ENSEMBLE")
                print_confusion_matrix(get_averaged_predictions(mini), dev_labels)
                print()

                if bEvalPhase:
                    print("VALIDATION VOTING ENSEMBLE")
                    print_confusion_matrix(get_averaged_predictions(mini_valid), valid_labels)
                    print()
                if bTestPhase:
                    print("TEST VOTING ENSEMBLE")
                    print_confusion_matrix(get_averaged_predictions(mini_test), test_labels)
                    print()

            all_probabilities.append(mini)
            all_test_probabilities.append(mini_test)
            all_valid_probabilities.append(mini_valid)

        # FASTTEXT
        # fasttext_df = fasttext_to_df('./new_fasttext/{0}/{1}/{2}_{1}_{0}.out'.format(str_mode, str_phase, sLang))

        # Use only if you want to use the fasttext from fasttext folder, not from new_fasttext
        # fasttext_path = './fasttext/{}_fasttext_outputs_dev-test_full-reduced/'.format(str_mode)
        # fasttext_file = '{}_{}_fasttext_{}.csv'.format(sLang, str_phase, str_full_reduced)
        # fasttext_df = pd.read_csv(fasttext_path + fasttext_file)
        # fasttext_df = fasttext_df.drop(columns='ID')

        print("FASTTEXT MODEL")
        dev_fasttext_df = pd.read_csv('./fasttext/probabilities/{}_ingeotec_model_1_dev.csv'.format(sLang), encoding='utf-8', sep='\t')
        dev_fasttext_probabilities = dev_fasttext_df.to_numpy()
        dev_fasttext_probabilities = numpy.delete(dev_fasttext_probabilities, 0, 1)
        dev_fasttext_predictions = dev_fasttext_probabilities.argmax(1)
        print_confusion_matrix(dev_fasttext_predictions, dev_labels)

        if bTestPhase:
            test_fasttext_df = pd.read_csv('./fasttext/probabilities/{}_ingeotec_model_2_test.csv'.format(sLang), encoding='utf-8', sep='\t')
            test_fasttext_probabilities = test_fasttext_df.to_numpy()
            test_fasttext_probabilities = numpy.delete(test_fasttext_probabilities, 0, 1)
            test_fasttext_predictions = test_fasttext_probabilities.argmax(1)
            print_confusion_matrix(test_fasttext_predictions, test_labels)

        print()

        # BERT
        print("BERT MODEL")
        model_number = '7'
        # bert_path = './bert/mono_bert_dev-test_full-reduced/'
        # bert_file = '{}_dev_bert_full.csv'.format(sLang)
        bert_path = './bert/dev/{}/dev_results_{}.csv'.format(sLang, model_number)

        dev_bert_df = pd.read_csv(bert_path, encoding='utf-8', sep='\t')
        dev_bert_predictions = dev_bert_df['predictions']
        dev_bert_df = dev_bert_df.drop(dev_bert_df.columns[0], axis=1)
        dev_bert_df = dev_bert_df.drop(columns='predictions')
        dev_bert_probabilities = dev_bert_df.to_numpy()

        # dev_bert_df = pd.read_csv(bert_path + bert_file)
        # dev_bert_df = dev_bert_df.drop(dev_bert_df.columns[0], axis=1)
        # dev_bert_df = dev_bert_df.drop(columns='id')
        # dev_bert_probabilities = dev_bert_df.to_numpy()
        # dev_bert_predictions = dev_bert_probabilities.argmax(1)

        print_confusion_matrix(dev_bert_predictions, dev_labels)

        if bTestPhase:
            # bert_path = './bert/mono_bert_dev-test_full-reduced/'
            # bert_file = '{}_test_bert_full.csv'.format(sLang)
            bert_path = './bert/test/{}/test_results_{}.csv'.format(sLang, model_number)

            test_bert_df = pd.read_csv(bert_path, encoding='utf-8', sep='\t')
            test_bert_predictions = test_bert_df['predictions']
            test_bert_df = test_bert_df.drop(test_bert_df.columns[0], axis=1)
            test_bert_df = test_bert_df.drop(columns='predictions')
            test_bert_probabilities = test_bert_df.to_numpy()

            # test_bert_df = pd.read_csv(bert_path + bert_file)
            # test_bert_df = test_bert_df.drop(test_bert_df.columns[0], axis=1)
            # test_bert_df = test_bert_df.drop(columns='id')
            # test_bert_probabilities = test_bert_df.to_numpy()
            # test_bert_predictions = test_bert_probabilities.argmax(1)

            print_confusion_matrix(test_bert_predictions, test_labels)


        print()

        #
        # # GLOVE + BPE + W2V
        # print("G+B+W MODEL")
        # gbw_path = './outputs_flair_bpe_glove_w2v_clean_May29/{}/{}/'.format(sLang, str_mode)
        # gbw_file = '{}_test_glove_word2vec_bpe_forTest_full.csv'.format(sLang, str_phase, str_mode)
        #
        # gbw_df = pd.read_csv(gbw_path + gbw_file, sep=',')
        # gbw_df = gbw_df.drop(gbw_df.columns[0], axis=1)
        # # gbw_df = gbw_df.drop(columns='ID')
        #
        # gbw_probabilities = gbw_df.to_numpy()
        # gbw_predictions = gbw_probabilities.argmax(1)
        # if bEvalPhase:
        #     print_confusion_matrix(gbw_predictions, eval_labels)
        # # else:
        #     # print_confusion_matrix(gbw_predictions, dev_labels)
        # print()
        #

        print('-------------- FINAL ENSEMBLE --------------')
        print()

        print('From Normal Models:')
        print()

        selected_classifier = 0  # See all the classifiers available above.
        probabilities_for_voting_ensemble_dev = [
            all_probabilities[0][selected_classifier],
            dev_fasttext_probabilities,
            dev_bert_probabilities
        ]

        if bTestPhase:
            probabilities_for_voting_ensemble_test = [
                all_test_probabilities[0][selected_classifier],
                test_fasttext_probabilities,
                test_bert_probabilities
            ]

        if bEvalPhase:
            probabilities_for_voting_ensemble_valid = [
                all_valid_probabilities[0][selected_classifier],
                all_valid_probabilities[1][selected_classifier]
            ]

        print_confusion_matrix(get_averaged_predictions(probabilities_for_voting_ensemble_dev), dev_labels)
        if bEvalPhase:
            print_confusion_matrix(get_averaged_predictions(probabilities_for_voting_ensemble_valid), valid_labels)
        if bTestPhase:
            print_confusion_matrix(get_averaged_predictions(probabilities_for_voting_ensemble_test), test_labels)
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

        # if os.path.exists(output_dir) is False:
        #     print("Creating output directory " + output_dir)
        #     os.makedirs(output_dir)
        #
        # # with open(data_path + "final_outputs/" + cross + LANGUAGE_CODE + ".tsv", 'w', newline='') as out_file:
        # #     tsv_writer = csv.writer(out_file, delimiter='\t')
        # #     for i, prediction in enumerate(LABEL_ENCODER.inverse_transform(lr_word_predictions)):
        # #         tsv_writer.writerow([dev_data['tweet_id'][i], prediction])
        #
        # with open(output_dir + sLang + ".tsv", 'w', newline='') as out_file:
        #     tsv_writer = csv.writer(out_file, delimiter='\t')
        #     for i, prediction in enumerate(LABEL_ENCODER.inverse_transform(submitting_predictions)):
        #         if bEvalPhase:
        #             tsv_writer.writerow([eval_data['tweet_id'][i], prediction])
        #         else:
        #             tsv_writer.writerow([dev_data['tweet_id'][i], prediction])

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
