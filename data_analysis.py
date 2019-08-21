import xml.etree.ElementTree as ET
import re
import nltk
import unidecode
import spacy
from textacy import keyterms
import hunspell

from re import finditer


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from nltk.tokenize.treebank import TreebankWordDetokenizer

from imblearn.over_sampling import RandomOverSampler

from collections import Counter

LANGUAGE_CODE = 'es'
dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")

emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

data_path = "./codalab/DATASETS/public_data_development/"
test_path = "./codalab/DATASETS/public_data_task1/"
parser_dev = ET.XMLParser(encoding='utf-8')
parser_train = ET.XMLParser(encoding='utf-8')

tree_dev = ET.parse(data_path + LANGUAGE_CODE + "/intertass_" + LANGUAGE_CODE + "_dev.xml", parser=parser_dev)
tree_train = ET.parse(data_path + LANGUAGE_CODE + "/intertass_" + LANGUAGE_CODE + "_train.xml", parser=parser_train)


def get_dataframe_from_xml(data):
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
                if element.text == 'NONE' or element.text == 'NEU':
                    sentiment.append(element.text)
                else:
                    sentiment.append(element.text)

    result_df = pandas.DataFrame()
    result_df['tweet_id'] = tweet_id
    # result_df['user'] = user
    result_df['content'] = content
    # result_df['lang'] = lang
    result_df['day_of_week'] = day_of_week
    result_df['month'] = month
    result_df['hour'] = hour

    # encoder = preprocessing.LabelEncoder()
    result_df['sentiment'] = sentiment  # encoder.fit_transform(sentiment)

    return result_df


def perform_upsampling(dataframe):
    ros = RandomOverSampler()
    x_resampled, y_resampled = ros.fit_resample(dataframe[['tweet_id', 'content']], dataframe['sentiment'])
    df = pandas.DataFrame(data=x_resampled[0:, 0:], columns=['tweet_id', 'content'])
    df['sentiment'] = y_resampled
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def extract_uppercase_feature(dataframe):
    regex = re.compile(r"\b[A-Z][A-Z]+\b")
    result = []
    for tweet in dataframe:
        result.append(len(regex.findall(tweet)))
    return result


def extract_length_feature(tokenized_dataframe):
    return [len(tweet) for tweet in tokenized_dataframe]


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
    positive_voc, negative_voc = get_sentiment_vocabulary(data_feed, 'P', 'N')
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


def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])


def libreoffice_processing(tokenized_data):
    print("Libreoffice processing")
    return [[word if dictionary.spell(word) is True else next(iter(dictionary.suggest(word)), word) for word in tweet] for tweet in tokenized_data]


def get_sentiment_vocabulary(data, positive, negative):
    pos_neg_tweets = []
    pos_neg_bool_labels = []
    for i, tweet in enumerate(data):
        sentiment = train_data['sentiment'][i]
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
    result = [re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), tweet) for tweet in result]  # Hashtag
    result = [tweet.lower() for tweet in result]  # Tweet to lowercase
    result = [re.sub(r"^.*http.*$", 'http', tweet) for tweet in result]  # Remove all http contents
    result = [re.sub(r"\B@\w+", 'username', tweet) for tweet in result]  # Remove all usernames
    result = [re.sub(r"(\w)(\1{2,})", r"\1", tweet) for tweet in result] # Remove all letter repetitions
    result = [re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"[a-zA-Z]*hah[a-zA-Z]*", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"\d+", '', tweet) for tweet in result]  # Remove all numbers
    result = [tweet.translate(str.maketrans('', '', string.punctuation)) for tweet in result]  # Remove punctuation

    return result


def remove_accents(tokenized_data):
    result = []
    for tweet in tokenized_data:
        result.append([unidecode.unidecode(word) for word in tweet])
    return result


def tokenize_list(datalist):
    result = []
    for row in datalist:
        result.append(nltk.word_tokenize(row))
    return result


def remove_stopwords(tokenized_data):
    result = []
    for row in tokenized_data:
        result.append([word for word in row if word not in [] ]) # nltk.corpus.stopwords.words('spanish')])
    return result


def stem_list(datalist):
    stemmer = nltk.stem.SnowballStemmer('spanish')
    result = []
    for row in datalist:
        stemmed_words = [stemmer.stem(word) for word in row]
        result.append(stemmed_words)
    return result


def lemmatize_list(datalist):
    print("Lemmatizing the data. Could take a while...")
    lemmatizer = spacy.load("es_core_news_sm")
    result = []
    for i, row in enumerate(datalist):
        mini_result = [token.lemma_ for token in lemmatizer(row)]
        result.append(mini_result)
        i += 1
    return result


def print_vocabulary_analysis(tokenized_train_list, tokenized_dev_list):
    all_train_words = [item for sublist in tokenized_train_list for item in sublist]
    all_dev_words = [item for sublist in tokenized_dev_list for item in sublist]
    train_vocabulary = []
    dev_vocabulary = []
    for word in all_train_words:
        if word not in train_vocabulary:
            train_vocabulary.append(word)
    for word in all_dev_words:
        if word not in dev_vocabulary:
            dev_vocabulary.append(word)
    train_word_counter = Counter(all_train_words)
    most_common_train_words = train_word_counter.most_common(10)
    dev_word_counter = Counter(all_dev_words)
    most_common_dev_words = dev_word_counter.most_common(10)
    print("The total number of words in TRAINING_DATA is: " + str(len(all_train_words)))
    print("The length of the vocabulary in TRAINING_DATA is: " + str(len(train_vocabulary)))
    print("Most common words in TRAINING_DATA:")
    print(most_common_train_words)
    print()
    print("The total number of words in DEVELOPMENT_DATA is: " + str(len(all_dev_words)))
    print("The length of the vocabulary in DEVELOPMENT_DATA is: " + str(len(dev_vocabulary)))
    print("Most common words in DEVELOPMENT_DATA:")
    print(most_common_dev_words)
    print()

    out_of_vocabulary = []
    for word in dev_vocabulary:
        if word not in train_vocabulary:
            out_of_vocabulary.append(word)
    print("The number of Out-Of-Vocabulary words is: " + str(len(out_of_vocabulary)))
    print("Which is the " + str(len(out_of_vocabulary) / len(dev_vocabulary) * 100) + "% of the Development Vocabulary")
    print()


def print_separator(string_for_printing):
    print('//////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print()
    print("                      " + string_for_printing)
    print()
    print('//////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print()


# GET THE DATA
train_data = get_dataframe_from_xml(tree_train)
dev_data = get_dataframe_from_xml(tree_dev)

train_data = perform_upsampling(train_data)
print(train_data)

# TEXT PREPROCESSING
processed_train_tweets = text_preprocessing(train_data['content'])
processed_dev_tweets = text_preprocessing(dev_data['content'])

print_separator("Length Analysis of the Tweets")

print("Number of Tweets in TRAINING_DATA: " + str(len(processed_train_tweets)))  # TOTAL TWEETS COUNT
print("Number of Tweets in DEVELOPMENT_DATA: " + str(len(processed_dev_tweets)))
print()

tokenized_train_tweets = tokenize_list(processed_train_tweets)
tokenized_dev_tweets = tokenize_list(processed_dev_tweets)

# FEATURE EXTRACTION
train_data['has_uppercase'] = extract_uppercase_feature(train_data['content'])
dev_data['has_uppercase'] = extract_uppercase_feature(dev_data['content'])

train_data['length'] = extract_length_feature(tokenized_train_tweets)
dev_data['length'] = extract_length_feature(tokenized_dev_tweets)

train_data['question_mark'] = extract_question_mark_feature(train_data['content'])
dev_data['question_mark'] = extract_question_mark_feature(dev_data['content'])

train_data['exclamation_mark'] = extract_exclamation_mark_feature(train_data['content'])
dev_data['exclamation_mark'] = extract_exclamation_mark_feature(dev_data['content'])

train_data['letter_repetition'] = extract_letter_repetition_feature(train_data['content'])
dev_data['letter_repetition'] = extract_letter_repetition_feature(dev_data['content'])

# VOCABULARY ANALYSIS

print("The maximum length of a Tweet in TRAINING_DATA is: " + str(max(train_data['length'])))
print("The minimum length of a Tweet in TRAINING_DATA is: " + str(min(train_data['length'])))
print("The average length of a Tweet in TRAINING_DATA is: " + str(sum(train_data['length'])/len(train_data['length'])))
print()
print("The maximum length of a Tweet in DEVELOPMENT_DATA is: " + str(max(dev_data['length'])))
print("The minimum length of a Tweet in DEVELOPMENT_DATA is: " + str(min(dev_data['length'])))
print("The average length of a Tweet in DEVELOPMENT_DATA is: " + str(sum(dev_data['length'])/len(dev_data['length'])))
print()

print_separator("Vocabulary Analysis after tokenize")

print_vocabulary_analysis(tokenized_train_tweets, tokenized_dev_tweets)

# print_separator("Vocabulary Analysis after Libreoffice Processing")
#
# libreoffice_train_tweets = libreoffice_processing(tokenized_train_tweets)
# libreoffice_dev_tweets = libreoffice_processing(tokenized_dev_tweets)
# print_vocabulary_analysis(libreoffice_train_tweets, libreoffice_dev_tweets)
#
# print_separator("After lemmatizing the data...")
#
# lemmatized_train_tweets = lemmatize_list(processed_train_tweets)
# lemmatized_dev_tweets = lemmatize_list(processed_dev_tweets)
#
# print_vocabulary_analysis(lemmatized_train_tweets, lemmatized_dev_tweets)
#
# print_separator("After removing accents to the data...")
#
# without_accents_train = remove_accents(lemmatized_train_tweets)
# without_accents_dev = remove_accents(lemmatized_dev_tweets)
#
# print_vocabulary_analysis(without_accents_train, without_accents_dev)

print_separator("Label Counts:")

print("Total count of each Label in TRAINING_DATA:")
print(train_data['sentiment'].value_counts('N'))
print()
print("Total count of each Label in DEVELOPMENT_DATA:")
print(dev_data['sentiment'].value_counts('N'))
print()

print_separator("Most discriminating words analysis")

print("Training data:")
train_pos_voc, train_neg_voc = get_sentiment_vocabulary(tokenized_train_tweets, 'P', 'N')
train_data['pos_voc'], train_data['neg_voc'], train_data['neu_voc'], train_data['none_voc'] = extract_sent_words_feature(tokenized_train_tweets, tokenized_train_tweets)
print("The most discriminating words between P and N are:")
print("Category P:")
print(train_pos_voc)
print("Category N:")
print(train_neg_voc)
print()

print("Development data:")
dev_pos_voc, dev_neg_voc = get_sentiment_vocabulary(tokenized_dev_tweets, 'P', 'N')
dev_data['pos_voc'], dev_data['neg_voc'], dev_data['neu_voc'], dev_data['none_voc'] = extract_sent_words_feature(tokenized_dev_tweets, tokenized_train_tweets)
print("The most discriminating words between P and N are:")
print("Category P:")
print(dev_pos_voc)
print("Category N:")
print(dev_neg_voc)
print()

print_separator("Correlation analysis in TRAINING_DATA:")


# with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
#     '''
#     print("Hour VS Sentiment")
#     print()
#     print(train_data.groupby(['hour', 'sentiment']).size())
#     print("Month VS Sentiment")
#     print()
#     print(train_data.groupby(['month', 'sentiment']).size())
#     print("Day of the Week VS Sentiment")
#     print()
#     print(train_data.groupby(['day_of_week', 'sentiment']).size())
#     print("Length VS Sentiment")
#     print()
#     print(train_data.groupby(['length', 'sentiment']).size())
#     '''
#     print("Uppercase VS Sentiment")
#     print()
#     print(train_data.groupby(['has_uppercase', 'sentiment']).size())
#     print("Question VS Sentiment")
#     print()
#     print(train_data.groupby(['question_mark', 'sentiment']).size())
#     print("Exclamation VS Sentiment")
#     print()
#     print(train_data.groupby(['exclamation_mark', 'sentiment']).size())
#     print("Positive Vocabulary VS Sentiment")
#     print()
#     print(train_data.groupby(['pos_voc', 'sentiment']).size())
#     print("Negative Vocabulary VS Sentiment")
#     print()
#     print(train_data.groupby(['neg_voc', 'sentiment']).size())
#     print("Neutral Vocabulary VS Sentiment")
#     print()
#     print(train_data.groupby(['neu_voc', 'sentiment']).size())
#     print("None Vocabulary VS Sentiment")
#     print()
#     print(train_data.groupby(['none_voc', 'sentiment']).size())
#
#     print("Correlation analysis in DEVELOPMENT_DATA:")
#     print()
#     '''
#     print("Hour VS Sentiment")
#     print()
#     print(dev_data.groupby(['hour', 'sentiment']).size())
#     print("Month VS Sentiment")
#     print()
#     print(dev_data.groupby(['month', 'sentiment']).size())
#     print("Day of the Week VS Sentiment")
#     print()
#     print(dev_data.groupby(['day_of_week', 'sentiment']).size())
#     print("Length VS Sentiment")
#     print()
#     print(dev_data.groupby(['length', 'sentiment']).size())
#     '''
#     print("Uppercase VS Sentiment")
#     print()
#     print(dev_data.groupby(['has_uppercase', 'sentiment']).size())
#     print("Question VS Sentiment")
#     print()
#     print(dev_data.groupby(['question_mark', 'sentiment']).size())
#     print("Exclamation VS Sentiment")
#     print()
#     print(dev_data.groupby(['exclamation_mark', 'sentiment']).size())
#     print("Positive Vocabulary VS Sentiment")
#     print()
#     print(dev_data.groupby(['pos_voc', 'sentiment']).size())
#     print("Negative Vocabulary VS Sentiment")
#     print()
#     print(dev_data.groupby(['neg_voc', 'sentiment']).size())
#     print("Neutral Vocabulary VS Sentiment")
#     print()
#     print(dev_data.groupby(['neu_voc', 'sentiment']).size())
#     print("None Vocabulary VS Sentiment")
#     print()
#     print(dev_data.groupby(['none_voc', 'sentiment']).size())

