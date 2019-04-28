import nltk
import re
import xml.etree.ElementTree as ET
import unidecode
import pandas
from sklearn import preprocessing
import spacy
from re import finditer
import matplotlib.pyplot as pyplot


lemmatizer = spacy.load("es_core_news_sm")


def lemmatize_list(datalist):
    result = []
    for row in datalist:
        lemmatized_words = [lemmatizer(token)[0].lemma_ for token in row]
        result.append(lemmatized_words)
    return result


def lemmatize_word(word):
    print(lemmatizer(word)[0].lemma_)


def camel_case_split(identifier):
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier)
    return [m.group(0) for m in matches]


def regex_tester(string):
    print(re.search(r"(\w)(\1{2,})", string))
    return


def regex_sub(string):
    print(re.sub(r"(\w)(\1{2,})", camel_case_split, string))
    return


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


train_data = get_dataframe_from_xml(tree_train)
dev_data = get_dataframe_from_xml(tree_dev)

fig = pyplot.figure()

ax = fig.add_subplot(1, 1, 1)

ax.hist(train_data['month'])

pyplot.title('Month distribution')
pyplot.xlabel('Month')
pyplot.ylabel('#Tweets')
pyplot.show()
