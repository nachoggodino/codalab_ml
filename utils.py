import pandas
from sklearn import preprocessing
import xml.etree.ElementTree as ET

from nltk.tokenize.treebank import TreebankWordDetokenizer


def parse_xml(filepath):
    parser = ET.XMLParser(encoding='utf-8')
    tree = ET.parse(filepath, parser=parser)
    return tree


def get_dataframe_from_xml(data, encode_label=True):
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

    if encode_label:
        encoder = preprocessing.LabelEncoder()
        sentiment = encoder.fit_transform(sentiment)

    result_df['sentiment'] = sentiment

    return result_df


def untokenize_sentence(simple_list):
    return TreebankWordDetokenizer().detokenize(simple_list)
