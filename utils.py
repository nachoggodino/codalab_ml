import pandas
from sklearn import preprocessing
import xml.etree.ElementTree as ET
import pandas as pd

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


def read_files(sLang, bStoreFiles=False):
    train_data = pd.DataFrame()
    dev_data = pd.DataFrame()
    test_data = pd.DataFrame()
    valid_data = pd.DataFrame()

    if bStoreFiles:
        train_data = get_dataframe_from_xml(parse_xml('./dataset/xml/intertass_{}_train.xml'.format(sLang)))
        dev_data = get_dataframe_from_xml(parse_xml('./dataset/xml/intertass_{}_dev.xml'.format(sLang)))

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


def csv_to_flair_format(data, labels, sLang, sPhase):
    result = pandas.DataFrame()
    result['labels'] = ['__label__' + str(label) for label in labels]
    result['content'] = data
    result.to_csv('dataset/flair/intertass_{}_{}.txt'.format(sLang, sPhase), header=None, index=None, sep=' ')
    return


