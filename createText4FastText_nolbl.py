import re
from re import finditer
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn import preprocessing
import string
import spacy
import argparse


emoji_pattern = re.compile("[" 
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

nlp = spacy.load("es_core_news_md")


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

    result_df = pd.DataFrame()
    result_df['tweet_id'] = tweet_id
    result_df['content'] = content
    #result_df['sentiment'] = sentiment
    return result_df


def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])


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


def tokenize_list(datalist):
    print("Tokenizing data...")
    return [' '.join([token.text for token in nlp(row)]) for row in datalist]


def read_files(filename):
    tree_train = ET.parse(filename)
    train_data = get_dataframe_from_xml(tree_train)
    return train_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='the filename to process')
    args = parser.parse_args()
    data2process = read_files(args.filename)

    # PRE-PROCESSING
    preprocessed_content = text_preprocessing(data2process['content'])
    data2process['content'] = tokenize_list(preprocessed_content)
    filename_out = args.filename.replace('.xml', '.ftx')
    with open(filename_out, 'w') as f_out:
        print('Writing file to ' + filename_out)
        for index, row in data2process.iterrows():
            f_out.write(row['content'] + '\n')

