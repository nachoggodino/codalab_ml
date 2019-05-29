import re
from re import finditer
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn import preprocessing
import string
import spacy
import argparse
nlp = spacy.load("es_core_news_md")


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
        bValue = False
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
            elif element.tag == 'value' and not bValue:
                bValue = True
                if element.text == 'N+' or element.text == 'P+':
                    sentiment.append(element.text[0])
                else:
                    sentiment.append(element.text)
                if element.text == 'NONE' or element.text == 'NEU':
                    ternary_sentiment.append('O')
                else:
                    ternary_sentiment.append(element.text)
            # else:
                # print("Unknown tag: " + element.tag)

    result_df = pd.DataFrame()
    result_df['tweet_id'] = tweet_id
    # result_df['user'] = user
    result_df['content'] = content
    # result_df['lang'] = lang
    # result_df['day_of_week'] = day_of_week
    # result_df['month'] = month
    # result_df['hour'] = hour
    print(len(sentiment))
    print(len(content))

    result_df['sentiment'] = sentiment
    # result_df['ternary_sentiment'] = ternary_sentiment
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
    result = [re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", '', tweet, flags=re.MULTILINE) for tweet in result]  # Remove all http contents
    result = [re.sub(r"\B@\w+", '', tweet) for tweet in result]  # Remove all usernames
    result = [re.sub(r"(\w)(\1{2,})", r"\1", tweet) for tweet in result]  # Remove all letter repetitions
    result = [re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"\d+", '', tweet) for tweet in result]  # Remove all numbers
    result = [tweet.translate(str.maketrans('', '', string.punctuation + '¡')) for tweet in result]  # Remove punctuation
    return result


def tokenize_list(datalist):
    print("Tokenizing data...")
    return [' '.join([token.text for token in nlp(row)]) for row in datalist]


def dataframe_to_ftfile(dataframe, filename):
    filename = './previous_years/ft_processed/ftx/' + filename
    preprocessed = text_preprocessing(dataframe['content'])
    dataframe['content'] = tokenize_list(preprocessed)
    with open(filename, 'w') as fout:
        print('Writing file to ' + filename)
        for index, row in dataframe.iterrows():
            fout.write('__label__' + row['sentiment'] + '\t' + row['content'] + '\n')

    return


def dataframe_to_csvfile(dataframe, filename):
    filename = './previous_years/ft_processed/csv/' + filename
    print("Writing file to " + filename)
    dataframe.to_csv(filename, encoding='utf-8', sep='\t')


if __name__ == '__main__':

    cr_test_df, es_test_df, pe_test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_tree = ET.parse("./previous_years/tagged/four_label/general-test-tagged-3l.xml")
    unsupervised_train_df = get_dataframe_from_xml(train_tree)
    print(unsupervised_train_df['sentiment'].value_counts())
    train_tree = ET.parse("./previous_years/tagged/four_label/general-train-tagged-3l.xml")
    test_df = get_dataframe_from_xml(train_tree)
    print(test_df['sentiment'].value_counts())
    for file in ['intertass-CR-development-tagged.xml', 'intertass-CR-train-tagged.xml']:
        cr_tree = ET.parse("./previous_years/tagged/four_label/" + file)
        cr_test_df = pd.concat([cr_test_df, get_dataframe_from_xml(cr_tree)], ignore_index=True)
    for file in ['intertass-ES-development-tagged.xml', 'intertass-ES-train-tagged.xml']:
        es_tree = ET.parse("./previous_years/tagged/four_label/" + file)
        es_test_df = pd.concat([es_test_df, get_dataframe_from_xml(es_tree)], ignore_index=True)
    for file in ['intertass-PE-development-tagged.xml', 'intertass-PE-train-tagged.xml']:
        pe_tree = ET.parse("./previous_years/tagged/four_label/" + file)
        pe_test_df = pd.concat([pe_test_df, get_dataframe_from_xml(pe_tree)], ignore_index=True)

    neu_test_values = test_df.loc[test_df['sentiment'] == 'NEU']
    none_test_values = test_df.loc[test_df['sentiment'] == 'NONE']
    neg_test_values = test_df.loc[test_df['sentiment'] == 'N']
    pos_test_values = test_df.loc[test_df['sentiment'] == 'P']

    es_final_test = pd.concat([neu_test_values.head(670),
                               none_test_values.head(644),
                               neg_test_values.head(2241),
                               pos_test_values.head(1545)], ignore_index=True).sample(frac=1).reset_index(drop=True)

    cr_final_test = pd.concat([neu_test_values.head(670),
                               none_test_values.head(979),
                               neg_test_values.head(1958),
                               pos_test_values.head(1520)], ignore_index=True).sample(frac=1).reset_index(drop=True)

    mx_final_test = pd.concat([neu_test_values.head(450),
                               none_test_values.head(330),
                               neg_test_values.head(2182),
                               pos_test_values.head(1200)], ignore_index=True).sample(frac=1).reset_index(drop=True)

    pe_final_test = pd.concat([neu_test_values.head(670),
                               none_test_values.head(1480),
                               neg_test_values.head(758),
                               pos_test_values.head(789)], ignore_index=True).sample(frac=1).reset_index(drop=True)

    uy_final_test = pd.concat([neu_test_values.head(670),
                               none_test_values.head(325),
                               neg_test_values.head(1375),
                               pos_test_values.head(1093)], ignore_index=True).sample(frac=1).reset_index(drop=True)

    # dataframe_to_ftfile(es_final_test, 'es_general_test.ftx')
    # dataframe_to_ftfile(cr_final_test, 'cr_general_test.ftx')
    # dataframe_to_ftfile(mx_final_test, 'mx_general_test.ftx')
    # dataframe_to_ft(pe_final_test, 'pe_general_test.ftx')
    # dataframe_to_ftfile(uy_final_test, 'uy_general_test.ftx')
    # dataframe_to_ftfile(cr_test_df, 'es_lang_test.ftx')
    # dataframe_to_ftfile(es_test_df, 'cr_lang_test.ftx')
    # dataframe_to_ftfile(pe_test_df, 'pe_lang_test.ftx')
    # dataframe_to_ftfile(unsupervised_train_df, 'unsupervised_train.ftx')

    dataframe_to_csvfile(es_final_test, 'es_general_test.csv')
    dataframe_to_csvfile(cr_final_test, 'cr_general_test.csv')
    dataframe_to_csvfile(mx_final_test, 'mx_general_test.csv')
    dataframe_to_csvfile(pe_final_test, 'pe_general_test.csv')
    dataframe_to_csvfile(uy_final_test, 'uy_general_test.csv')
    dataframe_to_csvfile(cr_test_df, 'es_lang_test.csv')
    dataframe_to_csvfile(es_test_df, 'cr_lang_test.csv')
    dataframe_to_csvfile(pe_test_df, 'pe_lang_test.csv')
    dataframe_to_csvfile(unsupervised_train_df, 'unsupervised_train.csv')


