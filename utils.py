import pandas as pd
import os
import xml.etree.ElementTree as ET
from sklearn import preprocessing

LABEL_ENCODER = preprocessing.LabelEncoder()
TERNARY_LABEL_ENCODER = preprocessing.LabelEncoder()


def get_labelencoder():
    return LABEL_ENCODER

def get_dataframe_from_xml(data, bReduceLabels=False):
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
    result_df['content'] = content
    result_df['day_of_week'] = day_of_week
    result_df['month'] = month
    result_df['hour'] = hour

    if bReduceLabels is True:
        sentiment = ['O' if (x == 'NEU' or x == 'NONE') else x for x in sentiment]

    LABEL_ENCODER.fit(sentiment)
    TERNARY_LABEL_ENCODER.fit(ternary_sentiment)
    result_df['sentiment'] = LABEL_ENCODER.transform(sentiment)
    result_df['ternary_sentiment'] = TERNARY_LABEL_ENCODER.transform(ternary_sentiment)
    return result_df


def read_files(data_path, sLang, bCross, LANGUAGE_CODE=['cr', 'es', 'mx', 'pe', 'uy']):
    train_data = pd.DataFrame()
    if bCross is True:
        dev_cross_data = pd.DataFrame()
        for sLangCross in [x for x in LANGUAGE_CODE if x != sLang]:
            df_train_cross = read_file(data_path + sLangCross + "/intertass_" + sLangCross + "_train.xml")
            df_dev_cross = read_file(data_path + sLangCross + "/intertass_" + sLangCross + "_dev.xml")
            train_data = pd.concat([train_data, df_train_cross], ignore_index=True)
            dev_cross_data = pd.concat([dev_cross_data, df_dev_cross], ignore_index=True)
        # We add train_cross + dev_cross (but never we can add dev_lang)
        train_data = pd.concat([train_data, dev_cross_data], ignore_index=True)

    else:
        train_data = read_file(data_path + sLang + "/intertass_" + sLang + "_train.xml")

    dev_data = read_file(data_path + sLang + "/intertass_" + sLang + "_dev.xml")
    tst_data = read_file(data_path + sLang + "/intertass_" + sLang + "_test.xml")
    return train_data, dev_data, tst_data


def read_file(file_path):
    tree_info = ET.parse(file_path)
    data = get_dataframe_from_xml(tree_info)
    return data


def save_file(data, filename_out):
    with open(filename_out , 'w', encoding="utf-8") as f_out:
        print('Writing dev file for FastText at: ' + filename_out)
        for txt, lbl in zip(data['content'], data['sentiment']):
            f_out.write('__label__' + str(LABEL_ENCODER.inverse_transform([lbl])[0]) + '\t' + txt + '\n')