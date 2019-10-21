import pandas
from sklearn import preprocessing
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tweet_preprocessing
import hunspell
import swifter

from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk

from imblearn.over_sampling import RandomOverSampler


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


def csv_to_flair_format(preprocess=False, postpreprocess=False):
    for sLang in ['es', 'cr', 'mx', 'pe', 'uy', 'all']:
        train_data = pd.read_csv('./dataset/csv/intertass_{}_train.csv'.format(sLang), encoding='utf-8', sep='\t')
        test_data = pd.read_csv('./dataset/csv/intertass_{}_test.csv'.format(sLang), encoding='utf-8', sep='\t')
        dev_data = pd.read_csv('./dataset/csv/intertass_{}_dev.csv'.format(sLang), encoding='utf-8', sep='\t')

        train_data = perform_upsampling(train_data)

        encoder = preprocessing.LabelEncoder()
        test_data['sentiment'] = encoder.fit_transform(test_data['sentiment'])

        if preprocess:
            train_data['content'] = tweet_preprocessing.preprocess(train_data['content'])
            dev_data['content'] = tweet_preprocessing.preprocess(dev_data['content'])
            test_data['content'] = tweet_preprocessing.preprocess(test_data['content'])

        if postpreprocess:
            dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")
            train_data['content'] = train_data.swifter.progress_bar(False).apply(lambda row: tokenize_sentence(row.content), axis=1)
            dev_data['content'] = dev_data.swifter.progress_bar(False).apply(lambda row: tokenize_sentence(row.content), axis=1)
            test_data['content'] = test_data.swifter.progress_bar(False).apply(lambda row: tokenize_sentence(row.content), axis=1)
            train_data['content'] = train_data.swifter.progress_bar(True).apply(lambda row: libreoffice_processing(row.content, dictionary), axis=1)
            dev_data['content'] = dev_data.swifter.apply(lambda row: libreoffice_processing(row.content, dictionary), axis=1)
            test_data['content'] = test_data.swifter.apply(lambda row: libreoffice_processing(row.content, dictionary), axis=1)

        csv2flair(train_data['content'], train_data['sentiment'], sLang, 'train')
        csv2flair(dev_data['content'], dev_data['sentiment'], sLang, 'dev')
        csv2flair(test_data['content'], test_data['sentiment'], sLang, 'test')


def encode_label(list_of_labels):
    encoder = preprocessing.LabelEncoder()
    return encoder.fit_transform(list_of_labels)


def print_confusion_matrix(predictions, labels, print_confusion_matrix=False, print_prec_and_rec=False):
    preds = pd.Series(predictions, name='Predicted')
    labs = pd.Series(labels, name='Actual')
    df_confusion = pd.crosstab(labs, preds)
    if print_confusion_matrix:
        print(df_confusion)
    prec = precision_score(labs, preds, average='macro')
    rec = recall_score(labs, preds, average='macro')
    score = 2*(prec*rec)/(prec+rec)
    print("F1-SCORE: " + str(score))
    if print_prec_and_rec:
        print("Recall: " + str(rec))
        print("Precision: " + str(prec))
    print()
    return


def libreoffice_processing(tokenized_sentence, dictionary):
    return [word if dictionary.spell(word) is True else next(iter(dictionary.suggest(word)), word) for word in tokenized_sentence]


def tokenize_sentence(sentence):
    return nltk.word_tokenize(sentence)


def perform_upsampling(dataframe):
    ros = RandomOverSampler(random_state=1234)
    x_resampled, y_resampled = ros.fit_resample(dataframe[['tweet_id', 'content']], dataframe['sentiment'])
    df = pd.DataFrame(data=x_resampled[0:, 0:], columns=['tweet_id', 'content'])
    df['sentiment'] = y_resampled
    df = df.sample(frac=1).reset_index(drop=True)
    return df


# AUXILIAR
def csv2flair(data, labels, sLang, sPhase):
    result = pandas.DataFrame()
    result['labels'] = ['__label__' + str(label) for label in labels]
    result['content'] = data
    result.to_csv('dataset/flair/intertass_{}_{}.txt'.format(sLang, sPhase), header=None, index=None, sep=' ')
    return




