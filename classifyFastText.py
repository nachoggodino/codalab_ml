import sys
import os
import fastText
import csv
import pandas as pd
from sklearn import preprocessing
import xml.etree.ElementTree as ET
import re
import string
import nltk
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
import argparse
import numpy as np

# Append script path to import path to locate tass_eval
sys.path.append(os.path.realpath(__file__))

# Import evalTask1 function fro tass_eval module
from tass_eval import evalTask1

# model = fastText.load_model('model.bin')

data_path = "../TASS2019/DATASETS/public_data/"
LANGUAGE_CODE = ['cr', 'es', 'mx', 'pe', 'uy']
# LANGUAGE_CODE = ['cr']
CROSS_LINGUAL = [False]
bTestPhase = False  # If we are doing test, then concatenate train + dev, if not use dev as test
bReduceLabels = False # To indicate if we use the 4 labels or only 3

LABEL_ENCODER = preprocessing.LabelEncoder()
TERNARY_LABEL_ENCODER = preprocessing.LabelEncoder()

replace_reducedLabel_perLanguage = {'cr': 'NONE', 'es': 'NEU', 'mx': 'NEU', 'pe': 'NONE', 'uy': 'NEU'}

print("Loading Spacy Model")
lemmatizer = spacy.load("es_core_news_sm")  # GLOBAL to avoid loading the model several times

emoji_pattern = re.compile("[" 
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U0001F910-\U0001F93F"  # emoticos (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U00002639-\U0000263B"  # more emoticons
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

    if bReduceLabels is True:
        sentiment = ['O' if (x == 'NEU' or x == 'NONE') else x for x in sentiment]

    LABEL_ENCODER.fit(sentiment)
    TERNARY_LABEL_ENCODER.fit(ternary_sentiment)
    result_df['sentiment'] = LABEL_ENCODER.transform(sentiment)
    result_df['ternary_sentiment'] = TERNARY_LABEL_ENCODER.transform(ternary_sentiment)
    return result_df

def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])

def text_preprocessing(data):
    result = data
    result = [tweet.replace('\n', '').strip() for tweet in result]  # Newline and leading/trailing spaces
    result = [emoji_pattern.sub(r'', tweet) for tweet in result]
    result = [tweet.replace(u'\u2018', "'").replace(u'\u2019', "'") for tweet in result]  # Quotes replace by general
    result = [re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), tweet) for tweet in result]  # Hashtag
    result = [tweet.lower() for tweet in result]
    result = [re.sub(r"^.*http.*$", 'http', tweet) for tweet in result]  # Remove all http contents
    result = [re.sub(r"\B@\w+", 'username', tweet) for tweet in result]  # Remove all usernames
    result = [re.sub(r"(\w)(\1{2,})", r"\1", tweet) for tweet in result]  # Remove all letter repetitions
    result = [re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"\d+", '', tweet) for tweet in result]  # Remove all numbers
    result = [tweet.translate(str.maketrans('', '', string.punctuation + '¡')) for tweet in result]  # Remove punctuation

    return result

def tokenize_list(datalist):
    print("Tokenizing")
    return [nltk.word_tokenize(row) for row in datalist]


def lemmatize_list(datalist):
    print("Lemmatizing data...")
    return [[token.lemma_ for token in lemmatizer(row)] for row in datalist]


def pipeline_processing_data(train_data, dev_data, tst_data):
    # PRE-PROCESSING
    preprocessed_train_content = text_preprocessing(train_data['content'])
    preprocessed_dev_content = text_preprocessing(dev_data['content'])
    preprocessed_tst_content = text_preprocessing(tst_data['content'])

    # # TOKENIZE
    # tokenized_train_content = tokenize_list(preprocessed_train_content)
    # tokenized_dev_content = tokenize_list(preprocessed_dev_content)
    # tokenized_tst_data = tokenize_list(preprocessed_tst_content)

    # clean_train_content = tokenized_train_content  # remove_stopwords(tokenized_train_content)
    # clean_dev_content = tokenized_dev_data  # remove_stopwords(tokenized_dev_data)
    # if bTestPhase is True:
    #     clean_tst_content = tokenized_tst_data  # remove_stopwords(tokenized_tst_data)

    # LIBRE OFFICE PROCESSING
    # libreoffice_train_tweets = [TreebankWordDetokenizer().detokenize(row)
    #                             for row in libreoffice_processing(tokenized_train_content)]
    # libreoffice_dev_tweets = [TreebankWordDetokenizer().detokenize(row)
    #                           for row in libreoffice_processing(tokenized_dev_data)]
    # libreoffice_tst_tweets = [TreebankWordDetokenizer().detokenize(row)
    #                           for row in libreoffice_processing(tokenized_tst_data)]

    # LEMMATIZING
    lemmatized_train_tweets = lemmatize_list(preprocessed_train_content)
    lemmatized_dev_tweets = lemmatize_list(preprocessed_dev_content)
    lemmatized_tst_tweets = lemmatize_list(preprocessed_tst_content)
    #
    # # REMOVING ACCENTS
    # without_accents_train = remove_accents(lemmatized_train_tweets)
    # without_accents_dev = remove_accents(lemmatized_dev_tweets)
    # if bTestPhase is True:
    #   without_accents_tst = remove_accents(lemmatized_tst_tweets)

    final_train_content = [TreebankWordDetokenizer().detokenize(row) for row in lemmatized_train_tweets]
    train_data['final_sentences'] = final_train_content

    final_dev_content = [TreebankWordDetokenizer().detokenize(row) for row in lemmatized_dev_tweets]
    dev_data['final_sentences'] = final_dev_content

    final_tst_content = [TreebankWordDetokenizer().detokenize(row) for row in lemmatized_tst_tweets]
    tst_data['final_sentences'] = final_tst_content

    return final_train_content, final_dev_content, final_tst_content


def read_files(sLang, bCross):
    train_data = pd.DataFrame()

    tree_tst = ET.parse(data_path + sLang + "/intertass_" + sLang + "_test.xml")
    tst_data = get_dataframe_from_xml(tree_tst)

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

    tree_dev = ET.parse(data_path + sLang + "/intertass_" + sLang + "_dev.xml")
    dev_data = get_dataframe_from_xml(tree_dev)

    return train_data, dev_data, tst_data


def write_for_fasttext(train_data, dev_data, tst_data, sLang, bCross):
    suffix = ''
    if bCross is True:
        suffix = '_cross'

    if bReduceLabels is True:
        suffix += '_reduced'
    else:
        suffix += '_full'

    filename_out = data_path + sLang + "/intertass_" + sLang + suffix + "_train.txt"
    if os.path.exists(filename_out) is False:
        with open(filename_out, 'w', encoding="utf-8") as f_out:
            print('Writing train file for FastText at: ' + filename_out)
            for txt, lbl in zip(train_data['final_sentences'], train_data['sentiment']):
                f_out.write('__label__' + str(LABEL_ENCODER.inverse_transform([lbl])[0]) + '\t' + txt + '\n')
    else:
        print('Skipping ' + filename_out)

    filename_out = data_path + sLang + "/intertass_" + sLang + suffix + "_dev.txt"
    if os.path.exists(filename_out) is False:
        with open(filename_out , 'w', encoding="utf-8") as f_out:
            print('Writing dev file for FastText at: ' + filename_out)
            for txt, lbl in zip(dev_data['final_sentences'], dev_data['sentiment']):
                f_out.write('__label__' + str(LABEL_ENCODER.inverse_transform([lbl])[0]) + '\t' + txt + '\n')
    else:
        print('Skipping ' + filename_out)

    filename_out = data_path + sLang + "/intertass_" + sLang + suffix + "_test.txt"
    if os.path.exists(filename_out) is False:    
        with open(filename_out, 'w', encoding="utf-8") as f_out:
            print('Writing tst file for FastText at: ' + filename_out)
            for txt, lbl in zip(tst_data['final_sentences'], tst_data['sentiment']):
                f_out.write(txt + '\n')
    else:
        print('Skipping ' + filename_out)


def write_fasttext_sentences(model_path, in_file, traindev, is_dev=False):
    print('Loading FastText model from: ' + model_path)
    model = fastText.load_model(model_path)

    suffix = '_sents_vectors_forDev'
    if traindev is True:
        suffix = '_sents_vectors_forTest'

    out_file = in_file.replace('.txt', suffix + '.txt')
    if os.path.exists(out_file) is False:        
        print("Writing FastText sentence vectors to: " + out_file)
        with  open(in_file) as f_in, open(out_file, 'w') as f_out:
            for line in f_in.readlines():
                if is_dev is True:  # Skip the label information
                    aLine = line.split('\t')
                    txt = aLine[1].strip()
                else:
                    txt = line.strip()
                np_line = np.array_str(model.get_sentence_vector(txt)).replace('\n', '')
                f_out.write(re.sub('[\[\]]', '', np_line) + '\n')
    else:
        print('Skipping ' + out_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--btest", help="if we want to work for test", action="store_true")
    parser.add_argument("-r", "--breduced", help="if we want to use reduced labels", action="store_true")
    parser.add_argument("-c", "--bcross", help="if we want to make the cross data", action="store_true")
    args = parser.parse_args()

    if args.btest:
        bTestPhase = True
    if args.breduced:
        bReduceLabels = True
    if args.bcross:
        CROSS_LINGUAL = [True]
    else:
        CROSS_LINGUAL = [False]

    # GET THE DATA
    for bCross in CROSS_LINGUAL:
        print('----------------- CROSS : ' + str(bCross) + ' ----------------')
        for sLang in LANGUAGE_CODE:
            print('** LANG: ' + sLang)
            
            output_dir = data_path + "final_outputs/"
            prefix = ''
            if bCross is True:
                output_dir = output_dir + '/cross/'
                prefix = '_cross'
            else:
                output_dir = output_dir + '/mono/'

            train_data, dev_data, tst_data = read_files(sLang, bCross)
            final_train_content, final_dev_content, final_tst_content = pipeline_processing_data(train_data, dev_data,
                                                                                                 tst_data)

            write_for_fasttext(train_data, dev_data, tst_data, sLang, bCross)
            if bReduceLabels is True:
                prefix += '_reduced'
                train_filename = data_path + sLang + "/intertass_" + sLang + prefix + "_train.txt"
                dev_filename = data_path + sLang + "/intertass_" + sLang + prefix + "_dev.txt"
                tst_filename = data_path + sLang + "/intertass_" + sLang + prefix + "_test.txt"
                labels = ['ID', 'N', 'O', 'P']
            else:
                prefix += '_full'
                train_filename = data_path + sLang + "/intertass_" + sLang + prefix + "_train.txt"
                dev_filename = data_path + sLang + "/intertass_" + sLang + prefix + "_dev.txt"
                tst_filename = data_path + sLang + "/intertass_" + sLang + prefix + "_test.txt"
                labels = ['ID', 'N', 'NEU', 'NONE', 'P']

            classifier_name = output_dir + "/intertass_" + sLang + prefix + '.bin'
            if bTestPhase is True:  # Combine train + dev
                import subprocess
                subprocess.call("cat " + train_filename + ' ' + dev_filename + " > " + data_path + sLang + "/intertass_" + sLang + prefix + "_train_dev.txt", shell=True)
                train_filename = data_path + sLang + "/intertass_" + sLang + prefix + "_train_dev.txt"
                classifier_name = output_dir + "/intertass_" + sLang + prefix + '_forTest.bin'

            if os.path.exists(classifier_name) is True:
                print('Loading classifier from ' + classifier_name)
                classifier = fastText.load_model(classifier_name)
            else:
                classifier = fastText.train_supervised(input=train_filename,
                                                   wordNgrams=2, dim=300, pretrainedVectors='./cc.es.300.vec')
                print('Saving classifier to ' + classifier_name)
                classifier.save_model(classifier_name)

            if bTestPhase is False:
                if os.path.exists(output_dir + sLang + "_dev_fasttext" + prefix + ".tsv") is False:
                    print("Writing " + output_dir + sLang + "_dev_fasttext" + prefix + ".tsv")
                    with open(dev_filename) as f_in,  \
                            open(output_dir + sLang + "_dev_fasttext" + prefix + ".tsv", 'w', newline='') as out_file, \
                            open(output_dir + sLang + "_dev_fasttext"+ prefix + ".csv", 'w', newline='') as out_csv_file:
                        tsv_writer = csv.writer(out_file, delimiter='\t')
                        csv_writer = csv.writer(out_csv_file)
                        csv_writer.writerow(labels)
                        row_values = dict()
                        for i, line in enumerate(f_in):
                            aLine = line.split('\t')
                            txt = aLine[1]
                            prediction = classifier.predict(txt.strip(), k=4)
                            for lbl, val in zip(prediction[0], prediction[1]):
                                row_values[lbl.replace('__label__', '')] = val
                            values = [str(row_values[column]) for column in labels if column != 'ID']
                            if bReduceLabels is True:
                                lbl = prediction[0][0].replace('__label__', '')
                                if lbl == 'O':
                                    lbl = replace_reducedLabel_perLanguage[sLang]
                                tsv_writer.writerow([dev_data['tweet_id'][i], lbl])
                                csv_writer.writerow([dev_data['tweet_id'][i], values[0], values[1], values[2]])
                            else:
                                tsv_writer.writerow(
                                    [dev_data['tweet_id'][i], prediction[0][0].replace('__label__', '')])
                                csv_writer.writerow([dev_data['tweet_id'][i], values[0], values[1], values[2], values[3]])
                else:
                    print("Skipping " + output_dir + sLang + "_dev_fasttext" + prefix + ".tsv")

            run_file = output_dir + sLang + "_dev_fasttext" + prefix + ".tsv"
            gold_file = data_path + sLang + "/intertass_" + sLang + "_dev_gold.tsv"

            scores = evalTask1(gold_file, run_file)
            with open(output_dir + prefix + '_' + sLang + "_fasttext.res", 'w', newline='') as out_file:
                print("f1_score: %f\n" % scores['maf1'])
                out_file.write("f1_score: %f\n" % scores['maf1'])
                print("precision: %f\n" % scores['map'])
                out_file.write("precision: %f\n" % scores['map'])
                print("recall: %f\n" % scores['mar'])
                out_file.write("recall: %f\n" % scores['mar'])
                print("%f\t%f\t%f\n" % (scores['map'], scores['mar'], scores['maf1']))

            if bTestPhase is True:
                if os.path.exists(output_dir + sLang + "_test_fasttext" + prefix + ".tsv") is False:
                    print("Writing " + output_dir + sLang + "_test_fasttext" + prefix + ".tsv")
                    with open(tst_filename) as f_in, \
                            open(output_dir + sLang + "_test_fasttext" + prefix + ".tsv", 'w', newline='') as out_file, \
                            open(output_dir + sLang + "_test_fasttext" + prefix + ".csv", 'w', newline='') as out_csv_file:
                        tsv_writer = csv.writer(out_file, delimiter='\t')
                        csv_writer = csv.writer(out_csv_file)
                        csv_writer.writerow(labels)
                        row_values = dict()
                        for i, txt in enumerate(f_in):
                            prediction = classifier.predict(txt.strip(), k=4)
                            for lbl, val in zip(prediction[0], prediction[1]):
                                row_values[lbl.replace('__label__', '')] = val
                            values = [str(row_values[column]) for column in labels if column != 'ID']
                            if bReduceLabels is True:
                                lbl = prediction[0][0].replace('__label__', '')
                                if lbl == 'O':
                                    lbl = replace_reducedLabel_perLanguage[sLang]
                                tsv_writer.writerow([tst_data['tweet_id'][i], lbl])
                                csv_writer.writerow([tst_data['tweet_id'][i], values[0], values[1], values[2]])
                            else:
                                tsv_writer.writerow(
                                    [tst_data['tweet_id'][i], prediction[0][0].replace('__label__', '')])
                                csv_writer.writerow([tst_data['tweet_id'][i], values[0], values[1], values[2], values[3]])

                else:
                    print("Skipping " + output_dir + sLang + "_test_fasttext" + prefix + ".tsv")


            # # Write the sentence vectors
            # print("Writing Sentence Vectors")
            # write_fasttext_sentences(model_path=output_dir + "/intertass_" + sLang + prefix + '.bin',
            #                          in_file=dev_filename,
            #                          traindev=bTestPhase, is_dev=True)
            #
            # if bTestPhase is True:
            #     write_fasttext_sentences(model_path=output_dir + "/intertass_" + sLang + prefix + '.bin',
            #                              in_file=tst_filename,
            #                              traindev=bTestPhase, is_dev=False)