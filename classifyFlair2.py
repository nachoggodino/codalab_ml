import argparse
import logging
import os
import csv
import sys
from pathlib import Path

import pandas as pd

from sklearn.utils import resample

from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, BertEmbeddings, DocumentRNNEmbeddings, \
    DocumentPoolEmbeddings, OneHotEmbeddings, ELMoEmbeddings, BytePairEmbeddings
from flair.models import TextClassifier
from flair.training_utils import EvaluationMetric

from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair.data import Sentence
import flair.datasets

# Append script path to import path to locate tass_eval
sys.path.append(os.path.realpath(__file__))

# Import evalTask1 function fro tass_eval module
from tass_eval import evalTask1
from utils import read_file, save_file, get_labelencoder

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


LANGUAGE_CODE = ['cr', 'es', 'mx', 'pe', 'uy']
replace_reducedLabel_perLanguage = {'cr': 'NONE', 'es': 'NEU', 'mx': 'NEU', 'pe': 'NONE', 'uy': 'NEU'}
data_path = "../TASS2019/DATASETS/public_data/"

FLAIR_EMBEDDINGS = False
BERT_EMBEDDINGS = False
GLOVE_EMBEDDINGS = False
WORD2VEC_EMBEDDINGS = False
FINETUNED_HOT_EMBEDDINGS = False
ELMO_EMBEDDINGS = False
WORD2VEC_GLOVE_EMBEDDINGS = False
BYTE_PAIR_EMBEDDINGS = True
BPE_WORD2VEC_GLOVE_EMBEDDINGS = False
WIKIFT_FLAIR = False
WIKIFT = False
BERT_GLOVE_BPE = False
CHAR_EMBEDDINGS = False
BPE_GLOVE_CHARS = False
GLOVE_CHARS = False
GLOVE_BPE = False


'''
import gensim
vectors = gensim.models.KeyedVectors.load_word2vec_format('./model.txt', binary=False, unicode_errors='ignore')
vectors.save('./w2v_es_conll17.gensim.vec', pickle_protocol=4)
'''

def resample_dataset(train_data):
    print('Nada')
    # FIND MAJORITY CLASS
    majority_class = ""
    minority_class = ""
    count_majority_class = 0
    count_minority_class = 9999999999
    lbl_encoder = get_labelencoder()
    dict_counts = dict()

    for item in ['N', 'NEU', 'NONE', 'P']:
        l = len(train_data.loc[train_data['sentiment'] == lbl_encoder.transform([item])[0]])
        dict_counts[item] = l
        if l > count_majority_class:
            majority_class = item
            count_majority_class = l
        if l < count_minority_class:
            minority_class = item
            count_minority_class = l

    # Upsample minority class
    lst_upsampled = list()
    for item in ['N', 'NEU', 'NONE', 'P']:
        if item != majority_class:
            print("Upsampling minority class: {} ({}) following majority class: {} ({})".format(item,
                                                                                                dict_counts[item],
                                                                                                majority_class,
                                                                                                count_majority_class))
            df_minority_upsampled = resample(
                train_data.loc[train_data['sentiment'] == lbl_encoder.transform([item])[0]],
                replace=True,  # sample with replacement
                n_samples=count_majority_class - dict_counts[item],  # to match majority class
                random_state=123)  # reproducible results
            lst_upsampled.append(df_minority_upsampled)
    lst_upsampled.append(train_data)
    df_upsampled = pd.concat(lst_upsampled)
    return df_upsampled


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--btest", help="if we want to work for test", action="store_true")
    parser.add_argument("-r", "--breduced", help="if we want to use reduced labels", action="store_true")
    parser.add_argument("-c", "--bcross", help="if we want to make the cross data", action="store_true")
    parser.add_argument("-l", "--lang", help="Language to process", required=True)
    parser.add_argument("-m", "--lr", help="Learning rate [default: 0.5]", type=float, default=0.5)
    parser.add_argument("-i", "--iters", help="Number of iterations [Default: 20]", type=int, default=20)
    parser.add_argument("-e", "--emb", help="Embeddings to use. Concat using :. Options: flair, bert, glove, word2vec, elmo, bpe, wiki, chars", required=True)
    parser.add_argument("-p", "--pooling", help="Pooling to use. Default: mean [max, min, mean]",
                        type=str,
                        default='mean',
                        choices=['mean', 'max', 'min'])

    cmd_args = parser.parse_args()

    if cmd_args.emb:
        embeddings = list()
        for type_emb in cmd_args.emb.split(":"):
            if type_emb == 'flair':
                embeddings.append(FlairEmbeddings('spanish-forward-fast'))
                embeddings.append(FlairEmbeddings('spanish-backward-fast'))
            elif type_emb == 'bert':
                embeddings.append(BertEmbeddings('bert-base-multilingual-cased'))
            elif type_emb == 'glove':
                embeddings.append(WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec'))
            elif type_emb == 'word2vec':
                embeddings.append(WordEmbeddings('../../../../Data/Models/Word2Vec/Spanish_CoNLL17/w2v_es_conll17.gensim.vec'))
            elif type_emb == 'elmo':
                embeddings.append(ELMoEmbeddings('../../../../Data/Models/Elmo/Spanish_CoNLL17/'))
            elif type_emb == 'bpe':
                embeddings.append(BytePairEmbeddings(language='es'))
            elif type_emb == 'wiki':
                embeddings.append(WordEmbeddings('../../../../Data/Models/FastText/wiki.es.gensim.vec'))
            elif type_emb == 'chars':
                embeddings.append(WordEmbeddings('../../../../Data/Models/Chars/lemma_lowercased_estenten11_freeling_v4_virt.gensim.vec'))
            else:
                print('ERROR: type of embedding no accepted' + cmd_args.emb + '. Options: flair, bert, glove, word2vec, elmo, bpe, wiki, chars')
                exit()

        prefix_model_output_dir = '_'.join(cmd_args.emb.split(":"))
        if cmd_args.pooling != 'mean':
            prefix_model_output_dir += '_' + cmd_args.pooling
        document_embeddings = DocumentPoolEmbeddings(embeddings, pooling=cmd_args.pooling, fine_tune_mode='linear')

    if cmd_args.btest:
        bTestPhase = True
    else:
        bTestPhase = False

    if cmd_args.breduced:
        bReduceLabels = True
    else:
        bReduceLabels = False

    if cmd_args.bcross:
        CROSS_LINGUAL = [True]
    else:
        CROSS_LINGUAL = [False]

    LANGUAGE_CODE = [cmd_args.lang]  # we can only process one language at time because of CUDA

    # GET THE DATA
    for bCross in CROSS_LINGUAL:
        print('----------------- CROSS : ' + str(bCross) + ' ----------------')
        for sLang in LANGUAGE_CODE:
            print('** LANG: ' + sLang)

            output_dir = data_path + "outputs_flair_" + prefix_model_output_dir + '/' + sLang
            prefix = ''
            if bCross is True:
                output_dir = output_dir + '/cross/'
                prefix = '_cross'
            else:
                output_dir = output_dir + '/mono/'

            if os.path.exists(output_dir) is False:
                print('Creating directory ' + output_dir)
                os.makedirs(output_dir)

            if bReduceLabels is True:
                prefix += '_reduced'
                train_filename = "intertass_" + sLang + prefix + "_train.txt"
                dev_filename =  "intertass_" + sLang + prefix + "_dev.txt"
                tst_filename = "intertass_" + sLang + "_test_prevTASS.txt"
                eval_filename = "intertass_" + sLang + prefix + "_test.txt"
                labels = ['ID', 'N', 'O', 'P']
            else:
                prefix += '_full'
                train_filename = "intertass_" + sLang + prefix + "_train.txt"
                dev_filename = "intertass_" + sLang + prefix + "_dev.txt"
                tst_filename = "intertass_" + sLang + "_test_prevTASS.txt"
                eval_filename = "intertass_" + sLang + prefix + "_test.txt"
                labels = ['ID', 'N', 'NEU', 'NONE', 'P']

            # train_data = read_file(data_path + sLang + "/intertass_" + sLang + "_train.xml")
            # train_data = resample_dataset(train_data)
            # train_filename = "intertass_" + sLang + "_train.txt2"
            # save_file(train_data, data_path + sLang + "/intertass_" + sLang + "_train.txt2")

            if bTestPhase is False:
                corpus = NLPTaskDataFetcher.load_classification_corpus(Path(data_path + sLang),
                                                                       train_file=train_filename,
                                                                       dev_file=dev_filename,
                                                                       test_file=tst_filename)
            else:
                train_filename = train_filename.replace('_train.txt', '_train_dev.txt')
                corpus = NLPTaskDataFetcher.load_classification_corpus(Path(data_path + sLang),
                                                                       train_file=train_filename,
                                                                       test_file=tst_filename)

            # prefix_model_output_dir = ''
            # if FLAIR_EMBEDDINGS is True:
            #     embeddings = [FlairEmbeddings('spanish-forward-fast'),
            #                        FlairEmbeddings('spanish-backward-fast')]
            #
            #     # document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512,
            #     #                                              reproject_words=True,
            #     #                                              reproject_words_dimension=256,
            #     #                                              dropout=0.35,
            #     #                                              rnn_layers=2,
            #     #                                              rnn_type='LSTM'
            #     #                                              )
            #     document_embeddings = DocumentPoolEmbeddings(embeddings)
            #     prefix_model_output_dir = "flair"
            #
            # elif BERT_EMBEDDINGS is True:
            #     embeddings = [BertEmbeddings('bert-base-multilingual-cased')]
            #
            #     # document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512,
            #     #                                              reproject_words=True,
            #     #                                              reproject_words_dimension=256,
            #     #                                              dropout=0.35,
            #     #                                              rnn_layers=2,
            #     #                                              rnn_type='LSTM'
            #     #                                              )
            #     document_embeddings = DocumentPoolEmbeddings(embeddings)
            #     prefix_model_output_dir = "bert"
            #
            # elif GLOVE_EMBEDDINGS is True:
            #     # instantiate pre-trained word embeddings
            #     embeddings = WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec')
            #     # document pool embeddings
            #     # document_embeddings = DocumentPoolEmbeddings([embeddings], fine_tune_mode='nonlinear')
            #     document_embeddings = DocumentPoolEmbeddings([embeddings])
            #     prefix_model_output_dir = "glove"
            #
            # elif WORD2VEC_EMBEDDINGS is True:
            #     embeddings = WordEmbeddings('../../../../Data/Models/Word2Vec/Spanish_CoNLL17/w2v_es_conll17.gensim.vec')
            #     document_embeddings = DocumentPoolEmbeddings([embeddings])
            #     prefix_model_output_dir = "w2v"
            #
            # elif FINETUNED_HOT_EMBEDDINGS is True:
            #     embeddings = OneHotEmbeddings(corpus)
            #     document_embeddings = DocumentPoolEmbeddings([embeddings])
            #     prefix_model_output_dir = "hot"
            #
            # elif ELMO_EMBEDDINGS is True:
            #     embeddings = ELMoEmbeddings('../../../../Data/Models/Elmo/Spanish_CoNLL17/')
            #     document_embeddings = DocumentPoolEmbeddings([embeddings])
            #     prefix_model_output_dir = "elmo"
            #
            # elif WORD2VEC_GLOVE_EMBEDDINGS is True:
            #     embeddings = [WordEmbeddings('../../../../Data/Models/Word2Vec/Spanish_CoNLL17/w2v_es_conll17.gensim.vec'),
            #                    WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec')]
            #     document_embeddings = DocumentPoolEmbeddings(embeddings)
            #     prefix_model_output_dir = "glove_word2vec"
            #
            # elif BYTE_PAIR_EMBEDDINGS is True:
            #     embeddings = BytePairEmbeddings(language='es')
            #     document_embeddings = DocumentPoolEmbeddings([embeddings])
            #     prefix_model_output_dir = "bpe"
            #
            # elif BPE_WORD2VEC_GLOVE_EMBEDDINGS is True:
            #     embeddings = [WordEmbeddings('../../../../Data/Models/Word2Vec/Spanish_CoNLL17/w2v_es_conll17.gensim.vec'),
            #                   WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec'),
            #                   BytePairEmbeddings(language='es')]
            #     document_embeddings = DocumentPoolEmbeddings(embeddings)
            #     prefix_model_output_dir = "bpe_glove_word2vec"
            #
            # elif WIKIFT_FLAIR is True:
            #     embeddings = [WordEmbeddings('../../../../Data/Models/FastText/wiki.es.gensim.vec'),
            #                   FlairEmbeddings('spanish-forward-fast'),
            #                   FlairEmbeddings('spanish-backward-fast')]
            #     document_embeddings = DocumentPoolEmbeddings(embeddings)
            #     prefix_model_output_dir = "fasttext_flair"
            #
            # elif WIKIFT is True:
            #     embeddings = [WordEmbeddings('../../../../Data/Models/FastText/wiki.es.gensim.vec')]
            #     document_embeddings = DocumentPoolEmbeddings(embeddings)
            #     prefix_model_output_dir = "fasttext"
            #
            #
            # elif BERT_GLOVE_BPE is True:
            #     embeddings = [BertEmbeddings('bert-base-multilingual-cased'),
            #                   WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec'),
            #                   BytePairEmbeddings(language='es')]
            #     document_embeddings = DocumentPoolEmbeddings(embeddings, pooling='max')
            #     prefix_model_output_dir = "bpe_glove_bert_maxpool"
            #
            # elif CHAR_EMBEDDINGS is True:
            #     # embeddings = [WordEmbeddings('../../../../Data/Models/Chars/estenten11_freeling_v4_virt.gensim.vec')]
            #     embeddings = [WordEmbeddings('../../../../Data/Models/Chars/lemma_lowercased_estenten11_freeling_v4_virt.gensim.vec')]
            #     document_embeddings = DocumentPoolEmbeddings(embeddings, pooling='max')
            #     prefix_model_output_dir = "lemma_chars"
            #
            # elif BPE_GLOVE_CHARS is True:
            #     embeddings = [WordEmbeddings('../../../../Data/Models/Chars/lemma_lowercased_estenten11_freeling_v4_virt.gensim.vec'),
            #                   WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec'),
            #                   BytePairEmbeddings(language='es')]
            #     document_embeddings = DocumentPoolEmbeddings(embeddings, pooling='max')
            #     prefix_model_output_dir = "bpe_glove_chars_maxpool"
            #
            # elif GLOVE_CHARS is True:
            #     embeddings = [WordEmbeddings('../../../../Data/Models/Chars/lemma_lowercased_estenten11_freeling_v4_virt.gensim.vec'),
            #                   WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec')]
            #     document_embeddings = DocumentPoolEmbeddings(embeddings, pooling='max')
            #     prefix_model_output_dir = "glove_chars_maxpool"
            #
            # elif GLOVE_BPE is True:
            #     embeddings = [BytePairEmbeddings(language='es'),
            #                   WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec')]
            #     document_embeddings = DocumentPoolEmbeddings(embeddings, pooling='max')
            #     prefix_model_output_dir = "glove_chars_maxpool"

            if bTestPhase is False:
                classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                            multi_label=False)
                trainer = ModelTrainer(classifier, corpus)
                trainer.train('./' + prefix_model_output_dir + '_' + sLang + prefix + '/',
                              learning_rate=cmd_args.lr,
                              mini_batch_size=16,
                              anneal_factor=0.5,
                              patience=1,
                              evaluation_metric=EvaluationMetric.MICRO_F1_SCORE,
                              max_epochs=cmd_args.iters)

                plotter = Plotter()
                plotter.plot_training_curves('./' + prefix_model_output_dir + '_' + sLang + prefix + '/loss.tsv')
                plotter.plot_weights('./' + prefix_model_output_dir + '_' + sLang + prefix + '/weights.txt')

                # 7. find learning rate
                learning_rate_tsv = trainer.find_learning_rate('./' + prefix_model_output_dir + '_' + sLang + prefix + '/learning_rate.tsv')

                plotter = Plotter()
                plotter.plot_learning_rate(learning_rate_tsv)
                del(classifier)
                del(trainer)

                classifier = TextClassifier.load('./' + prefix_model_output_dir + '_' + sLang + prefix + '/best-model.pt')
                dev_data = read_file(data_path + sLang + "/intertass_" + sLang + "_dev.xml")

                print("Writing " + output_dir + sLang + "_dev_" + prefix_model_output_dir + prefix + ".tsv")
                with open(data_path + sLang + '/' + dev_filename) as f_in, \
                        open(output_dir + sLang + "_dev_" + prefix_model_output_dir + prefix + ".tsv", 'w', newline='') as out_file, \
                        open(output_dir + sLang + "_dev_" + prefix_model_output_dir + prefix + ".csv", 'w',
                             newline='') as out_csv_file:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    csv_writer = csv.writer(out_csv_file)
                    csv_writer.writerow(labels)
                    for i, line in enumerate(f_in):
                        aLine = line.split('\t')
                        txt = aLine[1]
                        prediction = classifier.predict(Sentence(txt), multi_class_prob=True)
                        max_score = 0.0
                        row_values = dict()
                        for lbl in zip(prediction[0].labels):
                            label_to_write = lbl[0].value
                            if bReduceLabels is True:
                                if lbl[0].value == 'O':
                                    label_to_write = replace_reducedLabel_perLanguage[sLang]
                                    row_values[label_to_write] = lbl[0].score
                            else:
                                row_values[label_to_write] = lbl[0].score
                            if lbl[0].score > max_score:
                                max_score = lbl[0].score
                                max_arg = label_to_write
                        values = [str(row_values[column]) for column in labels if column != 'ID']
                        label_to_write = max_arg
                        if bReduceLabels is True:
                            tsv_writer.writerow([dev_data['tweet_id'][i], label_to_write])
                            csv_writer.writerow([dev_data['tweet_id'][i], values[0], values[1], values[2]])
                        else:
                            tsv_writer.writerow(
                                [dev_data['tweet_id'][i], label_to_write])
                            csv_writer.writerow(
                                [dev_data['tweet_id'][i], values[0], values[1], values[2], values[3]])

                tst_data = pd.read_csv(data_path + sLang + '/' + tst_filename.replace('_test_prevTASS.txt', '_test_prevTASS.csv'), sep="\t")
                print("Writing " + output_dir + sLang + "_test_prevTASS_predictions_" + prefix_model_output_dir + prefix + ".tsv")
                with open(output_dir + sLang + "_test_prevTASS_predictions_" + prefix_model_output_dir + prefix + ".tsv", 'w', newline='') as out_file, \
                     open(output_dir + sLang + "_test_prevTASS_predictions_" + prefix_model_output_dir + prefix + ".csv", 'w',
                             newline='') as out_csv_file:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    csv_writer = csv.writer(out_csv_file)
                    csv_writer.writerow(labels)
                    for index, row in tst_data.iterrows():
                        txt = row['content']
                        prediction = classifier.predict(Sentence(txt), multi_class_prob=True)
                        max_score = 0.0
                        row_values = dict()
                        for lbl in zip(prediction[0].labels):
                            label_to_write = lbl[0].value
                            if bReduceLabels is True:
                                if lbl[0].value == 'O':
                                    label_to_write = replace_reducedLabel_perLanguage[sLang]
                                    row_values[label_to_write] = lbl[0].score
                            else:
                                row_values[label_to_write] = lbl[0].score
                            if lbl[0].score > max_score:
                                max_score = lbl[0].score
                                max_arg = label_to_write
                        values = [str(row_values[column]) for column in labels if column != 'ID']
                        label_to_write = max_arg
                        if bReduceLabels is True:
                            tsv_writer.writerow([row['tweet_id'], label_to_write])
                            csv_writer.writerow([row['tweet_id'], values[0], values[1], values[2]])
                        else:
                            tsv_writer.writerow(
                                [row['tweet_id'], label_to_write])
                            csv_writer.writerow(
                                [row['tweet_id'], values[0], values[1], values[2], values[3]])

            run_file = output_dir + sLang + "_dev_" + prefix_model_output_dir + prefix + ".tsv"
            gold_file = data_path + sLang + "/intertass_" + sLang + "_dev_gold.tsv"

            scores = evalTask1(gold_file, run_file)
            with open(output_dir + sLang + "_dev_" + prefix_model_output_dir + prefix + ".res", 'w', newline='') as out_file:
                print("f1_score: %f\n" % scores['maf1'])
                out_file.write("f1_score: %f\n" % scores['maf1'])
                print("precision: %f\n" % scores['map'])
                out_file.write("precision: %f\n" % scores['map'])
                print("recall: %f\n" % scores['mar'])
                out_file.write("recall: %f\n" % scores['mar'])
                print("%f\t%f\t%f\n" % (scores['map'], scores['mar'], scores['maf1']))

            if bTestPhase is True:
                prefix_model_output_dir += '_forTest'
                classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                            multi_label=False)
                trainer = ModelTrainer(classifier, corpus)
                trainer.train('./' + prefix_model_output_dir + '_' + sLang + prefix + '/',
                              learning_rate=cmd_args.lr,
                              mini_batch_size=16,
                              anneal_factor=0.5,
                              patience=1,
                              train_with_dev=True,
                              evaluation_metric=EvaluationMetric.MICRO_F1_SCORE,
                              max_epochs=cmd_args.iters)

                plotter = Plotter()
                plotter.plot_training_curves('./' + prefix_model_output_dir + '_' + sLang + prefix + '/loss.tsv')
                plotter.plot_weights('./' + prefix_model_output_dir  + '_' + sLang + prefix + '/weights.txt')

                # 7. find learning rate
                learning_rate_tsv = trainer.find_learning_rate(prefix_model_output_dir + '_' + sLang + prefix + '/learning_rate.tsv')

                plotter = Plotter()
                plotter.plot_learning_rate(learning_rate_tsv)

                del(classifier)
                del(trainer)

                classifier = TextClassifier.load('./' + prefix_model_output_dir + '_' + sLang + prefix + '/final-model.pt')
                eval_data = read_file(data_path + sLang + "/intertass_" + sLang + "_test.xml")

                print("Writing " +  output_dir + sLang + "_test_" + prefix_model_output_dir + prefix + ".tsv")
                with open(data_path + sLang + '/' + eval_filename) as f_in, \
                        open( output_dir + sLang + "_test_" + prefix_model_output_dir + prefix + ".tsv", 'w', newline='') as out_file, \
                        open( output_dir + sLang + "_test_" + prefix_model_output_dir + prefix + ".csv", 'w',
                             newline='') as out_csv_file:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    csv_writer = csv.writer(out_csv_file)
                    csv_writer.writerow(labels)
                    for i, line in enumerate(f_in):
                        txt = line
                        prediction = classifier.predict(Sentence(txt), multi_class_prob=True)
                        max_score = 0.0
                        row_values = dict()
                        for lbl in zip(prediction[0].labels):
                            label_to_write = lbl[0].value
                            if bReduceLabels is True:
                                if lbl[0].value == 'O':
                                    label_to_write = replace_reducedLabel_perLanguage[sLang]
                                    row_values[label_to_write] = lbl[0].score
                            else:
                                row_values[label_to_write] = lbl[0].score
                            if lbl[0].score > max_score:
                                max_score = lbl[0].score
                                max_arg = label_to_write
                        values = [str(row_values[column]) for column in labels if column != 'ID']
                        label_to_write = max_arg
                        if bReduceLabels is True:
                            tsv_writer.writerow([eval_data['tweet_id'][i], label_to_write])
                            csv_writer.writerow([eval_data['tweet_id'][i], values[0], values[1], values[2]])
                        else:
                            tsv_writer.writerow(
                                [eval_data['tweet_id'][i], label_to_write])
                            csv_writer.writerow(
                                [eval_data['tweet_id'][i], values[0], values[1], values[2], values[3]])


