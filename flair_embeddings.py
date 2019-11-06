from pathlib import Path

from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
import hunspell
import pandas as pd
from flair.visual.training_curves import Plotter


from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings, BertEmbeddings, \
    ELMoEmbeddings, FastTextEmbeddings, BytePairEmbeddings
from flair.training_utils import EvaluationMetric

import utils
import tweet_preprocessing
import spacy
from flair.embeddings import WordEmbeddings
from flair.data import Sentence


if __name__ == '__main__':

    bStoreFiles = True

    bLibreOffice = False
    bLemmatize = False
    bTokenize = False
    bPreprocess = True
    bUpsampling = True

    labels = ['0', '1', '2', '3']
    model_number = '7'

    if bLibreOffice:
        print("Loading Hunspell directory")
        dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")

    if bLemmatize:
        print("Loading Spacy Model")
        lemmatizer = spacy.load("es_core_news_md")  # GLOBAL to avoid loading the model several times

    for sLang in ['pe']:

        print("Training on -{}-".format(sLang.upper()))

        train_data, dev_data, test_data, _ = utils.read_files(sLang)

        if bUpsampling:
            print('Upsampling the data...')
            train_data = utils.perform_upsampling(train_data)

        if bPreprocess:
            train_data['content'] = tweet_preprocessing.preprocess(train_data['content'], bLowercasing=True,
                                                                   bPunctuation=True, bAll=False)
            dev_data['content'] = tweet_preprocessing.preprocess(dev_data['content'], bLowercasing=True,
                                                                 bPunctuation=True, bAll=False)
            test_data['content'] = tweet_preprocessing.preprocess(test_data['content'], bLowercasing=True,
                                                                  bPunctuation=True, bAll=False)

        if bTokenize:
            print("Tokenizing...")
            train_data['content'] = train_data.swifter.progress_bar(False).apply(
                lambda row: utils.tokenize_sentence(row.content), axis=1)
            dev_data['content'] = dev_data.swifter.progress_bar(False).apply(
                lambda row: utils.tokenize_sentence(row.content), axis=1)
            test_data['content'] = test_data.swifter.progress_bar(False).apply(
                    lambda row: utils.tokenize_sentence(row.content), axis=1)

        if bLibreOffice:
            print("LibreOffice Processing... ")
            train_data['content'] = train_data.swifter.progress_bar(True).apply(
                lambda row: utils.libreoffice_processing(row.content, dictionary), axis=1)
            dev_data['content'] = dev_data.swifter.apply(
                lambda row: utils.libreoffice_processing(row.content, dictionary), axis=1)
            test_data['content'] = test_data.swifter.apply(
                    lambda row: utils.libreoffice_processing(row.content, dictionary), axis=1)

        if bLemmatize:
            print("Lemmatizing data...")
            train_data['content'] = train_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.content, lemmatizer), axis=1)
            dev_data['content'] = dev_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.content, lemmatizer), axis=1)
            test_data['content'] = test_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.content, lemmatizer), axis=1)

        if bTokenize:
            train_data['content'] = [utils.untokenize_sentence(sentence) for sentence in train_data['content']]
            dev_data['content'] = [utils.untokenize_sentence(sentence) for sentence in dev_data['content']]
            test_data['content'] = [utils.untokenize_sentence(sentence) for sentence in test_data['content']]

        if bStoreFiles:
            utils.csv2ftx(train_data.content, train_data.sentiment, sLang, 'train', 'flair')
            utils.csv2ftx(dev_data.content, dev_data.sentiment, sLang, 'dev', 'flair')
            utils.csv2ftx(test_data.content, test_data.sentiment, sLang, 'test', 'flair')

        corpus = Corpus = ClassificationCorpus('./dataset/flair/', train_file='intertass_{}_train.txt'.format(sLang),
                                               dev_file='intertass_{}_dev.txt'.format(sLang),
                                               test_file='intertass_{}_test.txt'.format(sLang))

        word_embeddings = [BertEmbeddings('bert-base-multilingual-cased')]

        document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, rnn_type='LSTM',
                                                    reproject_words_dimension=256, dropout=0.35, rnn_layers=2)
        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
        trainer = ModelTrainer(classifier, corpus)
        trainer.train('./bert/dev/{}/'.format(sLang), max_epochs=20, mini_batch_size=16, anneal_factor=0.5,
                      learning_rate=0.05, patience=1)

        # trainer.find_learning_rate('./bert/learning_rate/dev/learning_rate1.tsv')

        # plotter = Plotter()
        # plotter.plot_training_curves('./bert/dev/{}/loss.tsv'.format(sLang))
        # plotter.plot_weights('./bert/dev/{}/weights.txt'.format(sLang))

        best_model = TextClassifier.load('./bert/dev/{}/best-model.pt'.format(sLang))

        dev_predictions, dev_neg, dev_neu, dev_none, dev_pos = [], [], [], [], []
        for tweet in dev_data['content']:
            max_score = 0.0
            row_values = dict()
            prediction = best_model.predict(Sentence(tweet), multi_class_prob=True)
            for lbl in zip(prediction[0].labels):
                label_to_write = lbl[0].value
                row_values[label_to_write] = lbl[0].score
                if lbl[0].score > max_score:
                    max_score = lbl[0].score
                    max_arg = label_to_write
            values = [row_values[column] for column in labels]
            label_to_write = max_arg
            dev_predictions.append(int(label_to_write))
            dev_neg.append(values[0])
            dev_neu.append(values[1])
            dev_none.append(values[2])
            dev_pos.append(values[3])

        final_df = pd.DataFrame({
            'predictions': dev_predictions,
            'N': dev_neg,
            'NEU': dev_neu,
            'NONE': dev_none,
            'POS': dev_pos,
        })
        final_df.to_csv('./bert/dev/{}/dev_results_{}.csv'.format(sLang, model_number), encoding='utf-8', sep='\t')

        print('------------------------>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      DEV')
        # utils.print_confusion_matrix(dev_predictions, utils.encode_label(dev_data['sentiment']))

        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                    multi_label=False)
        trainer = ModelTrainer(classifier, corpus)
        trainer.train('./bert/test/{}/'.format(sLang), learning_rate=0.05, mini_batch_size=16, anneal_factor=0.5, patience=1,
                      train_with_dev=True, max_epochs=8)

        # trainer.find_learning_rate('./bert/learning_rate/test/learning_rate1.tsv')

        # plotter = Plotter()
        # plotter.plot_training_curves('./bert/test/{}/loss.tsv'.format(sLang))
        # plotter.plot_weights('./bert/test/{}/weights.txt'.format(sLang))

        best_model = TextClassifier.load('./bert/test/{}/final-model.pt'.format(sLang))

        test_predictions, test_neg, test_neu, test_none, test_pos = [], [], [], [], []
        for tweet in test_data['content']:
            max_score = 0.0
            row_values = dict()
            prediction = best_model.predict(Sentence(tweet), multi_class_prob=True)
            for lbl in zip(prediction[0].labels):
                label_to_write = lbl[0].value
                row_values[label_to_write] = lbl[0].score
                if lbl[0].score > max_score:
                    max_score = lbl[0].score
                    max_arg = label_to_write
            values = [row_values[column] for column in labels if column != 'ID']
            label_to_write = max_arg
            test_predictions.append(int(label_to_write))
            test_neg.append(values[0])
            test_neu.append(values[1])
            test_none.append(values[2])
            test_pos.append(values[3])
        final_df = pd.DataFrame({
            'predictions': test_predictions,
            'N': test_neg,
            'NEU': test_neu,
            'NONE': test_none,
            'POS': test_pos,
        })
        final_df.to_csv('./bert/test/{}/test_results_{}.csv'.format(sLang, model_number), encoding='utf-8', sep='\t')

        print('------------------------>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      TEST')
        # utils.print_confusion_matrix(test_predictions, utils.encode_label(test_data['sentiment']))

        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')
