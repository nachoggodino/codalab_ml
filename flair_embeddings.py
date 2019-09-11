from pathlib import Path

from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings, BertEmbeddings, ELMoEmbeddings, BytePairEmbeddings
from flair.training_utils import EvaluationMetric

import utils

from flair.embeddings import WordEmbeddings
from flair.data import Sentence

if __name__ == '__main__':

    # utils.csv_to_flair_format(preprocess=True)

    sLang = 'es'

    corpus = Corpus = ClassificationCorpus('./dataset/flair/', train_file='intertass_{}_train.txt'.format(sLang),
                                           dev_file='intertass_{}_dev.txt'.format(sLang),
                                           test_file='intertass_{}_test.txt'.format(sLang))

    search_space = SearchSpace()

    word_embeddings = [
        [BertEmbeddings('bert-base-multilingual-cased')],
        [FlairEmbeddings('spanish-forward-fast'), FlairEmbeddings('spanish-backward-fast'), BertEmbeddings('bert-base-multilingual-cased')],
        [FlairEmbeddings('spanish-forward-fast'), FlairEmbeddings('spanish-backward-fast')]
    ]

    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=word_embeddings)
    # search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[16, 32, 64, 128, 256, 512])
    # search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2, 3])
    # search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.01, 0.05, 0.1, 0.15, 0.2])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[4, 8, 16, 32])
    search_space.add(Parameter.PATIENCE, hp.choice, options=[1])
    search_space.add(Parameter.ANNEAL_FACTOR, hp.choice, options=[0.3, 0.5, 0.8])

    param_selector = TextClassifierParamSelector(
        corpus=corpus,
        multi_label=False,
        base_path='resources/results_flairs',
        document_embedding_type='max',
        max_epochs=20,
        training_runs=1,
        optimization_value=OptimizationValue.DEV_SCORE,
        evaluation_metric=EvaluationMetric.MACRO_F1_SCORE
    )
    param_selector.optimize(search_space, max_evals=2)

    # document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, rnn_type='LSTM',
    #                                             reproject_words_dimension=256, bidirectional=True)
    # classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
    # trainer = ModelTrainer(classifier, corpus)
    # trainer.train('./resources/results_flair/training2/', max_epochs=10, mini_batch_size=32, anneal_factor=0.3,
    #               learning_rate=0.05, patience=1)
