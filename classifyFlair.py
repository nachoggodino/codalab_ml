from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings, BertEmbeddings, ELMoEmbeddings, BytePairEmbeddings
from flair.data_fetcher import NLPTaskDataFetcher
from flair.training_utils import EvaluationMetric
from pathlib import Path


corpus = NLPTaskDataFetcher.load_classification_corpus(Path('../TASS2019/DATASETS/public_data/cr'),
                                                        train_file='intertass_cr_train.txt',
                                                        dev_file='intertass_cr_dev_prevTASS.txt',
                                                        test_file='intertass_cr_dev.txt')
#word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('spanish-forward-fast'), FlairEmbeddings('spanish-backward-fast')]
# word_embeddings = [WordEmbeddings('/media/lfdharo/97481d74-4cb5-4983-9a69-a748c32711ba/Data/Models/Glove/glove-sbwc_spanish.i25.vec'),
#                    FlairEmbeddings('spanish-forward-fast'),
#                    FlairEmbeddings('spanish-backward-fast')]
# word_embeddings = [BertEmbeddings('bert-base-multilingual-cased'),
#                    FlairEmbeddings('spanish-forward-fast'),
#                    FlairEmbeddings('spanish-backward-fast')]

# word_embeddings = [WordEmbeddings('../../../../Data/Models/Word2Vec/Spanish_CoNLL17/w2v_es_conll17.gensim.vec'),
#                    WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec'),
#                    ELMoEmbeddings('../../../../Data/Models/Elmo/Spanish_CoNLL17/')]

# word_embeddings = [FlairEmbeddings('spanish-forward-fast'), FlairEmbeddings('spanish-backward-fast')]


# document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)
# classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
# trainer = ModelTrainer(classifier, corpus)
# trainer.train('./', max_epochs=10)

search_space = SearchSpace()
# search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[word_embeddings])
# search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[8, 16, 32, 64, 128])
# search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
# search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
# search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.01, 0.025, 0.05, 0.1])
# search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])


# search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
#     [WordEmbeddings('../../../../Data/Models/Word2Vec/Spanish_CoNLL17/w2v_es_conll17.gensim.vec')],
#     [WordEmbeddings('../../../../Data/Models/Glove/glove-sbwc_spanish.i25.gensim.vec')],
#     [ELMoEmbeddings('../../../../Data/Models/Elmo/Spanish_CoNLL17/')],
#     [BytePairEmbeddings('es')],
# ])
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[[WordEmbeddings('../../../../Data/Models/Chars/lemma_lowercased_estenten11_freeling_v4_virt.gensim.vec')]])
# search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[16, 32, 64, 128])
# search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2, 3])
# search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.01, 0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[16, 32, 64])
search_space.add(Parameter.PATIENCE, hp.choice, options=[1])
search_space.add(Parameter.ANNEAL_FACTOR, hp.choice, options=[0.3, 0.5, 0.8])

param_selector = TextClassifierParamSelector(
    corpus=corpus,
    multi_label=False,
    base_path='resources/results_flairs',
    document_embedding_type='max',
    max_epochs=100,
    training_runs=1,
    optimization_value=OptimizationValue.DEV_SCORE,
    evaluation_metric=EvaluationMetric.MACRO_F1_SCORE
)
param_selector.optimize(search_space, max_evals=20)