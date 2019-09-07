from gensim.models import KeyedVectors
import pandas as pd
import utils
from sklearn import svm
import os
import re
import string
import hunspell
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.data import Sentence

from pathlib import Path

# # from wikipedia2vec import Wikipedia2Vec
#
# # Wikipedia2Vec.load('./embeddings/eswiki_20180420_300d.pkl')
#
#
# def get_sentiment_vocabulary():
#
#     pos_df = pd.read_csv('./lexicons/isol/positivas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
#     neg_df = pd.read_csv('./lexicons/isol/negativas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
#
#     return pos_df['words'].array, neg_df['words'].array
#
#
# positive_voc, negative_voc = get_sentiment_vocabulary()
#
# # model = KeyedVectors.load_word2vec_format('./embeddings/eswiki_20180420_300d.txt')
# # model.save('./embeddings/eswiki_20180420_gensim.bin')
# saved_model = KeyedVectors.load('./embeddings/eswiki_20180420_gensim.bin', mmap='r')
# print('Model Re-loaded')
#
# vectors, polarities = [], []
#
# for word in positive_voc:
#     if word not in saved_model.vocab:
#         continue
#     vectors.append(saved_model.get_vector(word))
#     polarities.append(1)
#
# for word in negative_voc:
#     if word not in saved_model.vocab:
#         continue
#     vectors.append(saved_model.get_vector(word))
#     polarities.append(-1)
#
# dataframe = pd.DataFrame({'vector': vectors, 'polarity': polarities}).sample(frac=1).reset_index(drop=True)
# print(dataframe)
#
# classifier = svm.SVR()
# print(dataframe['vector'].array)
# classifier.fit(dataframe['vector'].array, dataframe['polarity'])
# print(classifier.predict('bueno'))
# print(classifier.predict('idiota'))


# for sLang in ['es', 'cr', 'mx', 'pe', 'uy']:
#     test_data = utils.get_dataframe_from_xml(utils.parse_xml('./dataset/xml/intertass_{}_test.xml'.format(sLang)))
#     labels = pd.read_csv('./tass_test_gold/{}.tsv'.format(sLang), sep='\t', header=None)
#     labels.columns = ['tweet_id', 'sentiment']
#     test_data['sentiment'] = labels.sentiment
#     print(test_data.sentiment)
#     print(test_data)
#     test_data.to_csv('./dataset/csv/intertass_{}_test.csv'.format(sLang), sep='\t', encoding='utf-8')

train_data = pd.read_csv('./dataset/csv/intertass_es_train.csv', encoding='utf-8', sep='\t')
test_data = pd.read_csv('./dataset/csv/intertass_es_test.csv', encoding='utf-8', sep='\t')
dev_data = pd.read_csv('./dataset/csv/intertass_es_dev.csv', encoding='utf-8', sep='\t')
#

#
# dev_data.filter(['content', 'labels'], axis=1).to_csv('./dataset/flair/train.csv', sep='\t', index=False, header=False)
# train_data.filter(['content', 'labels'], axis=1).to_csv('./dataset/flair/dev.csv', sep='\t', index=False, header=False)
# test_data.filter(['content', 'labels'], axis=1).to_csv('./dataset/flair/test.csv', sep='\t', index=False, header=False)
#
# columns = {1: 'content', 5: 'labels'}
# corpus: Corpus = ColumnCorpus('./dataset/flair', columns)
#
# word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]
#
# document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256,)
# classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=True)
#
# trainer = ModelTrainer(classifier, corpus)
# trainer.train('./dataset/flair/', max_epochs=10)

sLang = 'uy'

utils.csv_to_flair_format(train_data['content'], train_data['sentiment'], sLang, 'train')
utils.csv_to_flair_format(dev_data['content'], dev_data['sentiment'], sLang, 'dev')
utils.csv_to_flair_format(test_data['content'], test_data['sentiment'], sLang, 'test')

