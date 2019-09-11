from gensim.models import KeyedVectors
import pandas as pd
import utils
from sklearn import svm, preprocessing
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
from tweet_preprocessing import preprocess

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

_, _, test_data, _ = utils.read_files('es')

classifier = TextClassifier.load('./resources/results_flair/training2/best-model.pt')
data = preprocess(test_data['content'])

sentences = [Sentence(tweet) for tweet in data]
result = classifier.predict(sentences)
predictions = [int(sentence.labels[0].value) for sentence in result]
utils.print_confusion_matrix(predictions, utils.encode_label(test_data['sentiment']))
print(result)

# sentence = Sentence("Hey! que tal estamos tio!!")
# result = classifier.predict(sentence)
# print(result[0].labels[0].value)

