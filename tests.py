from gensim.models import KeyedVectors
import pandas as pd
import utils
from sklearn import svm
import os

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

