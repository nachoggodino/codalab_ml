from gensim.models import KeyedVectors
import pandas as pd
import utils
from sklearn import svm
import os
import re
import string

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


def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])


emoji_pattern = re.compile("[" 
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

url_pattern = re.compile(".*http.*")

clean_tweet = "@ToniThrowdown ahora tengo un dilema. Ayúdame Toni Y en 3DS porque me salió a 25 euros nuevo https://t.co/2bSRMWnrZ5"

clean_tweet = clean_tweet.replace('\n', '').strip()
clean_tweet = " ".join([emoji_pattern.sub(r'EMOJI', word) for word in clean_tweet.split()])
clean_tweet = clean_tweet.replace(u'\u2018', "'").replace(u'\u2019', "'")
clean_tweet = re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), clean_tweet)
# clean_tweet = clean_tweet.lower()
clean_tweet = re.sub(r"http\S+", "HTTP", clean_tweet)
clean_tweet = re.sub(r"\B@\w+", 'USERNAME', clean_tweet)
clean_tweet = re.sub(r"(\w)(\1{2,})", r"\1", clean_tweet)
clean_tweet = re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'JAJAJA', clean_tweet)
clean_tweet = re.sub(r"\d+", '', clean_tweet)
# clean_tweet = clean_tweet.translate(str.maketrans('', '', string.punctuation + '¡'))

print(clean_tweet)




