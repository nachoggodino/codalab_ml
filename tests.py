import nltk
import re
import unidecode
import spacy


lemmatizer = spacy.load("es_core_news_sm")


def lemmatize_list(datalist):
    result = []
    for row in datalist:
        lemmatized_words = [lemmatizer(token)[0].lemma_ for token in row]
        result.append(lemmatized_words)
    return result


print(lemmatize_list([('buenos', 'días', 'me', 'preguntaba', 'como', 'estas')]))
