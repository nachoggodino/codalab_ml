import nltk
import re
import unidecode
import spacy
from re import finditer


lemmatizer = spacy.load("es_core_news_sm")


def lemmatize_list(datalist):
    result = []
    for row in datalist:
        lemmatized_words = [lemmatizer(token)[0].lemma_ for token in row]
        result.append(lemmatized_words)
    return result


def lemmatize_word(word):
    print(lemmatizer(word)[0].lemma_)


def camel_case_split(identifier):
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier)
    return [m.group(0) for m in matches]


def regex_tester(string):
    print(re.search(r"(\w)(\1{2,})", string))
    return

def regex_sub(string):
    print(re.sub(r"(\w)(\1{2,})", camel_case_split, string))
    return


print(camel_case_split("holaQueTal"))
print(camel_case_split("holaQueTal"))
print(camel_case_split("HolaBuenastardesComoEstas"))
print(camel_case_split("holaSoy EspacialRotoMas henoJulio"))
print(camel_case_split("JuanEsUnImbecil"))
print(camel_case_split("buenos diasComoEstas"))
print(camel_case_split("genialJajaja"))
print(camel_case_split("rodrigoNieto"))
print(camel_case_split("laMadreQueMePario"))
