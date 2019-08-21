import pandas as pd
import utils
from sklearn import preprocessing
import emoji
import re
import string


def read_files(sLang, bStoreFiles=False):
    train_data = pd.DataFrame()
    dev_data = pd.DataFrame()
    test_data = pd.DataFrame()
    valid_data = pd.DataFrame()

    if bStoreFiles:
        train_data = utils.get_dataframe_from_xml(utils.parse_xml('./dataset/xml/intertass_{}_train.xml'.format(sLang)))
        dev_data = utils.get_dataframe_from_xml(utils.parse_xml('./dataset/xml/intertass_{}_dev.xml'.format(sLang)))

        train_data.to_csv('./dataset/csv/intertass_{}_train.csv'.format(sLang), encoding='utf-8', sep='\t')
        dev_data.to_csv('./dataset/csv/intertass_{}_dev.csv'.format(sLang), encoding='utf-8', sep='\t')

    else:

        train_data = pd.read_csv('./dataset/csv/intertass_{}_train.csv'.format(sLang), encoding='utf-8', sep='\t')
        dev_data = pd.read_csv('./dataset/csv/intertass_{}_dev.csv'.format(sLang), encoding='utf-8', sep='\t')

    valid_data = pd.read_csv('./dataset/csv/intertass_{}_valid.csv'.format(sLang), encoding='utf-8', sep='\t')
    test_data = pd.read_csv('./dataset/csv/intertass_{}_test.csv'.format(sLang), encoding='utf-8', sep='\t')

    encoder = preprocessing.LabelEncoder()
    valid_data['sentiment'] = encoder.fit_transform(valid_data['sentiment'])
    test_data['sentiment'] = encoder.transform(test_data['sentiment'])

    return train_data, dev_data, test_data, valid_data


def preprocess(data):
    result = []
    for tweet in data:
        clean_tweet = tweet
        clean_tweet = clean_tweet.replace('\n', '').strip()
        clean_tweet = clean_tweet.replace(u'\u2018', "'").replace(u'\u2019', "'")

        # clean_tweet = " ".join([emoji_pattern.sub(r'EMOJI', word) for word in clean_tweet.split()])  # EMOJIS
        clean_tweet = emoji.demojize(clean_tweet, use_aliases=True)
        # clean_tweet = re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), clean_tweet)  # HASHTAGS
        clean_tweet = re.sub(r"http\S+", "URL", clean_tweet)  # URL
        clean_tweet = re.sub(r"\B@\w+", 'USERNAME', clean_tweet)  # USERNAME
        clean_tweet = re.sub(r"(\w)(\1{2,})", r"\1", clean_tweet)  # LETTER REPETITION
        clean_tweet = re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'JAJAJA', clean_tweet)
        clean_tweet = re.sub(r"[a-zA-Z]*hah[a-zA-Z]*", 'JAJAJA', clean_tweet)
        clean_tweet = re.sub(r"[a-zA-Z]*jej[a-zA-Z]*", 'JAJAJA', clean_tweet)  # LAUGHTER NOT WORKING
        clean_tweet = re.sub(r"[a-zA-Z]*joj[a-zA-Z]*", 'JAJAJA', clean_tweet)
        clean_tweet = re.sub(r"[a-zA-Z]*jij[a-zA-Z]*", 'JAJAJA', clean_tweet)
        clean_tweet = re.sub(r"[a-zA-Z]*lol[a-zA-Z]*", 'JAJAJA', clean_tweet)
        # clean_tweet = re.sub(r"\d+", '', clean_tweet)  # NUMBERS
        # clean_tweet = clean_tweet.translate(str.maketrans('', '', string.punctuation + '¡'))  # PUNCTUATION
        print(string.punctuation)

        # q=que, x=por, d=de, to=todos, xd,

        clean_tweet = clean_tweet.lower()
        result.append(clean_tweet)

    return result

def print_preprocess(data):
    hashtags, urls, usernames, letReps, laughters, numbers, emojis = list(), list(), list(), list(), list(), list(), list()
    qque, xpor, dde = list(), list(), list()
    for tweet in data:
        clean_tweet = tweet

        emoji_tweet = emoji.demojize(clean_tweet, use_aliases=True)
        emojis.extend(re.findall(r":[a-z_0-9]*?:", emoji_tweet, re.IGNORECASE))

        hashtags.extend(re.findall(r"\B#\w+", clean_tweet))  # HASHTAGS
        urls.extend(re.findall(r"http\S+", clean_tweet))  # URL
        usernames.extend(re.findall(r"\B@\w+", clean_tweet))  # URL
        letReps.extend(re.findall(r"(\w)(\1{2,})", clean_tweet))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*jaj[a-zA-Z]*", clean_tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*hah[a-zA-Z]*", clean_tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*jej[a-zA-Z]*", clean_tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*joj[a-zA-Z]*", clean_tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*jij[a-zA-Z]*", clean_tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*lol[a-zA-Z]*", clean_tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*hah[a-zA-Z]*", clean_tweet, re.IGNORECASE))  # URL
        numbers.extend(re.findall(r"\d+", clean_tweet))  # URL
        qque.extend(re.findall(r"\b(q)\b", clean_tweet, re.IGNORECASE))  # q = que
        xpor.extend(re.findall(r"\b(x)\b", clean_tweet, re.IGNORECASE))  # x = por
        dde.extend(re.findall(r"\b(d)\b", clean_tweet, re.IGNORECASE)) # d = de
        # clean_tweet = clean_tweet.translate(str.maketrans('', '', string.punctuation + '¡'))  # PUNCTUATION

        # q=que, x=por, d=de, to=todos, xd,
    return hashtags, urls, usernames, letReps, laughters, numbers, emojis, xpor, qque, dde



sc = {'¡', '!', '?', '¿'}
punctuation = ''.join([c for c in string.punctuation if c not in sc])

train_data, dev_data, test_data, valid_data = read_files('es')
hashtags, urls, usernames, letReps, laughters, numbers,emojis, xpor, qque, dde = print_preprocess(train_data['content'])
print(len(hashtags))
print(len(urls))
print(len(usernames))
print(len(letReps))
print(len(laughters))
print(len(numbers))
print(emojis)
print(len(xpor))
print(len(qque))
print(len(dde))




with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(train_data['content'])
    print()
