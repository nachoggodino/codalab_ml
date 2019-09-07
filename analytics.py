import pandas as pd
import utils
from sklearn import preprocessing
import emoji
import re
import string
from collections import Counter


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
    qque, xpor, dde, xqs, pqs = list(), list(), list(), list(), list()
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
        dde.extend(re.findall(r"\b(d)\b", clean_tweet, re.IGNORECASE))  # d = de
        xqs.extend(re.findall(r"\b(xq)\b", clean_tweet, re.IGNORECASE))  # xq = porque
        pqs.extend(re.findall(r"\b(pq)\b", clean_tweet, re.IGNORECASE))  # pq = porque
        # clean_tweet = clean_tweet.translate(str.maketrans('', '', string.punctuation + '¡'))  # PUNCTUATION


    return hashtags, urls, usernames, letReps, laughters, numbers, emojis, xpor, qque, dde, xqs, pqs



sc = {'¡', '!', '?', '¿'}
punctuation = ''.join([c for c in string.punctuation if c not in sc])

train_data, dev_data, test_data, valid_data = utils.read_files('all')
hashtags, urls, usernames, letReps, laughters, numbers,emojis, xpor, qque, dde, xqs, pqs = print_preprocess(dev_data['content'])

print("Occurrences: \n")

print(len(hashtags))
print(len(urls))
print(len(usernames))
print(len(letReps))
print(len(laughters))
print(len(numbers))
print(len(emojis))
print(len(xpor))
print(len(qque))
print(len(dde))
print(len(xqs))
print(len(pqs))

print("\n\nSingle Occurrence:\n")

print(len(list(dict.fromkeys(hashtags))))
print(len(list(dict.fromkeys(urls))))
print(len(list(dict.fromkeys(usernames))))
print(len(list(dict.fromkeys(letReps))))
print(len(list(dict.fromkeys(laughters))))
print(len(list(dict.fromkeys(numbers))))
print(len(list(dict.fromkeys(emojis))))

print("\n\nCounters: \n")

print(Counter(hashtags).most_common(5))
print(Counter(usernames).most_common(5))
print(Counter(numbers).most_common(5))
print(Counter(emojis).most_common(5))






with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(train_data['content'])
    print()
