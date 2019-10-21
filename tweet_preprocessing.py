import re
import string
import emoji
from num2words import num2words


# emoji_pattern = re.compile("["
#                            u"\U0001F600-\U0001F64F"  # emoticons
#                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            u"\U00002702-\U000027B0"
#                            u"\U000024C2-\U0001F251"
#                            "]+", flags=re.UNICODE)


def preprocess(data):
    result, emojis = [], list()
    for tweet in data:
        clean_tweet = tweet
        clean_tweet = clean_tweet.replace('\n', '').strip()
        clean_tweet = clean_tweet.replace(u'\u2018', "'").replace(u'\u2019', "'")

        # clean_tweet = " ".join([emoji_pattern.sub(r'EMOJI', word) for word in clean_tweet.split()])  # EMOJIS

        # clean_tweet = emoji.demojize(clean_tweet, use_aliases=True)
        clean_tweet = re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), clean_tweet)  # HASHTAGS
        clean_tweet = re.sub(r"http\S+", "URL", clean_tweet)  # URL
        # clean_tweet = re.sub(r"\B@\w+", '', clean_tweet)  # USERNAME
        # clean_tweet = re.sub(r"(\w)(\1{2,})", r"\1", clean_tweet)  # LETTER REPETITION
        # clean_tweet = re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
        # clean_tweet = re.sub(r"[a-zA-Z]*hah[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
        # clean_tweet = re.sub(r"[a-zA-Z]*jej[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)  # LAUGHTER NOT WORKING
        # clean_tweet = re.sub(r"[a-zA-Z]*joj[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
        # clean_tweet = re.sub(r"[a-zA-Z]*jij[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
        # clean_tweet = re.sub(r"[a-zA-Z]*lol[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
        # clean_tweet = re.sub(r"[a-zA-Z]*hah[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
        clean_tweet = re.sub(r"\b(x)\b", 'por', clean_tweet, re.IGNORECASE)  # x = por
        clean_tweet = re.sub(r"\b(d)\b", 'de', clean_tweet, re.IGNORECASE)  # d = de
        clean_tweet = re.sub(r"\b(q)\b", 'que', clean_tweet, re.IGNORECASE)  # q = que
        clean_tweet = re.sub(r"\b(xq)\b", 'porque', clean_tweet, re.IGNORECASE)  # xq = porque
        clean_tweet = re.sub(r"\b(pq)\b", 'porque', clean_tweet, re.IGNORECASE)  # pq = porque

        # clean_tweet = re.sub(r"\d+", lambda m: num2words(m.group(0)), clean_tweet)  # NUMBERS
        # clean_tweet = re.sub(r"\d+", 'NUMBER', clean_tweet)  # NUMBERS
        #
        sc = {'¡', '!', '?', '¿', '#', '@', '_', ':'}
        punctuation = ''.join([c for c in string.punctuation + '¡¿' if c not in sc])  # PUNCTUATION
        clean_tweet = clean_tweet.translate(str.maketrans('', '', punctuation + '¡'))

        clean_tweet = clean_tweet.lower()  # LOWERCASING
        result.append(clean_tweet)


    return result


def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])


