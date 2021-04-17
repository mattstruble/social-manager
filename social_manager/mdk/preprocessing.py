import re
from html.parser import HTMLParser

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, TweetTokenizer

from social_manager.mdk.text_replacement import (apostrophe_dict,
                                                 short_word_dict)

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

__stemming = SnowballStemmer("english")
__lemmatizing = WordNetLemmatizer()
__html_parser = HTMLParser()
__english_stopwords = set(stopwords.words("english"))
__tokenizer = TweetTokenizer()


def clean_words(text):
    text = text.lower()

    for key, value in apostrophe_dict.items():
        text = re.sub(key, value, text)

    for key, value in short_word_dict.items():
        text = re.sub("(\\s|^)" + key + "(\\s|$)", " %s " % value, text)

    text = re.sub("\n", " ", text)
    text = re.sub("\r", " ", text)
    text = re.sub("\t", " ", text)
    text = re.sub('"', "", text)
    text = re.sub("'", "", text)
    text = re.sub("[ ]+", " ", text)
    text = text.strip()
    text = re.sub("\d+", "", text)

    return text


def tokenize(text):
    text = __tokenizer.tokenize(text) #word_tokenize(text)
    text = filter(lambda x: not x.startswith("@"), text)
    text = filter(lambda x: not x.startswith("#"), text)
    text = filter(lambda x: not x.startswith("http"), text)
    text = map(lambda x: re.sub("[^A-Za-z\s]+", "", x), text)
    text = filter(lambda x: x not in __english_stopwords, text)
    text = map(lambda x: __lemmatizing.lemmatize(x, "v"), text)
    text = filter(lambda x: len(x) > 0, text)
    return list(text)

def preprocess(text):
    clean = clean_words(text)
    tokens = tokenize(clean)
    return tokens
