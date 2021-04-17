import re
from html.parser import HTMLParser

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, word_tokenize

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

    text = re.sub("\n|\r|\t|[ ]+", " ", text)
    text = re.sub("\"|'|\d+|[^A-Za-z\s@#]+", "", text)

    return text


def __tweet_filter(text):
    return (
        len(text) > 0
        and text not in __english_stopwords
        and not text.startswith("@")
        and not text.startswith("#")
        and not text.startswith("http")
    )


def tokenize(text):
    tokens = __tokenizer.tokenize(text)  # word_tokenize(text)
    tokens = filter(__tweet_filter, tokens)
    tokens = map(lambda x: __lemmatizing.lemmatize(x, "v"), tokens)
    return list(tokens)


def preprocess(text):
    return tokenize(clean_words(text))
