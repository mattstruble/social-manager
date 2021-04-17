import re
from html.parser import HTMLParser

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from social_manager.models.text_replacement import (apostrophe_dict,
                                                    short_word_dict)

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

stemming = SnowballStemmer("english")
lemmatizing = WordNetLemmatizer()
html_parser = HTMLParser()
english_stopwords = set(stopwords.words("english"))


def clean_words(text):
    for key, value in apostrophe_dict.items():
        text = re.sub(key, value, text)

    for key, value in short_word_dict.items():
        text = re.sub("(\\s|^)" + key + "(\\s|$)", " %s " % value, text)

    text = re.sub("\n", " ", text)
    text = re.sub("\r", " ", text)
    text = re.sub("\t", " ", text)
    text = re.sub('"', "", text)
    text = re.sub("'", "", text)
    text = re.sub("[^A-Za-z\s]+", "", text)
    text = re.sub("[ ]+", " ", text)
    text = text.strip()
    text = re.sub("\d+", "", text)

    return text


def tokenize(text):
    text = word_tokenize(text)
    text = filter(lambda x: x not in english_stopwords, text)
    text = map(lambda x: lemmatizing.lemmatize(x, "v"), text)
    return list(text)


from social_manager.models.base_model import Model

m = Model()
cleaned = clean_words(
    'The just do it latest lgtm ttyl Tweets from Tweet (@tweet): "WhatsApp cofounder: It\'s time to delete Facebook https://t.co/q7gnbEhJkH"'
)
tokenized = tokenize(cleaned)
print(cleaned)
print(tokenized)
