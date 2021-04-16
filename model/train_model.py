# https://www.ahmedbesbes.com/blog/sentiment-analysis-with-keras-and-word-2-vec
import os
import zipfile

import numpy as np
import pandas as pd
import wget
from gensim.models.doc2vec import TaggedDocument
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

import pickle

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

sentiment_dataset_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
data_dir = "data"


def get_training_file():
    return [
        os.path.join(data_dir, x)
        for x in os.listdir(data_dir)
        if "training" in x and x.endswith(".csv")
    ][0]


def load_data(url):
    if os.path.exists(data_dir):
        return get_training_file()

    os.mkdir(data_dir)

    wget.download(url, out=data_dir)

    zips = [
        os.path.join(data_dir, x) for x in os.listdir(data_dir) if x.endswith(".zip")
    ]

    for zip in zips:
        zip_ref = zipfile.ZipFile(zip, "r")
        zip_ref.extractall(data_dir)

    return get_training_file()


def ingest(training_file):
    data = pd.read_csv(
        training_file, names=["sentiment", "id", "date", "query", "user", "text"]
    )
    data.drop(["id", "date", "query", "user"], axis=1, inplace=True)
    data = data[data.sentiment.isnull() == False]
    data["sentiment"] = data["sentiment"].map(lambda x: int(min(1, x)))
    data = data[data.text.isnull() == False]
    data.reset_index(inplace=True)
    data.drop("index", axis=1, inplace=True)

    print(data.head(4))
    print(data.tail(4))
    print(data.shape)
    print(data["sentiment"].unique())

    return data


def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda x: not x.startswith("@"), tokens)
        tokens = filter(lambda x: not x.startswith("#"), tokens)
        tokens = filter(lambda x: not x.startswith("http"), tokens)
        return list(tokens)
    except Exception as e:
        print(e)
        return "NC"


def preprocess(data):
    print("Preprocessing data...")
    data["tokens"] = data["text"].progress_map(tokenize)
    data = data[data.tokens != "NC"]
    data.reset_index(inplace=True)
    data.drop("index", axis=1, inplace=True)
    return data


def tag_tweets(tweets, tag_name):
    tagged_tweets = []
    for i, v in tqdm(enumerate(tweets), desc="tagging tokens"):
        tag = "{}_{}".format(tag_name, i)
        tagged_tweets.append(TaggedDocument(v, [tag]))

    return tagged_tweets


def train_w2v(x_train, vector_size):
    w2v = Word2Vec(vector_size=vector_size, min_count=10)
    w2v.build_vocab(x_train)
    w2v.train(
        x_train,
        total_examples=w2v.corpus_count,
        epochs=w2v.epochs,
    )

    return w2v.wv


def train_tfidf(x_train):
    vectorizer = TfidfVectorizer(min_df=10, lowercase=False)
    _ = vectorizer.fit([" ".join(words) for words in x_train])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    return tfidf


def weighted_vector(tokens, n_dim, w2v, tfidf):
    vec = np.zeros((1, n_dim))
    count = 0.0

    for word in tokens:
        if word in w2v and word in tfidf:
            vec += w2v[word].reshape((1, n_dim)) * tfidf[word]
            count += 1.0

    if count != 0:
        vec /= count

    return vec


if __name__ == "__main__":
    n_dim = 200

    if not os.path.exists("preprocessed_data.plk"):
        training_file = load_data(sentiment_dataset_url)
        data = ingest(training_file)
        data = preprocess(data)

        data.to_pickle("preprocessed_data.plk")
    else:
        data = pd.read_pickle("preprocessed_data.plk")

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(data.tokens), np.array(data.sentiment), test_size=0.2
    )

    if not os.path.exists("word2vec.pkl") or not os.path.exists("tfidf.pkl"):
        w2v = train_w2v(x_train, n_dim)
        tfidf = train_tfidf(x_train)

        with open("word2vec.pkl", "wb") as f:
            pickle.dump(w2v, f)

        with open("tfidf.pkl", "wb") as f:
            pickle.dump(tfidf, f)
    else:
        with open("word2vec.pkl", "rb") as f:
            w2v = pickle.load(f)
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)

    train_vecs = scale(
        np.concatenate(
            [weighted_vector(tokens, n_dim, w2v, tfidf) for tokens in x_train]
        )
    )
    test_vecs = scale(
        np.concatenate(
            [weighted_vector(tokens, n_dim, w2v, tfidf) for tokens in x_test]
        )
    )

    print(train_vecs.shape, test_vecs.shape)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(32, activation="relu", input_dim=n_dim),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_vecs, y_train, epochs=10, batch_size=32)

    score = model.evaluate(test_vecs, y_test, batch_size=128)
    print(score)

    model.save("sentiment-model.h5")
