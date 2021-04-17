import os
import pickle
import shutil
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
import tensorflow as tf
import wget
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tqdm import tqdm

from social_manager.mdk.preprocessing import preprocess


class __Model:
    def __init__(self):
        self.data_url = None
        tqdm.pandas(desc=self.__name__)

        if self.model_files_exist():
            self.w2v = self.__load_pickle(self.word2vec_fname)
            self.tfidf = self.__load_pickle(self.tfidf_fname)
            self.model = tf.keras.models.load_model(self.model_fname)
        else:
            self.w2v = None
            self.tfidf = None
            self.model = None

        self.n_dim = 200
        self.min_count = 10
        self.epochs = 10

    def __load_pickle(self, fname):
        with open(fname, "rb") as f:
            return pickle.load(f)

    def __save_pickle(self, fname, obj):
        with open(fname, "wb") as f:
            pickle.dump(obj, f)

    @property
    def base_dir(self):
        return "model"

    @property
    def token_column_name(self):
        return "tokens"

    @property
    def target_column_name(self):
        return "target"

    @property
    @lru_cache(maxsize=None)
    def processed_data_fname(self):
        return os.path.join(self.model_dir, "processed_data.plk")

    @property
    @lru_cache(maxsize=None)
    def word2vec_fname(self):
        return os.path.join(self.model_dir, "word2vec.plk")

    @property
    @lru_cache(maxsize=None)
    def tfidf_fname(self):
        return os.path.join(self.model_dir, "tfidf.pkl")

    @property
    @lru_cache(maxsize=None)
    def model_fname(self):
        return os.path.join(self.model_dir, "{}_model.h5".format(self.__name__))

    @property
    @lru_cache(maxsize=None)
    def model_summary_fname(self):
        return os.path.join(self.model_dir, "model_summary.txt")

    @property
    @lru_cache(maxsize=None)
    def model_dir(self):
        return os.path.join(self.base_dir, self.__name__)

    @property
    @lru_cache(maxsize=None)
    def data_dir(self):
        return os.path.join(self.model_dir, "data")

    def get_weighted_vector(self, tokens):
        weighted_vec = np.zeros((1, self.n_dim))
        count = 0.0

        for word in tokens:
            if word in self.w2v and word in self.tfidf:
                weighted_vec += (
                    self.w2v[word].reshape((1, self.n_dim)) * self.tfidf[word]
                )
                count += 1.0

        if count != 0.0:
            weighted_vec /= count

        return weighted_vec

    def load_data(self, force=False):
        if force and os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

            wget.download(self.data_url, out=self.data_dir)

            zips = [
                os.path.join(self.data_dir, x)
                for x in os.listdir(self.data_dir)
                if x.endswith(".zip")
            ]

            for zip in zips:
                zip_ref = zipfile.ZipFile(zip, "r")
                zip_ref.extractall(self.data_dir)

    def model_files_exist(self):
        return (
            os.path.exists(self.word2vec_fname)
            and os.path.exists(self.tfidf_fname)
            and os.path.exists(self.model_fname)
        )

    def predict(self, batch):
        batch = map(preprocess, batch)
        batch = self.preprocess_model(batch)

        return self.model.predict(batch)

    def preprocess(self):
        raise NotImplementedError

    def preprocess_model(self, batch):
        vecs = scale(
            np.concatenate([self.get_weighted_vector(tokens) for tokens in batch])
        )

        return vecs

    def train(self, force=False):
        if force and os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

        if not os.path.exists(self.processed_data_fname):
            self.load_data(force)
            data = self.preprocess()
        else:
            with open(self.processed_data_fname, "rb") as f:
                data = pickle.load(f)

        if not self.model_files_exist():
            x_train, x_test, y_train, y_test = train_test_split(
                np.array(data[self.token_column_name]),
                np.array(data[self.target_column_name]),
                test_size=0.2,
            )

            self.w2v = self._train_w2v(x_train)
            self.tfidf = self._train_tfidf(x_train)

            train_vecs = self.preprocess_model(x_train)
            test_vecs = self.preprocess_model(x_test)

            self.model = self._train_model(train_vecs, y_train, test_vecs, y_test)

    def _train_w2v(self, x_train):
        w2v = Word2Vec(vector_size=self.n_dim, min_count=self.min_count)
        w2v.build_vocab(x_train)
        w2v.train(x_train, total_examples=w2v.corpus_count, epochs=w2v.epochs)

        self.__save_pickle(self.word2vec_fname, w2v.wv)
        return w2v.wv

    def _train_tfidf(self, x_train):
        vectorizer = TfidfVectorizer(min_df=self.min_count, lowercase=False)
        _ = vectorizer.fit([" ".join(words) for words in x_train])
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

        self.__save_pickle(self.tfidf_fname, tfidf)

        return tfidf

    def _train_model(self, x_train, y_train, x_test, y_test):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu", input_dim=self.n_dim),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=32)

        loss, acc = model.evaluate(x_test, y_test, batch_size=128)

        with open(self.model_summary_fname, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
            f.write("loss={} | acc={}".format(loss, acc))

        model.save(self.model_fname)

        return model


class SentimentModel(__Model):
    def __init__(self):
        self.__name__ = "sentiment"
        super().__init__()
        self.data_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        self.data_filename = os.path.join(
            self.data_dir, "training.1600000.processed.noemoticon.csv"
        )

        self.n_dim = 200
        self.min_count = 10

    def preprocess(self):
        data = pd.read_csv(
            self.data_filename,
            names=["sentiment", "id", "date", "query", "user", "text"],
        )
        data.drop(["id", "date", "query", "user"], axis=1, inplace=True)
        data = data[data.sentiment.isnull() == False]

        # swap sentiment so that 0 is positive and 1 is negative to match other models.
        data["sentiment"] = data["sentiment"].map(lambda x: 1 - int(min(1, x)))
        data = data[data.text.isnull() == False]

        data[self.token_column_name] = data["text"].progress_map(preprocess)
        data = data.rename(columns={"sentiment": self.target_column_name})
        data.reset_index(inplace=True)
        data.drop("index", axis=1, inplace=True)

        data.to_pickle(self.processed_data_fname)

        return data
