import logging
import os
import pickle
import re
import shutil
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
import tensorflow as tf
import wget
from appdirs import site_data_dir
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tqdm import tqdm

from social_manager.mdk.preprocessing import preprocess
from social_manager.utils import setup_logger, is_docker_container

logger = logging.getLogger(__name__)
setup_logger(logger)


class __Model:
    def __init__(self, name, **kwargs):
        self.__name__ = name
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

        self.loss = kwargs["loss"] if "loss" in kwargs else "binary_crossentropy"

    def __load_pickle(self, fname):
        with open(fname, "rb") as f:
            return pickle.load(f)

    def __save_pickle(self, fname, obj):
        with open(fname, "wb") as f:
            pickle.dump(obj, f)

    def _log(self, msg, level=logging.INFO):
        logger.log(level, "[%s] %s", self.__name__, msg)

    @property
    @lru_cache(maxsize=None)
    def base_dir(self):
        if is_docker_container():
            return "/models"
        return os.path.join(site_data_dir(appname="social_manager"), "models")

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
        self._log("Downloading training data into {}".format(self.data_dir))
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
        self._log("Data download successfully completed.")

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
        self._log("Beginning training for %s." % self.__name__)
        if force and os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

        if not os.path.exists(self.processed_data_fname):
            self._log("Couldn't find processed data. Loading new data.")
            self.load_data(force)
            self._log("Beginning data preprocess...")
            data = self.preprocess()
            self._log("Data successfully completed.")
        else:
            self._log("Processed data is already up-to-date!")
            with open(self.processed_data_fname, "rb") as f:
                data = pickle.load(f)

        if not self.model_files_exist():
            self._log(
                "Missing model files from {}, retraining models. ".format(
                    self.model_dir
                )
            )
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
        else:
            self._log("Models are already up-to-date!")

        if os.path.exists(self.data_dir):
            self._log("Cleaning up {}.".format(self.data_dir))
            shutil.rmtree(self.data_dir)

        self._log("Training completed successfully.")

    def _train_w2v(self, x_train):
        self._log("Training word2vec...")
        w2v = Word2Vec(vector_size=self.n_dim, min_count=self.min_count)
        w2v.build_vocab(x_train)
        w2v.train(x_train, total_examples=w2v.corpus_count, epochs=w2v.epochs)

        self.__save_pickle(self.word2vec_fname, w2v.wv)
        self._log("Word2Vec successfully completed.")
        return w2v.wv

    def _train_tfidf(self, x_train):
        self._log("Training TF-IDF...")
        vectorizer = TfidfVectorizer(min_df=self.min_count, lowercase=False)
        _ = vectorizer.fit([" ".join(words) for words in x_train])
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

        self.__save_pickle(self.tfidf_fname, tfidf)
        self._log("TF-IDF successfully completed.")
        return tfidf

    def _train_model(self, x_train, y_train, x_test, y_test):
        self._log("Training Neural Network...")
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu", input_dim=self.n_dim),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name=self.__name__,
        )

        model.compile(optimizer="rmsprop", loss=self.loss, metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=32)

        loss, acc = model.evaluate(x_test, y_test, batch_size=128)

        with open(self.model_summary_fname, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
            f.write("loss={} | acc={}".format(loss, acc))

        model.save(self.model_fname)
        self._log(
            "Neural Network successfully completed: [acc={:.2f}, loss={:.2f}]".format(
                acc, loss
            )
        )
        return model


class SentimentModel(__Model):
    """
    A model trained using the stanford sentiment140 dataset, containing 1,600,000 tweets. More information on the data
    can be found at http://help.sentiment140.com/for-students/.

    The model's output is a prediction of a sentiment (0=positive, 1=negative).
    """

    def __init__(self):
        super().__init__("sentiment")
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


class __WikipediaDetox(__Model):
    # https://meta.wikimedia.org/wiki/Research:Detox/Data_Release
    def __init__(self, name):
        super().__init__(name)

        self.comments_filename = os.path.join(
            self.data_dir, "%s_annotated_comments.tsv" % name
        )
        self.annotations_filename = os.path.join(
            self.data_dir, "%s_annotations.tsv" % name
        )
        self.demographics_filename = os.path.join(
            self.data_dir, "%s_worker_demographics.tsv"
        )

        # self.n_dim = 300
        # self.epochs = 20

    def preprocess(self):
        # Average the toxicity across the different reviewers.
        annotations = pd.read_csv(self.annotations_filename, sep="\t")
        annotations = annotations.groupby("rev_id").mean().reset_index()

        # Merge in comments
        comments = pd.read_csv(self.comments_filename, sep="\t")
        data = pd.merge(annotations, comments, how="outer", on="rev_id")

        # clean columns
        data = data[["comment", self.__name__]]
        data = data[data.comment.isnull() == False]
        data = data[data[self.__name__].isnull() == False]

        def replace_str_tokens(text):
            return re.sub("NEWLINE_TOKEN|TAB_TOKEN", " ", text)

        def map_values_to_binary(value):
            return 0.0 if value < 0.3 else 1.0

        data["comment"] = data["comment"].map(replace_str_tokens)
        data[self.__name__] = data[self.__name__].map(map_values_to_binary)

        data[self.token_column_name] = data["comment"].progress_map(preprocess)
        data = data.rename(columns={self.__name__: self.target_column_name})
        data.reset_index(inplace=True)
        data.drop("index", axis=1, inplace=True)

        data.to_pickle(self.processed_data_fname)

        return data


class ToxicityModel(__WikipediaDetox):
    """
    A model trained using the Wikipedia Detox project dataset.

    This data set includes over 100k labeled discussion
    comments from English Wikipedia. Each comment was labeled by multiple annotators via Crowdflower on whether it is
    a toxic or healthy contribution.

    More information can be found at https://meta.wikimedia.org/wiki/Research:Detox/Data_Release.

    The model's output is a prediction of toxicity (0=neutral/healthy, 1=toxic).
    """

    def __init__(self):
        super().__init__("toxicity")
        self.data_url = "https://ndownloader.figshare.com/articles/4563973/versions/2"


class AttackModel(__WikipediaDetox):
    """
    A model trained using the Wikipedia Detox project dataset.

    This data set includes over 100k labeled discussion
    comments from English Wikipedia. Each comment was labeled by multiple annotators via Crowdflower on whether it
    contains a personal attack.

    More information can be found at https://meta.wikimedia.org/wiki/Research:Detox/Data_Release.

    The model's output is a prediction of personal attack (0=non-attack, 1=attack).
    """

    def __init__(self):
        super().__init__("attack")
        self.data_url = "https://ndownloader.figshare.com/articles/4054689/versions/6"


class AggressionModel(__WikipediaDetox):
    """
    A model trained using the Wikipedia Detox project dataset.

    This data set includes over 100k labeled discussion
    comments from English Wikipedia. Each comment was labeled by multiple annotators via Crowdflower on whether it
    has an aggressive tone.

    More information can be found at https://meta.wikimedia.org/wiki/Research:Detox/Data_Release.

    The model's output is a prediction of aggression (0=neutral/friendly, 1=aggressive).
    """

    def __init__(self):
        super().__init__("aggression")
        self.data_url = "https://ndownloader.figshare.com/articles/4267550/versions/5"
