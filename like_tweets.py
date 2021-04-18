import numpy as np

from social_manager.mdk.models import (AggressionModel, AttackModel,
                                       SentimentModel, ToxicityModel)
from social_manager.social import TwitterHandler

sentiment = SentimentModel()
aggression = AggressionModel()
attack = AttackModel()
toxicity = ToxicityModel()

models = [sentiment, attack, aggression, toxicity]


def analyze_tweets(tweets):
    batch = [t.text for t in tweets]

    batch_size = len(batch)
    vec = np.zeros(batch_size, dtype=np.float32)
    for model in models:
        preds = model.predict(batch).reshape(batch_size)
        vec += preds

        print("\n ---- {} ----".format(model.__name__))
        for t, v in zip(tweets, preds):
            print("{:.2f}: [{}] {}".format(v, t.id, t.text))
            print("----")

    vec /= float(len(models))
    print("\n ---- AVG ----")

    for t, v in zip(tweets, vec):
        print("{:.2f}: [{}] {}".format(v, t.id, t.text))
        print("----")

    print(vec)


def analyze_mentions(twitter_handler: TwitterHandler):
    mentions = twitter_handler.get_mentions()
    analyze_tweets(mentions)


def analyze_search(twitter_handler: TwitterHandler):
    search_results = twitter_handler.get_search_results()
    analyze_tweets(search_results)


if __name__ == "__main__":
    handler = TwitterHandler()
    print("\n#### MENTIONS ####")
    # analyze_mentions(handler)
    print("\n#### SEARCH ####")
    analyze_search(handler)
