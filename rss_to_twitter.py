from social_manager.rss_reader import RSSReader
from social_manager.twitter_handler import TwitterHandler


def tweet_items(feed_reader: RSSReader, handler):
    for item in feed_reader.auto_saving_items_generator():
        message = handler.format_message(item.summary, item.title, item.link)
        handler.post(message)


if __name__ == "__main__":
    feed_reader = RSSReader()
    twitter_handler = TwitterHandler()
    tweet_items(feed_reader, twitter_handler)
