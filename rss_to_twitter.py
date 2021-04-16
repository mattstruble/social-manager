from social_manager.rss import RSSReader
from social_manager.rss.parsers import buzzsprout_parser
from social_manager.social import TwitterHandler


def tweet_items(feed_reader: RSSReader, handler):
    for item in feed_reader.auto_saving_items_generator():
        message = handler.format_message(item.summary, item.title, item.link)
        handler.post(message)


if __name__ == "__main__":
    feed_reader = RSSReader(buzzsprout_parser)
    twitter_handler = TwitterHandler()
    tweet_items(feed_reader, twitter_handler)
