import configparser
import os
import re
from copy import deepcopy
from dataclasses import dataclass

import feedparser
from twython import Twython, TwythonError

HTML_CLEANER = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


class ConfigReader:
    TWITTER = "Twitter"
    RSS = "RSS"
    LOG = "Log"
    MESSAGE = "Message"

    def __init__(self, file_path):
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(file_path)

        self.config = {}
        for section in self.config_parser.sections():
            self.config[section] = {}
            for key, val in self.config_parser.items(section):
                self.config[section][key] = deepcopy(val)

    def get_config(self, section=None):
        if section:
            return self.config[section]

        return self.config

    @property
    def twitter(self):
        return self.get_config(ConfigReader.TWITTER)

    @property
    def rss(self):
        return self.get_config(ConfigReader.RSS)

    @property
    def log(self):
        return self.get_config(ConfigReader.LOG)

    @property
    def message(self):
        return self.get_config(ConfigReader.MESSAGE)


@dataclass
class FeedItem:
    title: str
    summary: str
    link: str


class RSSReader:
    def __init__(self, rss_config):
        self.feed_url = rss_config["feed_url"]
        self.data_file = rss_config["data_file"]

        # Load in previous RSS etag to only collect newest feed information
        etag = ""
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as f:
                etag = f.readlines()[0]

        self.feed_parser = feedparser.parse(self.feed_url, etag=etag)

        self.feed_items = []
        if self.feed_parser:
            self._parse_feed_items()

        # Save etag to file
        with open(self.data_file, "w") as f:
            f.write(self.feed_parser.etag)

    def _parse_feed_items(self):
        for item in self.feed_parser["items"]:
            title = item["title"]
            link = item["links"][0]["href"]
            summary = item["summary"]

            feed_item = FeedItem(title=title, summary=summary, link=link)
            self.feed_items.append(deepcopy(feed_item))


def format_message(item: FeedItem, message_config):
    summary_split_val = message_config["summary_split_val"]
    summary_max_lines = int(message_config["summary_max_lines"])
    summary_max_len = int(message_config["summary_max_len"])
    string_format = message_config["string_format"]

    title = item.title
    link = item.link
    summary = item.summary

    # If there is a split val split the summary on it and take the first split.
    if summary_split_val:
        summary = summary.split(summary_split_val, 1)[0]

    # Clean out HTML tags, replacing <br> with \n.
    summary = summary.replace("<br />", "\n")
    summary = summary.replace("\n\n", "\n")
    summary = re.sub(HTML_CLEANER, "", summary)

    # Limit summary to the maximum number of lines.
    lines = summary.split("\n")
    summary = ""
    i = 0
    for line in lines:
        if i >= summary_max_lines:
            break

        if len(line) > 0:
            summary += line
            i += 1
            if i < summary_max_lines:
                summary += "\n\n"

    # Restrict summary to defined maximum length
    summary = (
        summary
        if len(summary) <= summary_max_len
        else summary[0 : summary_max_len - 3] + "..."
    )

    # Combine data dictionary and message format dictionaries to pass into string format
    format_dict = {"title": title, "link": link, "summary": summary}
    format_dict = {**format_dict, **message_config}

    message = string_format.format(**format_dict)
    message = message.replace("\\n", "\n")

    # Clean off opening and closing quotes
    message = message[1:-1]
    return message


def post_tweet(message: str, twitter_config):
    try:
        twitter = Twython(
            app_key=twitter_config["consumer_key"],
            app_secret=twitter_config["consumer_secret"],
            oauth_token=twitter_config["access_token"],
            oauth_token_secret=twitter_config["access_token_secret"],
        )

        twitter.update_status(status=message)
    except TwythonError as e:
        print(e)


def tweet_items(feed_reader: RSSReader, config_reader: ConfigReader):
    for item in feed_reader.feed_items:
        message = format_message(item, config_reader.message)
        post_tweet(message, config_reader.twitter)
        print(message)
        break


if __name__ == "__main__":
    config_reader = ConfigReader("bot.cfg")
    feed_reader = RSSReader(config_reader.rss)
    tweet_items(feed_reader, config_reader)
    print(config_reader.twitter)
