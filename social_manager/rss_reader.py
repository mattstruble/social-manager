import os
from copy import deepcopy
from dataclasses import dataclass

import feedparser

from .config_reader import ConfigReader
from .utils import get_data_dir


@dataclass
class FeedItem:
    title: str
    summary: str
    link: str


class RSSReader:
    def __init__(self):
        self.config = ConfigReader("configs/rss.cfg")

        self.feed_url = self.config["feed_url"]
        self.data_file = os.path.join(get_data_dir(), "rss.dat")

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
