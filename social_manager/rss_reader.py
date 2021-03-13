import os
import json
from copy import deepcopy
from dataclasses import dataclass

import feedparser

from .config_reader import ConfigReader
from .utils import get_data_dir


@dataclass
class FeedItem:
    id: int
    title: str
    summary: str
    link: str


class RSSReader:
    def __init__(self):
        self.config = ConfigReader("configs/rss.cfg")

        self.feed_url = self.config["feed_url"]
        self.data_file = os.path.join(get_data_dir(), "rss.json")

        # Load in previous RSS etag to only collect newest feed information
        etag = ""
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as f:
                self.saved_data = json.load(f)
        else:
            self.saved_data = {"etag": "", "last_id": 0}

        print(self.saved_data)

        self.feed_parser = feedparser.parse(self.feed_url, etag=self.saved_data["etag"])

        self.feed_items = []
        if self.feed_parser:
            self._parse_feed_items()

        self._filter_feed_items()

        self.saved_data["etag"] = self.feed_parser.etag

    def save(self):
        with open(self.data_file, "w") as f:
            json.dump(self.saved_data, f)

    def auto_saving_items_generator(self, limit=5):
        for feed_item in self.feed_items[:limit]:
            try:
                yield feed_item
                self.saved_data["last_id"] = feed_item.id
            finally:
                self.save()

    def _filter_feed_items(self):
        self.feed_items.sort(key=lambda x: x.id)

        i = 0
        for feed_item in self.feed_items:
            print(feed_item.id)
            if feed_item.id > int(self.saved_data["last_id"]):
                break
            i+=1

        self.feed_items = self.feed_items[i:]

        print(self.feed_items)

    def _parse_feed_items(self):
        for item in self.feed_parser["items"]:
            id = int(item['id'][11:])
            title = item["title"]
            link = item["links"][0]["href"]
            summary = item["summary"]

            feed_item = FeedItem(id = id, title=title, summary=summary, link=link)
            self.feed_items.append(deepcopy(feed_item))
