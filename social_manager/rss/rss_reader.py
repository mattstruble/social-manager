import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass

import feedparser

from social_manager.config_reader import ConfigReader
from social_manager.utils import get_data_dir, setup_logger

from types import FunctionType

logger = logging.getLogger(__name__)
setup_logger(logger)


@dataclass
class RSSItem:
    id: int
    title: str
    summary: str
    link: str


class RSSReader:
    def __init__(self, rss_parser: FunctionType):
        self.config = ConfigReader("configs/rss.cfg")

        self.feed_url = self.config["feed_url"]
        self.feed_limit = int(self.config["feed_limit"])
        self.data_file = os.path.join(get_data_dir(), "rss.json")

        self.rss_parser = rss_parser

        # Load in previous RSS etag to only collect newest feed information
        etag = ""
        if os.path.exists(self.data_file):
            logger.debug("Loading {}".format(self.data_file))
            with open(self.data_file, "r") as f:
                self.saved_data = json.load(f)
        else:
            self.saved_data = {"etag": "", "last_id": 0}

        self.feed_parser = feedparser.parse(self.feed_url)

        self.feed_items = list(map(self.rss_parser, self.feed_parser["items"]))

        self._filter_feed_items()

        self.saved_data["etag"] = self.feed_parser.etag

    def save(self):
        with open(self.data_file, "w") as f:
            logger.info("Saving {} to file {}".format(self.saved_data, self.data_file))
            json.dump(self.saved_data, f)

    def auto_saving_items_generator(self):
        try:
            for feed_item in self.feed_items[: self.feed_limit]:
                yield feed_item
                self.saved_data["last_id"] = feed_item.id
        finally:
            self.save()

    def _filter_feed_items(self):
        logger.debug("Filtering feed items...")
        self.feed_items.sort(key=lambda x: x.id)

        i = 0
        for feed_item in self.feed_items:
            if feed_item.id > int(self.saved_data["last_id"]):
                break
            i += 1

        self.feed_items = self.feed_items[i:]
        logger.debug(
            "Filtered feed items down to {} item(s).".format(len(self.feed_items))
        )
