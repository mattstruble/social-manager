import os
import logging

from dataclasses import dataclass

from twython import Twython

from social_manager.utils import setup_logger, CONFIG_DIR

from .base_handler import BaseHandler

logger = logging.getLogger(__name__)
setup_logger(logger)


@dataclass
class Tweet:
    id: int
    created_at: str
    screen_name: str
    text: str
    favorited: bool
    retweeted: bool

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id


class TwitterHandler(BaseHandler):
    def __init__(self):
        super().__init__(os.path.join(CONFIG_DIR, "twitter.cfg"))

        self.oauth_config = self.config["OAuth"]
        self.message_config = self.config["Message"]
        self.search_config = self.config["Search"]

        self.twitter = Twython(
            app_key=self.oauth_config["consumer_key"],
            app_secret=self.oauth_config["consumer_secret"],
            oauth_token=self.oauth_config["access_token"],
            oauth_token_secret=self.oauth_config["access_token_secret"],
        )

        # todo: save screen_name to data file
        timeline = self.twitter.get_home_timeline(count=1)
        self.screen_name = timeline[0]["user"]["screen_name"]

    def format_message(self, message, title="", link=""):
        logger.debug(
            "Formatting message={}, title={}, link={}".format(message, title, link)
        )

        summary_split_val = self.message_config["summary_split_val"]
        summary_max_lines = int(self.message_config["summary_max_lines"])
        summary_max_len = int(self.message_config["summary_max_len"])
        string_format = self.message_config["string_format"]

        summary = message

        # If there is a split val split the summary on it and take the first split.
        if summary_split_val:
            summary = message.split(summary_split_val, 1)[0]

        # Clean out HTML tags, replacing <br> with \n.
        summary = self._remove_html_tags(summary)

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
        format_dict = {**format_dict, **self.message_config}

        message = string_format.format(**format_dict)
        message = message.replace("\\n", "\n")

        # Clean off opening and closing quotes
        message = message[1:-1]
        return message

    def post(self, message):
        logger.debug("Posting message={}".format(message))

        with self.rate_limiter:
            try:
                self.twitter.update_status(status=message)
            except Exception as e:
                logger.exception(e)
                raise e

    def _search_result_generator(self, search_results, exclude_self=True):
        statuses = (
            search_results["statuses"]
            if "statuses" in search_results
            else search_results
        )
        for result in statuses:
            tweet = Tweet(
                result["id"],
                result["created_at"],
                result["user"]["screen_name"],
                result["full_text"],
                result["favorited"],
                result["retweeted"],
            )
            if exclude_self and tweet.screen_name == self.screen_name:
                continue

            yield tweet

    def get_search_results(self, exclude_self=True):
        logger.debug("get_search_results exclude_self={}".format(exclude_self))

        search_results = set()
        for hashtag in self.search_config["hashtags"].split(","):
            with self.rate_limiter:
                try:
                    search = self.twitter.search(
                        q="#{}".format(hashtag),
                        lang="en",
                        since_id=0,
                        tweet_mode="extended",
                    )
                except Exception as e:
                    logger.exception(e)
                    raise e

            for tweet in self._search_result_generator(search, exclude_self):
                search_results.add(tweet)

        return search_results

    def get_mentions(self, **params):
        logger.debug("get_mentions params={}".format(params))
        mention_results = set()

        with self.rate_limiter:
            try:
                mentions = self.twitter.get_mentions_timeline(
                    tweet_mode="extended", lang="en", **params
                )
            except Exception as e:
                logger.exception(e)
                raise e

        for tweet in self._search_result_generator(mentions):
            mention_results.add(tweet)

        return mention_results
