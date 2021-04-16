import re

from ratelimiter import RateLimiter

from social_manager.config_reader import ConfigReader
from social_manager.utils import get_data_dir, get_log_dir


class BaseHandler:
    HTML_CLEANER = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")

    def __init__(self, config_path):
        self.data_dir = get_data_dir()
        self.log_dir = get_log_dir()

        self.config = ConfigReader(config_path)

        self.rate_limit_config = self.config["RateLimiter"]
        max_calls = int(self.rate_limit_config["max_calls"])
        period = int(self.rate_limit_config["period"])
        self.rate_limiter = RateLimiter(max_calls=max_calls, period=period)

    def format_message(self, message, title="", link=""):
        raise NotImplementedError

    def post(self, message):
        raise NotImplementedError

    def _remove_html_tags(self, str):
        cleaned_str = str.replace("<br />", "\n")
        cleaned_str = re.sub(r"\n+", "\n", cleaned_str).strip()
        cleaned_str = re.sub(self.HTML_CLEANER, "", cleaned_str)

        return cleaned_str
