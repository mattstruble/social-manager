from twython import Twython, TwythonError
from .base_handler import BaseHandler

class TwitterHandler(BaseHandler):

    def __init__(self):
        super().__init__("configs/twitter.cfg")

        self.oauth_config = self.config["OAuth"]
        self.message_config = self.config["Message"]

        self.twitter = Twython(
            app_key=self.oauth_config["consumer_key"],
            app_secret=self.oauth_config["consumer_secret"],
            oauth_token=self.oauth_config["access_token"],
            oauth_token_secret=self.oauth_config["access_token_secret"],
        )

    def format_message(self, message, title="", link=""):
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
            else summary[0: summary_max_len - 3] + "..."
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
        try:
            self.twitter.update_status(status=message)
        except TwythonError as e:
            print(e)
