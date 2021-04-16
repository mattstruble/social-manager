from .rss_reader import RSSItem


def buzzsprout_parser(rss_dict):
    """
    Takes in an RSS FeedDict and transcribes it to the RSSItem based on the expected output from buzzsprout rss.

    :param rss_dict: Dictionary mapping of a single item from an RSS feed.
    :type rss_dict: :class:`FeedParserDict`
    :return: A dataclass containing only the relevant information.
    :rtype: :class:`RSSItem`
    """
    id = int(rss_dict["id"][11:])
    title = rss_dict["title"]
    link = rss_dict["links"][0]["href"].split(".mp3")[0]
    summary = rss_dict["summary"]

    rss_item = RSSItem(id=id, title=title, summary=summary, link=link)

    return rss_item