import sys
from typing import List
from conveyor.plugin.base_plugin import BasePlugin
from conveyor.utils import getLogger
from bs4 import BeautifulSoup
import requests
import time

logging = getLogger(__name__)

# requests.packages.urllib3.util.connection.HAS_IPV6 = False


class SearchPlugin(BasePlugin):
    def __init__(self, lazy: bool = False):
        super().__init__()
        self.query = None
        self.session = None
        self.lazy = lazy
        self.time = 0

    def process_new_dat(self, data: dict):
        if not self.lazy and self.session is None:
            self.session = requests.Session()
            self.session.get("https://www.google.com/generate_204")
        try:
            query = data.get("query")
            if query is not None:
                self.query = query
            return None
        except Exception as e:
            return e

    def finish(self):
        if self.query is None:
            return None
        else:
            if self.session is None:
                self.session = requests.Session()
            start = time.perf_counter()
            result = search(self.session, self.query, "en", "us")
            end = time.perf_counter()
            self.time = end - start
            print(f"<PLUGIN_INFO> {end - start}", file=sys.stderr)
            logging.info(f"SearchPlugin: taking {end - start} seconds")
            return result


# https://stackoverflow.com/questions/73149221/programmatically-searching-google-without-api-key-in-python
def search(session: requests.Session, query: str, language: str, country: str) -> List:
    if session is None:
        session = requests.Session()
    # https://docs.python-requests.org/en/master/user/quickstart/#passing-parameters-in-urls
    params = {
        "q": query,
        "hl": language,  # language
        "gl": country,  # country of the search, UK -> United Kingdom
        "start": 0,  # number page by default up to 0
        # "num": 100          # parameter defines the maximum number of results to return.
    }

    # https://docs.python-requests.org/en/master/user/quickstart/#custom-headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }
    data = []
    logging.debug("Searching...")
    html = session.get(
        "https://www.google.com/search", params=params, headers=headers, timeout=30
    )
    logging.debug(f"Status code: {html.status_code}")
    soup = BeautifulSoup(html.text, "lxml")
    for result in soup.select(".tF2Cxc"):
        title = result.select_one(".DKV0Md").text
        try:
            snippet = result.select_one(".lEBKkf span").text
        except Exception as _:
            snippet = None
        links = result.select_one(".yuRUbf a")["href"]

        data.append({"title": title, "snippet": snippet, "links": links})
    if soup.select_one(".d6cvqb a[id=pnnext]"):
        params["start"] += 10
    return data
