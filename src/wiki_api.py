import requests
import urllib.parse
import re
import threading
class MetaAPI:
    def __init__(self, api_key, api_name: str, base_url: str, proxy=None):
        self.proxy = proxy
        self.session = requests.Session()
        if self.proxy:
            proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            self.session.proxies = proxies
        self.api_key = api_key
        self.api_name = api_name
        self.base_url = base_url
        self.lock = threading.Lock()
class WikiSearchAPI(MetaAPI):
    def __init__(self, proxy=None):
        api_name = 'Wiki Search'
        base_url = 'https://en.wikipedia.org/w/api.php?'
        super(WikiSearchAPI, self).__init__(api_name=api_name, base_url=base_url, api_key=None, proxy=proxy)
    def call(self, query, num_results=5):
        def remove_html_tags(text):
            clean = re.compile('<.*?>')
            return re.sub(clean, '', text)
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
        }
        call_url = self.base_url + urllib.parse.urlencode(params)
        r = self.session.get(call_url)
        data = r.json()['query']['search']
        data = [d['title'] + ": " + remove_html_tags(d["snippet"]) for d in data][:num_results]
        return data
    
# api = WikiSearchAPI()
# s = api.call('Jin Li is the president of Fudan university')
# print(s)
