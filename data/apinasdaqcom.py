import quandl
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import urllib

class apinasdaqcom:
    url = None
    key = None 

    def __init__(self, url, key):
        self.url = url
        self.key = key


    def download(self, symbol, start_date):
        if (self.key != None) and self.key:
            quandl.ApiConfig.api_key = self.key
            now = datetime.today()
            end_date = now.strftime('%Y-%m-%d')
            if(start_date == None):
                tenyears = relativedelta(years=10)
                fromdate = now - tenyears
                start_date = fromdate.strftime('%Y-%m-%d')
            print("Request data for {} from nasdaq via {}".format(symbol, self.key if self.url is None else self.url))

            #return quandl.get(symbol, start_date = start_date, end_date = end_date)
            resp =  requests.get(self.url.replace('[symbol]', symbol), 
                                timeout=5,
                                headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                                         'Accept-Encoding' : 'gzip, deflate, br',
                                         'Accept-Language' : 'en-US,en;q=0.7',
                                         'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
                                         })
            if resp.ok:
                return resp.content['data']
            else:
                print("Get error {}".format(resp.reason))
                return None