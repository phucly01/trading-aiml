from datetime import datetime
from dateutil.relativedelta import relativedelta
from pydoc import locate

class Downloader:

    dl = None

    def __init__(self, name, url, key):
        
        module = locate("data.{}".format(name))
        if(module != None):
            classname = getattr(module, name)
            if classname != None:
                now = datetime.today()
                tenyears = relativedelta(years=10)
                fromdate = now - tenyears
                date = now.strftime('%Y-%m-%d')
                url = url.replace('[date]', fromdate.strftime('%Y-%m-%d'), 1).replace('[date]', date)
                self.dl = classname(url, key)
            else:
                print("Class data.{}.{} is not found".format(name, name))
        else:
            print("Module data.{} is not found".format(name))

    def download(self, symbol, start_date):
        print("Downloading data for {} from {}".format(symbol, self.dl.url))
        return self.dl.download(symbol, start_date)
    
