# -*- coding: utf-8 -*-

import quandl


import pandas as pd

quandl.ApiConfig.api_key = "qbRQz4bJfYktgvzox9gx"
data = quandl.get("WIKI/AAPL", rows = 5)

data1 = quandl.get_table("SMA/INSP", paginate=True)

print(data1)
