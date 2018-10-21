# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import quandl
from keras import Sequential

quandl.ApiConfig.api_key = "yc4aJkVNasWtdExZobzx"

data = quandl.get("WIKI/AAPL", rows = 10)

print(data)