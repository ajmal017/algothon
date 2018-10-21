#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:17:22 2018

Utils file on stocks

@author: prem
"""

import quandl

# getting stocks from quandl
df = quandl.get('WIKI/GOOGL')

# we're selecting these specific features of the stock analysis
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# the percentage the stock missed between highest and closing
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

# how much the stock changed in percent
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#print the data
print (df)

data = quandl.get()