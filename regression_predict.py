#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:38:05 2018

@author: prem
"""


import quandl, math
import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
# For plotting
import datetime
import matplotlib.pyplot as plt
from matplotlib import style



df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
# X = X[:-forecast_out]

# df.dropna(inplace=True)

# y = np.array(df['label'])

try:
    pickle_in = open('linearregression.pickle', 'rb')
    clf = pickle.load(pickle_in)
except FileNotFoundError:
    print("\nlinearregression.pickle not found. Please run regression_fit.py first.")
    exit(0)

forecast_set = clf.predict(X_lately)

# print(forecast_set, confidence, forecast_out)

# Plotting data
style.use('ggplot')

df['Forecast'] = np.nan

last_date = df.iloc[-1].name

last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()