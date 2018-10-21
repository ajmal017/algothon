#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:36:42 2018

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

print(df)

dx = quandl.get_table("SMA/FBD", brand_ticker = 'GOOGL')
dx = dx[['new_fans', 'fans']]

print(dx)

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
# X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# clf = svm.SVR(kernel=k)
clf = LinearRegression(n_jobs=-1)

clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
    
print("\nModel has been trained. Please run regression_predict.py")
