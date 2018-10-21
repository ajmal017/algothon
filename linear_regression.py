#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:06:08 2018

@author: prem
"""
import quandl
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

data = quandl.get_table('SMA/FBD', brand_ticker='MCD')

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

def date_to_integer(date):
    return 10000*date.year + 100*data.month + data.year

x = np.array(data['date'])
x_train = np.array(x)
print(x_train)
y_train = np.array(data['new_fans'])
print(y_train)

#model.fit(x_train, y_train, epochs=150, batch_size=10)
#scores = model.evaluate(x_train, y_train)