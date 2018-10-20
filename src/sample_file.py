import quandl
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, activation='linear', input_dim=1))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])



API_KEY = "3ppcK5PSDfRn7osHWktG"

quandl.ApiConfig.api_key = '3ppcK5PSDfRn7osHWktG'
data = quandl.get_table('SMA/FBD' , brand_ticker='MCD')

#quandl.ApiConfig.api_key = API_KEY

#data = quandl.get_table("SMA1/SMA-FBD")

def date_to_int(date):
    result = 0
    year, month, day = date.year, date.month, date.day
    #print(year, month, day)

    def is_leap_year(y):
        if y % 4:
            return False
        elif y % 100:
            return True
        elif y % 400:
            return False
        else:
            return True

    def extra_day(y):
        if is_leap_year(y):
            return 1
        else:
            return 0

    for i in range(1, year+1):
        result += 365 + extra_day(i)

    def days_in_month(m):
        if m in {1, 3, 5, 7, 8, 10, 12}:
            return 31
        elif m == 2:
            return 28
        else:
            return 30

    for j in range(1, month+1):
        result += days_in_month(j)

    result += day

    return result

x_train = np.array(list(map(date_to_int, data['date'])))
y_train = np.array(data['fans'])

print(x_train.shape, y_train.shape)

model.fit(x_train, y_train, epochs=5, batch_size=1)
print(model.predict(np.array([date_to_int(pd.Timestamp(2014, 10, 20)), date_to_int(pd.Timestamp(2014, 10, 21))])))
