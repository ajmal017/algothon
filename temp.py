import quandl
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

quandl.ApiConfig.api_key = "yc4aJkVNasWtdExZobzx"


prices = quandl.get("WIKI/MCD")
prices.head(200)

sentiment = quandl.get_table("IFT/NSA", ticker = 'MCD')
sentiment.head(200)

#sentiment['sentiment_rating'] = ((sentiment['sentiment_high'] * (sentiment['news_buzz'] + sentiment['news_volume'])) +( sentiment['sentiment_low']) * (sentiment['news_volume'] + sentiment['news_buzz']))
#sentiment = sentiment[['sentiment_rating']]
sentiment = sentiment[['news_buzz']]
print(sentiment)
sent_dataset = sentiment.values
sent_dataset = sent_dataset.astype('float32')
print(sent_dataset)

prices = prices[['Adj. Close']]
price_dataset = prices.values
price_dataset = price_dataset.astype('float32')
print(price_dataset)
plt.plot(price_dataset)
plt.show()

price_train_size = int(len(price_dataset) * 0.67)
price_test_size = len(price_dataset) - price_train_size
price_train, price_test = price_dataset[0:price_train_size,:], price_dataset[price_train_size:len(price_dataset),:]

sent_train_size = int(len(sent_dataset) * 0.67)
sent_test_size = len(sent_dataset) - sent_train_size
sent_train, sent_test = sent_dataset[0:sent_train_size,:], sent_dataset[sent_train_size:len(sent_dataset),:]

# convert an array of values into a dataset matrix
def create_dataset(price_dataset, sent_dataset, look_back=10):
	dataX, dataY = [], []
	for i in range(len(price_dataset)-look_back-1):
		a = price_dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(price_dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_dataset(price_train, sent_train, look_back)
print(trainX)
print(trainY)
testX, testY = create_dataset(price_test, sent_test, look_back)

print(testX)
print(testY)

#Create the model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=5, batch_size=2, verbose=2)

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(price_dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(price_dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(price_dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(price_dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

