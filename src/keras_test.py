import numpy as np
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, activation='linear', input_dim=1))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

x_train = np.array([1, 2, 3, 4])
y_train = np.array([0,-1,-2,-3])

model.fit(x_train, y_train, epochs=10, batch_size=1)
print(model.predict(np.array([5, 6])))
