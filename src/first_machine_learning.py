# Python code for Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

X = [[1, 2, 3, 4]]
y = [0,-1,-2,-3]
z = [0, 0, 0, 0]

model.fit(X,y)
predicted = model.predict(z)
