import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def marks_prediction(marks):
    # X = pd.read_excel("C:/Users/joaka/Desktop/flask_demo/X_train.xlsx")
    # y = pd.read_excel("C:/Users/joaka/Desktop/flask_demo/y_train.xlsx")

    # X = X.values
    # y = y.values

    X = np.array([2,4,7,1,8,9,3,6])
    y = np.array([4,8,10,2,10,10,6,10])

    X  = X.reshape((-1, 1))
    y = y.reshape((-1, 1))

    model = LinearRegression()
    model.fit(X,y)

    X_test = np.array(marks)
    X_test = X_test.reshape((1,-1))

    return model.predict(X_test)[0]

