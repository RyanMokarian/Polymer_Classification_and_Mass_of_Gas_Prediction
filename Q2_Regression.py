"""
Mass of Gas Prediction using Regression Model
Ryan Mokarian, 2023-01-17
"""

from numpy.random import seed
seed(1)

import Utility

import numpy as np
import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt
import random
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statistics
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier # The k-nearest neighbor classifier
from sklearn.feature_selection import VarianceThreshold # Feature selector
from sklearn.pipeline import Pipeline # For setting up pipeline
from sklearn.model_selection import GridSearchCV # For optimization
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from time import time
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score
import tensorflow
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import utils
from keras import optimizers

def train_and_predict_linear_regression(data, result):
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=test_size, random_state=0)
    lin_reg_model = LinearRegression().fit(X_train, y_train.reshape(-1,1))
    # print('lin_reg_model coefficients: ', lin_reg_model.coef_)
    y_pred = lin_reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"coefficient of determination (Linear, training data): {lin_reg_model.score(X_train, y_train.reshape(-1,1))}")
    print('MAE_linear_regression: ',mae) # 3.501352064491701e-05 as baseline
    print()
    pass

def train_and_predict_polynomial_regression(data, result):
    test_size = 0.2
    poly_degree = 3 # can be a hyper-parameter
    poly = PolynomialFeatures(degree = poly_degree, include_bias = False) # LinearRegression() will take care of bias (intercept)
    poly_features = poly.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, result, test_size=test_size, random_state=0)
    poly_reg_model = LinearRegression().fit(X_train, y_train.reshape(-1,1))
    # print('poly_reg_model coefficients: ', poly_reg_model.coef_) # for n= 3: x3, x2y, xy2, y3, x2, xy, y2, x, y
    y_pred = poly_reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"coefficient of determination (Polynomial, testing data): {poly_reg_model.score(X_test, y_test.reshape(-1,1))}")
    print('MAE_polynomial_regression: ',mae)
    # for n=2 is 3.2742803750848476e-05 (6.5% better).
    # For n=3 is 3.225479061992506e-05 (8% better).
    # For n=4 is 3.2159969444923665e-05 (8% better). Therefore, n=3 is selected

    # question_conditions = [[800,50],[2000, 100],[500, 200]]
    # X_question = poly.fit_transform(question_conditions)
    # y_pred = poly_reg_model.predict(X_question)
    # print('Polynomial Regression Model prediction for 3 conditions: ', y_pred)

    pass

def plot(df):
    values = df.values
    # specify columns to plot
    groups = [2, 3, 4]
    i = 1
    # plot each column
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(df.columns[group], y=0.5, loc='right')
        i += 1
    plt.savefig('timestamp_feature_plots')
    # plt.show()
    plt.close()
    pass

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
     """
     Frame a time series as a supervised learning dataset.
     Arguments:
     data: Sequence of observations as a list or NumPy array.
     n_in: Number of lag observations as input (X).
     n_out: Number of observations as output (y).
     dropnan: Boolean whether or not to drop rows with NaN values.
     Returns:
     Pandas DataFrame of series framed for supervised learning.
     """
     n_vars = 1 if type(data) is list else data.shape[1]
     df = pd.DataFrame(data)
     cols, names = list(), list()
     # input sequence (t-n, ... t-1)
     for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
     names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
     # forecast sequence (t, t+1, ... t+n)
     for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
     # put it all together
     agg = concat(cols, axis=1)
     agg.columns = names
     # drop rows with NaN values
     if dropnan:
        agg.dropna(inplace=True)
     return agg


if __name__ == "__main__":

    df = pd.read_csv('regression.csv').reset_index().rename(columns={"Unnamed: 0": "timestamp"})
    data = df[['pressure', 'temperature']]
    result = df['massflow']

    plot(df)

    # ________________________________________________________________________________________________________
    #
    # Regressor Models
    # ________________________________________________________________________________________________________

    # Regression Model (Linear and ploynomial)
    train_and_predict_linear_regression(data.values, result.values)
    train_and_predict_polynomial_regression(data.values, result.values)

    # Check other regressors
    regressors = [
        KNeighborsRegressor(),
        GradientBoostingRegressor(),
        ExtraTreesRegressor(),
        RandomForestRegressor(),
        Ridge()
    ]

    X_train, X_test, y_train, y_test = train_test_split(data.values, result.values, test_size=0.2, random_state=0)

    print()
    for model in regressors:#
        start = time()
        model.fit(X_train, y_train)
        train_time = time() - start
        start = time()
        y_pred = model.predict(X_test)
        predict_time = time() - start
        print(model)
        print("\tTraining time: %0.3fs" % train_time)
        print("\tPrediction time: %0.3fs" % predict_time)
        print("\tExplained variance:", explained_variance_score(y_test, y_pred))
        print("\tMean absolute error:", mean_absolute_error(y_test, y_pred))
        print("\tR2 score:", r2_score(y_test, y_pred))
        print()

        # predict for Gradient Boosting Regressor
    question_conditions = [[800,50],[2000, 100],[500, 200]]
    model_GB = GradientBoostingRegressor().fit(data.values, result.values)
    y_pred = model_GB.predict(question_conditions)
    print('Gradient Boosting Model prediction for 3 conditions: ', y_pred) #
    # mass flow for the 1st condition (temp=50, pres=800) should be around 0.0056, while it was predicted 0.0146

    # ________________________________________________________________________________________________________
    #
    # Long Short-Term Memory (LSTM) Model to consider long term dependency
    # ________________________________________________________________________________________________________

    df2 = df.copy().set_index('timestamp').drop(columns=['index', 'ambient_temperature'])
    values = df2.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
    print(reframed.head())

    # split into train and test sets
    values = reframed.values
    n_train_timestamp = int(0.8*len(values))
    train = values[:n_train_timestamp, :]
    test = values[n_train_timestamp:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('LSTM_loss')
    plt.title('Loss (MAE)')
    # plt.show()
    plt.close()

    # Evaluate Model
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    print("R2 score:", r2_score(inv_y, inv_yhat))

    # Predict
    # Scenario (1): massflow(t-1) = Average(massflow)
    # question_conditions = np.array([[[0.004323049,800,50]],[[0.004323049,2000,100]],[[0.004323049,500,200]]])
    # Scenario (2): massflow(t-1) = 0
    question_conditions = np.array([[[0,800,50]],[[0,2000,100]],[[0,500,200]]])
    yhat = model.predict(question_conditions)
    question_conditions = question_conditions.reshape((question_conditions.shape[0], question_conditions.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, question_conditions[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    print('LSTM prediction for 3 conditions: ',inv_yhat)