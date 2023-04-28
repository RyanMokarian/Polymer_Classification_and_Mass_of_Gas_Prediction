"""
Q1. Polymer Classification
Ryan Mokarian, 2023-01-17
"""
import Utility

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import tensorflow
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils
from keras import optimizers


def knn_imbalanced_data(df_train):
    X = df_train.drop('finalform', axis=1)
    y = df_train['finalform']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)
    model_knn = KNeighborsClassifier(5).fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # X.hist()
    # plt.show()
    # plt.savefig('features_histogram')
    # plt.close()
    print('KNN - Imbalanced Data:')
    print(classification_report(y_test, y_pred))
    print()
    return model_knn

def knn_balanced_data(df_train):
    X_imbalanced = df_train.drop('finalform', axis=1)
    y_ = df_train['finalform']
    X, y = Utility.over_sampling(X_imbalanced, y_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=66) # default size = 0.25

    model_knn = KNeighborsClassifier(5).fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('KNN - Balanced Data:')
    print(classification_report(y_test, y_pred))
    print()
    return model_knn


def ann_balanced_data():
    global history
    X_imbalanced = df_train.drop('finalform', axis=1)
    y_ = df_train['finalform']
    X, y = Utility.over_sampling(X_imbalanced, y_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66) # default size = 0.25
    model = keras.Sequential()
    model.add(Dense(16, input_dim=2, activation='relu'))
    model.add(Dense(32, input_dim=16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, input_dim=32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, input_dim=16, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # One-hot encode the finalform column
    y_train_enc = pd.get_dummies(y_train)
    num_epochs = 100
    history = model.fit(X_train, y_train_enc, epochs=num_epochs, validation_split=0.2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('Neural_Network_accuracy')
    plt.close()
    epoch_nums = range(1, num_epochs + 1)
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('Neural_Network_loss')
    plt.close()
    y_pred_ = model.predict(X_test)
    y_pred = np.argmax(y_pred_, axis=-1)
    y_pred_enc = pd.get_dummies(y_pred)
    y_test_enc = pd.get_dummies(y_test)
    print('ANN accuracy_score: ', accuracy_score(y_pred_enc, y_test_enc))  # 0.76
    print(classification_report(y_test_enc, y_pred_enc))


if __name__ == "__main__":

    # Extract user and current file folder
    subdir = "NOVA_technical_assignment"
    try:
        # working directory should be the parent folder
        wd = os.path.dirname(os.path.abspath(__file__))  #
        wd = os.path.dirname(wd)  #
    except:
        wd = '.'

    #_________________________________________________________________________________________________________
    # Initial investigation and train-test data preparation
    #_________________________________________________________________________________________________________

    Utility.classification_initial_investigation('classification_density.csv', 'classification_labels.csv')

    df_density_raw = pd.read_csv('classification_density.csv').sort_values('sample')
    df_density = pd.pivot_table(data=df_density_raw, index='sample', columns='parameter', aggfunc={'value': 'mean'}).reset_index()
    df_density.columns = ['_'.join(col) for col in df_density.columns.values]
    df_density = df_density.rename(columns={'sample_':'sample', 'value_bulk density':'bulk_density', 'value_particle density':'particle_density'})
    df_label = pd.read_csv('classification_labels.csv')
    df_train, df_test = Utility.classification_scaled_train_test_dfs(df_density, df_label)

    # save df_train, df_test as csv files
    path = os.path.join(wd, subdir, 'df_train.csv')
    df_train.to_csv(path)
    path = os.path.join(wd, subdir, 'df_test.csv')
    df_test.to_csv(path)

    #_________________________________________________________________________________________________________
    #
    # Train a KNN Classifier
    # Note dataset has been scaled and then once used imbalanced data in training the KNN model and
    # another time it used balanced using SMOTE oversampling method
    #_________________________________________________________________________________________________________

    # KNN with imbalanced data
    model_knn_imbalanced_data = knn_imbalanced_data(df_train)

    # KNN with balanced data
    model_knn_balanced_data = knn_balanced_data(df_train)


    # ________________________________________________________________________________________________________
    # Train a Neural Network
    # Note dataset has been scaled and balanced by SMOTE for oversampling
    # ________________________________________________________________________________________________________

    ann_balanced_data()


    #________________________________________________________________________________________________________
    #
    # Use the KNN model trained on all balanced training data to predict physical form of 160 unlabelled samples.
    #________________________________________________________________________________________________________

    X_imbalanced = df_train.drop('finalform', axis=1)
    y_ = df_train['finalform']
    X, y = Utility.over_sampling(X_imbalanced, y_)
    model_knn = KNeighborsClassifier(5).fit(X, y)

    y_pred = model_knn.predict(df_test.values)

    df_predict = df_test.copy()
    df_predict['finalform'] = y_pred.tolist()

    # save results as a csv file
    path = os.path.join(wd, subdir, 'Samples_label_prediction.csv')
    df_predict.to_csv(path)
