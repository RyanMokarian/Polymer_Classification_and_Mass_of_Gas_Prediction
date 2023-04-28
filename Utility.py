"""
Utility functions
Ryan Mokarian, 2023-01-17
"""

import Utility

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
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
import tensorflow
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils
from keras import optimizers

def classification_initial_investigation(density, label):

    df_density = pd.read_csv(density)
    print(density+' Shape :', df_density.shape) # (3547, 3)
    print('NaN values :', df_density.isna().sum()) # 0
    print('Number of unique samples: ', len(set(df_density['sample']))) # 1769

    df_label = pd.read_csv(label)
    print(label+' Shape :', df_label.shape) # (1609, 2)
    print('NaN values :', df_label.isna().sum()) #
    print('Number of unique samples: ', len(set(df_label['sample']))) # 1609

    # Identify samples with less or more than two entries
    sample_with_1entry = list((df_density.groupby('sample').size()==1).index[df_density.groupby('sample').size()==1])
    print('Samples with one entry :',sample_with_1entry)
    sample_with_3entries = list((df_density.groupby('sample').size()==3).index[df_density.groupby('sample').size()==3])
    print('Samples with three entries :',sample_with_3entries)
    sample_with_4entries = list((df_density.groupby('sample').size()==4).index[df_density.groupby('sample').size()==4])
    print('Samples with four entries :',sample_with_4entries)
    sample_with_more_than_4entries = list((df_density.groupby('sample').size()>4).index[df_density.groupby('sample').size()>4])
    print('Samples with more than four entries :',sample_with_more_than_4entries)
    return 1

def classification_scaled_train_test_dfs(df_density, df_label):
    df_train_notScaled = pd.merge(df_density, df_label, how='inner', left_on='sample', right_on='sample')
    df_train = df_train_notScaled.copy()
    col_names = ['bulk_density', 'particle_density']
    features = df_train[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df_train[col_names] = features

    df_leftJoin = pd.merge(df_density, df_label, how='left', left_on='sample', right_on='sample')
    df_test_notScaled = df_leftJoin[df_leftJoin['finalform'].isnull()].drop(columns=['finalform'])
    df_test = df_test_notScaled.copy()
    features2 = df_test[col_names]
    scaler = StandardScaler().fit(features2.values)
    features2 = scaler.transform(features2.values)
    df_test[col_names] = features2
    df_train = df_train.set_index('sample')
    df_test = df_test.set_index('sample')

    print('df_train before scaling ',df_train_notScaled.head())
    print('df_train after scaling ',df_train.head())
    print('df_test before scaling ',df_test_notScaled.head())
    print('df_test after scaling ',df_test.head())
    print('df_density ',df_density.shape)
    print('df_label ', df_label.shape)
    print('df_train', df_train.shape)
    print('df_test', df_test.shape)
    print('Final form class types :', list(set(df_train['finalform'])))
    print('Final form frequency :', df_train['finalform'].value_counts())

    return df_train, df_test


def over_sampling(X_imbalanced, y_):
    # Balance the data
    print('Frequency of forms before balancing\n', y_.value_counts())  #
    sm = SMOTE(random_state=2)
    X, y = sm.fit_resample(X_imbalanced, y_)
    print('Frequency of forms after balancing\n', y.value_counts())  #
    return X, y