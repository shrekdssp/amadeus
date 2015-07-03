# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:57:55 2015

@author: pietro
"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = GradientBoostingRegressor( n_estimators = 1500 , max_features = 20 )

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)