from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator ):
    def __init__(self):
        self.clf = GradientBoostingRegressor( n_estimators = 1500 , max_depth = 6 , max_features = 20 , verbose = 1)
        print("N. estimators:\t%i" % (self.clf.n_estimators) )
        print("Max features:\t%i" %  (self.clf.max_features) )
        print("Max features:\t%i" % (self.clf.max_depth) ) 
    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
