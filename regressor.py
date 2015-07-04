from sklearn.base import BaseEstimator
import xgboost as xgb
 
class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = xgb.XGBRegressor(max_depth=17, n_estimators=1000, learning_rate=0.05)
 
    def fit(self, X, y):
        self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)