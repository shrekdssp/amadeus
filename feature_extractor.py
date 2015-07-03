# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:58:43 2015

@author: pietro
"""
import pandas as pd
class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        
        #uncomment the line below in the submission
        #path = os.path.dirname(__file__)
        data_weather = pd.read_csv("data_weather.csv")
        
        X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC']]
        X_weather = X_weather.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
        X_encoded = X_encoded.set_index(['DateOfDeparture', 'Arrival'])
        X_weather = X_weather.set_index(['DateOfDeparture', 'Arrival'])
        X_encoded = X_encoded.join(X_weather).reset_index()
        
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_array = X_encoded.values
        return X_array