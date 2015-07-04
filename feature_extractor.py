import pandas as pd
import numpy as np
import os

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, data_encoded):
        ## 
        data_encoded['DateOfDeparture'] = pd.to_datetime(data_encoded['DateOfDeparture'])
        
        path = os.path.dirname(__file__)
        weather = pd.read_csv( os.path.join( path , "weather_data.csv" ) )
        #weather = pd.read_csv('/mnt/datacamp/14_15DataCamp/DataLake/weather_data.txt',parse_dates=['Date'])
        weather = weather.join(pd.get_dummies(weather[u'Events'], prefix='event_'))
        weather.fillna(0,inplace=True)
        weather.drop(['Precipitationmm',u'Events'],axis=1,inplace=True)
        weather.rename(columns = {'Date':'DateOfDeparture',u'AirPort':'Arrival'},inplace=True)
        
        data_encoded = pd.merge( data_encoded, weather , on=['Arrival','DateOfDeparture'],how='left')
        
        
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Departure'], prefix='d'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Arrival'], prefix='a'))
        data_encoded = data_encoded.drop('Departure', axis=1)
        data_encoded = data_encoded.drop('Arrival', axis=1)

        
        
        data_encoded['year'] = data_encoded['DateOfDeparture'].dt.year
        data_encoded['month'] = data_encoded['DateOfDeparture'].dt.month
        data_encoded['day'] = data_encoded['DateOfDeparture'].dt.day
        data_encoded['weekday'] = data_encoded['DateOfDeparture'].dt.weekday
        data_encoded['week'] = data_encoded['DateOfDeparture'].dt.week
        data_encoded['n_days'] = data_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['year'], prefix='y'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['month'], prefix='m'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['day'], prefix='d'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['weekday'], prefix='wd'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['week'], prefix='w'))
        
        data_encoded = data_encoded.drop(['DateOfDeparture'], axis=1)
        
        X_array = np.array(data_encoded)
        print("\nX train shape:\t(%i,%i)\n" % (X_array.shape[0],X_array.shape[1]))
        
        return X_array