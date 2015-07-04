import pandas as pd
import numpy as np
import os

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, data_encoded):
        
        data_encoded['DateOfDeparture'] = pd.to_datetime(data_encoded['DateOfDeparture'])
        
        path = os.path.dirname(__file__)
        weather = pd.read_csv( os.path.join( path , "weather_data.csv" ) )
        #weather = pd.read_csv('weather_data.csv',parse_dates=['Date'])
        weather.fillna(0,inplace=True)
        weather = weather.join(pd.get_dummies(weather[u' Events'], prefix='event_'))
        
        weather.drop(['Precipitationmm',u' Events'],axis=1,inplace=True)
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
        
        
        keep_cols = ['WeeksToDeparture', 'std_wtd', 'weekday', 'n_days', 'WindDirDegrees', 'week', 'd_DFW', 'day', ' Min Humidity', ' Mean Humidity', ' Min Sea Level PressurehPa', ' Max Sea Level PressurehPa', 'Max TemperatureC', 'd_ATL', 'Min TemperatureC', ' Max Gust SpeedKm/h', 'Mean TemperatureC', ' Mean Sea Level PressurehPa', 'Min DewpointC', 'Max Humidity', 'wd_5', 'Dew PointC', 'MeanDew PointC', ' Mean Wind SpeedKm/h', ' Max Wind SpeedKm/h', 'd_SFO', 'a_BOS', 'a_SFO', 'd_EWR', 'd_ORD', 'd_LAS', 'd_BOS', 'a_ATL', 'd_LAX', 'd_PHL', 'a_LGA', ' CloudCover', 'd_LGA', 'd_JFK', 'a_ORD', 'a_MCO', 'a_DFW', ' Min VisibilitykM', ' Mean VisibilityKm', 'd_DEN', 'month', 'a_LAX', 'a_JFK', 'wd_6', 'wd_1', 'w_47', 'd_MCO', 'a_EWR', 'd_DTW', 'd_PHX', 'd_SEA', 'a_SEA', 'd_MIA', 'wd_4', 'a_DEN', 'wd_0', 'wd_3', 'd_MSP', 'a_DTW', 'a_MSP', 'wd_2', 'd_IAH', 'd_CLT', 'a_LAS', 'w_1', 'a_PHL', 'w_27', 'a_MIA', 'w_52', 'w_44', 'a_IAH', 'a_PHX', 'a_CLT', 'w_21', 'year', 'w_36', 'w_7', 'w_22', 'y_2012', 'w_46', 'w_35', 'y_2011', 'd_4', 'w_8', 'd_3', 'd_23', 'w_51', 'm_11', 'y_2013', 'm_10', 'm_6', 'd_31', 'd_5', 'd_25', 'm_1', 'd_11', 'w_14', 'w_48', 'm_12', 'event__Rain-Thunderstorm', 'd_29', 'd_21', 'd_28', 'w_45', 'd_24', 'w_2', 'event__Rain', 'd_6', 'm_5', 'm_2', 'm_3', 'd_20', 'w_40', 'w_5', 'event__Fog', 'w_38', 'd_26', 'd_15', 'd_1', 'w_41', 'w_50', 'd_18', 'd_13', 'd_9', 'd_8', 'm_7', 'd_27', 'w_39', 'd_12', 'w_26', 
                     'w_37', 'd_30', 'm_8', 'd_10', 'd_16', 'd_2', 'w_3', 'd_22', 'w_10', 'd_14', 'w_13', 'w_15']
        
        print list(data_encoded.columns)
        
        X_array = np.array(data_encoded)
        print("\nX train shape:\t(%i,%i)\n" % (X_array.shape[0],X_array.shape[1]))
        
        return X_array