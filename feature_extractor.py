import pandas as pd
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, data_encoded):
         
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Departure'], prefix='d'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Arrival'], prefix='a'))
        data_encoded = data_encoded.drop('Departure', axis=1)
        data_encoded = data_encoded.drop('Arrival', axis=1)

        
        data_encoded['DateOfDeparture'] = pd.to_datetime(data_encoded['DateOfDeparture'])
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
        
        keep_cols = ['weekday', 'wd_5', 'WeeksToDeparture', 'n_days', 'week', 'd_ORD', 'a_LGA', 
                     'a_ORD', 'a_JFK', 'd_JFK', 'a_LAX', 'd_LGA', 'd_ATL', 'd_LAX', 'day', 'w_47', 
                     'd_SFO', 'a_DFW', 'a_ATL', 'd_MIA', 'a_LAS', 'd_LAS', 'd_DTW', 'std_wtd', 
                     'a_SFO', 'a_BOS', 'a_IAH', 'a_PHX', 'a_DTW', 'w_27', 'd_DFW', 'd_IAH', 
                     'a_MCO', 'd_PHX', 'd_BOS', 'wd_1', 'd_CLT', 'a_MIA', 'a_EWR', 'd_SEA', 'a_SEA', 'wd_3', 'a_DEN',
                     'd_EWR', 'd_DEN', 'a_CLT', 'wd_6', 'w_1', 'month', 'w_44', 'w_35', 'd_MCO', 'w_52', 'd_PHL']
        data_encoded = data_encoded[keep_cols]
        
        X_array = np.array(data_encoded)
        print("\nX train shape:\t(%i,%i)\n" % (X_array.shape[0],X_array.shape[1]))
        
        return X_array