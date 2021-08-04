__author__ = 'Li Bai'


"""the available data are loaded from nordhavn3_april.csv, and the explanatory variables are weather forecasts 
('temperature', 'humidity', 'DNI (Direct normal irradiance)', 'windspeed') and 
the output is heat load. Considering the time-frequency domain analysis, 'Day sin',
       'Day cos', 'Week sin', 'Week cos' are added as input features for feature selection
       
with PACF analysis, it can be seen that all the weather related variables are 3-lagged correlated. therefore, 
all the above variables are considered at time t, t-1 and t-2; However, for the heat load for day-ahead forecast, 
only the heat load before t-24 can be known ahead of time, thus only heat load at t-24 and t-25 are used as inputs

The feature selection results in this dataset are ['heat-lag-0','heat-lag-24', 'heat-lag-25', 'temperature-lag-0', 'temperature-lag-1',
 'temperature-lag-2', 'humidity-lag-2', 'windspeed-lag-2','DNI-lag-2', 'Day cos-lag-2']; such results are used for 
 artifiical intelligence methods (SK-learn package and Tensorflow packages) and online learning methods"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from helpers import add_day_week_features, data_gene, feature_selection, LAG_DICT, SHIFT_HEAT, LAG_DICT1, SHIFT_HEAT1



df1_ww = pd.read_csv ('D:\\OneDrive\\OneDrive - Danmarks Tekniske '
                   'Universitet\\energydataDTU\\venv\\data_gene'
                 '\\nornhavn3_april'
                 '.csv',sep=',', index_col=0)


#  load files
df1_ww.index=pd.to_datetime(df1_ww.index)
df1_ww['windspeed']=np.sqrt(df1_ww['windx'].to_numpy()**2+df1_ww[
    'windy'].to_numpy()**2)

df_ww_copy = df1_ww.copy()

df_ww_copy=pd.DataFrame(columns=['heat', 'temperature', 'humidity',
                                 'DNI','windspeed'], index=df1_ww.index)
df_ww_copy['heat']=df1_ww['Counter [MWh]']
df_ww_copy['temperature']=df1_ww['temperature']
df_ww_copy['DNI']=df1_ww['solarflux']
df_ww_copy['windspeed']=df1_ww['windspeed']
df_ww_copy['humidity']=df1_ww['humidity']



# plot PACF or ACF and plot FFT spectrum
# plot_acf_or_pacf(df_ww_copy)
# fft_analy(df_ww_copy)


# # heat load comes from space heating!
# fall = df_ww_copy[(df_ww_copy.index >= '2018-1-21 00:00:00')
#      & (df_ww_copy.index < '2020-07-05 00:00:00')]

df=add_day_week_features(df_ww_copy)
df1_new=data_gene(LAG_DICT, SHIFT_HEAT, df)

index_start=24-df1_new.index[0].hour
index_end=1+df1_new.index[-1].hour
df1_new=df1_new.iloc[index_start:-index_end,:]

df1_new_copy=df1_new.copy()
# '2018-01-21 00:00:00' ~ '2020-07-05 23:00:00'


# select the heating season data
start0=datetime.datetime(2018,1,22,0,0,0);
end0=datetime.datetime(2018,5,31,23,0,0);
start1=datetime.datetime(2018,9,24,0,0,0);
end1=datetime.datetime(2019,5,31,23,0,0);
start2=datetime.datetime(2019,9,24,0,0,0);
end2=datetime.datetime(2020,5,31,23,0,0);

date_gene0 = pd.date_range(start=start0, end=end0, freq='H').tolist()
date_gene1 = pd.date_range(start=start1, end=end1, freq='H').tolist()
date_gene2 = pd.date_range(start=start2, end=end2, freq='H').tolist()

dates = date_gene0 + date_gene1 + date_gene2


# 3:1 for train and test
df1_new=df1_new.loc[dates,:]





N_total = len(df1_new)
N_train=int(int(N_total*0.75/24)*24);
train_df=df1_new[0:N_train]


#  normalization!!!
train_df_copy=train_df.copy()
train_df_mean=train_df_copy.mean()
train_df_std=train_df_copy.std()
train_df_copy=(train_df_copy-train_df_mean)/train_df_std

y_train=train_df_copy['heat-lag-0']
train_df_copy.pop('heat-lag-0')
X_train=train_df_copy.copy()



feature_set_lasso, feature_set_xtree, feature_set_info=feature_selection(df1_new, X_train, y_train, alpha=0.05,
                                                                         n_estimators=20)

# ========================feature selection results========================
columns=['heat-lag-0','heat-lag-24', 'heat-lag-25', 'temperature-lag-0', 'temperature-lag-1',
 'temperature-lag-2', 'humidity-lag-2', 'windspeed-lag-2','DNI-lag-2', 'Day cos-lag-2']


# ===========the selected columns are saved to "nordhavn_terminal3_selected.csv"================
# df1_out=df1_new[columns]
# df1_out.to_csv("nordhavn_terminal3_selected.csv")
#




