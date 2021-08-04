__author__ = 'Li Bai'
"""based on one year and a half bornholm data, the heating season data is selected out to feed into the data directly
as one whole time series considering OLS, RLS and transformed versions"""

"""We include OLS, RLS KRLS, Logit-Normal RLS, Jeo-RLS and these transforms 
with RML recursive maximum likelihood"""

"""The whole data will be segmented into 3 parts：the first 75% used for 
training, and the rest 25% is used as test dataset!  Among the first 75%, 
it will be further split into 7:3 as training and validation dataset; (or 
cross-validation) is considered for finding the hyperparameters..
In a nutshell, the data will be split into 52.5% as training, 22.5% as valid; 
25% as final test dataset"""

"""HEADS-UP FOR THE DATA: the whole data starts from the Jan 2018 to 
the end of May 2020 before summer came; only the data at heating season are considered. Heating season is referred to 
the months 24, Sep to the  which assumes that heat load is 
mainly caused by heating space; while in summer, it is random, depending on the 
hot-tap water usage """


import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize, Bounds
from scipy.optimize import root
import statsmodels.tsa.seasonal
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.tsa.api import acf, pacf, graphics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.colors as mcolors
import matplotlib as mpl
from helpers_online import OLS_Yeo, OLS_LG, RLS, RLS_Yeo, RLS_LG, RML, RML_Yeo, RML_LG



mpl.rc('xtick', labelsize=20)
mpl.rc('ytick', labelsize=20)
plt.rcParams.update({'font.size': 24})

mpl.rcParams['figure.figsize'] = (25,10)
mpl.rcParams['axes.grid'] = False

df= pd.read_csv ("nordhavn_terminal3_selected.csv",
                          sep=',', index_col=0)
df.index=pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

print(df.keys())
keys=df.keys()


# add dummy or not
# df_out=add_dummy_hour(df_out)
# ==============data to be and paritioned as train, val, test (7:2:1) and  standarized==================


df1=df.copy()

N_total = len(df1)
N1=int(int(N_total*0.75/24)*24);
df1=df[0:N1]

# df1 = df1.sample(frac=1).reset_index(drop=True)
N_train=int(int(0.7*len(df1)/24)*24)


df_features_un=df.iloc[:,1:].copy()
df_output_un=df.iloc[:,0:1].copy()


df_features_train_un=df_features_un.iloc[0:N_train,:]
df_features_test_un=df_features_un.iloc[N_train:,:]

df_output_train_un=df_output_un.iloc[0:N_train]
df_output_test_un=df_output_un.iloc[N_train:]



train_df=df1.iloc[0:N_train,:]
val_df = df1.iloc[N_train:,:]
test_df = df.iloc[N1:,:]


train_df_max = train_df.max()
train_df_min = train_df.min()

HEAT_MAX_REAL=train_df_max[['heat-lag-0','heat-lag-24','heat-lag-25']].max()
HEAT_MIN_REAL=train_df_min[['heat-lag-0','heat-lag-24','heat-lag-25']].min()

EPSILON_HEAT=0.02

HEAT_MAX_APPRO=(HEAT_MAX_REAL-HEAT_MIN_REAL)*(EPSILON_HEAT)+HEAT_MAX_REAL
HEAT_MIN_APPRO=HEAT_MIN_REAL-(HEAT_MAX_REAL-HEAT_MIN_REAL)*(EPSILON_HEAT)

train_df_max['heat-lag-0']=(HEAT_MAX_REAL-HEAT_MIN_REAL)*(EPSILON_HEAT)+HEAT_MAX_REAL
train_df_max['heat-lag-24']=(HEAT_MAX_REAL-HEAT_MIN_REAL)*(EPSILON_HEAT)+HEAT_MAX_REAL
train_df_max['heat-lag-25']=(HEAT_MAX_REAL-HEAT_MIN_REAL)*(EPSILON_HEAT)+HEAT_MAX_REAL

train_df_min['heat-lag-0']=HEAT_MIN_REAL-(HEAT_MAX_REAL-HEAT_MIN_REAL)*(EPSILON_HEAT)
train_df_min['heat-lag-24']=HEAT_MIN_REAL-(HEAT_MAX_REAL-HEAT_MIN_REAL)*(EPSILON_HEAT)
train_df_min['heat-lag-25']=HEAT_MIN_REAL-(HEAT_MAX_REAL-HEAT_MIN_REAL)*(EPSILON_HEAT)

# if the real values in the end are over the HEAT_MAX_REAL, then they will be limited to HEAT_MAX_REAL; if lower than
# HEAT_MIN_REAL, limited to HEAT_MIN_REAL

df_copy=df.copy()
# min_max normalization make everything limited to (0,1)
df_copy_norm=(df_copy-train_df_min)/(train_df_max-train_df_min)
df_features=df_copy_norm.iloc[:,1:]
df_output=df_copy_norm.iloc[:,0:1]

df_features['const']=np.ones(df_features.shape[0])


NUM_FEATURES=df_features.shape[1]

df_output_max=train_df_max.iloc[0]
df_output_min=train_df_min.iloc[0]

df_features_max=train_df_max.iloc[1:].to_numpy()
df_features_min=train_df_min.iloc[1:].to_numpy()

df_features_train=df_features.iloc[0:N_train,:]
df_features_test=df_features.iloc[N_train:,:]

df_output_train=df_output.iloc[0:N_train,:]
df_output_test=df_output.iloc[N_train:,:]






#
# y_test, y_train, sigma2s = OLS_Yeo(nu, df_features, df_output, N_train)
# y_test, y_train, sigma2s = OLS_LG(nu, df_features, df_output, N_train)
#
# y_test, y_train, _, sigma2s = RLS(lam_fgt, df_features, df_output, N_train)
# y_test, y_train, _, sigma2s = RLS_Yeo(lam_fgt, nu, df_features, df_output, N_train)
# y_test, y_train, _, sigma2s = RLS_LG(lam_fgt, nu, df_features, df_output, N_train)
#
# y_test, y_train, _, sigma2s = RML(lam_fgt, df_features, df_output, N_train, NUM=2000)
# y_test, y_train, _, sigma2s = RML_Yeo(lam_fgt, nu, df_features, df_output, N_train, NUM=2000)
# y_test, y_train, _, sigma2s = RML_LG(lam_fgt, nu, df_features, df_output, N_train, NUM=2000)
#
# df_final_test['Method' + str(s)] = y_test
# df_final_train['Method' + str(s)] = y_train
# sigma2ss.append(np.array(sigma2s))
#
# df_output_max['Method' + str(s)] = df_output_max['heat-lag-0']
# df_output_min['Method' + str(s)] = df_output_min['heat-lag-0']




"""CRPS for ad single point and CRPS for average; for time lead"""

def hyper_para_val(df_features, df_output, method):
    df_output_train= df_output.iloc[0:N_train,:]
    df_output_test=df_output.iloc[N_train:,:]

    df_final_train = df_output_train.copy()
    df_final_test = df_output_test.copy()

    df_output_max=train_df_max[['heat-lag-0']]
    df_output_min=train_df_min[['heat-lag-0']]

    df_origin_test = (df_final_test) * (
                df_output_max - df_output_min)+df_output_min

    df_origin_train = (df_final_train ) * (
                df_output_max - df_output_min)+df_output_min




    sigma2ss=[]
    nus=[]

    lam_fgts = np.arange(0.95, 1, 0.001)
    nu_trans = np.arange(0.1, 3, 0.1)  #

    for j in range(len(nu_trans)):
        nu=nu_trans[j]
        for k in range(len(lam_fgts)):
            nus.append(nu)
            s=j*len(lam_fgts)+k; print(s)
            lam_fgt = lam_fgts[k]
            # if method == 'OLS_Yeo':
            #     y_test, y_train, sigma2s = OLS_Yeo(nu, df_features, df_output, N_train)
            # if method=="OLS_LG":
            #     y_test, y_train, sigma2s = OLS_LG(nu, df_features, df_output, N_train)
            # if method=="RLS":
            #     y_test, y_train, _, sigma2s = RLS(lam_fgt, df_features, df_output, N_train)
            # if method=="RLS_Yeo":
            #     y_test, y_train, _, sigma2s = RLS_Yeo(lam_fgt, nu, df_features, df_output, N_train)
            # if method=="RLS_LG":
            #     y_test, y_train, _, sigma2s = RLS_LG(lam_fgt, nu, df_features, df_output, N_train)
            # if method=="RML":
            #     y_test, y_train, _, sigma2s = RML(lam_fgt, df_features,df_output, N_train,NUM=2000)
            # if method=="RML_Yeo":
            #     y_test, y_train, _, sigma2s = RML_Yeo(lam_fgt, nu,df_features,df_output, N_train,NUM=2000)
            # if method=="RML_LG":
            #     y_test, y_train, _, sigma2s = RML_LG(lam_fgt, nu, df_features, df_output, N_train, NUM=2000)

            if method == 'OLS_Yeo':
                y_test, y_train, sigma2s = OLS_Yeo(df_features, df_output, N_train, nu=nu)
            if method == "OLS_LG":
                y_test, y_train, sigma2s = OLS_LG(df_features, df_output, N_train, nu=nu)
            if method == "RLS":
                y_test, y_train, _, sigma2s = RLS(df_features, df_output, N_train, lam_fgt=lam_fgt)
            if method == "RLS_Yeo":
                y_test, y_train, _, sigma2s = RLS_Yeo(df_features, df_output, N_train, lam_fgt=lam_fgt, nu=nu)
            if method == "RLS_LG":
                y_test, y_train, _, sigma2s = RLS_LG(df_features, df_output, N_train, lam_fgt=lam_fgt, nu=nu)
            if method == "RML":
                y_test, y_train, _, sigma2s = RML(df_features, df_output, N_train, NUM=2000, lam_fgt=lam_fgt)
            if method == "RML_Yeo":
                y_test, y_train, _, sigma2s = RML_Yeo(df_features, df_output, N_train, NUM=2000, lam_fgt=lam_fgt, nu=nu)
            if method == "RML_LG":
                y_test, y_train, _, sigma2s = RML_LG(df_features, df_output, N_train, NUM=2000, lam_fgt=lam_fgt, nu=nu)

            df_final_test['Method'+str(s)]=y_test
            df_final_train['Method'+str(s)]=y_train
            sigma2ss.append(np.array(sigma2s))

            df_output_max['Method'+str(s)]=df_output_max['heat-lag-0']
            df_output_min['Method'+str(s)]=df_output_min['heat-lag-0']


    df_origin_test = (df_final_test) * (
                df_output_max - df_output_min)+ df_output_min
    df_origin_train = (df_final_train) * (
                df_output_max - df_output_min)+ df_output_min

    sigma2ss=np.array(sigma2ss)
    N_val=int(int(df_origin_train.shape[0]/24/4)*24)#int(50*24)
    print("N_val:{}".format(N_val))
    df_origin_val=df_origin_train.iloc[-N_val:,:]
    sigma2s_val=sigma2ss[:, N_train-N_val:N_train]
    # df_final_test.plot()
    rmse_vals=[]; mae_vals=[];
    for k in range(df_origin_val.shape[1]-1):
        nu=nus[k]
        rmse_val=np.sqrt(mean_squared_error(df_origin_val['heat-lag-0'].to_numpy(),
                                        df_origin_val['Method'+str(
                                            k)].to_numpy()))
        mae_val=mean_absolute_error(df_origin_val['heat-lag-0'].to_numpy(),
                                        df_origin_val['Method'+str(
                                            k)].to_numpy())
        # crps_val=crps_general_array(df_origin_val['heat'].to_numpy(),df_origin_val['Method'+str(
        #                                     k)].to_numpy(), sigma2s_val[k,:],
        #                             nu, col="RLS")

        rmse_vals.append(rmse_val)
        mae_vals.append(mae_val)
        # crps_vals.append(crps_val)

    rmse_vals=np.array(rmse_vals)
    mae_vals=np.array(mae_vals)
    # crps_vals=np.array(crps_vals)
    import math
    loc1=rmse_vals.argmin();
    nu1=nu_trans[int(loc1//len(lam_fgts))]
    lam_fgt1=lam_fgts[int(math.fmod(loc1,len(lam_fgts)))]

    print("rmse min:{}, {}".format(rmse_vals.min(), rmse_vals.argmin()))
    print("optimal nu: {}".format(nu1))
    print("optimal forgetting factor lambda:{}".format(lam_fgt1))

    return rmse_vals


methods=['OLS_LG','OLS_Yeo']+['RLS','RLS_LG','RLS_Yeo']+['RML','RML_LG', 'RML_Yeo']
# methods=['RML_LG', 'RML_Yeo']
for method in methods:
    print(method)
    rmse_vals=hyper_para_val(df_features, df_output, method)
    plt.figure()
    plt.plot(rmse_vals)


