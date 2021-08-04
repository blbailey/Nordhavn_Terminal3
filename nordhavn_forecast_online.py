__author__ = 'Li Bai'


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

"""HEADS-UP FOR THE DATA: the whole data starts from the late September to 
the May 2020 before summer came; which assumes that heat load is mainly 
caused by heating space; while in summer, it is random, depending on the 
hot-tap water usage"""


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
from helpers_online import OLS, OLS_Yeo, OLS_LG, RLS, RLS_Yeo, RLS_LG, RML, RML_Yeo, RML_LG
from helpers_online import train_empirical_prob, train_hour_probs, similar_day_select_probs
from helpers_online import crps_general_array
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
N_train=int(int(N_total*0.75/24)*24);
df1=df[0:N_train]

# df1 = df1.sample(frac=1).reset_index(drop=True)
# N_train=int(int(0.7*len(df1)/24)*24)


df_features_un=df.iloc[:,1:].copy()
df_output_un=df.iloc[:,0:1].copy()


df_features_train_un=df_features_un.iloc[0:N_train,:]
df_features_test_un=df_features_un.iloc[N_train:,:]

df_output_train_un=df_output_un.iloc[0:N_train]
df_output_test_un=df_output_un.iloc[N_train:]



train_df=df1.iloc[0:N_train,:]
test_df = df1.iloc[N_train:,:]



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


df_final_train = pd.DataFrame(index=df_output_train.index)#df_output_train.copy()
df_final_test = pd.DataFrame(index=df_output_test.index)#df_output_test.copy()

df_final_train['Real']=df_output_train.to_numpy()[:,0]
df_final_test['Real']=df_output_test.to_numpy()[:,0]
dict_max={"Real": train_df_max['heat-lag-0']};dict_min={"Real": train_df_min['heat-lag-0']};

df_output_max=pd.Series(data=dict_max, index=['Real'])
df_output_min=pd.Series(data=dict_min, index=['Real'])

# df_output_max = train_df_max[['heat-lag-0']]
# df_output_min = train_df_min[['heat-lag-0']]

df_final_train['Persistent'] = df_features_train['heat-lag-24'].to_numpy()
df_final_test['Persistent'] = df_features_test['heat-lag-24'].to_numpy()
df_output_max['Persistent']=train_df_max['heat-lag-0']
df_output_min['Persistent']=train_df_min['heat-lag-0']


methods = ['OLS', 'OLS_LG', 'OLS_Yeo'] + ['RLS', 'RLS_LG', 'RLS_Yeo'] + ['RML', 'RML_LG', 'RML_Yeo']
df_sigma2s = pd.DataFrame(index=df_features.index)

for method in methods:
    if method == 'OLS':
        y_test, y_train, sigma2s = OLS(df_features, df_output, N_train)
    if method == 'OLS_Yeo':
        y_test, y_train, sigma2s = OLS_Yeo(df_features, df_output, N_train, lam=0.9)
    if method == "OLS_LG":
        y_test, y_train, sigma2s = OLS_LG(df_features, df_output, N_train, nu=0.1)
    if method == "RLS":
        y_test, y_train, _, sigma2s = RLS(df_features, df_output, N_train, lam_fgt=0.998)
    if method == "RLS_Yeo":
        y_test, y_train, _, sigma2s = RLS_Yeo(df_features, df_output, N_train, lam_fgt=0.998, lam=0.4)
    if method == "RLS_LG":
        y_test, y_train, _, sigma2s = RLS_LG(df_features, df_output, N_train,lam_fgt=0.998, nu=0.3)
    if method == "RML":
        y_test, y_train, _, sigma2s = RML(df_features, df_output, N_train, NUM=2000, lam_fgt=0.996)
    if method == "RML_Yeo":
        y_test, y_train, _, sigma2s = RML_Yeo(df_features, df_output, N_train, NUM=2000, lam_fgt=0.997, lam=0.4)
    if method == "RML_LG":
        y_test, y_train, _, sigma2s = RML_LG(df_features, df_output, N_train, NUM=2000, lam_fgt=0.998, nu=0.2)

    df_final_test[method] = y_test
    df_final_train[method] = y_train
    df_sigma2s[method] = sigma2s
    # sigma2ss.append(np.array(sigma2s))

    df_output_max[method] = train_df_max['heat-lag-0']
    df_output_min[method] = train_df_min['heat-lag-0']

df_origin_test = (df_final_test) * (
        df_output_max - df_output_min) + df_output_min
df_origin_train = (df_final_train) * (
        df_output_max - df_output_min) + df_output_min


df_origin_test[df_origin_test<=0]=0.
df_origin_test[df_origin_test>=HEAT_MAX_REAL]=HEAT_MAX_REAL


# df_origin_test.to_csv("nordhavn_test_pred_online.csv")




# ==========================crps=================================
crps_emp=train_empirical_prob(df_output_test_un,df_output_train_un)
crps_hour=train_hour_probs(df_output_test_un, df_output_train_un)
crps_sds=similar_day_select_probs(
    df_features_train, df_features_test, df_output_train,df_output_test)
#
# # =================CRPS comment==========================
df_sigma2s_test=df_sigma2s.iloc[N_train:,:]
print("I am running crps")
cols_new=['OLS', 'OLS_LG', 'OLS_Yeo', 'RLS', 'RLS_LG', 'RLS_Yeo', 'RML', 'RML_LG',
       'RML_Yeo']
nus={'OLS': 0, 'OLS_LG': 0.1, 'OLS_Yeo': 0.9, 'RLS': 0, 'RLS_LG':0.3, 'RLS_Yeo':0.4, 'RML':0, 'RML_LG':0.2,
       'RML_Yeo':0.4}

df_crps_time_lead=pd.DataFrame(columns=cols_new, index=np.arange(0, 24))

for col in cols_new:
    print("column {}".format(col))
    crps_vals=crps_general_array(df_origin_test["Real"].to_numpy(),
                             df_origin_test[col].to_numpy(), df_sigma2s_test[col], nus[col], col)


    crpss2=crps_vals[2]
    df_crps_time_lead[col]=crpss2
    crpss=crps_vals[0]



df_crps_time_lead['empirical']=crps_emp[2]
df_crps_time_lead['empirical-SDS-hourly']=crps_sds[2]
df_crps_time_lead['empirical-hourly']=crps_hour[2]
#
# df_crps_time_lead.to_csv("nordhavn_crps_online.csv")
# plt.figure()
# plt.plot(df_crps_time_lead.iloc[:,3:9], '-', LineWidth=3)
# # plt.plot(df_crps_time_lead.iloc[:,9:], '--',LineWidth=3)
# plt.legend(df_crps_time_lead.columns.to_list(), loc='upper right', #bbox_to_anchor=(0.5, 1.05),
#           ncol=2, fancybox=True, shadow=True)
# plt.ylabel("CRPS (MWh)")
# plt.xlabel("Time lead (hour)")
# plt.tight_layout()
#
# df_crps_test=df_crps_time_lead.mean(axis=0)
#
# fig=plt.figure()
# plt.gcf().subplots_adjust(bottom=0.25)
# BarWidth=0.2
# ax = fig.add_subplot(111)
# br1=np.arange(len(df_crps_test))
# ax.bar(br1, df_crps_test.to_numpy(), color='C0', width=BarWidth, alpha=0.5, edgecolor='grey', label='IRMSE')
# plt.xticks([r+BarWidth for r in range(len(df_crps_test))], labels=df_crps_test.index, rotation=70)
# plt.ylabel("CRPS (MWh)")
# plt.hlines(df_crps_test['empirical-SDS-hourly'], -0.5, len(df_crps_test)-0.5, color='black', LineWidth=1.5,
#            linestyles='--')
# # plt.ylim(30,60)
# plt.show()
# plt.tight_layout()
#
#
# df_icrps_test=100*(df_crps_test['empirical-SDS-hourly']-df_crps_test[0:9])/df_crps_test['empirical-SDS-hourly']
#
# fig=plt.figure()
# plt.gcf().subplots_adjust(bottom=0.25)
# BarWidth=0.2
# ax = fig.add_subplot(111)
# br1=np.arange(len(df_icrps_test))
# ax.bar(br1, df_icrps_test.to_numpy(), color='C0', width=BarWidth, alpha=0.5, edgecolor='grey', label='IRMSE')
# plt.xticks([r+BarWidth for r in range(len(df_icrps_test))], labels=df_icrps_test.index, rotation=70)
# plt.ylabel("ICRPS (%)")
# plt.hlines(0, -0.5, len(df_icrps_test)-0.5, color='black', LineWidth=1.5,
#            linestyles='--')
# # plt.ylim(30,60)
# plt.show()
# plt.tight_layout()
#
#
# df_icrps_time_lead=pd.DataFrame(columns=df_crps_time_lead.columns[3:9], index=np.arange(0,24))
# for col in df_icrps_time_lead.columns:
#
#     df_icrps_time_lead[col]=(df_crps_time_lead['empirical-SDS-hourly'].to_numpy()-df_crps_time_lead[col].to_numpy(
#
#     ))/df_crps_time_lead[
#     'empirical-SDS-hourly'].to_numpy()*100
#
#
#
# plt.figure()
# plt.plot(df_icrps_time_lead, '-', LineWidth=3)
# # plt.plot(df_crps_time_lead.iloc[:,9:], '--',LineWidth=3)
# plt.legend(df_icrps_time_lead.columns.to_list(), loc='upper right', #bbox_to_anchor=(0.5, 1.05),
#           ncol=2, fancybox=True, shadow=True)
# plt.ylabel("ICRPS (%)")
# plt.xlabel("Time lead (hour)")
# plt.tight_layout()
# plt.hlines(0, -0.5, len(df_icrps_time_lead)-0.5, color='black', LineWidth=1.5,
#            linestyles='--')