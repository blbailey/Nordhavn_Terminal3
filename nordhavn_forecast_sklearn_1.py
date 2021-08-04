__author__ = 'Li Bai'

"""hourly forecasting are made for the segmented data (during 2019 winter as heating period)
Input features: (['Counter [MWh]', 'temperature', 'humidity', 'DNI', 'Wx', 'Wy','Day sin', 'Day cos', 'Week sin', 'Week cos'] 
Obviously, the Counter is the heat demand, the dependent variable; the others are predictors related to weather features or temperal features, besides the
history heat demand as predictors as well (a day before)

SK-linear regression models are considered including linear regression, support vector regression, decision regression 
tree, 
SGD (linear regression), decision tree based estimator for ensemble methods such as ExtraTree, AdaBoost and 
GradientBoost.

On top of this, Naive and persistent models are used as benchmarks. In addition, a naive ensemble of Linear, 
SGD-Linear, Decision tree and SVR is considered as well"""


"""dummy variables are optional: hourly models are proposed by introducing dummy variables in one-hot code manner for each hour"""
"""In the previous literature, it is found that outdoor temperature is highly used as the most influencing factor for the inputs"""
# the basic step goes like this:
# 1> basic feature analysis through ACF or PACF, seasonality
# 2> feature correlation to see linear relationship whether they make further information from similairty sense from information theory analysis
# 3> to see whether day sin day cos week sin week cos is helpful to find a weekly or daily pattern as they make contribution to the similarity
# import IPython
# import IPython.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
from sklearn.utils import shuffle
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
from helpers import rmse_func, mae_func, r2_func, bias_func,irmse_func
from helpers import rmse_array_func, mae_array_func, r2_array_func, bias_array_func
from helpers import irmse_array_func
from nordhavn_forecast_sklearn import df_main, add_dummy_hour_flip
from sklearn.utils import shuffle



# df1 = pd.read_csv ("D:\\OneDrive\\OneDrive - Danmarks Tekniske Universitet\\energydataDTU\\venv\\data_gene\\terminal3_weather.csv",
#                           sep=',', index_col=0)

df_test_pred, df_train_pred=df_main()
print(df_test_pred.columns)
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

train_df=df1[0:N_train]
val_df = df1[N_train:]

test_df = df[N1:]
NUM_FEATURES=df.shape[1]

train_df_mean = train_df.mean()
train_df_std = train_df.std()

train_df1 = ((train_df - train_df_mean) / train_df_std)
val_df1 = ((val_df - train_df_mean) / train_df_std)
test_df1 = ((test_df - train_df_mean) / train_df_std)

# =============comment them out if dummy variables are not requried============
train_df1 = add_dummy_hour_flip(train_df1)
val_df1 = add_dummy_hour_flip(val_df1)
test_df1 = add_dummy_hour_flip(test_df1)


X_train=train_df1.iloc[:,1:].to_numpy();y_train=train_df1.iloc[:,0].to_numpy()
X_val=val_df1.iloc[:,1:].to_numpy();y_val=val_df1.iloc[:,0].to_numpy()
X_test=test_df1.iloc[:,1:].to_numpy();y_test=test_df1.iloc[:,0].to_numpy()


# df_test_pred_norm=pd.DataFrame(index=test_df1.index);df_test_pred=pd.DataFrame(index=test_df1.index)
# df_val_pred_norm=pd.DataFrame(index=val_df1.index);df_val_pred=pd.DataFrame(index=val_df1.index)
# df_train_pred_norm=pd.DataFrame(index=train_df1.index);df_train_pred=pd.DataFrame(index=train_df1.index)

# df_test_pred_norm['Real']=test_df1['heat-lag-0'].to_numpy()
# df_val_pred_norm['Real']=val_df1['heat-lag-0'].to_numpy()
# df_train_pred_norm['Real']=train_df1['heat-lag-0'].to_numpy()
#
# df_test_pred['Real']=test_df1['heat-lag-0'].to_numpy()*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
# df_val_pred['Real']=val_df1['heat-lag-0'].to_numpy()*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
# df_train_pred['Real']=train_df1['heat-lag-0'].to_numpy()*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
#
#
# y_train_mtx=y_train.reshape(-1,24);
# y_val_mtx=y_val.reshape(-1,24);
# y_test_mtx=y_test.reshape(-1,24)
#
#
#
# df_test_pred_norm['Persistent']=test_df1['heat-lag-24'].to_numpy()
# df_val_pred_norm['Persistent']=val_df1['heat-lag-24'].to_numpy()
# df_train_pred_norm['Persistent']=train_df1['heat-lag-24'].to_numpy()
#
# df_test_pred['Persistent']=test_df1['heat-lag-24'].to_numpy()*train_df_std['heat-lag-24']+train_df_mean['heat-lag-24']
# df_val_pred['Persistent']=val_df1['heat-lag-24'].to_numpy()*train_df_std['heat-lag-24']+train_df_mean['heat-lag-24']
# df_train_pred['Persistent']=train_df1['heat-lag-24'].to_numpy()*train_df_std['heat-lag-24']+train_df_mean['heat-lag-24']
#



def gridCV_SVR(X_tmp, y_tmp):
    # 2， 1
    from sklearn import svm
    reg_svm=svm.SVR(kernel='rbf')
    from sklearn.model_selection import GridSearchCV
    param_grid={'C':[0.5,1,2,4,6,8],'gamma':[0.1, 1, 2, 4, 8]}
    search=GridSearchCV(reg_svm, param_grid, cv=5, refit=True)
    search.fit(X_tmp, y_tmp)
    mdl_best=search.best_estimator_
    print(mdl_best.C, mdl_best.gamma)

    return mdl_best.C, mdl_best.gamma

def gridCV_tree(X_tmp, y_tmp):
    from sklearn import tree
    from sklearn.model_selection import GridSearchCV
    param_grid = {'max_depth': [3, 5, 10],
                  'min_samples_split': [2, 5, 10]}
    regr_tree = tree.DecisionTreeRegressor(random_state=0)
    search=GridSearchCV(regr_tree, param_grid, cv=5, refit=True)
    search.fit(X_tmp, y_tmp)
    mdl_best=search.best_estimator_
    print(mdl_best.max_depth, mdl_best.min_samples_split)
    return mdl_best.max_depth, mdl_best.min_samples_split

# ==================hyperparameter optimization=======================
def svr_pso_fun(para):
    from sklearn import svm
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import mean_squared_error

    num_particle=para.shape[0]
    dim=para.shape[1]
    values=[]
    print(para)
    for k in range(num_particle):
        C=para[k,0]; gamma=para[k,1]
        reg_svm = svm.SVR(kernel='rbf', C=C, gamma=gamma)
        cv_results = cross_validate(reg_svm, X_train, y_train, cv=5, return_estimator=True)
        mdl_best_loc=np.argmax(cv_results['test_score'])
        mdl_best=cv_results['estimator'][mdl_best_loc]
        values.append(mean_squared_error(mdl_best.predict(X_train),y_train))
    return np.array(values)
def PSO_hyperpara_optimize():
    import pyswarms as ps
    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    bounds=(np.array([0, 0]), np.array([2,2]))
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, bounds=bounds, options=options)
    best_cost, best_pos = optimizer.optimize(svr_pso_fun, iters=20)
    # we get the result of array([9.92468751, 0.45858203]) after 20 iterations for searching
    return best_cost, best_pos
# best_cost, best_pos=PSO_hyperpara_optimize()
# =================basic models======================================
# def Benchmark(X_yes_train,y_train,X_yes_test,y_test):
#
#     # use the training hourly average for the test output for benchmark
#     nums_train=int(len(X_yes_train)/24);nums_test=int(len(X_yes_test)/24)
#     y_ben_train_mtx=np.tile(np.mean(X_yes_train[:,0].reshape(-1,24),axis=0), (nums_train, 1))
#     y_train_mtx=X_yes_train[:,0].reshape(-1,24)
#
#     y_ben_test_mtx=np.tile(np.mean(X_yes_train[:,0].reshape(-1,24),axis=0), (nums_test, 1))
#     y_test_mtx=X_yes_test[:,0].reshape(-1,24)
#
#
#     mse_ben_train_mtx=mean_squared_error(y_ben_train_mtx,y_train_mtx, multioutput='raw_values')
#     mse_ben_test_mtx=mean_squared_error(y_ben_test_mtx, y_test_mtx, multioutput='raw_values')
#
#     mae_ben_train_mtx=mean_absolute_error(y_train_mtx, y_ben_train_mtx, multioutput='raw_values')
#     mae_ben_test_mtx=mean_absolute_error(y_test_mtx, y_ben_test_mtx, multioutput='raw_values')
#
#     bias_ben_mtx=np.mean(y_test_mtx, axis=0)-np.mean(y_ben_test_mtx,axis=0)
#         # np.sqrt(mse_ben_test-np.var(y_ben_test))
#     print(mse_ben_test_mtx)
#     print(np.var(y_ben_test_mtx))
#
#     r_sqr_ben_mtx=1-np.var(y_test_mtx-y_ben_test_mtx,axis=0)/np.var(y_test_mtx, axis=0)
#
#     ben_metric_train_mtx=[mse_ben_train_mtx,  mae_ben_train_mtx]
#     rmse_ben_test_mtx = np.sqrt(mse_ben_test_mtx)
#     ben_metric_test_mtx=[mse_ben_test_mtx, mae_ben_test_mtx, rmse_ben_test_mtx, mse_ben_test_mtx, bias_ben_mtx, r_sqr_ben_mtx]
#
#
#
#     # np.multiply(np.ones(shape=(nums,24)),(np.mean(X_yes_train[:,0].reshape(-1,24),axis=0)).reshape(-1,1))
#     y_ben_train=np.tile(np.mean(X_yes_train[:,0].reshape(-1,24),axis=0), (nums_train, 1)).reshape(-1,)
#     # make it for every hour instead of it...
#     nums_test=int(len(X_yes_test)/24);
#     y_ben_test=np.tile(np.mean(X_yes_train[:,0].reshape(-1,24),axis=0), (nums_test, 1)).reshape(-1,)
#
#     y_ben = np.concatenate((y_ben_train, y_ben_test))
#
#     mse_ben_train=mean_squared_error(y_train, y_ben_train)
#     mse_ben_test=mean_squared_error(y_test, y_ben_test)
#
#     mae_ben_train=mean_absolute_error(y_train, y_ben_train)
#     mae_ben_test=mean_absolute_error(y_test, y_ben_test)
#
#     bias_ben=np.mean(y_test)-np.mean(y_ben_test)
#         # np.sqrt(mse_ben_test-np.var(y_ben_test))
#     print(mse_ben_test)
#     print(np.var(y_ben_test))
#
#     r_sqr_ben=1-np.var(y_test-y_ben_test)/np.var(y_test)
#
#     ben_metric_train=[mse_ben_train,  mae_ben_train]
#     rmse_ben_test = np.sqrt(mean_squared_error(y_test, y_ben_test))
#     ben_metric_test=[mse_ben_test, mae_ben_test, rmse_ben_test, mse_ben_test, bias_ben, r_sqr_ben]
#
#     print("Benchmark train MSE : {}, MAE : {}".format(mse_ben_train, mae_ben_train))
#     print("Benmark test MSE : {}, MAE : {}".format(mse_ben_test, mae_ben_test))
#
#     return y_ben_train, y_ben_test, y_ben, ben_metric_train, ben_metric_test,ben_metric_train_mtx,ben_metric_test_mtx

def SVR(X_train,y_train,X_test,y_test):
    from sklearn import svm

    # X_tmp=np.vstack((X_train, X_val))
    # y_tmp=np.hstack((y_train, y_val))

    # C_opt, gamma_opt=gridCV_SVR(X_tmp, y_tmp)
    C_opt=2;gamma_opt=1;
    reg_svm=svm.SVR(kernel='rbf', C=C_opt, gamma=gamma_opt).fit(X_train,y_train)

    y_svm_train=reg_svm.predict(X_train);
    # y_svm_val=reg_svm.predict(X_val);
    y_svm_test=reg_svm.predict(X_test);

    return reg_svm, y_svm_train, y_svm_test,
def linear_reg(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LinearRegression
    linreg=LinearRegression().fit(X_train, y_train)
    y_lin_train=linreg.predict(X_train)
    # y_lin_val=linreg.predict(X_val)
    y_lin_test=linreg.predict(X_test)
    return linreg, y_lin_train, y_lin_test

def Tree(X_train,y_train,X_test,y_test):
    from sklearn import tree
    max_depth_opt=5;
    min_samples_split_opt=2;
    # in the default tree, no pruning is performed since hyperparameter ccp_alpha is set to default 0.
    # no pruning...
    regr_tree=tree.DecisionTreeRegressor(max_depth=max_depth_opt, min_samples_split=min_samples_split_opt, random_state=0).fit(X_train,y_train)

    # print(regr_tree.cost_complexity_pruning_path(X_train,y_train))
    y_tree_train = regr_tree.predict(X_train)
    # y_tree_val = regr_tree.predict(X_val)
    y_tree_test  = regr_tree.predict(X_test)

    return regr_tree, y_tree_train, y_tree_test
def SGD_lin(X_train,y_train,X_test,y_test):

    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline

    sgd_reg = make_pipeline(SGDRegressor(loss='huber',max_iter=1000, tol=1e-3,shuffle=False)).fit(X_train,y_train)


    y_sgd_train=sgd_reg.predict(X_train)
    # y_sgd_val=sgd_reg.predict(X_val)
    y_sgd_test = sgd_reg.predict(X_test)


    return sgd_reg, y_sgd_train, y_sgd_test
def GP(X_train,y_train,X_test,y_test):
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, RationalQuadratic, Exponentiation
    from sklearn.base import clone
    # RBF math k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)
    # where :math:`l` is the length scale of the kernel and
    #     :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    from sklearn.gaussian_process import GaussianProcessRegressor
    # kernel=1.0 * RBF(0.02)
    # the rational quadratic kernel can be seen as a scale mixture (an infinite sum) of RBF kernels with different characteristic length scales.
    #  different kernel provides great differences!!! for GP
    # gpr depends on too much on the kernel;
    kernel = Exponentiation(RationalQuadratic(), exponent=2)
        # 1.0 * RBF(length_scale=2, length_scale_bounds=(1e-2, 1e3))
    # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    # + ConstantKernel(constant_value=0.1)
    gpr_reg = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b").fit(X_train, y_train)


    y_gpr_train=gpr_reg.predict(X_train)
    y_gpr_val = gpr_reg.predict(X_val)
    y_gpr_test = gpr_reg.predict(X_test)


    return gpr_reg, y_gpr_train, y_gpr_val, y_gpr_test
# =====================ensemble methods=======================================
# such as bagging, forest trees, adaboost and gradient tree bodsting..
def ensemble_ranfor(X_train, y_train, X_test, y_test):
    N_ESTIMATORS = 300

    from sklearn.ensemble import RandomForestRegressor
    # . max_depth, min_samples_leaf can be used to control over-fitting...
    # randomforecast is based on bagging tree, and then in the process of individual tree building a random splitting is
    # considered in which a subset of features are selected for splitting instead of full features.. feature bagging
    # to avoid building multiple correlated trees since one or a few predictors could be strongly correlated (multicollinearity)
    # max_leaf_nodes=5 avoid overgrowing trees
    # here no CCP pruning is considered...
    regr_ranfor=RandomForestRegressor(max_depth=3, n_estimators=N_ESTIMATORS, max_features='log2', max_leaf_nodes=5, random_state=0).fit(X_train, y_train)
    y_ranfor_train=regr_ranfor.predict(X_train)
    y_ranfor_test = regr_ranfor.predict(X_test)
    return regr_ranfor, y_ranfor_train, y_ranfor_test
def ensemble_xtree(X_train, y_train, X_test, y_test):
    N_ESTIMATORS = 300

    from sklearn.ensemble import ExtraTreesRegressor
    # extra trees similar to random trees, while slightly different, 1) each tree is trained using full datasets instead of bootstrap
    # top-down splitting is random; instead of finding local optimal cut-point for each feature, a random one is selected from a uniform distribution
    # based on the feature's empirical range; among all the generated cut-points corresponding to the selected features, the optimal one is selected corresponding to a feature

    # in effect, the parameter bootstrap=True can still use bootstrap..
    # random_state is related to 3 things: 1) boostrap=True 2) sampling of the features selection 3) draw of cut-points for the selected features
    regr_xtree=ExtraTreesRegressor(n_estimators=N_ESTIMATORS, random_state=0, max_features='log2', max_leaf_nodes=5, max_depth=3, bootstrap=False).fit(X_train, y_train)
    y_xtree_train=regr_xtree.predict(X_train)
    y_xtree_test = regr_xtree.predict(X_test)

    return regr_xtree, y_xtree_train, y_xtree_test
def ensemble_ada(X_train, y_train, X_test, y_test):
    N_ESTIMATORS = 300

    from sklearn.ensemble import AdaBoostRegressor
    # default base estimator is DecisionTreeRegressor(max_depth=3)...
    # Learning rate shrinks the contribution of each regressor by learning_rate.
    # There is a trade off between learning_rate and n_estimators.

    regr_ada = AdaBoostRegressor(random_state=0, n_estimators=N_ESTIMATORS, learning_rate=0.05, loss='square').fit(X_train, y_train)
    y_ada_train=regr_ada.predict(X_train)
    y_ada_test = regr_ada.predict(X_test)

    return regr_ada, y_ada_train, y_ada_test
def ensemble_gb(X_train, y_train, X_test, y_test):
    N_ESTIMATORS = 300

    from sklearn.ensemble import GradientBoostingRegressor
    # The fraction of samples to be used for fitting the individual base learners.
    # If smaller than 1.0 this results in Stochastic Gradient Boosting.
    # subsample interacts with the parameter n_estimators.
    # Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.

    # all features are selected in default mode
    regr_gb = GradientBoostingRegressor(n_estimators=N_ESTIMATORS, learning_rate=0.05,
    max_depth=3, random_state=0, loss='huber',max_leaf_nodes=5).fit(X_train, y_train)
    y_gb_train=regr_gb.predict(X_train)
    y_gb_test = regr_gb.predict(X_test)


    return regr_gb, y_gb_train, y_gb_test



_, y_lin_train, y_lin_test=linear_reg(X_train, y_train, X_test, y_test)
_, y_svm_train, y_svm_test=SVR(X_train,y_train,X_test,y_test)
_, y_tree_train, y_tree_test=Tree(X_train,y_train,X_test,y_test)
_, y_sgd_train, y_sgd_test =SGD_lin(X_train,y_train,X_test,y_test)

y_mix_train=(y_lin_train+y_svm_train+y_tree_train+y_sgd_train)/4
y_mix_test=(y_lin_test+y_svm_test+y_tree_test+y_sgd_test)/4


# _, y_gpr_train, y_gpr_test=GP(X_train,y_train,X_test,y_test)
_, y_ranfor_train, y_ranfor_test=ensemble_ranfor(X_train, y_train, X_test, y_test)
_, y_xtree_train, y_xtree_test =ensemble_xtree(X_train, y_train, X_test, y_test)
_, y_ada_train, y_ada_test =ensemble_ada(X_train, y_train, X_test, y_test)
_, y_gb_train, y_gb_test  =ensemble_gb(X_train, y_train, X_test, y_test)

df_train_pred['SK-Linear']=y_lin_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_train_pred['SVR']=y_svm_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_train_pred['DTree']=y_tree_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_train_pred['SGD-Linear']=y_sgd_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_train_pred['Naive Ensemble']=y_mix_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_train_pred['RanFor']=y_ranfor_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_train_pred['Xtree']=y_xtree_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_train_pred['Ada']=y_ada_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_train_pred['GB']=y_gb_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']


df_test_pred['SK-Linear']=y_lin_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_test_pred['SVR']=y_svm_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_test_pred['DTree']=y_tree_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_test_pred['SGD-Linear']=y_sgd_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_test_pred['Naive Ensemble']=y_mix_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_test_pred['RanFor']=y_ranfor_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_test_pred['Xtree']=y_xtree_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_test_pred['Ada']=y_ada_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
df_test_pred['GB']=y_gb_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']



df_test_pred[df_test_pred<=0]=0.

df_test_rmse=rmse_func(df_test_pred);
df_test_mae=mae_func(df_test_pred);
df_test_r2=r2_func(df_test_pred);
df_test_bias=bias_func(df_test_pred);

df_test_irmse=irmse_func(df_test_rmse)

#
df_test_array_rmse=rmse_array_func(df_test_pred);
df_test_array_mae=mae_array_func(df_test_pred);
df_test_array_r2=r2_array_func(df_test_pred);
df_test_array_bias=bias_array_func(df_test_pred);



df_test_array_irmse=irmse_array_func(df_test_array_rmse)

# plt.figure()
# plt.plot(df_test_array_irmse.iloc[:,0:4], '--', LineWidth=2.5)
# plt.plot(df_test_array_irmse.iloc[:,4:], LineWidth=2.5)
# plt.legend(df_test_array_rmse.columns.to_list(), loc='upper right')
# plt.ylabel("IRMSE (MWh)")
# plt.xlabel("Time lead (hour)")
#
df_test_pred.to_csv("nordhavn_test_pred_dummy.csv")
# plt.figure()
# plt.plot(df_test_array_rmse.iloc[:,0:5], '--', LineWidth=2.5)
# plt.plot(df_test_array_rmse.iloc[:,5:], LineWidth=2.5)
# plt.legend(df_test_array_rmse.columns.to_list(), loc='upper right')
# plt.ylabel("RMSE (MWh)")
# plt.xlabel("Time lead (hour)")
# import matplotlib as mpl
#
# mpl.rc('xtick', labelsize=20)
# mpl.rc('ytick', labelsize=20)
# plt.rcParams.update({'font.size': 24})
#
# mpl.rcParams['figure.figsize'] = (18,10)
# mpl.rcParams['axes.grid'] = False
# # plt.figure()
# # plt.bar(df_test_rmse.to_numpy(), LineWidth=3)
# # plt.xticks(np.arange(df_test_rmse.shape[0]), df_test_rmse.index, rotation=15)
# fig=plt.figure()
# BarWidth=0.25
# ax = fig.add_subplot(111)
# br1=np.arange(len(df_test_irmse))
# ax.bar(br1, df_test_irmse.to_numpy()[:,0], color='C0', width=BarWidth, alpha=0.5, edgecolor='grey', label='IRMSE')
# plt.xticks([r+BarWidth for r in range(len(df_test_irmse))], labels=df_test_irmse.index, rotation=15)
# plt.ylabel("IRMSE (%)")
# plt.hlines(0, -0.5, len(df_test_irmse)-0.5, color='black', LineWidth=1.5,
#            linestyles='--')
# plt.show()
#
# fig=plt.figure()
# BarWidth=0.25
# ax = fig.add_subplot(111)
# br1=np.arange(len(df_test_rmse))
# ax.bar(br1, df_test_rmse.to_numpy()[:,0], color='C0', width=BarWidth, alpha=0.5, edgecolor='grey', label='RMSE')
# plt.xticks([r+BarWidth for r in range(len(df_test_rmse))], labels=df_test_rmse.index, rotation=15)
# plt.ylabel("RMSE (MWh)")
# plt.hlines(df_test_rmse.loc['Persistent','RMSE'], -0.5, len(df_test_rmse)-0.5, color='black', LineWidth=1.5,
#            linestyles='--')
# plt.show()
# fig=plt.figure()
# BarWidth=0.25
# ax = fig.add_subplot(111)
# br1=np.arange(len(df_test_mae))
# ax.bar(br1, df_test_mae.to_numpy()[:,0], color='C0', width=BarWidth, alpha=0.5, edgecolor='grey', label='MAE')
# plt.xticks([r+BarWidth for r in range(len(df_test_mae))], labels=df_test_mae.index, rotation=15)
# plt.ylabel("MAE (MWh)")
# plt.hlines(df_test_mae.loc['Persistent','MAE'], -0.5, len(df_test_mae)-0.5, color='black', LineWidth=1.5,
#            linestyles='--')
# plt.show()
#
# fig=plt.figure()
# BarWidth=0.25
# ax = fig.add_subplot(111)
# br1=np.arange(len(df_test_r2))
# ax.bar(br1, df_test_r2.to_numpy()[:,0], color='C0', width=BarWidth, alpha=0.5, edgecolor='grey', label='R2')
# plt.xticks([r+BarWidth for r in range(len(df_test_r2))], labels=df_test_r2.index, rotation=15)
# plt.ylabel("R2")
# plt.hlines(df_test_r2.loc['Persistent','R2'], -0.5, len(df_test_r2)-0.5, color='black', LineWidth=1.5,
#            linestyles='--')
# plt.show()
#
# fig=plt.figure()
# BarWidth=0.25
# ax = fig.add_subplot(111)
# br1=np.arange(len(df_test_bias))
# ax.bar(br1, df_test_bias.to_numpy()[:,0], color='C0', width=BarWidth, alpha=0.5, edgecolor='grey', label='RMSE')
# plt.xticks([r+BarWidth for r in range(len(df_test_bias))], labels=df_test_bias.index, rotation=15)
# plt.ylabel("Bias (MWh)")
# plt.hlines(df_test_bias.loc['Persistent','Bias'], -0.5, len(df_test_bias)-0.5, color='black', LineWidth=1.5,
#            linestyles='--')
# plt.show()
#
