__author__ = 'Li Bai'

"""NN-Linear, NN-Dense, LSTM and CNN are used; NN is a simple 3-layer models including input layer, hiddle layer and 
output layer. In the hidden layer, the basis function is y=x; NN-Dense: the basis function of the hidden lyaer is 
tanh function; CNN considers one convolutional layer."""
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from helpers import add_day_week_features, data_gene, data_gene_dict, feature_selection, LAG_DICT, LAG_DICT1, \
    SHIFT_HEAT, SHIFT_HEAT1
from helpers import rmse_func, mae_func, r2_func, bias_func
from helpers import rmse_array_func, mae_array_func, r2_array_func, bias_array_func

from sklearn.utils import shuffle



left = 0.125  # the left side of the subplots of the figure
right = 0.90  # the right side of the subplots of the figure
bottom = 0.18  # the bottom of the subplots of the figure
top = 0.9  # the top of the subplots of the figure
wspace = 0.2  # the amount of width reserved for space between subplots,
# expressed as a fraction of the average axis width
hspace = 0.2  # the amount of height reserved for space between subplots,





mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False




def NN_Linear(X_train, X_val, X_test, y_train, y_val, y_test):
    print("NN_Linear is running")
    BATCH_SIZE = 256
    SHUFFLE_BUFFER_SIZE = 500

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    patience=50
    MAX_EPOCHS = 300
    #  Linear
    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    linear.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                 metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError(), tf.metrics.MeanSquaredError()])

    history = linear.fit(train_dataset, epochs=MAX_EPOCHS,
                        validation_data=val_dataset,
                        callbacks=[early_stopping], verbose=0)

    y_linear_train=linear.predict(X_train)
    y_linear_val=linear.predict(X_val)
    y_linear_test=linear.predict(X_test)


    return  linear, y_linear_train, y_linear_val, y_linear_test


def NN_Dense(X_train, X_val, X_test, y_train, y_val, y_test):
    print("NN-dense is running")
    patience=50
    MAX_EPOCHS = 600
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 256
    SHUFFLE_BUFFER_SIZE = 500
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    # dense
    dense = tf.keras.Sequential([
      tf.keras.layers.Dense(units=64, activation='sigmoid'),
      tf.keras.layers.Dense(units=64, activation='sigmoid'),
      tf.keras.layers.Dense(units=1)
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    dense.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError(), tf.metrics.MeanSquaredError()])

    history = dense.fit(train_dataset, epochs=MAX_EPOCHS,
                        validation_data=val_dataset,
                        callbacks=[early_stopping], verbose=0)

    y_dense_train = dense.predict(X_train)
    y_dense_test = dense.predict(X_test)
    y_dense_val = dense.predict(X_val)


    return dense, y_dense_train, y_dense_val, y_dense_test



def LSTM(X_train, X_val, X_test, y_train, y_val, y_test):
    print("LSTM is running")
    # LSTM
    patience = 40

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 128#32#*24
    SHUFFLE_BUFFER_SIZE = 500
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # inpu shape (None, 24, 10)
    MAX_EPOCHS=600
    # inputs: A 3D tensor with shape [batch, timesteps, feature].
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(16, return_sequences=False),
        tf.keras.layers.Dropout(.4),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1),
        # tf.keras.layers.Dense(units=1)
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError(), tf.metrics.MeanSquaredError()])

    history = lstm_model.fit(train_dataset, epochs=MAX_EPOCHS,
                        validation_data=val_dataset,
                        callbacks=[early_stopping], verbose=0)

    # IPython.display.clear_output()
    # val_performance['LSTM'] = lstm_model.evaluate(val_dataset)
    # performance['LSTM'] = lstm_model.evaluate(test_dataset, verbose=0)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    y_lstm_train=lstm_model.predict(X_train)
    y_lstm_val=lstm_model.predict(X_val)

    # mse_lstm_test=(lstm_model.evaluate(test_dataset, verbose=0))[0]
    y_lstm_test=lstm_model.predict(X_test)

    return lstm_model, y_lstm_train, y_lstm_val, y_lstm_test



    # bias_lstm = np.mean(y_test)-np.mean(y_lstm_test)# np.sqrt(mse_lstm_test - np.var(y_lstm_test))
    # r_sqr_lstm = 1 - np.var(y_test-y_lstm_test) / np.var(y_test)
    # bias['LSTM']=bias_lstm
    # r_sqr['LSTM']=r_sqr_lstm
    #
    # y_lstm_test_mtx=y_lstm_test.reshape(-1,24)
    # mse_lstm_test_mtx=mean_squared_error(y_lstm_test_mtx, y_test_mtx, multioutput='raw_values')
    # mae_lstm_test_mtx=mean_absolute_error(y_lstm_test_mtx, y_test_mtx, multioutput='raw_values')
    # rmse_lstm_test_mtx=np.sqrt(mse_lstm_test_mtx)
    # performance_mtx['LSTM']=[mse_lstm_test_mtx, mae_lstm_test_mtx, rmse_lstm_test_mtx,mse_lstm_test_mtx]
    # bias_lstm_mtx = np.mean(y_test_mtx, axis=0)-np.mean(y_lstm_test_mtx, axis=0)# np.sqrt(mse_lstm_test - np.var(y_lstm_test))
    # r_sqr_lstm_mtx= 1 - np.var(y_test_mtx-y_lstm_test_mtx,axis=0) / np.var(y_test_mtx,axis=0)
    # bias_mtx['LSTM']=bias_lstm_mtx
    # r_sqr_mtx['LSTM']=r_sqr_lstm_mtx

def CNN(X_train, X_val, X_test, y_train, y_val, y_test):
    print("CNN is running")
    # CNN
    patience = 100

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 500
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    # CNN
    CONV_WIDTH=24
    MAX_EPOCHS=600

    print("batch size:{}, max_epoch：{}".format(BATCH_SIZE, MAX_EPOCHS))
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH,),
                               activation='relu'),
        # tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=16, activation='sigmoid'),
        tf.keras.layers.Dense(units=1),
    ])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    conv_model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError(), tf.metrics.MeanSquaredError()])

    history = conv_model.fit(train_dataset, epochs=MAX_EPOCHS,
                        validation_data=val_dataset,
                        callbacks=[early_stopping], verbose=0)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    # IPython.display.clear_output()
    # val_performance['Conv'] = conv_model.evaluate(val_dataset)
    # performance['Conv'] = conv_model.evaluate(test_dataset, verbose=0)
    y_conv_train=conv_model.predict(X_train)
    y_conv_val=conv_model.predict(X_val)
    # mse_conv_test=(conv_model.evaluate(test_dataset, verbose=0))[0]
    y_conv_test=conv_model.predict(X_test)


    return conv_model, y_conv_train, y_conv_val, y_conv_test




def Benchmark(X_train, X_val, X_test, y_train, y_val, y_test):#(X_yes_train,y_train,X_yes_test,y_test):

    # use the training hourly average for the test output for benchmark
    nums_train=int(len(X_train)/24);
    nums_val=int(len(X_val)/24)
    nums_test=int(len(X_test)/24)
    # y_ben_train_mtx=np.tile(np.mean(y_train.reshape(-1,24),axis=0), (nums_train, 1))
    # y_train_mtx=X_yes_train[:,0].reshape(-1,24)
    y_ben_val_mtx=np.tile(np.mean(y_train.reshape(-1,24),axis=0), (nums_val, 1))
    y_ben_test_mtx=np.tile(np.mean(y_train.reshape(-1,24),axis=0), (nums_test, 1))


    # np.multiply(np.ones(shape=(nums,24)),(np.mean(X_yes_train[:,0].reshape(-1,24),axis=0)).reshape(-1,1))
    y_ben_train=y_train
    # make it for every hour instead of it...
    y_ben_val=y_ben_val_mtx.reshape(-1,)
    y_ben_test=y_ben_test_mtx.reshape(-1,)



    return y_ben_train, y_ben_val, y_ben_test


def add_dummy_hour_flip(df_out):
    dummy=np.zeros(shape=(df_out.shape[0], 24))
    for k, date in enumerate(df_out.index):
        dummy[k, date.hour]=1
    names=['hour'+ str(23-k) for k in range(24)]
    for k, name in enumerate(names):
        df_out[name]=dummy[:,k]
    return df_out

def df_main():
# def main():

    df= pd.read_csv ("nordhavn_terminal3_selected.csv",
                              sep=',', index_col=0)

    df.index=pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")


    # df_out=add_dummy_hour(df_out)

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

    train_df1=add_dummy_hour_flip(train_df1)
    val_df1=add_dummy_hour_flip(val_df1)
    test_df1=add_dummy_hour_flip(test_df1)

    X_train=train_df1.iloc[:,1:].to_numpy();y_train=train_df1.iloc[:,0].to_numpy()
    X_val=val_df1.iloc[:,1:].to_numpy();y_val=val_df1.iloc[:,0].to_numpy()
    X_test=test_df1.iloc[:,1:].to_numpy();y_test=test_df1.iloc[:,0].to_numpy()


    df_test_pred_norm=pd.DataFrame(index=test_df1.index);df_test_pred=pd.DataFrame(index=test_df1.index)
    df_val_pred_norm=pd.DataFrame(index=val_df1.index);df_val_pred=pd.DataFrame(index=val_df1.index)
    df_train_pred_norm=pd.DataFrame(index=train_df1.index);df_train_pred=pd.DataFrame(index=train_df1.index)


    df_test_pred_norm['Real']=test_df1['heat-lag-0'].to_numpy()
    df_val_pred_norm['Real']=val_df1['heat-lag-0'].to_numpy()
    df_train_pred_norm['Real']=train_df1['heat-lag-0'].to_numpy()

    df_test_pred['Real']=test_df1['heat-lag-0'].to_numpy()*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
    df_val_pred['Real']=val_df1['heat-lag-0'].to_numpy()*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
    df_train_pred['Real']=train_df1['heat-lag-0'].to_numpy()*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']


    y_train_mtx=y_train.reshape(-1,24);
    y_val_mtx=y_val.reshape(-1,24);
    y_test_mtx=y_test.reshape(-1,24)

    y_ben_train, y_ben_val, y_ben_test=Benchmark(X_train, X_val, X_test, y_train, y_val, y_test)


    df_test_pred_norm['Naive'] = y_ben_test
    df_val_pred_norm['Naive'] = y_ben_val
    df_train_pred_norm['Naive'] = y_ben_train


    df_test_pred['Naive'] = y_ben_test*train_df_std['heat-lag-24']+train_df_mean['heat-lag-24']
    df_val_pred['Naive'] = y_ben_val*train_df_std['heat-lag-24']+train_df_mean['heat-lag-24']
    df_train_pred['Naive'] = y_ben_train*train_df_std['heat-lag-24']+train_df_mean['heat-lag-24']

    df_test_pred_norm['Persistent']=test_df1['heat-lag-24'].to_numpy()
    df_val_pred_norm['Persistent']=val_df1['heat-lag-24'].to_numpy()
    df_train_pred_norm['Persistent']=train_df1['heat-lag-24'].to_numpy()

    df_test_pred['Persistent']=test_df1['heat-lag-24'].to_numpy()*train_df_std['heat-lag-24']+train_df_mean['heat-lag-24']
    df_val_pred['Persistent']=val_df1['heat-lag-24'].to_numpy()*train_df_std['heat-lag-24']+train_df_mean['heat-lag-24']
    df_train_pred['Persistent']=train_df1['heat-lag-24'].to_numpy()*train_df_std['heat-lag-24']+train_df_mean['heat-lag-24']

    #
    linear, y_linear_train, y_linear_val, y_linear_test=NN_Linear(X_train, X_val, X_test, y_train, y_val, y_test)

    df_test_pred_norm['NN-Linear']=y_linear_test
    df_val_pred_norm['NN-Linear']=y_linear_val
    df_train_pred_norm['NN-Linear']=y_linear_train

    df_test_pred['NN-Linear']=y_linear_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
    df_val_pred['NN-Linear']=y_linear_val*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
    df_train_pred['NN-Linear']=y_linear_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
    #
    #
    #

    dense, y_dense_train, y_dense_val, y_dense_test=NN_Dense(X_train, X_val, X_test, y_train, y_val, y_test)
    df_test_pred_norm['NN-Dense']=y_dense_test
    df_val_pred_norm['NN-Dense']=y_dense_val
    df_train_pred_norm['NN-Dense']=y_dense_train


    df_test_pred['NN-Dense']=y_dense_test*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
    df_val_pred['NN-Dense']=y_dense_val*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']
    df_train_pred['NN-Dense']=y_dense_train*train_df_std['heat-lag-0']+train_df_mean['heat-lag-0']



    df_out=data_gene_dict(LAG_DICT1, SHIFT_HEAT1)




    keys=df_out.keys()

    from sklearn.utils import shuffle
    N_total1 = len(df_out)
    N2=int(int(N_total1*0.75/24)*24);
    df_out1=df_out[0:N2]

    # df_out1 = df_out1.sample(frac=1).reset_index(drop=True)
    N_train1=int(int(0.7*len(df_out1)/24)*24)

    train_df1=df_out1.iloc[0:N_train1,:]
    val_df1 = df_out1.iloc[N_train1:,:]

    test_df1 = df_out.iloc[N2:,:]
    NUM_FEATURES=df_out1.shape[1]

    train_df1_mean = train_df1.mean()
    train_df1_std = train_df1.std()

    train_df1 = ((train_df1 - train_df1_mean) / train_df1_std)
    val_df1 = ((val_df1 - train_df1_mean) / train_df1_std)
    test_df1 = ((test_df1 - train_df1_mean) / train_df1_std)

    print(train_df1.shape)
    train_df1 = add_dummy_hour_flip(train_df1)
    val_df1 = add_dummy_hour_flip(val_df1)
    test_df1 = add_dummy_hour_flip(test_df1)


    # print(train_df1.shape)

    N_mid=int((train_df1.shape[1]-1)/24)

    # print("N_mid:{}".format(N_mid))
    # print("df_out1 shape:{}".format(df_out1.shape))

    X_train1=train_df1.iloc[:,1:].to_numpy().reshape(-1, N_mid, 24);
    X_train1=np.swapaxes(X_train1, 1, 2);
    X_train1=np.flip(X_train1, 1);
    y_train1=train_df1.iloc[:,0].to_numpy()
    X_val1=val_df1.iloc[:,1:].to_numpy().reshape(-1, N_mid, 24);
    X_val1=np.swapaxes(X_val1, 1, 2);
    X_val1=np.flip(X_val1, 1);
    y_val1=val_df1.iloc[:,0].to_numpy();
    X_test1=test_df1.iloc[:,1:].to_numpy().reshape(-1, N_mid, 24);
    X_test1=np.swapaxes(X_test1, 1, 2);
    X_test1=np.flip(X_test1, 1)
    y_test1=test_df1.iloc[:,0].to_numpy()



    lstm_model, y_lstm_train, y_lstm_val, y_lstm_test= LSTM(X_train1, X_val1, X_test1, y_train1, y_val1, y_test1)
    df_test_pred_norm['LSTM']=y_lstm_test
    df_val_pred_norm['LSTM']=y_lstm_val
    df_train_pred_norm['LSTM']=y_lstm_train

    df_test_pred['LSTM']=y_lstm_test*train_df1_std['heat-lag-0']+train_df1_mean['heat-lag-0']
    df_val_pred['LSTM']=y_lstm_val*train_df1_std['heat-lag-0']+train_df1_mean['heat-lag-0']
    df_train_pred['LSTM']=y_lstm_train*train_df1_std['heat-lag-0']+train_df1_mean['heat-lag-0']
    #

    #
    conv_model, y_conv_train, y_conv_val, y_conv_test=CNN(X_train1, X_val1, X_test1, y_train1, y_val1, y_test1)

    df_test_pred_norm['CNN']=y_conv_test
    # df_val_pred_norm['CNN']=y_conv_val
    # df_train_pred_norm['CNN']=y_conv_train


    # since the validation and trainning dataset are shuffled for training
    df_test_pred['CNN']=y_conv_test*train_df1_std['heat-lag-0']+train_df1_mean['heat-lag-0']
    # df_val_pred['CNN']=y_conv_val*train_df1_std['heat-lag-0']+train_df1_mean['heat-lag-0']
    # df_train_pred['CNN']=y_conv_train*train_df1_std['heat-lag-0']+train_df1_mean['heat-lag-0']

    df_test_pred[df_test_pred<=0]=0.



    df_test_rmse=rmse_func(df_test_pred);
    df_test_mae=mae_func(df_test_pred);
    df_test_r2=r2_func(df_test_pred);
    df_test_bias=bias_func(df_test_pred);


    # df_train_array_rmse=rmse_array_func(df_train_pred);
    # df_train_array_mae=mae_array_func(df_train_pred);
    # df_train_array_r2=r2_array_func(df_train_pred);
    # df_train_array_bias=bias_array_func(df_train_pred);
    #
    #
    # df_val_array_rmse=rmse_array_func(df_val_pred);
    # df_val_array_mae=mae_array_func(df_val_pred);
    # df_val_array_r2=r2_array_func(df_val_pred);
    # df_val_array_bias=bias_array_func(df_val_pred);


    df_test_array_rmse=rmse_array_func(df_test_pred);
    df_test_array_mae=mae_array_func(df_test_pred);
    df_test_array_r2=r2_array_func(df_test_pred);
    df_test_array_bias=bias_array_func(df_test_pred);


    # plt.figure();
    # # plt.plot(df_train_rmse, label="Train RMSE")
    # # plt.plot(df_val_rmse, label="Val RMSE")
    # plt.plot(df_test_rmse, label="Test RMSE")
    # plt.legend()
    # plt.figure();
    # # plt.plot(df_train_mae, label="Train MAE")
    # # plt.plot(df_val_mae, label="Val MAE")
    # plt.plot(df_test_mae, label="Test MAE")
    # plt.legend()
    # plt.figure();
    # # plt.plot(df_train_r2, label="Train R2")
    # # plt.plot(df_val_r2, label="Val R2")
    # plt.plot(df_test_r2, label="Test R2")
    # plt.legend()
    # plt.figure();
    # # plt.plot(df_train_bias, label="Train BIAS")
    # # plt.plot(df_val_bias, label="Val BIAS")
    # plt.plot(df_test_bias, label="Test BIAS")
    # plt.legend()
    #
    # df_test_array_rmse.plot()
    # df_test_array_mae.plot()
    # df_test_array_r2.plot()
    # df_test_array_bias.plot()

    return df_test_pred, df_train_pred

# # if __main__=="__main__":
# main()
# df_main()
