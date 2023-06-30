from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.layers import LSTM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def normalize_dataset(df, scale_cols):
    scaler = StandardScaler()

    df_scaled = scaler.fit_transform(df[scale_cols])

    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.column = scale_cols
    return df_scaled

def make_dataset(winSize, feature, label):
    feature_list = []
    label_list = []
    for i in range(len(feature) - winSize):
        feature_list.append(np.array(feature.iloc[i:i+winSize]))
        label_list.append(np.array(label.iloc[i+winSize]))
    return np.array(feature_list), np.array(label_list)

def create_trainNvalid_dataset(winSize, trainDF, testDF, feature_cols, label_cols, only_train=False, test_dataset=None):
    train_dataset = trainDF
    train_feature = train_dataset[feature_cols]
    train_label = train_dataset[label_cols]

    train_feature, train_label = make_dataset(winSize, train_feature, train_label)
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

    test_dataset = testDF
    test_feature = test_dataset[feature_cols]
    test_label = test_dataset[label_cols]

    test_feature, test_label = make_dataset(winSize, test_feature, test_label)
    return x_train, x_valid, y_train, y_valid, train_feature, train_label, test_feature, test_label

def create_model(WINDOW_SIZE, feature_cols):
    #model_path
    model = Sequential()
    model.add(LSTM(16, input_shape=(WINDOW_SIZE, len(feature_cols)), activation='relu', return_sequences=False))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer='adam')
    return model

def test(model, test_feature, test_label):
    pred = model.predict(test_feature)
    plt.figure(figsize=(12,9))
    plt.plot(test_label, label='actual')
    plt.plot(pred, label='prediction')
    plt.legend()
    plt.show()

#/home/nano/time_datas/05_Pump/over100mb

if __name__=='__main__':

    # pumpDF = pd.read_csv('/home/nano/time_datas/05_Pump/over100mb/pump_sensor.csv')
    # loadNconvert_dataset(pumpDF)
    # show_dataset_infos(pumpDF, 'machine_status')

    # pumpDF = pumpDF.drop('sensor_15',axis=1)
    # pumpDF = pumpDF.fillna(0)

    # pumpDF = pumpDF.sort_values(by='Date', ascending=True)

    # df_masking(pumpDF, 2, 3)
    # df_normal(pumpDF, 1)
    # devidePUMP_TrainNTest()

    trainDF = pd.read_csv('/home/nano/time_datas/05_Pump/trainDF.csv')
    testDF = pd.read_csv('/home/nano/time_datas/05_Pump/testDF.csv')
    scale_cols = ['sensor_00','sensor_01','sensor_02','sensor_03','sensor_04','sensor_05','sensor_06','sensor_07','sensor_08','sensor_09','sensor_10','sensor_11','sensor_12','sensor_13','sensor_14','sensor_16','sensor_17','sensor_18','sensor_19','sensor_20','sensor_21','sensor_22','sensor_23','sensor_24','sensor_25','sensor_26','sensor_27','sensor_28','sensor_29','sensor_30','sensor_31','sensor_32','sensor_33','sensor_34','sensor_35','sensor_36','sensor_37','sensor_38','sensor_39','sensor_40','sensor_41','sensor_42','sensor_43','sensor_44','sensor_45','sensor_46','sensor_47','sensor_48','sensor_49','sensor_50','sensor_51','Month','Day','Hour','machine_status']

    feature_cols = ['sensor_00','sensor_01','sensor_02','sensor_03','sensor_04','sensor_05','sensor_06','sensor_07','sensor_08','sensor_09','sensor_10','sensor_11','sensor_12','sensor_13','sensor_14','sensor_16','sensor_17','sensor_18','sensor_19','sensor_20','sensor_21','sensor_22','sensor_23','sensor_24','sensor_25','sensor_26','sensor_27','sensor_28','sensor_29','sensor_30','sensor_31','sensor_32','sensor_33','sensor_34','sensor_35','sensor_36','sensor_37','sensor_38','sensor_39','sensor_40','sensor_41','sensor_42','sensor_43','sensor_44','sensor_45','sensor_46','sensor_47','sensor_48','sensor_49','sensor_50','sensor_51','Month','Day','Hour']
    label_cols = ['machine_status']
    trainDF = normalize_dataset(trainDF, scale_cols)
    testDF = normalize_dataset(testDF, scale_cols)

    WinSize = 30
    x_train, x_valid, y_train, y_valid, train_feature, train_label, test_feature, test_label = create_trainNvalid_dataset(WinSize, trainDF, testDF, testDF, feature_cols, label_cols)

    model_path = 'Path/model'
    model = create_model(WinSize, featrue_cols)

    #train(model_path, model, x_train, y_train, x_valid, y_valid, epochs=500, batch_size=128)

    model_name = 'epoch_0314.h5'
    model.load_weights(os.path.join(model_path, model_name))
    test(model.test_feature, test_label)
    performance_evaluation(model, test_feature, test_label)