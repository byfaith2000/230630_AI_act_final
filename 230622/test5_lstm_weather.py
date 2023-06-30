import LSTM1

def weatherSetData(weather):
    weather.info()
    weather.loadNconvert_dataset()
    weather.info()
    weather.show_dataset_infos('T (degC)')
    weather.scale_cols = ['p (mbar)', 'T (degC)', 'VPmax (mbar)', 'sh (g/kg)', 'wv (m/s)']
    weather.normalize_dataset()
    weather.feature_cols = ['p (mbar)', 'VPmax (mbar)', 'sh (g/kg)', 'wv (m/s)']
    weather.label_cols = ['T (degC)']
    weather.create_trainNtest_dataset()
    weather.create_model('/home/nano/Desktop/LSTM5')

if __name__=="__main__":
    weather = LSTM1.L('/home/nano/Desktop/LSTM/time_datas/03_Weather Data/03_weather.csv')

    weatherSetData(weather)

    # 1.learning
    weather.train()

    # 2.predict
    #weather.load_weights(MODEL)
    #weather.test()
    weather.performance_evaluation()