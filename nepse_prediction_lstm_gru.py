import numpy as np
import math as math
import pandas as pd
import datetime as dt
import pandas_datareader as web
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,GRU
import json
from datetime import timedelta, datetime
import mysql.connector
from sqlalchemy import create_engine
import requests

def db_connect():
    database_name = 'stock'  ########## change according to need ########
    db = mysql.connector.connect(host='localhost', user='root', password='', database=database_name)
    return db



def fetch_nepse_data():
    db = db_connect()
    cursor = db.cursor(buffered=True)
    sql = f"select * from historic"
    dsql = f"delete from nepse_prediction_test"
    cursor.execute(dsql)
    cursor.execute(sql)
    datas = cursor.fetchall()
    print("data fetched from database")
    return datas


def data_preprocessing(datas):
    df = pd.DataFrame(data = datas, columns=['Scrip','Time','Close'])
    df['Close'] = df['Close'].astype(float)
    df2 = df.set_index('Time')
    df3=df2[['Close']]
    data = df3.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil( len(dataset) * .7 )
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len , :]
    x_train = []
    y_train = []
    for i in range(70, len(train_data)):
        x_train.append(train_data[i-70:i, 0])
        y_train.append(train_data[i, 0])
        if i<=71:
            print(x_train)
            print(y_train)
            print()
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print("data preprocessing done successfully")
    return x_train,y_train,training_data_len,scaled_data,dataset,scaler,df3,df


def create_train_lstm_model(x_train, y_train, training_data_len, scaled_data,dataset,scaler,df3):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    test_data = scaled_data[training_data_len - 70:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(70, len(test_data)):
        x_test.append(test_data[i - 70:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    new_df = df3.filter(['Close'])
    last_70_days = new_df[-70:].values
    last_70_days_scaled = scaler.transform(last_70_days)
    X_test = []
    X_test.append(last_70_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)
    int_price = pred_price[0][0]
    int_price1 = format(int_price, ".2f")
    price = float(int_price1)
    today_price = df3.tail(1)
    result = today_price.to_string(index=False)
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['lstm_prediction']
    predictions['lstm_prediction'] = predictions['lstm_prediction'].astype(float)
    print("LSTM model created successfully")
    return predictions,last_70_days_scaled,model


def create_train_gru_model(x_train, y_train, training_data_len,scaled_data,dataset,scaler,df3):
    model = Sequential()
    model.add(GRU(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(GRU(100, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    test_data = scaled_data[training_data_len - 70:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(70, len(test_data)):
        x_test.append(test_data[i - 70:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    new_df = df3.filter(['Close'])
    last_70_days = new_df[-70:].values
    last_70_days_scaled = scaler.transform(last_70_days)
    X_test = []
    X_test.append(last_70_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)
    int_price = pred_price[0][0]
    int_price1 = format(int_price, ".2f")
    price = float(int_price1)
    today_price = df3.tail(1)
    result = today_price.to_string(index=False)
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['gru_prediction']
    predictions['gru_prediction'] = predictions['gru_prediction'].astype(float)
    print("GRU model created successfully")
    return predictions,model

def future_value_pred(last_70_days_scaled,model,scaler):
    data_list = last_70_days_scaled.tolist()
    pred_val_list = []
    for i in range(0,8):
        data_list_x = data_list[i:]
        x_input = np.array(data_list_x)
        x_input = x_input.reshape(1,-1,1)
        pred_val = model.predict(x_input)
        data_list_end_ele = pred_val[0].tolist()
        pred_val = scaler.inverse_transform(pred_val)
        pred_val_list.append(float(pred_val[0][0]))
        data_list = data_list + [data_list_end_ele]
    return pred_val_list

def send_prediction_to_db(lstm_df, gru_df, df,ten_day_lstm,ten_day_gru):

    last_thirty_df = df.loc[len(df) - len(df) * 0.3:len(df), ['Time', 'Close']]
    last_thirty_df.reset_index(drop=True, inplace=True)
    final_df = pd.concat([last_thirty_df, lstm_df, gru_df], axis=1, join='inner')
    df_last_date = float(last_thirty_df['Time'].iloc[-1])

    next_seven_days = [] #### contains next seven day date
    one_day_unix_time = 24*60*60
    df_last_date = int(last_thirty_df['Time'].iloc[-1])
    liv = final_df.index.values[-1]
    for i in range(0,8):
        df_last_date = df_last_date + one_day_unix_time
        next_seven_days.append(df_last_date)

    lp = ten_day_lstm
    gp = ten_day_gru
    for date,l,g in zip(next_seven_days,lp,gp):
        liv=liv+1
        final_df.loc[liv] = [date, '0', l,g]


    final_df['Scrip'] = 'Nepse'
    final_df = final_df[['Scrip', 'Time', 'Close', 'lstm_prediction', 'gru_prediction']]
    engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')
    final_df.to_sql(name='nepse_prediction_test', con=engine, if_exists='replace', index=False)
    print("Prediction data sent to nespe_prediction table successfully")

def main():
    df = fetch_nepse_data()
    x_train, y_train, training_data_len, scaled_data, dataset,scaler,df3,df = data_preprocessing(df)
    lstm_df,last_70_days_scaled,lstm_model = create_train_lstm_model(x_train,y_train,training_data_len,scaled_data,dataset,scaler,df3)
    gru_df,gru_model = create_train_gru_model(x_train,y_train,training_data_len,scaled_data,dataset,scaler,df3)
    tdl = future_value_pred(last_70_days_scaled,lstm_model,scaler)
    tdg = future_value_pred(last_70_days_scaled,gru_model,scaler)
    send_prediction_to_db(lstm_df,gru_df,df,tdl,tdg)

def main():
    df = fetch_nepse_data()
    x_train, y_train, training_data_len, scaled_data, dataset,scaler,df3,df = data_preprocessing(df)
    lstm_df,last_70_days_scaled,lstm_model = create_train_lstm_model(x_train,y_train,training_data_len,scaled_data,dataset,scaler,df3)
    gru_df,gru_model = create_train_gru_model(x_train,y_train,training_data_len,scaled_data,dataset,scaler,df3)
    tdl = future_value_pred(last_70_days_scaled,lstm_model,scaler)
    tdg = future_value_pred(last_70_days_scaled,gru_model,scaler)
    send_prediction_to_db(lstm_df,gru_df,df,tdl,tdg)

if __name__ == "__main__":
    #create_nepse_tb()
    #nepse_scrapper()
    main()
