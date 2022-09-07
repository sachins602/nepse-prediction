import numpy as np
import math as math
import pandas as pd
import datetime as dt
import pandas_datareader as web
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import json
from datetime import timedelta, datetime
import mysql.connector
from sqlalchemy import create_engine

sectors = ['corporate_debentures_predictions', 'microfinance_predictions', 'commercial_banks_predictions',
           'non_life_insurance_predictions', 'hydro_powers_predictions', 'life_insurance_predictions', 'finance_predictions',
           'tradings_predictions', 'manufacturing_and_processing_predictions', 'investment_predictions', 'hotels_predictions',
           'development_banks_predictions', 'mutual_fund_predictions', 'other_predictions']


db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
cursor = db.cursor(buffered =True)
# database_name = 'stock'
# cursor.execute("CREATE DATABASE "+database_name) #remove this code if you have already created a Db named 'stocks'
for sector in sectors:
   print("DELETE FROM " + sector)
   cursor.execute("DELETE FROM " + sector)
   db.commit()

cursor.close()   

def future_value_pred(last_70_days_scaled,model,scaler):
    data_list = last_70_days_scaled.tolist()
    pred_val_list = []
    for i in range(0,7):
        data_list_x = data_list[i:]
        x_input = np.array(data_list_x)
        x_input = x_input.reshape(1,-1,1)
        pred_val = model.predict(x_input)
        data_list_end_ele = pred_val[0].tolist()
        pred_val = scaler.inverse_transform(pred_val)
        pred_val_list.append(float(pred_val[0][0]))
        data_list = data_list + [data_list_end_ele]
    return pred_val_list


def corporate_debentures_prediction():
   
    corporate_debentures = ['NICAD8283']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in corporate_debentures:
        sql=f"select * from corporate_debentures where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')

        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler)
        
        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='corporate_debentures_predictions', con=engine, if_exists = 'append', index=False)


def microfinance_prediction():
   
    microfinance = ['ACLBSL','ALBSL','CBBL','CLBSL','DDBL','FMDBL','FOWAD','GMFBS','GILB','GBLBS','GLBSL','ILBS','JSLBB','JBLB','KMCDB','KLBSL','LLBS','MLBSL','MSLB','MKLB','MLBS','MERO','MMFDB','MLBBL','NSLB','NLBBL','NICLBSL','NUBL','RMDC','RSDC','SABSL','SDLBSL','SMATA','SLBSL','SKBBL','SMFDB','SMB','SWBBL','SMFBS','SLBBL','USLB','VLBS','WNLB']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in microfinance:
        sql=f"select * from microfinance where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='microfinance_predictions', con=engine, if_exists = 'append', index=False)

        
        
        
def commercial_bank_prediction():
    commercial_banks=['ADBL','BOKL','CCBL','CZBIL','CBL','EBL','GBIME','KBL','LBL','MBL','MEGA','NABIL','NBL','NCCB','SBI','NICA','NMB','PRVU','PCBL','SANIMA','SBL','SCB','SRBL']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in commercial_banks:
        sql=f"select * from commercial_banks where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-30:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='commercial_banks_predictions', con=engine, if_exists = 'append', index=False)    
    
    

def non_life_insurance_prediction():
   
    non_life_insurance = ['AIL','EIC','GIC','HGI','IGI','LGIL','NIL','NICL','NLG','PRIN','PIC','PICL','RBCL','SIC','SGI','SICL','SIL','UIC']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in non_life_insurance:
        sql=f"select * from non_life_insurance where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='non_life_insurance_predictions', con=engine, if_exists = 'append', index=False)

    


def hydro_powers_prediction():
   
    hydro_powers = ['AKJCL','API','AKPL','AHPC','BARUN','BNHC','BPCL','CHL','CHCL','DHPL','GHL','GLH','HDHPC','HURJA','HPPL','JOSHI','KPCL','KKHC','LEC','MBJC','MKJC','MEN','MHNL','NHPC','NHDL','NGPL','NYADI','PMHPL','PPCL','RADHI','RHPL','RURU','SAHAS','SPC','SHPC','SJCL','SSHL','SHEL','SPDL','TPC','UNHPL','UMRH','UMHL','UPCL','UPPER']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in hydro_powers:
        sql=f"select * from hydro_powers where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='hydro_powers_predictions', con=engine, if_exists = 'append', index=False)    
    
    

def life_insurance_prediction():
   
   
    life_insurance = ['ALICL','GLICL','JLI','LICN','NLICL','NLIC','PLI','PLIC','RLI','SLI','SLICL','ULI']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in life_insurance:
        sql=f"select * from life_insurance where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='life_insurance_predictions', con=engine, if_exists = 'append', index=False)    
    
    

def finance_prediction():
   
   
    finance = ['BFC','CFCL','GFCL','GMFIL','GUFL','ICFC','JFL','MFIL','MPFL','NFS','PFL','PROFL','RLFL','SFCL','SIFC']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in finance:
        sql=f"select * from finance where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='finance_predictions', con=engine, if_exists = 'append', index=False)    
    

def tradings_prediction():
   
    
    tradings = ['BBC', 'STC']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in tradings:
        sql=f"select * from tradings where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='tradings_predictions', con=engine, if_exists = 'append', index=False)
    
    
    
    
    
def manufacturing_and_processing_prediction():
   
   
    manufacturing_and_processing = ['BNT', 'HDL', 'SHIVM', 'UNL']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in manufacturing_and_processing:
        sql=f"select * from manufacturing_and_processing where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='manufacturing_and_processing_predictions', con=engine, if_exists = 'append', index=False)    
    
    
    
def investment_prediction():
   
   
    investment = ['CHDC', 'CIT', 'HIDCL', 'NIFRA', 'NRN']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in investment:
        sql=f"select * from investment where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='investment_predictions', con=engine, if_exists = 'append', index=False)
    
    

def hotels_prediction():
   
    
    hotels = ['CGH', 'OHL', 'SHL', 'TRH']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in hotels:
        sql=f"select * from hotels where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='hotels_predictions', con=engine, if_exists = 'append', index=False)    

    
def development_banks_prediction():
   
    
    development_banks = ['CORBL','EDBL','GBBL','GRDBL','JBBL','KSBBL','KRBL','LBBL','MLBL','MDB','MNBBL','NABBC','SAPDBL','SADBL','SHINE','SINDU']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in development_banks:
        sql=f"select * from development_banks where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='development_banks_predictions', con=engine, if_exists = 'append', index=False)
    

       
def mutual_fund_prediction():
   
    
    mutual_fund = ['KEF','LUK','NEF','NIBLPF']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in mutual_fund:
        sql=f"select * from mutual_fund where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='mutual_fund_predictions', con=engine, if_exists = 'append', index=False)



        
def other_prediction():
   
    
    other = ['NTC', 'NRIC']
    db = mysql.connector.connect(host='localhost', user='root', password='', database='stock')
    cursor = db.cursor(buffered=True)
    for bank in other:
        sql=f"select * from other where Scrip='{bank}'"
        cursor.execute(sql)
        result = cursor.fetchall()
       

          
#prediction starts

        df = pd.DataFrame(data = result, columns=['Scrip','Time','Close','Open','High','Low','Volume'])

        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        df2 = df.set_index('Time')
        df2


        df3=df2[['Close']]
        df3


        # plt.figure(figsize=(14,6))
        # plt.title('Closing Price History')
        # plt.plot(df3['Close'])
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price NPR', fontsize=18)
        # plt.show()


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
        x_train.shape

        
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))


        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=10, epochs=50)


        test_data = scaled_data[training_data_len - 70: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(70, len(test_data)):
            x_test.append(test_data[i-70:i, 0])
            
            
        x_test = np.array(x_test)


        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
        rmse




        new_df = df3.filter(['Close'])
        last_70_days = new_df.iloc[-70:].values
        last_70_days_scaled = scaler.transform(last_70_days)
        X_test = []
        X_test.append(last_70_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        #yaa bata gareko bujhdainas taile chodde vai
        int_price = pred_price[0][0]
        int_price1 = format(int_price, ".2f")

        price = float(int_price1)

        today_price = df3.tail(1)
        result = today_price.to_string(index = False)

        # today = datetime.today()
        #today_date = today.strftime("%Y/%m/%d")
        #today = datetime.now() # get date and time today
        #delta = timedelta(days=1) #initialize delta
        #date = today + delta # add the delta days
        #tomorrow_date = date # format it



        predictions = pd.DataFrame(predictions)
        predictions.columns =['Prediction']
        predictions['Prediction'] = predictions['Prediction'].astype(float)

        last_thirty_df = df.loc[len(df)-len(df)*0.3:len(df),['Time','Close']]
        last_thirty_df.reset_index(drop=True,inplace=True)
        final_df = pd.concat([last_thirty_df,predictions],axis=1,join='inner')


        next_seven_days = [] #### contains next seven day date
        one_day_unix_time = 24*60*60
        df_last_date = last_thirty_df['Time'].iloc[-1]
        liv = final_df.index.values[-1]
        for i in range(0,7):
            df_last_date = df_last_date + one_day_unix_time
            next_seven_days.append(df_last_date)
    
        next_seven_days_prediction = future_value_pred(last_70_days_scaled,model,scaler) ###### create yourself
        for date,pred_price in zip(next_seven_days,next_seven_days_prediction):
            liv=liv+1
            final_df.loc[liv] = [date, '0', pred_price]


        final_df['Scrip'] = bank
        final_df = final_df[['Scrip','Time','Close','Prediction']]
        engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/stock')

        #table and columns aafai bancha banaunu pardaina
        final_df.to_sql(name='other_predictions', con=engine, if_exists = 'append', index=False)

        

        

        
if __name__=="__main__":
        corporate_debentures_prediction()
        microfinance_prediction()
        commercial_bank_prediction()
        non_life_insurance_prediction()
        hydro_powers_prediction()
        life_insurance_prediction()
        finance_prediction()
        tradings_prediction()
        manufacturing_and_processing_prediction()
        investment_prediction()
        hotels_prediction()
        development_banks_prediction()
        other_prediction()
        mutual_fund_prediction()
