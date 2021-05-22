import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import stock_pred as sp

app = dash.Dash()
server = app.server

def predictPrice(file_name="INE883A01011"):
    abs_path = "CSV/"+file_name+".csv"
    try:
        df_nse = pd.read_csv(abs_path)
    except FileNotFoundError as e:
        return {"error_code":"102","message":"File Not Found","data":[]}
        
    sp.createModel(file_name)
    scaler=MinMaxScaler(feature_range=(0,1))
    df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%d-%b-%Y")
    df_nse.index=df_nse['Date']


    data=df_nse.sort_index(ascending=True,axis=0)
    new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close Price'])
    for i in range(0,len(data)):
        new_data["Date"][i]=data['Date'][i]
        new_data["Close Price"][i]=data["Close Price"][i]
    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)

    dataset=new_data.values

    train=dataset[0:240,:]
    valid=dataset[240:,:]

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    x_train,y_train=[],[]

    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
        
    x_train,y_train=np.array(x_train),np.array(y_train)

    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    model_path = "Models/"+file_name+".h5"
    model=load_model(model_path)

    inputs=new_data[len(new_data)-len(valid)-60:].values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)

    X_test=[]
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    train = new_data[:240]
    valid = new_data[240:]
    valid['Predictions'] = closing_price
    predicted_list = []
    for index, row in valid.iterrows():
        # print(index.timestamp())
        # print("index",index)
        # print(row['Close'],row['Predictions'],row['Date'])
        predicted_list.append({"Date":index.timestamp(),"Close":row['Close Price'],"Predictions":row['Predictions']})
    print(predicted_list)
    return {"error_code":"100","message":"Record Found","data":predicted_list}
