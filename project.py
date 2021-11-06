# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 19:25:05 2021

@author: amalj
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date,timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression,Lasso,Ridge

import pmdarima as pmd
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.metrics import mean_squared_error
from math import sqrt

st.title('Stock Analysis.com')

st.text('Analyse your stock price here')



sel_box_var=st.selectbox("Select the stock you want to analyse",['Pfizer','Moderna'],index=0)

if sel_box_var=='Pfizer':
    df = yf.download('PFE', 
                      start= date.today()- timedelta(days = 300), 
                      end=date.today()+ timedelta(days = 1), 
                      progress=False)
   
    st.image('Pfizer.jpg')
    st.video('https://www.youtube.com/watch?v=nt5sgSnyb1s')
    st.text ('Last 7 days stock data given below -')
    st.text(df.tail(7))
    
    st.text('Number of rows and columns used for analysis-')
    st.text(df.shape)
    plt.figure(figsize=(16,8))
    plt.plot(df["Close"],label='Close Price history')
    plt.legend(loc='upper left', fontsize=10)
    plt.title("Historical Close price of Pfizer",size=30)
    fig=st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.text('Correlation Matrix of Infosys')
    st.text(df.corr())
     
    st.text('Heatmap of correlation Matrix')
    plt.figure(figsize=(5,5))
    sns.heatmap(df.corr(), annot=True)
    heatmap=st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
   
     
    sel_box_var=st.selectbox("Select Method you want to analyse the stock",['Linear','Auto ARIMA'],index=0)
    
     
    if sel_box_var=='Linear':
        #split into train and validation
        train = df[:160]
        test = df[160:]
        y_train = train[['Close']]
        x_train = train.drop(['Close','Adj Close'], axis=1)
           
        x_test = test.drop(['Close','Adj Close'], axis=1)
        y_test = test[['Close']]
        model=LinearRegression()
        reg=model.fit(x_train,y_train)
        Y_pred =reg.predict(x_test)
        st.text('intercept='+str(reg.intercept_))
        st.text("coefficients="+str(reg.coef_))
        st.text('Accuracy of model is:'+str(r2_score(y_test,Y_pred)*100)+' %')
        
        test['Predictions'] = 0
        test['Predictions'] = Y_pred
       
       
        train.index = df[:160].index
        test.index = df[160:].index
       
        plt.figure(figsize=(15,15))
        plt.plot(train["Close"],label='Training')
        plt.plot(test[['Close']],label='Actual')
        plt.plot(test[['Predictions']],label='Predicted')
        plt.legend(loc='upper left', fontsize=10)
        plt.title("Actual Vs Predicted Close price",size=30)
        pred=st.pyplot()
    else:
         #split into train and validation
        train = df[:160]
        test = df[160:]
        y_train = train[['Close']]
        x_train = train.drop(['Close','Adj Close'], axis=1)
           
        x_test = test.drop(['Close','Adj Close'], axis=1)
        y_test = test[['Close']]
        
        training = train['Close']
        validation = test['Close']
        #model = auto_arima(training, trace=True, error_action='ignore',seasonal=True, suppress_warnings=True)
        model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
        model.fit(training)
        forecast = model.predict(n_periods=len(test))
        forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
                    #plot
        plt.figure(figsize=(15,15))
        plt.plot(train['Close'],label='training')
        plt.plot(test['Close'], label='actual')
        plt.plot(forecast['Prediction'],label='forecast')
        plt.legend(loc='upper left', fontsize=10)
        arma=st.pyplot()
        st.text("Summary of the model - ")
        st.text(model.summary())
        rmse=sqrt(mean_squared_error(validation, forecast))
        st.text("Root-mean-square deviation - "+str(rmse))
        
    st.text('Pairplot of Pfizer to highlight overall relationship')
    sns.pairplot(df)
    pairplot=st.pyplot()
        
        
else: 
     df = yf.download('MRNA', 
                      start= date.today()- timedelta(days = 300), 
                      end=date.today()+ timedelta(days = 1), 
                      progress=False)
    
     st.image('moderna.jpg')
     st.video('https://www.youtube.com/watch?v=qb-AAvUP6mQ')
     st.text ('Last 7 days stock data given below -')
     st.text(df.tail(7))
     
     st.text('Number of rows and columns used for analysis-')
     st.text(df.shape)
     
     plt.figure(figsize=(16,8))
     plt.plot(df["Close"],label='Close Price history')
     plt.legend(loc='upper left', fontsize=10)
     plt.title("Historical Close price of moderna",size=30)
     fig=st.pyplot()
     
     st.text('Correlation Matrix of Infosys')
     st.text(df.corr())
     
     st.text('Heatmap of correlation Matrix')
     plt.figure(figsize=(5,5))
     sns.heatmap(df.corr(), annot=True)
     heatmap=st.pyplot()
     
     st.set_option('deprecation.showPyplotGlobalUse', False)
     
     sel_box_var=st.selectbox("Select Method you want to analyse the stock",['Linear','Auto ARIMA'],index=0)
     if sel_box_var=='Linear':
         #split into train and validation
         train = df[:160]
         test = df[160:]
         y_train = train[['Close']]
         x_train = train.drop(['Close','Adj Close'], axis=1)
            
         x_test = test.drop(['Close','Adj Close'], axis=1)
         y_test = test[['Close']]
         model=LinearRegression()
         reg=model.fit(x_train,y_train)
         Y_pred =reg.predict(x_test)
         st.text('intercept='+str(reg.intercept_))
         st.text("coefficients="+str(reg.coef_))
         st.text('Accuracy of model is:'+str(r2_score(y_test,Y_pred)*100)+' %')
         
         test['Predictions'] = 0
         test['Predictions'] = Y_pred
        
        
         train.index = df[:160].index
         test.index = df[160:].index
        
         plt.figure(figsize=(15,15))
         plt.plot(train["Close"],label='Training')
         plt.plot(test[['Close']],label='Actual')
         plt.plot(test[['Predictions']],label='Predicted')
         plt.legend(loc='upper left', fontsize=10)
         plt.title("Actual Vs Predicted Close price",size=30)
         pred=st.pyplot()
     else:
          #split into train and validation
         train = df[:160]
         test = df[160:]
         y_train = train[['Close']]
         x_train = train.drop(['Close','Adj Close'], axis=1)
            
         x_test = test.drop(['Close','Adj Close'], axis=1)
         y_test = test[['Close']]
         
         training = train['Close']
         validation = test['Close']
         #model = auto_arima(training, trace=True, error_action='ignore',seasonal=True, suppress_warnings=True)
         model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
         model.fit(training)
         forecast = model.predict(n_periods=len(test))
         forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
                     #plot
         plt.figure(figsize=(15,15))
         plt.plot(train['Close'],label='training')
         plt.plot(test['Close'], label='actual')
         plt.plot(forecast['Prediction'],label='forecast')
         plt.legend(loc='upper left', fontsize=10)
         arma=st.pyplot()
         st.text("Summary of the model - ")
         st.text(model.summary())
         rmse=sqrt(mean_squared_error(validation, forecast))
         st.text("Root-mean-square deviation - "+str(rmse))
     st.text('Pairplot of Moderna to highlight overall relationship')
     sns.pairplot(df)
     pairplot=st.pyplot()
        
