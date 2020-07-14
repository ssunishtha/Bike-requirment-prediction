# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:49:06 2020

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
bike=pd.read_csv('hour.csv')

bikes_prep = bike.copy()
bikes_prep = bikes_prep.drop(['index','date','casual','registered'])

bikes_prep.isnull().sum()

bikes_prep.hist(rwidth=0.9)
plt.tight_layout()

plt.subplot(2,2,1)
plt.title('temperature vs demand')
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],s=2,c='g')


plt.subplot(2,2,2)
plt.title('atemp vs demand')
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s=2, c='b')

plt.subplot(2,2,3)
plt.title('humidity vs demand')
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'],s=2, c='m')

plt.subplot(2,2,4)
plt.title('windspeed vs demand')
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'],s=2, c='c')

plt.tight_layout()

bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05,0.1,o.15,0.9,0.95,0.99])

correlation = bikes_prep[['temp', 'atemp', 'humidity', 'windspeed','demand']].corr()

bikes_prep=bikes_prep_drop(['weekday','year','workingday','atemp','windspeed'],axis=1)
df1 = pd.to_numeric(bikes_prep['demand'],downcast='float')
plt.accor(df1,maxlags=12)


cat_list = bikes_prep['season'].unique()
cat_average=bikes_prep.groupby('season').mean()['demand']
plt.bar(cat_list, cat_average)
cat_list = bikes_prep['season'].unique()
cat_average=bikes_prep.groupby('season').mean()['demand']
plt.bar(cat_list, cat_average)
colors=['g','r','m','b']
plt.subplot(3,3,1)
plt.title('average demand per season')
cat_list = bikes_prep['season'].unique()
cat_average=bikes_prep.groupby('season').mean()['demand']
plt.bar(cat_list, cat_average,color=colors)
bikes_prep['demand'].describe()
df1=bikes_prep['demand']
df2=np.log(df1)
plt.figure()
df1.hist(rwidth=0.9,bins=20)

plt.figure()
df2.hist(rwidth=0.9,bins=20)
bikes_prep_lag['season']=bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday']=bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather']=bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month']=bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour']=bikes_prep_lag['hour'].astype('category')

bikes_prep_lag = pd.get_dumies(bikes_prep_lag, drop_first=True)
bikes_prep_lag.dtypes
bikes_prep_lag['season']=bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday']=bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather']=bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month']=bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour']=bikes_prep_lag['hour'].astype('category')

bikes_prep_lag = pd.get_dumies(bikes_prep_lag, drop_first=True)
bikes_prep_lag=pd.concat([bikes_prep,t_1,t_2,t_3],axis=1)

bikes_prep_lag=bikes_prep_lag.dropna()


bikes_prep_lag.dtypes
bikes_prep_lag['season']=bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday']=bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather']=bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month']=bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour']=bikes_prep_lag['hour'].astype('category')

bikes_prep_lag = pd.get_dumies(bikes_prep_lag, drop_first=True)
t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns= ['t-1']

t_2 = bikes_prep['demand'].shift(+1).to_frame()
t_2.columns= ['t-2']

t_3 = bikes_prep['demand'].shift(+1).to_frame()
t_3.columns= ['t-3']

bikes_prep_lag=pd.concat([bikes_prep,t_1,t_2,t_3],axis=1)

bikes_prep_lag=bikes_prep_lag.dropna()


bikes_prep_lag.dtypes
bikes_prep_lag['season']=bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday']=bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather']=bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month']=bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour']=bikes_prep_lag['hour'].astype('category')

bikes_prep_lag = pd.get_dumies(bikes_prep_lag, drop_first=True)
bikes_prep_lag = pd.get_dummies(bikes_prep_lag, drop_first=True)
y=bikes_prep_lag[['demand']]
x=bikes_prep_lag.drop(['demand'],axis=1)
tr_size=0.7 * len(x)
tr_size=0.7 * len(x)
tr_size=int(tr_size)
x_train=x.values[0:tr_size]
x_test=x.values[tr_size:len(x)]

y_train=y.values[0:tr_size]
y_test=y.values[tr_size:len(y)]

from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()
std_reg.fit(x_train,y_train)
r2_train=std_reg.score(x_train,y_train)
r2_test=std_reg.score(x_test,y_test)
y_predict = std_reg.predict(x_test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test,y_predict))
y_test_e = []
y_predict_e = []

for i in range(0,len(y_test)):
    
    y_test_e.append(math.exp(y_test[i]))
    y_predict_e.append(math.exp(y_predict[i]))
    
log_sq_sum = 0.0
    
for i in range(0.len(y_test_e)):
    log_a = math.log(y_test_e[i] + 1)
    log_p = math.log(y_predict_e[i] + 1)
    log_diff =(log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff
    
rmsle = math.sqrt(log_sq_sum/len(y_test))
print(rmsle)
