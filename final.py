# Jiabei Fu
# Financial Engineering project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
path = './project_data'
files = os.listdir(path)
files.remove('.DS_Store')

# merge all adj prices data into a dataframe
Adj_price = []
for i in range(len(files)):
    cur = pd.read_csv(path +'/' + files[i]).iloc[:, [0, 5]]
    cur.columns = ['Date', files[i][:-4]]
    Adj_price.append(cur)
price_total = Adj_price[0]
for i in range(len(Adj_price) - 1):
    price_total = pd.merge(price_total, Adj_price[i + 1], how = 'outer', on = 'Date')
   
   
# delete the duplicate values
price_total = price_total[price_total['Date']>'2015-01-05'].sort_values(by = 'Date')
unique_index = [0]
for i in range(len(price_total) - 1):
    if price_total['Date'].iloc[i+1] != price_total['Date'].iloc[i]:
        unique_index.append(i+1)
price_total = price_total.iloc[unique_index]
price_total.index = price_total['Date']
price_total = price_total.iloc[:,1:]

# merge all volume data into a dataframe
vol = []
for i in range(len(files)):
    cur = pd.read_csv(path +'/' + files[i]).iloc[:, [0, -1]]
    cur.columns = ['Date', files[i][:-4]]
    vol.append(cur)
vol_total = vol[0]
for i in range(len(vol) - 1):
    vol_total = pd.merge(vol_total, vol[i + 1], how = 'outer', on = 'Date')
   
# delete the duplicate values
vol_total = vol_total[vol_total['Date']>'2015-01-05'].sort_values(by = 'Date')
unique_index = [0]
for i in range(len(vol_total) - 1):
    if vol_total['Date'].iloc[i+1] != vol_total['Date'].iloc[i]:
        unique_index.append(i+1)
vol_total = vol_total.iloc[unique_index]
vol_total.index = vol_total['Date']
vol_total = vol_total.iloc[:,1:]


price_total = price_total.fillna(method = 'ffill')
vol_total = vol_total.fillna(0)





'''
the aim of following is to synthesize financial factors
'''
# calculate the daily return
stock_return = (price_total / price_total.shift(1)).dropna() - 1


# calculate short term factors
std_20 = stock_return.rolling(window = 20, center = False, axis = 0).std() #std of 20 days
mean_20 = stock_return.rolling(window = 20, center = False, axis = 0).mean() #average of 20 days
R = 0.07 # American interest without any risk
Sharpe_20 = (mean_20 - R) / mean_20#Sharpe ratio of 20 days
skew_20 = stock_return.rolling(window = 20, center = False, axis = 0).skew() #skewness of 20 days
kurt_20 = stock_return.rolling(window = 20, center = False, axis = 0).kurt() #kurtness of 20 days
bolling_up_20 = (price_total.rolling(window = 20,center = False, axis = 0).mean() + 2*price_total.rolling(window = 20,center = False,axis = 0).std())/np.array(price_total)
bolling_down_20 = (price_total.rolling(window = 20,center = False, axis = 0).mean() - 2*price_total.rolling(window = 20,center = False,axis = 0).std())/np.array(price_total)
MAC_20 = price_total.rolling(window = 20, center = False, axis = 0).mean() / np.array(price_total)#MAC of 20 days

#calculate long term factors
std_120 = stock_return.rolling(window = 120, center = False, axis = 0).std() #std of 120 days
mean_120 = stock_return.rolling(window = 120, center = False, axis = 0).mean() #average of 120 days
R = 0.07 # American interest without any risk
Sharpe_120 = (mean_120 - R) / mean_120#Sharpe ratio of 20 days
skew_120 = stock_return.rolling(window = 120, center = False, axis = 0).skew() #skewness of 120 days
kurt_120 = stock_return.rolling(window = 120, center = False, axis = 0).kurt() #kurtness of 120 days
bolling_up_120 = (price_total.rolling(window = 120,center = False, axis = 0).mean() + 2*price_total.rolling(window = 120,center = False,axis = 0).std())/np.array(price_total)
bolling_down_120 = (price_total.rolling(window = 120,center = False, axis = 0).mean() - 2*price_total.rolling(window = 120,center = False,axis = 0).std())/np.array(price_total)
MAC_120 = price_total.rolling(window = 120, center = False, axis = 0).mean() / np.array(price_total)#MAC of 120 days



