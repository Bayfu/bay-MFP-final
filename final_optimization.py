# Jiabei Fu
# Financial programming final project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxopt import matrix, solvers
import scipy as sp
from  scipy.optimize import  minimize
from scipy import linalg as la

#real return
stock_return = pd.read_csv('/Users/fujiabei/Desktop/first_semester/financial_programming/final_project/factor_data/etf_ret.csv')
stock_return.index = stock_return['Date']
stock_return = stock_return.iloc[:,1:]
stock_return = 100 * stock_return

#predicted return
ret_pred = pd.read_csv('/Users/fujiabei/Desktop/first_semester/financial_programming/final_project/predicted_return/2020_2022.csv')
ret_pred.index = ret_pred['Date']
ret_pred = ret_pred.iloc[:, 1:]

# method to deal with the non-full rank covriance matrix
def shrinkage(cov, F, delta):
    for i in range(np.shape(cov)[0]):
        cov[i] = delta * cov[i] + (1 - delta) * F
    return cov

# use the data of past 90 days to calculate covariance matrix
window = 90

# position
weight_daily = []
# name of stocks held
stock_sel_daily = []
for i in range(np.shape(ret_pred)[0]):   
    
    exp_ret_cur = ret_pred.iloc[i]
    # top 10 highest predicted return
    top10_cur = exp_ret_cur.sort_values().iloc[-10:]
    # date
    date_cur = top10_cur.name
    # the stock selected(highest 10)
    stock_sel = top10_cur.index
    # mathc the data in the real return matrix
    date_index = np.where(stock_return.index == date_cur)[0][0]
    #calculate the covariance matrix
    his_data = stock_return.loc[:, stock_sel].iloc[date_index - window : date_index]
    cov_cur = his_data.cov()
    # shrinkage: keep the diagonal elements unchanged
    F = np.diag(cov_cur)
    delta = 0.8
    cov_shrinkage_cur = delta * cov_cur + (1 - delta) * F
    ones = np.ones(len(stock_sel))
    
    # if the predicted yearly return is greater than 18%, we consider it as a bullish market
    # pursue the maxium Sharpe ratio
    if np.mean(exp_ret_cur) > (1.18 ** (1/252) - 1) * 100:
        # max sharpe ration to pursue more profit
        obj = lambda w: -(w.T @ np.array(top10_cur) / (w.T @ cov_shrinkage_cur @ w))
        con = ({'type':'eq','fun':lambda w : 1 - ones @ w}
               )
        bd=[(0,1) for w in range(10)]
        res=minimize(obj,[0.1 for w in range(10)],method = 'SLSQP' , constraints = con, bounds=bd, tol = 1e-2)
        weight_cur = res.x
    # if the predicted yearly return is between 0%-18%, we minimize the volatility 
    # but also want to achieve certain return
    elif np.mean(exp_ret_cur) > 0:
        obj = lambda w: w.T @ cov_shrinkage_cur @ w
        con = ({'type':'eq','fun':lambda w : 1 - ones @ w},
               {'type':'eq','fun':lambda w : np.array(top10_cur) @ w - (1.10 ** (1/252) - 1) * 100}
              )
        bd=[(0,1) for w in range(10)]
        res=minimize(obj,[0 for w in range(10)],method = 'SLSQP' , constraints = con, bounds=bd, tol = 1e-2)
        weight_cur = res.x
    # if the predicted yearly return is negative, we consider it as a bearish market
    # what we want is just minimize the volatility
    else:
        obj = lambda w: w.T @ cov_shrinkage_cur @ w
        con = {'type':'eq','fun':lambda w : 1 - ones @ w}
        bd=[(0,1) for w in range(10)]
        res=minimize(obj,[0 for w in range(10)],method = 'SLSQP' , constraints = con, bounds=bd, tol = 1e-2)
        weight_cur = res.x
    #record the daily postion and stocks held
    weight_daily.append(weight_cur)
    stock_sel_daily.append(stock_sel)


'''calculate the portfolio return'''
date_begin = ret_pred.index[0]
stock_ret_real = stock_return.loc[date_begin:]
portfolio_ret = []
for i in range(len(weight_daily)):
    real_ret = stock_ret_real[stock_sel_daily[i]].iloc[i]
    portfolio_ret.append(np.array(weight_daily[i]) @ np.array(real_ret))
    


portfolio_ret = np.array(portfolio_ret)
portfolio_ret[np.where(abs(portfolio_ret) > 3)] = 0
portfolio_ret = pd.DataFrame(portfolio_ret, index = stock_ret_real.index, columns = ['portfolio return'])
portfolio_ret_cumprod = (portfolio_ret / 100 +1).cumprod()


''' compare with the SPY'''
# benchmark
SPY = pd.read_csv('/Users/fujiabei/Desktop/first_semester/financial_programming/final_project/SPY.csv')[['Date', 'Adj Close']]
SPY.index = SPY['Date']
SPY = SPY.iloc[:,1]    
SPY_2022 = SPY.loc['2020-01-02': '2022-10-07'] / SPY.loc['2020-01-02']  
SPY_2022 = SPY_2022.fillna(method = 'ffill')

portfolio_ret_cumprod['SPY'] = SPY_2022
portfolio_ret_cumprod = portfolio_ret_cumprod.fillna(method = 'ffill')

portfolio_ret_cumprod.plot()
plt.title('2020-2022')
plt.show()

   
portfolio_ret_cumprod.to_csv('/Users/fujiabei/Desktop/first_semester/financial_programming/final_project/predicted_return/portfolio_return_2020_2022.csv', encoding='utf_8_sig')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

