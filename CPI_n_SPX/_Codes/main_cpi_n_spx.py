# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:15:09 2024

@author: drgpap
"""

import os
import pathlib2 as pl
import re
from itertools import combinations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.impute import SimpleImputer

data_folder = "D:/user/George/Data/Kaggle/CPI_n_SPX/"

data_exts = [".csv"]

date_col_srch_pattern = 'DATE'

def uniform_format(date_str: str, split_char: str) -> str:
    str_vec = re.split(split_char, date_str)
    str_vec_new = str_vec[:]
    if len(str_vec[-1])==2:
        str_vec_new[-1] = f"20{str_vec[-1]}"
    str_new = '-'.join(str_vec_new)
    return str_new 

if __name__ == '__main__':
    data_files_dct = dict()
    data_df_dct = dict()
    for ext in data_exts:
        data_files_dct[ext] = list(pl.Path(data_folder).glob(f"*{ext}"))
        for one_file in data_files_dct[ext]:
            rootname = os.path.splitext(os.path.basename(one_file))[0]
            data_df_dct[rootname] = pd.read_csv(one_file)
            col_names = data_df_dct[rootname]
            date_cols = list(filter(lambda x: re.search(date_col_srch_pattern, x, re.IGNORECASE), 
                                    col_names)) 
            if len(date_cols)>0:
                for dc in date_cols:
                    data_df_dct[rootname].loc[:,dc] = data_df_dct[rootname].loc[:,dc].apply(lambda x: uniform_format(x,'/'))

                    data_df_dct[rootname].loc[:,dc] = pd.to_datetime(data_df_dct[rootname].loc[:,dc], 
                                                                     format="%m-%d-%Y", errors='coerce')

                    data_df_dct[rootname][dc] = pd.to_datetime(data_df_dct[rootname].loc[:,dc], 
                                                                     errors='coerce')
                         
                    data_df_dct[rootname] = data_df_dct[rootname].dropna(subset=[dc]).copy()
                    is_in_order = data_df_dct[rootname][dc].is_monotonic_increasing
                    if not is_in_order:
                        print("Dates not in order")
                    
    data_df_dct['SP500']['rets_c2c'] = data_df_dct['SP500'].loc[:,'close'].pct_change(periods=1)
    data_df_dct['SP500']['rets_c2c'] = data_df_dct['SP500']['rets_c2c'].fillna(value=0)
    data_df_dct['SP500']['cum_rets_c2c'] = (1 + data_df_dct['SP500']['rets_c2c']).cumprod()

    data_df_dct['CPI']['Date'] = pd.to_datetime(data_df_dct['CPI'][['Year','Month','Day']])
    cpi_old_cols = data_df_dct['CPI'].columns.to_list()
    cpi_new_cols = list(map(lambda x: f"cpi_{x}", cpi_old_cols))
    data_df_dct['CPI'] = data_df_dct['CPI'].rename(columns=dict(zip(cpi_old_cols,cpi_new_cols)))
    
    data_df = data_df_dct['SP500'].merge(data_df_dct['CPI'], how='left', 
                                         left_on='Date', right_on='cpi_Date')
    data_df[cpi_new_cols] = data_df[cpi_new_cols].fillna(method='ffill')
    data_df = data_df.dropna(subset=['cpi_Date'])
    data_df['cpi_lag'] = data_df['Date'] - data_df['cpi_Date']
    #
    # Stationarity test ------------------
    adf_test_result = adfuller(data_df['rets_c2c'])
    
    df_statistic, p_value, n_lags, n_obs, critical_values, ic_best = adf_test_result
    print(f'ADF Statistic: {df_statistic}')
    print(f'p-value: {p_value}')
    print(f'Num Lags: {n_lags}')
    print(f'Num Observations used: {n_obs}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')
     # Determine stationarity based on the p-value
    if p_value <= 0.05:
        print("The series is likely stationary.")
    else:
        print("The series is likely non-stationary.")
    
    # check for cointegration -----------
    coint_test_cols = ['open','high','low','close']
    pairwise_combinations = list(combinations(coint_test_cols, 2))
    coint_t = dict()
    coint_p_value = dict() 
    coint_crit_value = dict()
    for one_pair in pairwise_combinations:
        coint_t[one_pair], coint_p_value[one_pair], coint_crit_value[one_pair] = \
                coint(data_df[one_pair[0]], data_df[one_pair[1]])
        if coint_p_value[one_pair]<0.05:
            print(f"Series {one_pair[0]} and {one_pair[1]} are likely cointegrated.")
        else:
            print(f"Series {one_pair[0]} and {one_pair[1]} are likely not cointegrated.")    
    # check for outliers ----------------
    data_df.describe()
    
    # plot the cumulative returns
    plt.figure(1)
    sns.jointplot(data=data_df, x='Date', y='close')
    plt.show()
    #
    plt.figure(2)
    sns.boxplot(data_df, y='rets_c2c')
    plt.show()
    #
    plt.figure(3)
    sns.histplot(data_df['rets_c2c'])
    plt.show()
    #
    plt.figure(4)
    sns.jointplot(data=data_df, x='cpi_Date', y='cpi_Actual')
    plt.show()
    #
    plt.figure(5)
    sns.jointplot(data=data_df, x='cpi_Actual', y='rets_c2c', kind='reg',
                  color="m")
    #

    X = sm.add_constant(data_df['cpi_Actual'])
    Y = data_df.loc[:,'rets_c2c']
    model_ols_01 = sm.OLS(Y, X).fit()
    print(model_ols_01.summary())
    #
    data_df_lag = dict()
    model_ols_dct = dict()
    model_ols_summary_dct = dict()
    pvalues_summary_01 = dict()
    for lag in data_df['cpi_lag'].sort_values().unique():
        data_df_lag[lag] = data_df.loc[data_df['cpi_lag']==lag,:]
        X = sm.add_constant(data_df_lag[lag]['cpi_Actual'])
        Y = data_df_lag[lag]['rets_c2c']
        if (Y.shape[0]>5):
            model_ols_dct[lag] = sm.OLS(Y,X).fit()
            model_ols_summary_dct[lag] = model_ols_dct[lag].summary()
            pvalues_summary_01[lag] = model_ols_dct[lag].pvalues['cpi_Actual']
            print (lag)
            print (model_ols_summary_dct[lag])
        else:
            model_ols_summary_dct[lag] = None
            pvalues_summary_01[lag] = np.nan
    
    pvalues_summary_01_df = pd.DataFrame.from_dict(pvalues_summary_01, orient='index')  

    #
    X = sm.add_constant(data_df['cpi_Diff_with_prev'])
    Y = data_df.loc[:,'rets_c2c']
    model_ols_02 = sm.OLS(Y, X).fit()
    print(model_ols_02.summary())
    #
    data_df_lag_02 = dict()
    model_ols_02_dct = dict()
    model_ols_02_summary_dct = dict()
    pvalues_summary_02 = dict()
    for lag in data_df['cpi_lag'].sort_values().unique():
        data_df_lag[lag] = data_df.loc[data_df['cpi_lag']==lag,:]
        X = sm.add_constant(data_df_lag[lag]['cpi_Diff_with_prev'])
        Y = data_df_lag[lag]['rets_c2c']
        if (Y.shape[0]>5):
            model_ols_02_dct[lag] = sm.OLS(Y,X).fit()
            model_ols_02_summary_dct[lag] = model_ols_02_dct[lag].summary()
            pvalues_summary_02[lag] = model_ols_02_dct[lag].pvalues['cpi_Diff_with_prev']
            print (lag)
            print (model_ols_02_summary_dct[lag])
        else:
            model_ols_02_summary_dct[lag] = None
            pvalues_summary_02[lag] = np.nan   
            
    pvalues_summary_02_df = pd.DataFrame.from_dict(pvalues_summary_02, orient='index')
    
    