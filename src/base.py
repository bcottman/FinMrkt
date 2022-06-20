# !/usr/bin/env python
# -*- coding: utf-8 -*-
__coverage__ = 0.0
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
import warnings

warnings.filterwarnings("ignore")

import math


from pandas import to_datetime, DataFrame
from statistics import stdev
from pandas_datareader.data import DataReader


### pasoDecorators class
# adapted from pandas-flavor 11/13/2019
from pandas.api.extensions import register_dataframe_accessor
from functools import wraps

from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.api.extensions import register_dataframe_accessor
import numpy as np
from functools import wraps
from numba import jit

import matplotlib
import matplotlib.dates as mdates

import matplotlib.pyplot as plt


import seaborn as sns; sns.set()  # for plot stylin

def register_DataFrame_method(method):
    def inner(*args, **kwargs):
        class AccessorMethod(object):
            def __init__(self, pandas_obj):
                self._obj = pandas_obj

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)
        register_dataframe_accessor(method.__name__)(AccessorMethod)
        return method
    return inner()


def FutureValue_(principle=0, deposit=0, withdrawal=0, return_=0, nperiod=1):
    if (principle + deposit + withdrawal) == 0:
        return 0
    if return_ <= 0:
        return principle + deposit + withdrawal  # default is end of period
    amount = deposit - withdrawal
    cumamount = 0
    for n in range(1, nperiod + 1):
        #        print( 'before cumamount',cumamount)
        cumamount = cumamount + deposit * (1 + return_) ** n
    #        print( 'after cumamount',cumamount)
    return principle * (1 + return_ ** nperiod) + cumamount


def nppy_(EFT_df):
    t_dayspy = 252
    t_weekspy = 53
    t_monthsspy = 12
    t_ypy = 1

    nlen = len(EFT_df.index.day)
    ndays = ((EFT_df.index[-1] - EFT_df.index[0])).days + 1
    nweeks = ndays // 7
    nmonths = ndays // 30.5
    if ndays >= nlen:
        return t_dayspy
    elif ndays >= nlen:
        return t_monthsspy
    elif nlen >= nmonths:
        return t_monthsspy
    elif nlen >= nweeks:
        return t_weekspy
    else:
        return t_ypy
    
def return_fit(nperiod: int = 1, principle: float = 0.0, return_: float = 0.0) -> float:
    if (principle) == 0:
        return 0
    if return_ <= 0:
        return principle  # default is end of period
    return principle * (1 + return_) ** nperiod



def price_std(EFT_df, column_name, ticker, rd_type, start_date, end_date):

    EFT_df = DataReader(ticker, rd_type, start=start_date, end=end_date)
    EFT_df.index = to_datetime(EFT_df.index, format="%Y-%m-%d")

    return stdev(EFT_df[column_name])


@register_DataFrame_method
@jit
def price_std_f(EFT_df, column_name, ticker, rd_type, start_date, end_date):

    EFT_df = DataReader(ticker, rd_type, start=start_date, end=end_date)
    EFT_df.index = to_datetime(EFT_df.index, format="%Y-%m-%d")

    return stdev(EFT_df[column_name])


def p_or_r_std(
    EFT_df, asset_value_type, rd_type, column_name, ticker,
    start_date, end_date):
    EFT_df = DataReader(ticker, rd_type, start=start_date, end=end_date)
    EFT_df.index = to_datetime(EFT_df.index, format="%Y-%m-%d")
    nppy = nppy_(EFT_df)
    if asset_value_type.lower() == "price":

        ave_price = EFT_df[column_name].mean()
        return (ave_price * nppy, statistics.stdev(EFT_df[column_name]))
    else:
        EFT_df["return"] = (
            EFT_df[column_name] - EFT_df[column_name].shift(1)
        ) / EFT_df[column_name].shift(1)

        ave_return = (EFT_df["return"]).mean(skipna=True)

        return (ave_return * nppy, EFT_df["return"].std(skipna=True))

@register_DataFrame_method
@jit
def p_or_r_std_f(
    EFT_df, asset_value_type, rd_type, column_name, ticker, 
    start_date, end_date):
    EFT_df = DataReader(ticker, rd_type, start=start_date, end=end_date)
    EFT_df.index = to_datetime(EFT_df.index, format="%Y-%m-%d")
    nppy = nppy_(EFT_df)
    if asset_value_type.lower() == "price":

        ave_price = EFT_df[column_name].mean()
        return (ave_price * nppy, statistics.stdev(EFT_df[column_name]))
    else:
        EFT_df["return"] = (
            EFT_df[column_name] - EFT_df[column_name].shift(1)
        ) / EFT_df[column_name].shift(1)

        ave_return = (EFT_df["return"]).mean(skipna=True)

        return (ave_return * nppy, EFT_df["return"].std(skipna=True))


##  Annotate
def ticker_annotate(axis, ticker, principal, ave_return, return_fit, error_):
    astr = "TICKER= {:}\nPRINCIPAL= ${:.2f} \nAveRETURN= {:.2%}/year\
    \nFitted\RETURN= {:.2%}/year\nReturnError= {:.2%}/year"\
    .format(ticker,principal, ave_return, return_fit, error_) 
    axis.annotate(
        astr,
        xy=(0.5, 0.5), xycoords=axis.transAxes,
        xytext=(-90, 12), textcoords='offset points',
        size=20,
        bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),)
    
def plot_ticker(df, nper, xddata, y_data, yfit_data, ticker, popt, ave_return, nppy, error_):
    
    # trend line, high error, low error
    df['y_fit'] = yfit_data
    yfit_data_low =  yfit_data *  (1 - error_/ave_return)
    df['y_fit_low'] = yfit_data_low
    yfit_data_high = yfit_data * (1 + error_/ave_return)
    df['y_fit_high'] = yfit_data_high
    
    sns.set(rc={'figure.figsize':(12, 6)})
    fig, axis = plt.subplots(figsize=(12,6))
    fig.autofmt_xdate(rotation=45)
    #
    ntm = 19
    interv = nper//ntm  # interval between 25 tick marks
    if interv < 1: interv = 1
        # weekly instead of yearly x axis labels
    if (nppy == 52): 
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=interv))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
#    monthly instead of yearly x axis labels
    elif (nppy == 12): 
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interv))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    # We change the fontsize of minor ticks label 
    
    
    axis.tick_params(axis='both', which='major', labelsize=14)
    # plot raw data
    axis.plot(xddata, y_data, color='black', label='yield')
    # plot fitted yield trend line
    axis.plot(xddata, yfit_data, color='tab:blue', label='yield fit')
    # plot fitted yield lowabs error trend line
    axis.plot(xddata, yfit_data_low, color='tab:blue', alpha=0.1)
    # plot fitted yield high error trend line
    axis.plot(xddata, yfit_data_high, color='tab:blue', alpha=0.1)
    # plot error band
    axis.fill_between(xddata, yfit_data_low, yfit_data_high, alpha=0.2)
    axis.set_ylabel('price($)',size=20)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


    ticker_annotate(axis, ticker, popt[0], ave_return,  popt[1]*nppy, error_)
    
from scipy.optimize import curve_fit

def peb(df, ticker, column_name, nppy, ave_return, error_):
    # nppy: number of periods per year
    xddata = df.index  #dates array for axis
    nper = df.index.size # number of periods
    xdata = np.linspace(1,nper,num=nper).astype(int)
    df['ID'] = xdata
    y_data = df[column_name].to_numpy()

             
    # Compute upper and lower bounds using chosen uncertainty measure: here
    # it is a fraction of the standard deviation of measurements at each
    # time point based on the unbiased sample variance
    lp = df[column_name][0]
    hp = 1.1*lp ; ly =0.05; hy = 0.50
    low_bounds = [lp,ly/nppy] ; high_bounds = [hp, hy/nppy]
    popt, pcov = curve_fit(return_fit, xdata, y_data, method='trf', 
                           bounds=(low_bounds, high_bounds))

    yfit_data= return_fit(xdata, *popt)
    plot_ticker(df, nper, xddata, y_data, yfit_data,
    ticker, popt, ave_return, nppy, error_)
    plt.show()
    
    
def ticket_read(start_date, end_date, ticker, rd_type = 'yahoo' ):
    # GET PRICE DATA
    asset_value_type = 'return'
    EFT_df = DataReader(ticker, rd_type, start = start_date, end = end_date)
    EFT_df.index = to_datetime(EFT_df.index, format ='%Y-%m-%d')
    # determine nummber of peruods per year from 
    nppy = nppy_(EFT_df)
    # BIND OTHER VARIABLES
    column_name =  'Adj Close'
    # RETURNS
    ave_return, r_std = p_or_r_std(EFT_df, asset_value_type, rd_type, column_name,ticker, 
     start_date, end_date)
    
    peb(EFT_df, ticker, column_name, nppy, ave_return, r_std)