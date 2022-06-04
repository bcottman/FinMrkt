# !/usr/bin/env python
# -*- coding: utf-8 -*-
__coverage__ = 0.0
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
import warnings

warnings.filterwarnings("ignore")

import math
from pandas import to_datetime
from statistics import stdev
from pandas_datareader.data import DataReader


### pasoDecorators class
# adapted from pandas-flavor 11/13/2019
from pandas.api.extensions import register_dataframe_accessor
from functools import wraps

from pandas.core.dtypes.generic import ABCDataFrame
def register_DataFrame_method(method):
    """Register a function as a method attached to the Pandas DataFrame.
    Example
    -------
    for a function
        @pf.register_dataframe_method
        def row_by_value(df, col, value):
        return df[df[col] == value].squeeze()

    for a class method
        @pf.register_dataframe_accessor('Aclass')
        class Aclass(object):

        def __init__(self, data):
        self._data

        def row_by_value(self, col, value):
            return self._data[self._data[col] == value].squeeze()
    """

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


def p_or_r_std(
    asset_value_type, EFT_df, column_name, ticker, rd_type, start_date, end_date
):
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
