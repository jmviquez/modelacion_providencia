# -*- coding: utf-8 -*-
"""
Metrics for model validation

@author: zaula
"""

import numpy as np
import pandas as pd


#%% Metrics functions

def correlation(yobs, ysim):
    """
    Pearson correlation
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    return data.corr().iloc[1, 0]


def mean_absolute_error(yobs, ysim):
    """
    Nash Sutcliffe Efficiency Criteria (NSE)
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    return (np.abs(data.iloc[:, 0] - data.iloc[:, 0]) / data.shape[0]).sum()


def root_mean_square_error(yobs, ysim):
    """
    Root Mean Square Error (RMSE)
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    return np.sum((data.iloc[:, 0] - data.iloc[:, 1]) ** 2.0 / data.shape[0]) ** 0.5


def nash_sutcliffe_efficiency(yobs, ysim):
    """
    Nash Sutcliffe Efficiency Criteria (NSE)
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    rm = data.iloc[:, 0].mean()
    part1 = ((data.iloc[:, 1] - data.iloc[:, 0]) ** 2).sum()
    part2 = ((data.iloc[:, 0] - rm) ** 2).sum()

    return 1.0 - part1 / part2


def kling_gupta_efficiency(yobs, ysim):
    """
    kling Gupta Efficiency Criteria (KGE)
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    mean = data.mean().values
    std = data.std().values
    cc = data.corr().iloc[0, 1]
    part1 = (cc - 1.0) ** 2.0
    part2 = (std[1] / std[0] - 1.0) ** 2.0
    part3 = (mean[1] / mean[0] - 1.0) ** 2.0
    return 1 - (part1 + part2 + part3) ** 0.5


def bias(yobs, ysim):
    """
    Bias respect mean value

    bias = 1.0 - ((mean_sim / mean_obs - 1.0) ** 2) ** 0.5
    """
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    mean = data.mean().values
    part = (mean[1] / mean[0] - 1.0) ** 2.0
    return 1 - part ** 0.5

