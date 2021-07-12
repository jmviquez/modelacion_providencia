# -*- coding: utf-8 -*-
"""
Techniques for model evaluation

@author: zaula
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% Hydrological indexes

def monthly_series(data, method="sum"):
    if  method.lower() == "sum":
        data_m = data.resample("1M").sum()
    else:
        data_m = data.resample("1M").mean()
    data_m.index = data_m.index - pd.offsets.MonthBegin()
    return data_m
    

def daily_mean(data):
    """
    Computes the daily mean time serie

    INPUTS:
        data:      (Serie, DataFrame) input timesieries

    OUTPUTS:
        result:    (Serie, DataFrame) mean daily time serie
    """
    return data.groupby(data.index.day).mean()


def monthly_mean(data, method="sum"):
    """
    Computes the monthly mean time serie

    INPUTS:
        data:      (Serie, DataFrame) input timesieries
        method:    (string) aggregation method to compute monthly time series
                          "mean" or "sum". By default "sum"

    OUTPUTS:
        result:    (Serie, DataFrame) mean monthly time serie
    """
    method = method.lower()
    if method == "suma":
        data_m = data.resample("1M").sum()
    else:
        data_m = data.resample("1M").mean()
    return data_m.groupby(data_m.index.month).mean()


def flow_duration_curve(qt):
    """
    Returns the flow duration curve from an input serie
    
    INPUTS:
        qt:     (Serie) input flow Series
    
    OUTPUTS:
        fdc:    (Series) output flow duration curve
    """
    qt_sort = qt.sort_values(ascending=False)
    n       = len(qt_sort)   # numero de elementos en qt
    qt_sort.index = np.arange(n) / n  # asignamos probabilidad empirica
    return qt_sort


#%% Graphics tools

def plot_model_evaluation(prec, qobs, qsim, date1, date2, date3, figsize=None):
    """
    Generate an evaluation plot using daily precipitation, daily streamflow, mean monthly streamflow,
    flow duration curve and scatterplot

    INPUTS:
        prec:   (Serie) input daily precipitation
        qobs:   (Serie) input observed daily streamflow
        qsim:   (Serie) input simulated daily streamflow
        date1:  (string) initial date for calibration
        date2:  (string) final date for calibration and initial date for validation
        date3:  (string) final date for validation
    """
    if figsize is None:
        figsize = (10, 8)

    # Sort data
    data  = pd.concat((prec, qobs, qsim), axis=1)
    data.columns = ["prec", "qobs", "qsim"]
    prec = data.loc[date1:date3, "prec"]
    # Daily data
    qobs1 = data.loc[date1:date2, "qobs"]
    qsim1 = data.loc[date1:date2, "qsim"]
    qobs2 = data.loc[date2:date3, "qobs"]
    qsim2 = data.loc[date2:date3, "qsim"]
    # Monthly data
    qobsm1 = monthly_mean(qobs1, method="mean")
    qsimm1 = monthly_mean(qsim1, method="mean")
    qobsm2 = monthly_mean(qobs2, method="mean")
    qsimm2 = monthly_mean(qsim2, method="mean")
    # Flow duration curve
    qobs_fdc1 = flow_duration_curve(qobs1)
    qsim_fdc1 = flow_duration_curve(qsim1)
    qobs_fdc2 = flow_duration_curve(qobs2)
    qsim_fdc2 = flow_duration_curve(qsim2)

    # Create Axis
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, :])   # precipitation axis
    ax2 = fig.add_subplot(gs[1, :])   # streamflow axis
    ax3 = fig.add_subplot(gs[2, 0])   # mean monthly streamflow
    ax4 = fig.add_subplot(gs[2, 1])   # flow duration curve
    ax5 = fig.add_subplot(gs[2, 2])   # streamflow correlation

    # plot precipitation
    prec.plot(color="#212f3c", ax=ax1)
    ax1.set_ylabel("Precipitacion (mm)", fontsize=12)
    ax1.set_xlabel("")

    # plot streamflow
    qobs1.plot(color="#212f3c", label="Obs calibcación", ax=ax2)
    qsim1.plot(color="#c0392b", label="Sim calibcación", ax=ax2)
    qobs2.plot(color="#212f3c", label="Obs validación", ax=ax2)
    qsim2.plot(color="#1f618d", label="Sim validación", ax=ax2)
    ax2.set_ylabel("Caudal (m$^3$/s)", fontsize=12)
    ax2.set_xlabel("")

    # monthly streamflow
    qobsm1.plot(color="#212f3c", label="Obs calibración", ax=ax3)
    qsimm1.plot(color="#c0392b", label="Sim calibración", ax=ax3)
    qobsm2.plot(color="k", label="Obs validación", ax=ax3)
    qsimm2.plot(color="#1f618d", label="Sim validación", ax=ax3)
    ax3.set_ylabel("Caudal (m$^3$/s/mes)", fontsize=12)
    ax3.set_xlabel("")
    ax3.set_xticks(np.arange(1, 13))

    # flow duration curve
    qobs_fdc1.plot(color="#212f3c", label="Obs calibración", ax=ax4)
    qsim_fdc1.plot(color="#c0392b", label="Sim calibración", ax=ax4)
    qobs_fdc2.plot(color="k", label="Obs validación", ax=ax4)
    qsim_fdc2.plot(color="#1f618d", label="Sim validación", ax=ax4)
    ax4.set_ylabel("Caudal (m$^3$/s)", fontsize=12)
    ax4.set_xlabel("")
    ax4.set_yscale("log")

    # graficar correlacion
    ax5.plot(qobs1, qsim1, '.', color="#212f3c", label="Calibración", alpha=0.5)
    ax5.plot(qobs2, qsim2, '.', color="#1f618d", label="Validación", alpha=0.5)
    
    vmax = max(data.loc[date1:date3, "qobs"].max(), data.loc[date1:date3, "qsim"].max())
    ax5.plot([0, vmax], [0, vmax], 'k', label="linea 1:1")
    ax5.set_xlim(0, vmax)
    ax5.set_ylim(0, vmax)
    ax5.set_xlabel("Caudal obs (m$^3$/s)", fontsize=12)
    ax5.set_ylabel("Caudal sim (m$^3$/s)", fontsize=12)

    fig.tight_layout()

    return fig, [ax1, ax2, [ax3, ax4, ax5]]


def plot_correlation(yobs, ysim, log=False, **kwargs):
    """
    Create a correlation plot
    """
    
    data = pd.concat((yobs, ysim), axis=1).dropna(how="any")
    if "ax" not in kwargs:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (4, 4)))
    ax.plot(data.iloc[:, 0], data.iloc[:, 1], '.', color="#212f3c", alpha=0.5)

    vmax = data.max().max()
    ax.plot([0, vmax], [0, vmax], 'r')
    ax.set_xlabel(data.columns[0], fontsize=12)
    ax.set_ylabel(data.columns[1], fontsize=12)
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()
    return ax


