# -*- coding: utf-8 -*-
"""
Modello Idrologico Lumped in Continuo (MILC)

Rain-Runoff Model for daily simulation



Author:
    Saul Arciniega Esparza
    Faculty of Engineering, UNAM, Mexico
    zaul.ae@gmail.com
    
    
Reference:
    Lucca Broca (luca.brocca@irpi.cnr.it)
    
    Brocca, L., Liersch, S., Melone, F., Moramarco, T., Volk, M. (2013).
    Application of a model-based rainfall-runoff database as efficient tool for flood risk management.
    Hydrology and Earth System Sciences Discussion, 10, 2089-2115.
"""

#%% Import libraries
import os
import numpy as np
import pandas as pd
from math import factorial

import warnings
warnings.filterwarnings("ignore")


# Load IUH data
IUH_DATA = np.loadtxt(os.path.join(os.path.dirname(__file__), 'IUH.txt'))


#%% Main class. Hydrological model

class MILc(object):
    
    def __init__(self, area=100, lat=0, params=None):
        """
        
        INPUTS:
            area    >     [float] catchment area in km2
            lat     >     [float] catchment latitude at centroid
            params  >     [dict] model parameters
        
        Model Parameters
            gamma   >     [float] routing coefficient for HUI lag-time relationship
            w0      >     [float] initial water content as a fraction of W_max (-)
            wmax    >     [float] maximum water content (mm)
            m       >     [float] drainage exponent of percolation (-)
            ks      >     [float] satured hydraulic conductivity (mm/h)
            nu      >     [float] fraction of drainage vs interflow (-)
            alpha   >     [float] runoff exponent (-)
        """
        
        self.area = area    # catchment area in km2
        self.lat  = lat     # catchment latitude
        
        self.params = {
            "gamma": 1.0,   # routing coefficient for HUI lag-time relationship
            "w0": 0.5,      # initial water content as a fraction of W_max (-)
            "wmax": 1.0,    # maximum water content (mm)
            "m": 1.0,       # drainage exponent of percolation (-)
            "ks": 1.0,      # satured hydraulic conductivity (mm/h)
            "kc": 1.0,      # crop coefficient
            "nu": 1.0,      # fraction of drainage vs interflow (-)
            "alpha": 1.0    # runoff exponent (-)
        }
        
        if params is not None:
            for key, value in params.items():
                key = key.lower()
                if key in self.params:
                    self.params[key] = value
    
    def __repr__(self):
        return "MILC.hydrological.model"
    
    def __str__(self):
        text = "\n\n______________MILC structure______________\n"
        text += "Catchment properties:\n"
        text += "    Area (km2): {:.3f}\n".format(self.area)
        text += "    Latitude  : {:.4f}\n".format(self.lat)
        text += "Model Parameters:\n"
        text += "    gamma > routing coefficient (adim)               : {:.3f}\n".format(self.params["gamma"])
        text += "    w0    > Initial Water Content (adim)             : {:.3f}\n".format(self.params["w0"])
        text += "    wmax  > Maximum Water Capacity (mm)              : {:.3f}\n".format(self.params["wmax"])
        text += "    alpha > Runoff parameter (adim)                  : {:.3f}\n".format(self.params["alpha"])
        text += "    kc    > Vegetation/Crop Coeficient (adim)        : {:.3f}\n".format(self.params["kc"])
        text += "    m     > Drainage exponent (adim)                 : {:.3f}\n".format(self.params["m"])
        text += "    ks    > Satured hydraulic conductivity (mm/hr)   : {:.3f}\n".format(self.params["ks"])
        text += "    nu    > Fraction of drainage vs interflow (adim) : {:.3f}\n".format(self.params["nu"])
        return text
    
    def compute_eto(self, tmin, tmax, lat, doy, tmean=None, radiation=None):
        """
        Reference evapotranspiration over grass (ETo) using the Hargreaves
        equation
        """
        if tmean is None:
            tmean = (tmax + tmin) / 2.0
        lat_rad = degrees2rad(lat)
        if radiation is None:
            radiation = et_radiation(lat_rad, doy)
    
        return 0.0023 * (tmean + 17.8) * (tmax - tmin) ** 0.5 * 0.408 * radiation

    def compute_pet(self, eto):
        """
        Potential evapotranspiration (PET) using the Allen method
        """
        pet = eto * self.params["kc"]
        return pet
    
    def compute_et(self, w, pet):
        """
        Compute evpotranspiration

        """
        return max(0.0, pet * w / self.params["wmax"])
    
    def compute_runoff(self, prec, infil):
        """
        Compute runoff
        """
        return max(0.0, prec - infil)
    
    def compute_infiltration(self, w, prec):
        """
        Compute infiltlration
        """
        infil = max(0.0, prec * (1. - (w / self.params["wmax"]) ** self.params["alpha"]))
        # Check soil saturation
        if infil + w >= self.params["wmax"]:
            water_excess = infil + w - self.params["wmax"]
            infil -= water_excess
        return infil

    def compute_baseflow(self, w):
        """
        Compute baseflow
        """
        baseflow = (1 - self.params["nu"]) * self.params["ks"] * 24.0 * (w / self.params["wmax"]) ** self.params["m"]
        return max(0.0, baseflow)
    
    def compute_percolation(self, w):
        """
        Compute percolation
        """
        perc = self.params["nu"] * self.params["ks"] * 24.0 * (w / self.params["wmax"]) ** self.params["m"]
        return max(0.0, perc)
    
    def compute_water_content(self, w):
        """
        Compute Water Content
        """
        return w / self.params["wmax"]
    
    def convolution_giuh(self, runoff, baseflow, dt):
        """
        Compute streamflow hydrograph at the catchment outlet 
        using Geomorphological Instantaneos Unit Hydrograph for runoff and
        Nash Instantaneos Unit Hydrograph for baseflow.
        
        Inputs:
            runoff     >   [array] runoff serie
            baseflow   >   [array] baseflow serie
            dt         >   [float] computational time step for flood
                               event simulation, in hours
            delta_T    >   [float] input time step of the time series
        """
        delta_T = 24  # time delta in hours
        area = self.area
        gamma = self.params["gamma"]
        n = len(runoff)
        
        # Compute runoff IUH
        IUH1 = iuh_comp(gamma, area, dt, delta_T) * dt
        IUH1 /= np.sum(IUH1)
        
        # Compute baseflow IUH
        IUH2 = iuh_nash(1.0, 0.5*gamma, area, dt, delta_T) * dt
        IUH2 /= np.sum(IUH2)
        
        # Convolution to compute hydrographs
        qs_int = np.interp(np.arange(0, n, dt), np.arange(n), runoff)
        bf_int = np.interp(np.arange(0, n, dt), np.arange(n), baseflow)
        
        temp1 = np.convolve(IUH1, qs_int)
        temp2 = np.convolve(IUH2, bf_int)
        
        # Compute total flow and baseflow
        dt1 = np.round(1. / dt)
        idx = np.arange(0, n * dt1, dt1, dtype=int)
        
        factor = area * 1000. / delta_T / 3600.
        
        qd = temp1[idx] / factor  # routed runoff
        qb = temp2[idx] / factor  # routed baseflow
        
        return qd, qb
    
    def water_balance(self, prec, pet):
        """
        Compute the Water Balance
        """
        
        # Initial parameters
        n = len(prec)
        w  = self.params["w0"] * self.params["wmax"]  # water storage in mm
        # ks * 24              # convert mm/hr to mm     
        
        # Create empty arrays
        runoff   = np.zeros(n, dtype=np.float32)   # direct flow
        baseflow = np.zeros(n, dtype=np.float32)   # baseflow
        et       = np.zeros(n, dtype=np.float32)   # evapotranspiration
        infil    = np.zeros(n, dtype=np.float32)   # infiltration
        perc     = np.zeros(n, dtype=np.float32)   # percolation
        ww       = np.zeros(n, dtype=np.float32)   # water content
        
        for t in range(n):
            # Surface processes
            infil[t]  = self.compute_infiltration(w, prec[t])
            runoff[t] = self.compute_runoff(prec[t], infil[t])
            w += infil[t]  # update Water Storage
            # Subsurface processes
            et[t] = self.compute_et(w, pet[t])
            w -= et[t]  # update Water Storage
            # Deep processes
            baseflow[t] = self.compute_baseflow(w)
            perc[t]     = self.compute_percolation(w)
            w -= baseflow[t] + perc[t]  # update Water Storage
            # Compute Water Content
            ww[t] = self.compute_water_content(w)
        
        return runoff, baseflow, et, infil, perc, ww
    
    def run(self, forcings, start=None, end=None, save_state=False, dt=0.2, **kwargs):
        """
        Run the MILC model
        

        Parameters
        ----------
        forcings : DataFrame
            Input data with columns prec (precipitation), tmin, tmax, and
            pet(potential evapotranspiration, optional)
        start : string, optional
            Start date for simulation in format. Example: '2001-01-01'
        end : string, optional
            End date for simulation in format. Example: '2010-12-31'
        save_state : bool, optional
            If True (default), last storage is saved as w0 parameter
        dt : float, optional
            Time step in hours to compute IUH
        **kwargs :
            Model parameters can be changed for the simulation
                area    >     [float] catchment area in km2
                gamma   >     [float] routing coefficient for HUI lag-time relationship
                w0      >     [float] initial water content as a fraction of W_max (-)
                wmax    >     [float] maximum water content (mm)
                m       >     [float] drainage exponent of percolation (-)
                ks      >     [float] satured hydraulic conductivity (mm/h)
                nu      >     [float] fraction of drainage vs interflow (-)
                alpha   >     [float] runoff exponent (-)

        Returns
        -------
        Simulations : DataFrame
            Qt       > Streamflow (Qd+Qb) at catchment output (m3/s)
            Qd       > Directflow at catchment output (m3/s)
            Qb       > Baseflow at catchment output (m3/s)
            pet      > Potential Evapotranspiration in mm
            et       > Evapotranspiration in mm
            runoff   > Runoff in mm
            baseflow > Baseflow in mm
            infil    > Infiltration in mm
            perc     > Percolation in mm
            ww       > Water Content adimensional (w/wmax)
        """
        if start is None:
            start = forcings.index.min()
        if end is None:
            end = forcings.index.max()
        forcings = forcings.loc[start:end, :]
        
        # load new parameters
        self.area = kwargs.get("area", self.area)
        self.lat  = kwargs.get("lat", self.lat)
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value 
        
        # Create Output DataFrame
        output_vars = ["Qt", "Qd", "Qb", "pet", "et", "runoff", "baseflow", "infil", "perc", "ww"]
        rows = forcings.shape[0]
        cols = len(output_vars)
        outputs = pd.DataFrame(
            np.zeros((rows, cols), dtype=np.float32),
            index=forcings.index,
            columns=output_vars
        )
        
        # Get Forcings
        prec = forcings["prec"].values
        tmin = forcings["tmin"].values
        tmax = forcings["tmax"].values
        doy  = forcings.index.dayofyear.values  # day of year
        eto  = self.compute_eto(tmin, tmax, self.lat, doy)
        pet  = self.compute_pet(eto)
        
        # Compute Water Balance
        runoff, baseflow, et, infil, perc, ww = self.water_balance(prec, pet)
        
        # Flow routing using IUH
        qd, qb = self.convolution_giuh(runoff, baseflow, dt)
        
        # Save final storage state
        if save_state:
            self.params["w0"] = ww[-1] / self.params["wmax"]
        
        # Save Outputs
        outputs.loc[:, "Qt"]       = qd + qb
        outputs.loc[:, "Qd"]       = qd
        outputs.loc[:, "Qb"]       = qb
        outputs.loc[:, "pet"]      = pet
        outputs.loc[:, "et"]       = et
        outputs.loc[:, "runoff"]   = runoff
        outputs.loc[:, "baseflow"] = baseflow
        outputs.loc[:, "infil"]    = infil
        outputs.loc[:, "perc"]     = perc
        outputs.loc[:, "ww"]       = ww
    
        return outputs


#%% Evapotranspiration functions
SOLAR_CONSTANT = 0.0820  # [MJ m-2 min-1]


def degrees2rad(degrees):
    return degrees * (np.pi / 180.0)


def rad2degrees(radians):
    return radians * (180.0 * np.pi)


def sun_dec(doy):
    return 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)


def sunset_hour_angle(latitude, sun_decline):
    cos_sha = - np.tan(latitude) * np.tan(sun_decline)
    out_var = np.zeros_like(cos_sha)
    for i in range(len(cos_sha)):
        out_var[i] = np.arccos(min(max(cos_sha[i], -1.0), 1.0))
    return out_var


def inv_rel_dist_earth_sun(doy):
    return 1.0 + (0.033 * np.cos(2.0 * np.pi / 365.0 * doy))


def et_radiation(latitude, doy):
    """
    Global radiation computed for Hargreaves method
    """
    sd = sun_dec(doy)
    sha = sunset_hour_angle(latitude, sd)
    ird = inv_rel_dist_earth_sun(doy)

    tmp1 = (24.0 * 60.0) / np.pi
    tmp2 = sha * np.sin(latitude) * np.sin(sd)
    tmp3 = np.cos(latitude) * np.cos(sd) * np.sin(sha)
    return tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)


#%% Runoff Functions

def iuh_comp(gamma, area, dt, delta_T):
    """
    Calculation of geomorphological Instantaneos Unit Hydrograph
    using the geomorphological approach of Gupta et al. (1980)
    
    Inputs:
        gamma      >   [float] coefficient lag-time relationship
                           Lag = gamma * 1.19 * area ^ 0.33
        area       >   [float] basin area in squared kilometers
        dt         >   [float] computational time step for flood
                           event simulation, in hours
        delta_T    >   [float] input time step of the time series
    
    Outputs:
        IUH        >   [array] output Instaneous Unit Hydrograph
    """
    lag = (gamma * 1.19 * area ** 0.33) / delta_T
    hp = 0.8 / lag
    t = IUH_DATA[:,0] * lag
    IUH0 = IUH_DATA[:,1] * hp
    t_i = np.arange(0, np.max(t), dt)
    IUH = np.interp(t_i, t, IUH0)
    return IUH


def iuh_nash(n, gamma, area, dt, delta_T):
    """
    Nash Instantaneos Unit Hydrograph used for baseflow routine
    
    Inputs:
        n          >   [float] model parameter
        gamma      >   [float] coefficient lag-time relationship
                           Lag = gamma * 1.19 * area ^ 0.33
        area       >   [float] basin area in squared kilometers
        dt         >   [float] computational time step for flood
                           event simulation, in hours
        delta_T    >   [float] input time step of the time series
    """
    
    K = (gamma * 1.19 * area ** 0.33) / delta_T
    t = np.arange(0, 100, dt)
    IUH = (t/K) ** (n-1) * np.exp(-t/K) / factorial(n-1) / K
    
    return IUH 

