# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
modele du Genie Rural a 4 parametres Journalier (GR4j)

Rain-Runoff Model



Author:
    Saul Arciniega Esparza
    Faculty of Engineering, UNAM, Mexico
    zaul.ae@gmail.com
    
    
Based on:
    Andrew MacDonald (andrew@maccas.net)
    https://github.com/amacd31/pygr4j
    
    
Reference:
    Irstea (formerly Cemagref), Hydrosystems Research Unit, Antony, France
    Charles Perrin (charles.perrin@irstea.fr),
    Vazken Andreassian (vazken.andreassian@irstea.fr)
"""

#%% Import libraries
import os
import numpy as np
import pandas as pd
import math
from math import tanh

import warnings
warnings.filterwarnings("ignore")


#%% Main class. Hydrological model

class GR4J(object):
    
    def __init__(self, area=100, lat=0, params=None):
        """
        GR4J hydrological model for daily simulation
        
        
        Inputs:
            area    >     [float] catchment area in km2
            lat     >     [float] catchment latitude at centroid
            params  >     [dict] model parameters
        
        Model Parameters
            ps0     >     [float] initial production storage as a fraction ps0=(ps/X1)
            rs0     >     [float] initial routing storage as a fraction pr0=(rs/X3)
            x1      >     [float] maximum production capacity (mm)
            x2      >     [float] discharge parameter (mm)
            x3      >     [float] routing maximum capacity (mm)
            x4      >     [float] delay (days)
        """
        
        self.area = area    # catchment area in km2
        self.lat  = lat     # catchment latitude
        
        self.params = {
            "ps0": 1.0,
            "rs0": 0.5,
            "x1": 1.0,
            "x2": 1.0,
            "x3": 1.0,
            "x4": 1.0,
        }
        
        if params is not None:
            for key, value in self.params.items():
                self.params[key] = params.get(key, value)
    
    def __repr__(self):
        return "GR4J.hydrological.model"
    
    def __str__(self):
        text = "\n\n______________GR4J structure______________\n"
        text += "Catchment properties:\n"
        text += "    Area (km2): {:.3f}\n".format(self.area)
        text += "    Latitude  : {:.4f}\n".format(self.lat)
        text += "Model Parameters:\n"
        text += "    x1  > Maximum production capacity (mm)     : {:.3f}\n".format(self.params["x1"])
        text += "    x2  > Discharge parameter (mm)             : {:.3f}\n".format(self.params["x2"])
        text += "    x3  > Routing maximum capacity (mm)        : {:.3f}\n".format(self.params["x3"])
        text += "    x4  > Delay (days)                         : {:.3f}\n".format(self.params["x4"])
        text += "    ps0 > Initial production storage (psto/x1) : {:.3f}\n".format(self.params["ps0"])
        text += "    rs0 > Initial routing storage (rsto/x3)    : {:.3f}\n".format(self.params["rs0"])
        return text
    
    @staticmethod
    def global_radiation(lat, doy):
        """
        Global radiation computed for Oudin method
        """
        glob_rad = np.zeros(len(doy), dtype=np.float32)
        
        for i in range(len(doy)):
            teta  = 0.4093 * np.sin(doy[i] / 58.1 - 1.405)
            cosgz = max(0.001, np.cos(lat / 57.3 - teta))
            cosom = max(
                -1.0,
                min(
                    1.0 - cosgz / np.cos(lat / 57.3) / np.cos(teta),
                    1.0
                )
            )
            om = np.arccos(cosom)
            eta = 1.0 + np.cos(doy[i] / 58.1) / 30.0
            cospz = cosgz + np.cos(lat / 57.3) * np.cos(teta) * (np.sin(om) / om - 1.0)
            glob_rad[i] =  446 * om * eta * cospz
        
        return glob_rad
    
    def compute_pet(self, tmean, doy, lat):
        """
        Estimate potential evapotranspiration (PET) using the Oudin method
        
        Inputs:
            tmean   >  [array] input mean daily temperature in degrees 
            doy     >  [array] day of year for each tmean record
            lat     >  [float] latitude
        """
        radiation = self.global_radiation(lat, doy)
        pet = radiation * (tmean + 5.0) / 28.5 / 100.0
        pet[pet < 0.] = 0.0 
        return pet
    
    def reservoirs_evaporation(self, prec, pet, ps):
        """
        Estimate net evapotranspiration and reservoir production
        
        Inputs:
            tmean   >  [array] input mean daily temperature in degrees 
            doy     >  [array] day of year for each tmean record
            lat     >  [float] latitude
        """
        if prec > pet:
            evap = 0.
            snp = (prec - pet) / self.params["x1"]  # scaled net precipitation
            snp = min(snp, 13.)
            tsnp = tanh(snp) # tanh_scaled_net_precip
            # reservoir production
            res_prod = ((self.params["x1"] * (1. - (ps/ self.params["x1"]) ** 2.) * tsnp)
                        / (1. + ps / self.params["x1"] * tsnp))
            # routing pattern
            rout_pat = prec - pet - res_prod
        else:
            sne = (pet - prec) / self.params["x1"]  # scaled net evapotranspiration
            sne = min(sne, 13.)
            tsne = tanh(sne)  # tanh_scaled_net_evap
            ps_div_x1 = (2. - ps / self.params["x1"]) * tsne
            evap = ps * ps_div_x1 / (1. + (1. - ps / self.params["x1"]) * tsne)
            
            res_prod = 0  # reservoir_production
            rout_pat = 0  # routing_pattern
        
        return evap, res_prod, rout_pat
    
    def s_curves1(self, t):
        """
        Unit hydrograph ordinates for UH1 derived from S-curves.
        """
        if t <= 0:
            return 0
        elif t < self.params["x4"]:
            return (t / self.params["x4"]) ** 2.5
        else: # t >= x4
            return 1

    def s_curves2(self, t):
        """
        Unit hydrograph ordinates for UH2 derived from S-curves.
        """
        if t <= 0:
            return 0
        elif t < self.params["x4"]:
            return 0.5 * (t / self.params["x4"]) ** 2.5
        elif t < 2 * self.params["x4"]:
            return 1 - 0.5 * (2 - t / self.params["x4"]) ** 2.5
        else: # t >= x4
            return 1
    
    def compute_unitary_hydrograph(self):
        
        nuh1 = int(math.ceil(self.params["x4"]))
        nuh2 = int(math.ceil(2.0 * self.params["x4"]))
        self.uh1  = np.zeros(nuh1)
        self.uh2  = np.zeros(nuh2)
        uh1_ordinates = np.zeros(nuh1)
        uh2_ordinates = np.zeros(nuh2)
        
        for t in range(1, nuh1 + 1):
            uh1_ordinates[t-1] = self.s_curves1(t) - self.s_curves1(t-1)
    
        for t in range(1, nuh2 + 1):
            uh2_ordinates[t-1] = self.s_curves2(t) - self.s_curves2(t-1)
        
        self.ouh1 = uh1_ordinates
        self.ouh2 = uh2_ordinates
    
    def compute_hydrograph(self, rout_pat):
        """
        Daily hydrpgraph for catchment routine
        """
        for i in range(0, len(self.uh1) - 1):
            self.uh1[i] = self.uh1[i+1] + self.ouh1[i] * rout_pat
        self.uh1[-1] = self.ouh1[-1] * rout_pat

        for j in range(0, len(self.uh2) - 1):
            self.uh2[j] = self.uh2[j+1] + self.ouh2[j] * rout_pat
        self.uh2[-1] = self.ouh2[-1] * rout_pat
        
        return self.uh1, self.uh2
    
    def compute_exchange(self, uh1, rout_sto):
        # groundwater exchange
        gw_exc = self.params["x2"] * (rout_sto / self.params["x3"]) ** 3.5
        rout_sto = max(0, rout_sto + uh1[0] * 0.9 + gw_exc)
        return gw_exc, rout_sto
    
    def compute_discharge(self, uh2, gw_exc, rout_sto):
        new_rout_sto = rout_sto / (1. + (rout_sto / self.params["x3"]) ** 4.0) ** 0.25
        qr = rout_sto - new_rout_sto
        rout_sto = new_rout_sto
        qd =  max(0, uh2[0] *0.1 + gw_exc)
        return qr, qd, rout_sto
    
    def run(self, forcings, start=None, end=None, save_state=False, **kwargs):
        """
        Run the GR4J model
        

        Parameters
        ----------
        forcings : DataFrame
            Input data with columns prec (precipitation), tmean, and
            pet(potential evapotranspiration, optional)
        start : string, optional
            Start date for simulation in format. Example: '2001-01-01'
        end : string, optional
            End date for simulation in format. Example: '2010-12-31'
        save_state : bool, optional
            If True (default), last storage is saved as w0 parameter
        **kwargs :
            Model parameters can be changed for the simulation
                area    >     [float] catchment area in km2
                lat     >     [float] catchment latitude
                ps0     >     [float] initial production storage as a fraction ps0=(ps/X1)
                rs0     >     [float] initial routing storage as a fraction pr0=(rs/X3)
                x1      >     [float] maximum production capacity (mm)
                x2      >     [float] discharge parameter (mm)
                x3      >     [float] routing maximum capacity (mm)
                x4      >     [float] delay (days)

        Returns
        -------
        Simulations : DataFrame
            Qt       > Streamflow (Qd+Qb) at catchment output (m3/s)
            Qd       > Directflow at catchment output (m3/s)
            Qb       > Baseflow at catchment output (m3/s)
            pet      > Potential Evapotranspiration in mm
            gwe      > Groundwater Exchange in mm
            psto     > Production storage as a fraction of x1 (adim)
            rsto     > Routing storage as a fraction of x3 (adim)
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
        
                
        # Get Forcings
        prec  = forcings["prec"].values
        tmean = forcings["tmean"].values
        doy   = forcings.index.dayofyear.values  # day of year
        pet   = self.compute_pet(tmean, doy, self.lat)
        
        # Unitary hydrograph
        self.compute_unitary_hydrograph()
        
        # Create empty arrays
        n = len(prec)
        qtarray = np.zeros(n, dtype=np.float32)
        qdarray = np.zeros(n, dtype=np.float32)
        qrarray = np.zeros(n, dtype=np.float32)
        gwarray = np.zeros(n, dtype=np.float32)
        psarray = np.zeros(n, dtype=np.float32)
        rsarray = np.zeros(n, dtype=np.float32)
        
        # Compute Water Balance
        psto = self.params["ps0"] * self.params["x1"]
        rsto = self.params["rs0"] * self.params["x3"]
        
        # Compute water partioning 
        for t in range(n):
            res = self.reservoirs_evaporation(prec[t], pet[t], psto)
            evap, res_prod, rout_pat = res
            
            psto = psto - evap + res_prod
            perc = psto / (1. + (psto / 2.25 / self.params["x1"]) ** 4.) ** 0.25
            rout_pat = rout_pat + (psto - perc)
            psto = perc
            
            uh1, uh2 = self.compute_hydrograph(rout_pat)
            
            gw_exc, rsto = self.compute_exchange(uh1, rsto)
            
            qr, qd, rsto = self.compute_discharge(uh2, gw_exc, rsto)
            qt = qr + qd
            
            # Save outputs
            qtarray[t] = qt                        # total flow
            qdarray[t] = qd                        # runoff
            qrarray[t] = qr                        # baseflow
            gwarray[t] = gw_exc                    # groundwater exchange
            psarray[t] = psto / self.params["x1"]  # production storage
            rsarray[t] = rsto / self.params["x3"]  # routing storage            
        
        # Create Output DataFrame
        outputs = pd.DataFrame(
            {
                    "Qt": qtarray,
                    "Qd": qdarray,
                    "Qb": qrarray,
                    "pet": pet,
                    "gwe": gwarray,
                    "psto": psarray,
                    "rsto": rsarray,
            },
            index=forcings.index,
        )
        
        # Convert units
        outputs.loc[:, "Qt"] = outputs.loc[:, "Qt"] * self.area / 86.4
        outputs.loc[:, "Qd"] = outputs.loc[:, "Qd"] * self.area / 86.4
        outputs.loc[:, "Qb"] = outputs.loc[:, "Qb"] * self.area / 86.4
        
        # Save final storage state
        if save_state:
            self.params["ps0"] = psto / self.params["x1"]
            self.params["rs0"] = rsto / self.params["x3"]
        
        return outputs
    

    