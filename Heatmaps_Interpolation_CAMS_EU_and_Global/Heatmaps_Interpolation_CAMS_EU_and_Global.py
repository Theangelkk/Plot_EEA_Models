# Script for comparing the heatmaps of CAMS Europe Reanalyses and Global Reanalyses
# conda activate Plot_all_air_pol

# Link: https://www.meteoblue.com/en/weather-maps/#map=particulateMatter~pm2.5~CAMSEU~sfc~none&coords=3.09/46.54/26.12

from ast import arg
import os
from datetime import datetime, timedelta
from pickle import FALSE
import xarray as xr
import numpy as np
import math

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from time import sleep
import copy
import shutil
import argparse
import json
from scipy import interpolate
from skimage.transform import resize
from sklearn.metrics import mean_absolute_percentage_error

from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

import warnings
warnings.filterwarnings("ignore")

def valid_datetime(dt):
    for fmt in ('%Y-%m-%dT%H:%M', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(dt, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("Invalid date: '{0}'.".format(dt))

def valid_date(d):
    t = 'T00:00'
    return valid_datetime(d + t)

list_air_pollutant = ["CO", "NO2", "O3", "PM2p5", "PM10", "SO2"]

list_numeric_model_cams_eu = [  "chimere", "ensemble", "EMEP", "LOTOS-EUROS", "MATCH", \
                                "MINNI", "MOCAGE", "SILAM", "EURAD-IM", "DEHM", "GEM-AQ"]

parser = argparse.ArgumentParser(description='Compare CAMS Global - GEOS CF - CAMS EU')
parser.add_argument('-a', '--air_pollutant', help='Air Pollutant CO - NO2 - O3 - PM2p5 - PM10 - SO2', choices=list_air_pollutant, required=True)
parser.add_argument('-cams_eu', '--cams_eu', help='chimere - ensemble - EMEP - LOTOS-EUROS - MATCH - MINNI - MOCAGE - SILAM - EURAD-IM - DEHM - GEM-AQ', \
                     choices=list_numeric_model_cams_eu, required=True)
parser.add_argument('-m_air_pol', '--m_air_pol', help='Model level for air pollution', type=int, required=True)
parser.add_argument('-m_pm', '--m_pm', help='Model level for Particulate', type=int, required=True)
parser.add_argument('-s_date', '--start_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-e_date', '--end_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-delta_h', '--delta_hours', type=int, required=True)
parser.add_argument('-order_interp', '--order_interp', type=int, required=True, default=0)
parser.add_argument('-compute_air_dens', '--compute_air_density', help='Compute with formula the air density', action='store_true')
parser.add_argument('-co_ug_m^3', '--co_ug_m^3', help='CO in ug/m^3', action='store_true')
args = vars(parser.parse_args())

# CO [0] - NO2 [1] - O3 [2] - PM2p5 [3] - PM10 [4] - SO2 [5]
idx_air_pollutant_selected = list_air_pollutant.index(args["air_pollutant"])

idx_numeric_model_cams_eu = list_numeric_model_cams_eu.index(args["cams_eu"])

# Model level 55: about 288 meters [CAMS Global, GEOS CF, CAMS Europe]
# Model level 60: about 10 meters [CAMS Global, CAMS Europe]
if args["m_air_pol"] != 60 and args["m_air_pol"] != 55:
    print("Error: Model level of air pollution must be equal to 60 or 55")
    exit(-1)
else:
    model_level_air_pollution = int(args["m_air_pol"])

if args["m_pm"] != 60 and args["m_pm"] != 55:
    print("Error: Model level of particulate must be equal to 60 or 55")
    exit(-1)
else:
    model_level_pm = int(args["m_pm"])

cams_eu = args["cams_eu"]
fixed_air_density = not bool(args["compute_air_density"])

air_pollutant_selected = list_air_pollutant[idx_air_pollutant_selected]

start_date_time_to_display = args["start_date"]
end_date_time_to_display = args["end_date"]

# If it asked the visualization of CO in ug/m^3
co_in_ug_m3 = args["co_ug_m^3"]

if args["delta_hours"] > 0:
    delta_time_hours = int(args["delta_hours"])

# Order --> https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
# 0: Nearest-neighboor
# 1: Bi-linear
# 2: Bi-quadratic
# 3: Bi-cubic
# 4: Bi-quartic
# 5: Bi-quintic

order = int(args["order_interp"])

# Limit values Directive 2008/50/EC
dict_limit_air_pollutants = {}

# NO2 limit value 1 hour: 18.0 ug/m3
dict_limit_air_pollutants["NO2"] = 18.0 * 2

# CO limit value for daily max 8 hours: 10.0 mg/m3
dict_limit_air_pollutants["CO"] = 10.0 * 2

# Limit CO in ug/m3
dict_limit_air_pollutants["CO_ug_m3"] = dict_limit_air_pollutants["CO"] * 1000

# SO2 limit value for 1 hour: 350.0 ug/m3
dict_limit_air_pollutants["SO2"] = 350.0 * 2

# O3 limit value for daily max 8 hours: 120.0 ug/m3
dict_limit_air_pollutants["O3"] = 120.0 * 2

# PM2.5 limit value for 1 year: 25.0 ug/m3
dict_limit_air_pollutants["PM2.5"] = 15.0 * 2

# PM10 limit value for 1 year: 50.0 ug/m3
dict_limit_air_pollutants["PM10"] = 50.0 * 2

# Italy coordinates
lat_italy_bnds, lon_italy_bnds = [32,50], [5,21]

def joinpath(rootdir, targetdir):
    return os.path.join(os.sep, rootdir + os.sep, targetdir)

def load_ds_datasets(current_date):

    global  dict_start_time_numeric_models_cams_eu_reanalyses, numeric_model_selected, \
            end_time_cams_eu_reanalyses, end_time_cams_global, air_pollutant_selected, \
            DATADIR_CAMS_EU_reanalyses, DATADIR_CAMS_GLOBAL, \
            not_available_cams_eu_reanalyses, not_available_cams_global, \
            list_idx_lat, list_idx_lon, lat_italy_bnds, lon_italy_bnds, co_in_ug_m3
    
    ds_cams_eu_reanalyses = None
    ds_cams_global = None

    list_idx_lat = []
    list_idx_lon = []

    if  current_date >= dict_start_time_numeric_models_cams_eu_reanalyses[numeric_model_selected] and \
        current_date <= end_time_cams_eu_reanalyses:
        DATADIR_CURRENT_MONTH_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_cams_eu_reanalyses = joinpath(DATADIR_CURRENT_MONTH_CAMS_EU_reanalyses, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_cams_eu_reanalyses = xr.open_dataset(file_netcdf_path_cams_eu_reanalyses)
        list_idx_lat.append(ds_cams_eu_reanalyses.indexes["lat"])
        list_idx_lon.append(ds_cams_eu_reanalyses.indexes["lon"])

        if co_in_ug_m3 and air_pollutant_selected == "CO":
            ds_cams_eu_reanalyses["co"] *= 1000

        not_available_cams_eu_reanalyses = False    
    else:
        not_available_cams_eu_reanalyses = True
        list_idx_lat.append([*range(lat_italy_bnds[0], lat_italy_bnds[1], 1)])
        list_idx_lon.append([*range(lon_italy_bnds[0], lon_italy_bnds[1], 1)])
        print("Dataset not available for CAMS EU " +  str(numeric_model_selected))

    if current_date >= start_time_cams_global and current_date <= end_time_cams_global:
        DATADIR_CURRENT_MONTH_CAMS_GLOABL = joinpath(DATADIR_CAMS_GLOBAL, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_cams_global = joinpath(DATADIR_CURRENT_MONTH_CAMS_GLOABL, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_cams_global = xr.open_dataset(file_netcdf_path_cams_global)
        list_idx_lat.append(ds_cams_global.indexes["latitude"])
        list_idx_lon.append(ds_cams_global.indexes["longitude"])

        if co_in_ug_m3 and air_pollutant_selected == "CO":
            ds_cams_global["co"] *= 1000

        not_available_cams_global = False
    else:
        not_available_cams_global = True
        list_idx_lat.append([*range(lat_italy_bnds[0], lat_italy_bnds[1], 1)])
        list_idx_lon.append([*range(lon_italy_bnds[0], lon_italy_bnds[1], 1)])
        print("Dataset not available for CAMS GLOBAL")
              
    return ds_cams_eu_reanalyses, ds_cams_global

# ------------ Information on CAMS EUROPE Reanalyses ------------
not_available_cams_eu_reanalyses = False

# Path of CAMS Europe
path_main_dir_CAMS_Europe_data_reanalyses = os.environ['CAMS_Europe_Reanalyses']

if path_main_dir_CAMS_Europe_data_reanalyses == "":
    print("Error: set the environmental variables of CAMS_Europe_Reanalyses")
    exit(-1)

if air_pollutant_selected == "PM2p5" or air_pollutant_selected == "PM10":
    DATADIR_CAMS_EU_reanalyses = joinpath(path_main_dir_CAMS_Europe_data_reanalyses, "model_level_" + str(model_level_pm))
else:
    DATADIR_CAMS_EU_reanalyses = joinpath(path_main_dir_CAMS_Europe_data_reanalyses, "model_level_" + str(model_level_air_pollution))

DATADIR_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, "italy_ext")

numeric_model_selected = list_numeric_model_cams_eu[idx_numeric_model_cams_eu]
DATADIR_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, numeric_model_selected)

if air_pollutant_selected == "PM2.5":
    DATADIR_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, "PM2p5")
else:
    DATADIR_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, air_pollutant_selected)

# Time resolution of CAMS Europe Reanalyses
time_res_cams_eu_reanalyses = 1

dict_start_time_numeric_models_cams_eu_reanalyses = {}
dict_start_time_numeric_models_cams_eu_reanalyses["SILAM"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_reanalyses["MOCAGE"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_reanalyses["MINNI"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_reanalyses["MATCH"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_reanalyses["LOTOS-EUROS"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_reanalyses["EURAD-IM"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_reanalyses["ensemble"] = datetime(2016, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_reanalyses["EMEP"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_reanalyses["DEHM"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_reanalyses["chimere"] = datetime(2018, 1, 1, 0, 0)

end_time_cams_eu_reanalyses = datetime(2022, 12, 31, 0, 0)

# ------------ Information on CAMS GLOBAL Reanalyses ------------
not_available_cams_global = False

# Path of CAMS_Global
path_main_dir_CAMS_Global_data = os.environ['CAMS_Global_Reanalyses']

if path_main_dir_CAMS_Global_data == "":
    print("Error: set the environmental variables of CAMS_Global_Reanalyses")
    exit(-1)

DATADIR_CAMS_GLOBAL = joinpath(path_main_dir_CAMS_Global_data, "datasets_model_level_" + str(model_level_air_pollution))

if fixed_air_density:
    DATADIR_CAMS_GLOBAL = joinpath(DATADIR_CAMS_GLOBAL, "italy_ext_fixed")
else:
    DATADIR_CAMS_GLOBAL = joinpath(DATADIR_CAMS_GLOBAL, "italy_ext_formula")

DATADIR_CAMS_GLOBAL = joinpath(DATADIR_CAMS_GLOBAL, air_pollutant_selected)

# Time resolution of CAMS Global
time_res_cams_global = 3

start_time_cams_global = datetime(2003, 1, 1, 0, 0)
end_time_cams_global = datetime(2022, 12, 1, 0, 0)

'''
# ------------ Informationi on GEOS CF ------------
not_available_goes_cf = False

# Path of GEOS_CF_data
path_main_dir_GEOS_CF_data = os.environ['GEOS_CF_data']

if path_main_dir_GEOS_CF_data == "":
    print("Error: set the environmental variables of GEOS_CF_data")
    exit(-1)

DATADIR_GEOS_CF = joinpath(path_main_dir_GEOS_CF_data, "Google_data")

if fixed_air_density:
    DATADIR_GEOS_CF = joinpath(DATADIR_GEOS_CF, "Fixed_denisity")
else:
    DATADIR_GEOS_CF = joinpath(DATADIR_GEOS_CF, "Formula_denisity")

if model_level_air_pollution == 55 or model_level_air_pollution == 60:
    if air_pollutant_selected == "PM2p5":
        air_pollutant_pm25_geos_cf = "PM25_RH35_GCC"
        #air_pollutant_pm25_geos_cf = "PM25_RH35_GOCART"

        DATADIR_GEOS_CF = joinpath(DATADIR_GEOS_CF, air_pollutant_pm25_geos_cf)

    elif air_pollutant_selected == "PM10":
        print("PM10 not present for GEOS CF")

    else:
        DATADIR_GEOS_CF = joinpath(DATADIR_GEOS_CF, air_pollutant_selected)
else:
    print("Model level " + str(model_level_air_pollution) + " not available for GEOS CF")

# Time resolution of GEOS CF
time_res_geos_cf = 1

start_time_geos_cf = datetime(2018, 1, 1, 0, 0)
end_time_geos_cf = datetime(2023, 9, 30, 0, 0)
'''

def plot_heatmap(lon_bounds, lat_bounds, list_idx_lat, list_idx_lon, list_air_pollutant_ds, vmax, title, savefig = False, path_fig="", dpi=300):

    global  not_available_cams_eu_reanalyses, not_available_cams_global, \
            list_numeric_model_cams_eu, idx_numeric_model_cams_eu
    
    fig = plt.figure()

    # ------------ PLOT CAMS GLOBAL ------------
    ax_cams_global = fig.add_subplot(211)
    mp = Basemap(   
                    projection='merc',
                    llcrnrlon=lon_bounds[0],     # lower longitude
                    llcrnrlat=lat_bounds[0],     # lower latitude
                    urcrnrlon=lon_bounds[1],     # uppper longitude
                    urcrnrlat=lat_bounds[1],     # uppper latitude
                    resolution = 'i'
            )

    lon_cams_global, lat_cams_global = np.meshgrid(list_idx_lon[1], list_idx_lat[1])
    x_cams_global, y_cams_global = mp(lon_cams_global,lat_cams_global)
    
    if not_available_cams_global == False and list_air_pollutant_ds[1] is not None:
        c_scheme_cams_global = mp.pcolor(x_cams_global, y_cams_global, np.squeeze(list_air_pollutant_ds[1].to_array()), cmap = 'jet', vmin=0.0, vmax=vmax)
        cbar = mp.colorbar(c_scheme_cams_global, location='right', pad = '10%') # map information

    # consider this as the outline for the map that is to be created
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    ax_cams_global.set_title(title + " CAMS GLOBAL (0,75° x 0,75°)")

    '''
    # ------------ PLOT GEOS CF ------------
    ax_geos_cf = fig.add_subplot(312)
    mp = Basemap(   
                    projection='merc',
                    llcrnrlon=lon_bounds[0],     # lower longitude
                    llcrnrlat=lat_bounds[0],     # lower latitude
                    urcrnrlon=lon_bounds[1],     # uppper longitude
                    urcrnrlat=lat_bounds[1],     # uppper latitude
                    resolution = 'i'
            )

    lon_geos_cf, lat_geos_cf = np.meshgrid(list_idx_lon[2], list_idx_lat[2])
    x_geos_cf, y_geos_cf = mp(lon_geos_cf,lat_geos_cf)

    if not_available_goes_cf == False:
        c_scheme_geos_cf = mp.pcolor(x_geos_cf, y_geos_cf, np.squeeze(list_air_pollutant_ds[2].to_array().T), cmap = 'jet', vmin=0.0, vmax=vmax)
        cbar = mp.colorbar(c_scheme_geos_cf, location='right', pad = '10%') # map information

    # consider this as the outline for the map that is to be created
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    ax_geos_cf.set_title(title + " GEOS CF (0,25° x 0,25°)")
    '''

    # ------------ PLOT CAMS EUROPE Reanalyses ------------
    ax_cams_eu = fig.add_subplot(212)
    mp = Basemap(   
                    projection='merc',
                    llcrnrlon=lon_bounds[0],     # lower longitude
                    llcrnrlat=lat_bounds[0],     # lower latitude
                    urcrnrlon=lon_bounds[1],     # uppper longitude
                    urcrnrlat=lat_bounds[1],     # uppper latitude
                    resolution = 'i'
            )

    # Conversion of latitude and longitude coordinate in a 2D grid
    lon_cams_eu, lat_cams_eu = np.meshgrid(list_idx_lon[0], list_idx_lat[0])
    x_cams_eu, y_cams_eu = mp(lon_cams_eu,lat_cams_eu)

    if not_available_cams_eu_reanalyses == False:
        c_scheme_cams_eu = mp.pcolor(x_cams_eu, y_cams_eu, np.squeeze(list_air_pollutant_ds[0].to_array()), cmap = 'jet', vmin=0.0, vmax=vmax)
        cbar = mp.colorbar(c_scheme_cams_eu, location='right', pad = '10%') # map information

    # consider this as the outline for the map that is to be created
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    ax_cams_eu.set_title(title + " CAMS EU Reanalyses " + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]) + " (0,1° x 0,1°)")

    if savefig:
        plt.savefig(path_fig, dpi=dpi) #saves the image generated

    plt.close()

def plot_heatmap_0_75(  lon_bounds, lat_bounds, list_idx_lat, list_idx_lon, list_air_pollutant_ds, \
                        vmax, title_cams_eu, title_cams_global, savefig = False, path_fig="", dpi=300
                    ):

    global  not_available_cams_eu_reanalyses, not_available_cams_global, list_numeric_model_cams_eu, idx_numeric_model_cams_eu
    
    fig = plt.figure()

    # ------------ PLOT CAMS GLOBAL ------------
    ax_cams_global = fig.add_subplot(211)
    mp = Basemap(   
                    projection='merc',
                    llcrnrlon=lon_bounds[0],     # lower longitude
                    llcrnrlat=lat_bounds[0],     # lower latitude
                    urcrnrlon=lon_bounds[1],     # uppper longitude
                    urcrnrlat=lat_bounds[1],     # uppper latitude
                    resolution = 'i'
            )

    lon_cams_global, lat_cams_global = np.meshgrid(list_idx_lon[0], list_idx_lat[0])
    x_cams_global, y_cams_global = mp(lon_cams_global,lat_cams_global)
    
    if not_available_cams_global == False and list_air_pollutant_ds[0] is not None:
        c_scheme_cams_global = mp.pcolor(x_cams_global, y_cams_global, np.squeeze(list_air_pollutant_ds[0]), cmap = 'jet', vmin=0.0, vmax=vmax)
        cbar = mp.colorbar(c_scheme_cams_global, location='right', pad = '10%') # map information

    # consider this as the outline for the map that is to be created
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    ax_cams_global.set_title(title_cams_global)

    '''
    # ------------ PLOT GEOS CF ------------
    ax_geos_cf = fig.add_subplot(312)
    mp = Basemap(   
                    projection='merc',
                    llcrnrlon=lon_bounds[0],     # lower longitude
                    llcrnrlat=lat_bounds[0],     # lower latitude
                    urcrnrlon=lon_bounds[1],     # uppper longitude
                    urcrnrlat=lat_bounds[1],     # uppper latitude
                    resolution = 'i'
            )

    lon_geos_cf, lat_geos_cf = np.meshgrid(list_idx_lon[0], list_idx_lat[0])
    x_geos_cf, y_geos_cf = mp(lon_geos_cf,lat_geos_cf)

    if not_available_goes_cf == False:
        c_scheme_geos_cf = mp.pcolor(x_geos_cf, y_geos_cf, np.squeeze(list_air_pollutant_ds[1]), cmap = 'jet', vmin=0.0, vmax=vmax)
        cbar = mp.colorbar(c_scheme_geos_cf, location='right', pad = '10%') # map information

    # consider this as the outline for the map that is to be created
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    ax_geos_cf.set_title(title_geos_cf)
    '''

    # ------------ PLOT CAMS EUROPE Reanalyses ------------
    ax_cams_eu = fig.add_subplot(212)
    mp = Basemap(   
                    projection='merc',
                    llcrnrlon=lon_bounds[0],     # lower longitude
                    llcrnrlat=lat_bounds[0],     # lower latitude
                    urcrnrlon=lon_bounds[1],     # uppper longitude
                    urcrnrlat=lat_bounds[1],     # uppper latitude
                    resolution = 'i'
            )

    # Conversion of latitude and longitude coordinate in a 2D grid
    lon_cams_eu, lat_cams_eu = np.meshgrid(list_idx_lon[0], list_idx_lat[0])
    x_cams_eu, y_cams_eu = mp(lon_cams_eu,lat_cams_eu)

    if not_available_cams_eu_reanalyses == False:
        c_scheme_cams_eu = mp.pcolor(x_cams_eu, y_cams_eu, np.squeeze(list_air_pollutant_ds[1]), cmap = 'jet', vmin=0.0, vmax=vmax)
        cbar = mp.colorbar(c_scheme_cams_eu, location='right', pad = '10%') # map information

    # consider this as the outline for the map that is to be created
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    ax_cams_eu.set_title(title_cams_eu)

    if savefig:
        plt.savefig(path_fig, dpi=dpi) #saves the image generated
    else:
        plt.show()
        sleep(0.01)

    plt.close()

'''
def plot_heatmap_0_25(  
                        lon_bounds, lat_bounds, list_idx_lat, list_idx_lon, list_air_pollutant_ds, \
                        vmax, title_cams_eu, title_geos_cf, savefig = False, path_fig="", dpi=300
                    ):

    global  not_available_cams_eu_reanalyses, not_available_cams_global, \
            list_numeric_model_cams_eu, idx_numeric_model_cams_eu
    
    fig = plt.figure()

    # ------------ PLOT GEOS CF ------------
    ax_geos_cf = fig.add_subplot(311)
    mp = Basemap(   
                    projection='merc',
                    llcrnrlon=lon_bounds[0],     # lower longitude
                    llcrnrlat=lat_bounds[0],     # lower latitude
                    urcrnrlon=lon_bounds[1],     # uppper longitude
                    urcrnrlat=lat_bounds[1],     # uppper latitude
                    resolution = 'i'
            )

    lon_geos_cf, lat_geos_cf = np.meshgrid(list_idx_lon[0], list_idx_lat[0])
    x_geos_cf, y_geos_cf = mp(lon_geos_cf,lat_geos_cf)

    if not_available_goes_cf == False:
        c_scheme_geos_cf = mp.pcolor(x_geos_cf, y_geos_cf, np.squeeze(list_air_pollutant_ds[0]).T, cmap = 'jet', vmin=0.0, vmax=vmax)
        cbar = mp.colorbar(c_scheme_geos_cf, location='right', pad = '10%') # map information

    # consider this as the outline for the map that is to be created
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    ax_geos_cf.set_title(title_geos_cf)

    # ------------ PLOT CAMS EUROPE Reanalyses ------------
    ax_cams_eu = fig.add_subplot(312)
    mp = Basemap(   
                    projection='merc',
                    llcrnrlon=lon_bounds[0],     # lower longitude
                    llcrnrlat=lat_bounds[0],     # lower latitude
                    urcrnrlon=lon_bounds[1],     # uppper longitude
                    urcrnrlat=lat_bounds[1],     # uppper latitude
                    resolution = 'i'
            )

    # Conversion of latitude and longitude coordinate in a 2D grid
    lon_cams_eu, lat_cams_eu = np.meshgrid(list_idx_lon[0], list_idx_lat[0])
    x_cams_eu, y_cams_eu = mp(lon_cams_eu,lat_cams_eu)

    if not_available_cams_eu_reanalyses == False:
        c_scheme_cams_eu = mp.pcolor(x_cams_eu, y_cams_eu, np.squeeze(list_air_pollutant_ds[1]), cmap = 'jet', vmin=0.0, vmax=vmax)
        cbar = mp.colorbar(c_scheme_cams_eu, location='right', pad = '10%') # map information

    # consider this as the outline for the map that is to be created
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    ax_cams_eu.set_title(title_cams_eu)

    if savefig:
        plt.savefig(path_fig, dpi=dpi) #saves the image generated
    else:
        plt.show()
        sleep(0.01)

    plt.close()
'''

# Downsampling function
def downsampling_np_matrix(np_hr, shape_lr, order):

    # It is require a flip operation on a high-resolution image    
    np_hr_flip = np.flip(np_hr, 0)

    np_lr = resize(np_hr_flip, shape_lr, order=order)  #order = 3 for cubic spline
        
    return np_lr

def empty_dict_global_similarity():

    dict_global_info = {}

    dict_global_info["CAMS_EU_Reanalyses_vs_CAMS_Global"] = {}
    #dict_global_info["GEOS_CF_vs_CAMS_Global"] = {}
    #dict_global_info["CAMS_EU_vs_GEOS_CF"] = {}

    for key in dict_global_info:
        dict_global_info[key]["MSE"] = []
        dict_global_info[key]["RMSE"] = []
        dict_global_info[key]["UQI"] = []
        dict_global_info[key]["ERGAS"] = []
        dict_global_info[key]["SCC"] = []
        dict_global_info[key]["RASE"] = []
        dict_global_info[key]["SAM"] = []
        dict_global_info[key]["VIF"] = []
        dict_global_info[key]["Pearson_coeff"] = []
        dict_global_info[key]["Perc_change"] = []
        dict_global_info[key]["Perc_change_abs"] = []

    return dict_global_info

# Function for computing the global similarity between two heatmap
def compute_global_similarity(np_1, np_2):
    
    dict_current_sim = {}

    # It cannot be computed the PSNR and SSIM

    dict_current_sim["MSE"] = mse(np_1, np_2)
    dict_current_sim["RMSE"] = rmse(np_1, np_2)
    dict_current_sim["UQI"] = uqi(np_1, np_2)
    dict_current_sim["ERGAS"] = ergas(np_1, np_2)
    dict_current_sim["SCC"] = scc(np_1, np_2)
    dict_current_sim["RASE"] = rase(np_1, np_2)
    dict_current_sim["SAM"] = sam(np_1, np_2)
    dict_current_sim["VIF"] = vifp(np_1, np_2)

    dict_current_sim["Pearson_coeff"] = np.abs(np.corrcoef(np_1.flatten(), np_2.flatten()))[0,1]

    dict_current_sim["Perc_change"] = np.sum((((np_1 - np_2) * 100) / np_1)) / (np_1.shape[0] * np_1.shape[1])
    dict_current_sim["Perc_change_abs"] = np.sum(np.abs(((np_1 - np_2) * 100) / np_1)) / (np_1.shape[0] * np_1.shape[1])

    return dict_current_sim
          
def compute_statistical_into(np_air_pol):

    print("Mean: ", np.mean(np_air_pol))
    print("Standard deviation: ", np.std(np_air_pol))

def perc_abs_err(y_tilde,y):

    perc_abs_err_np = (np.absolute(y - y_tilde) / y) * 100.0

    return perc_abs_err_np

# ------------------ PLOT -----------------------
# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_LATEX = joinpath(PATH_DIR_PLOTS, "Latex")

if not os.path.exists(PATH_DIR_LATEX):
  os.mkdir(PATH_DIR_LATEX)

if fixed_air_density:
    PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, "FIXED_AIR_DENSITY")
else:
    PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, "FORMULA_AIR_DENSITY")

if not os.path.exists(PATH_DIR_LATEX):
    os.mkdir(PATH_DIR_LATEX)

PATH_DIR_LPATH_DIR_LATEXATEX = joinpath(PATH_DIR_LATEX, air_pollutant_selected)

if not os.path.exists(PATH_DIR_LATEX):
  os.mkdir(PATH_DIR_LATEX)

if air_pollutant_selected == "CO" and co_in_ug_m3:
    PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, "CO_ug_m^3")
elif air_pollutant_selected == "CO":
    PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, "CO_mg_m^3")

if not os.path.exists(PATH_DIR_LATEX):
  os.mkdir(PATH_DIR_LATEX)

PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, "Order_interpolation_" + str(order))

if not os.path.exists(PATH_DIR_LATEX):
  os.mkdir(PATH_DIR_LATEX)

PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, "Delta_h_" + str(delta_time_hours))

if not os.path.exists(PATH_DIR_LATEX):
  os.mkdir(PATH_DIR_LATEX)

path_latex_file_global_similarity_all_period = joinpath(PATH_DIR_LATEX, start_date_time_to_display.date().strftime("%Y-%m-%d") + "_" + end_date_time_to_display.date().strftime("%Y-%m-%d") + ".txt")

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "Heatmap_Interp_CAMS_EU_" + str(cams_eu) + "_vs_Global_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm))

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

if fixed_air_density:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "FIXED_AIR_DENSITY")
else:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "FORMULA_AIR_DENSITY")

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, air_pollutant_selected)

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

if air_pollutant_selected == "CO" and co_in_ug_m3:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_ug_m^3")
elif air_pollutant_selected == "CO":
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_mg_m^3")

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "Order_interpolation_" + str(order))

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "Delta_h_" + str(delta_time_hours))

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

idx_cams_eu_reanalyses = 0
idx_cams_global = 0
#idx_geos_cf = 0

previous_date = start_date_time_to_display
current_date = start_date_time_to_display

ds_cams_eu_reanalyses = None
ds_cams_global = None
#ds_geos_cf = None

list_idx_lat = []
list_idx_lon = []

list_idx_0_75_lat = []
list_idx_0_75_lon = []

#list_idx_0_25_lat = []
#list_idx_0_25_lon = []

diff_dates = end_date_time_to_display - start_date_time_to_display
diff_dates_hours = int(diff_dates.total_seconds() / (60*60*delta_time_hours))
delta = timedelta(hours=delta_time_hours)

ds_cams_eu_reanalyses, ds_cams_global = load_ds_datasets(current_date)

PATH_CURRENT_MONTH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, str(current_date.year) + "-" + str(current_date.month).zfill(2))

if not os.path.exists(PATH_CURRENT_MONTH_DIR_PLOTS):
  os.mkdir(PATH_CURRENT_MONTH_DIR_PLOTS)

path_json_file_global_similarity_current_month = joinpath(PATH_CURRENT_MONTH_DIR_PLOTS, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".json")
path_json_file_global_similarity_all_period = joinpath(PATH_DIR_PLOTS, "All_period.json")
dict_global_similarity_current_month = empty_dict_global_similarity()
dict_global_similarity_all_period = empty_dict_global_similarity()

for time in range(diff_dates_hours):
    
    # The month is changed
    if previous_date.year != current_date.year or previous_date.month != current_date.month:
        ds_cams_eu_reanalyses, ds_cams_global = load_ds_datasets(current_date)

        PATH_CURRENT_MONTH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, str(current_date.year) + "-" + str(current_date.month).zfill(2))

        if not os.path.exists(PATH_CURRENT_MONTH_DIR_PLOTS):
            os.mkdir(PATH_CURRENT_MONTH_DIR_PLOTS)

        # Save dict_global_similarity_current_month
        for key in dict_global_similarity_current_month:
            
            for key_1 in dict_global_similarity_current_month[key]:
                dict_global_similarity_current_month[key][key_1] = np.mean(np.array(dict_global_similarity_current_month[key][key_1]))

        with open(path_json_file_global_similarity_current_month, 'w') as json_file:
            json_dumps_str = json.dumps(dict_global_similarity_current_month, indent=4)
            json_file.write(json_dumps_str)

        path_json_file_global_similarity_current_month = joinpath(PATH_CURRENT_MONTH_DIR_PLOTS, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".json")
        dict_global_similarity_current_month = empty_dict_global_similarity()

    # Loading datasets
    if not_available_cams_eu_reanalyses == False:
        ds_current_date_cams_eu_reanalyses = ds_cams_eu_reanalyses.sel(time=current_date.isoformat())
    else:
        ds_current_date_cams_eu_reanalyses = None

    if (time*delta_time_hours) % time_res_cams_global == 0 and not_available_cams_global == False:
        ds_current_date_cams_global = ds_cams_global.sel(time=current_date.isoformat())
    else:
        ds_current_date_cams_global = None
    
    '''
    if not_available_goes_cf == False:
        ds_current_date_geos_cf = ds_geos_cf.sel(datetime=current_date.isoformat())
        ds_current_date_geos_cf = ds_current_date_geos_cf.rename({'__xarray_dataarray_variable__': air_pollutant_selected.lower()})
    else:
        ds_current_date_geos_cf = None
    '''

    print("Current date (" + air_pollutant_selected + "): " + current_date.isoformat())
    print("CAMS EU: " + str(not not_available_cams_eu_reanalyses))
    print("CAMS GLOBAL: " + str(not not_available_cams_global))

    if not_available_cams_eu_reanalyses == False:
        np_current_date_air_pol_cams_eu_reanalyses = ds_current_date_cams_eu_reanalyses[air_pollutant_selected.lower()].values

    '''
    if not_available_goes_cf == False:
        np_current_date_air_pol_geos_cf = ds_current_date_geos_cf[air_pollutant_selected.lower()].values.T
    '''
    
    if not_available_cams_global == False and ds_current_date_cams_global is not None:
        np_current_date_air_pol_cams_global = ds_current_date_cams_global[air_pollutant_selected.lower()].values

    '''
    # Downsampling CAMS EU --> GEOS CF
    if not_available_cams_eu == False and not_available_goes_cf == False:
        np_current_date_air_pol_cams_eu_geos_cf = downsampling_np_matrix(   
                                                                            np_current_date_air_pol_cams_eu, 
                                                                            np_current_date_air_pol_geos_cf.shape, 
                                                                            order
                                                                        )

    else:
        np_current_date_air_pol_cams_eu_geos_cf = np.zeros_like(np_current_date_air_pol_geos_cf)
    '''

    # Downsampling CAMS EU --> CAMS GLOBAL
    if not_available_cams_eu_reanalyses == False and not_available_cams_global == False and ds_current_date_cams_global is not None:
        np_current_date_air_pol_cams_eu_cams_global = downsampling_np_matrix(   
                                                                                np_current_date_air_pol_cams_eu_reanalyses, 
                                                                                np_current_date_air_pol_cams_global.shape,
                                                                                order
                                                                            )
    else:
        np_current_date_air_pol_cams_eu_cams_global = np.zeros_like(np_current_date_air_pol_cams_global)
    
    '''
    # Downsampling GEOS CF --> CAMS GLOBAL
    if not_available_goes_cf == False and not_available_cams_global == False and ds_current_date_cams_global is not None:
        np_current_date_air_pol_geos_cf_cams_global = downsampling_np_matrix(   
                                                                                np_current_date_air_pol_geos_cf, 
                                                                                np_current_date_air_pol_cams_global.shape,
                                                                                order
                                                                            )
    else:
        np_current_date_air_pol_geos_cf_cams_global = np.zeros_like(np_current_date_air_pol_cams_global)
    '''
    
    # -------------------- Visualization of Heatmap --------------------
    max_value_dict = dict_limit_air_pollutants[air_pollutant_selected]
        
    title = air_pollutant_selected + " "

    if air_pollutant_selected == "CO" and co_in_ug_m3 == False:
        title += "(mg/m^3) "
    elif air_pollutant_selected == "CO" and co_in_ug_m3:
        max_value_dict = dict_limit_air_pollutants["CO_ug_m3"]
        title += "(ug/m^3) "
    else:
        title += "(ug/m^3) "

    title += current_date.isoformat()

    #title_0_25_cams_eu =  title + " CAMS EU " + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]) + " (0,25° x 0,25°)"
    #title_0_25_geos_cf =  title + " GEOS CF (0,25° x 0,25°)"

    title_0_75_cams_eu =  title + " CAMS EU " + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]) + " (0,75° x 0,75°)"
    #title_0_75_geos_cf =  title + " GEOS CF (0,75° x 0,75°)"
    title_0_75_cams_global =  title + " CAMS GLOBAL (0,75° x 0,75°)"

    path_img_org = joinpath(PATH_CURRENT_MONTH_DIR_PLOTS, current_date.strftime("%Y-%m-%dT%H-%M-%S") + "_org.jpg")
    #path_img_0_25 = joinpath(PATH_CURRENT_MONTH_DIR_PLOTS, current_date.strftime("%Y-%m-%dT%H-%M-%S") + "_0_25.jpg")
    path_img_0_75 = joinpath(PATH_CURRENT_MONTH_DIR_PLOTS, current_date.strftime("%Y-%m-%dT%H-%M-%S") + "_0_75.jpg")

    if len(list_idx_0_75_lat) == 0:
        list_idx_0_75_lat.append(ds_current_date_cams_global.indexes["latitude"])
        list_idx_0_75_lon.append(ds_current_date_cams_global.indexes["longitude"])
    
    '''
    if len(list_idx_0_25_lat) == 0:
        list_idx_0_25_lat.append(ds_current_date_geos_cf.indexes["latitude"])
        list_idx_0_25_lon.append(ds_current_date_geos_cf.indexes["longitude"])
    '''

    #list_air_pollutant_ds = [ds_current_date_cams_eu, ds_current_date_cams_global, ds_current_date_geos_cf]
    list_air_pollutant_ds = [ds_current_date_cams_eu_reanalyses, ds_current_date_cams_global]
    
    '''
    # All methods in 0,25° x 0,25° --> GEOS CF
    list_air_pollutant_0_25_ds = [
                                    ds_current_date_geos_cf.to_array(), 
                                    np_current_date_air_pol_cams_eu_geos_cf,
                                ]
    '''
    
    # All methods in 0,75° x 0,75° --> CAMS GLOBAL
    if ds_current_date_cams_global is not None:
        np_0_75_cams_global = ds_current_date_cams_global.to_array()

        list_air_pollutant_0_75_ds = [
                                    np_0_75_cams_global,
                                    np_current_date_air_pol_cams_eu_cams_global
                                ]

        plot_heatmap(lon_italy_bnds, lat_italy_bnds, list_idx_lat, list_idx_lon, list_air_pollutant_ds, max_value_dict, title, savefig = True, path_fig=path_img_org, dpi=300)
        
        '''
        plot_heatmap_0_25(  lon_italy_bnds, lat_italy_bnds, list_idx_0_25_lat, list_idx_0_25_lon, list_air_pollutant_0_25_ds, max_value_dict, \
                            title_0_25_cams_eu, title_0_25_geos_cf, savefig = True, path_fig=path_img_0_25, dpi=300)
        '''

        plot_heatmap_0_75(  lon_italy_bnds, lat_italy_bnds, list_idx_0_75_lat, list_idx_0_75_lon, list_air_pollutant_0_75_ds, max_value_dict, \
                            title_0_75_cams_eu, title_0_75_cams_global, savefig = True, path_fig=path_img_0_75, dpi=300)

    else:
        np_0_75_cams_global = None

    # ---------------------------------------- Similarity ----------------------------------------
    # Link: https://towardsdatascience.com/measuring-similarity-in-two-images-using-python-b72233eb53c6
    
    # ------------- 0,75° x 0,75° --> CAMS GLOBAL -------------

    # CAMS EU vs CAMS Global
    if not_available_cams_eu_reanalyses == False and not_available_cams_global == False and ds_current_date_cams_global is not None:
        dict_cams_eu_vs_cams_global_current_datetime = compute_global_similarity(   np_current_date_air_pol_cams_eu_cams_global, \
                                                                                    np_current_date_air_pol_cams_global
                                                                                )
        
        for key in dict_global_similarity_current_month["CAMS_EU_Reanalyses_vs_CAMS_Global"]:
            
            can_append = True

            if key != "Perc_change" and key != "Perc_change_abs":
                if math.isnan(dict_cams_eu_vs_cams_global_current_datetime[key]):
                    can_append = False

            if can_append:
                dict_global_similarity_current_month["CAMS_EU_Reanalyses_vs_CAMS_Global"][key].append(dict_cams_eu_vs_cams_global_current_datetime[key])
                dict_global_similarity_all_period["CAMS_EU_Reanalyses_vs_CAMS_Global"][key].append(dict_cams_eu_vs_cams_global_current_datetime[key])

    '''
    # GEOS CF vs CAMS Global
    if not_available_goes_cf == False and not_available_cams_global== False and ds_current_date_cams_global is not None:
        dict_geos_cf_vs_cams_global_current_datetime = compute_global_similarity(   np_current_date_air_pol_geos_cf_cams_global, \
                                                                                    np_current_date_air_pol_cams_global
                                                                                )

        for key in dict_global_similarity_current_month["GEOS_CF_vs_CAMS_Global"]:
            dict_global_similarity_current_month["GEOS_CF_vs_CAMS_Global"][key] += dict_geos_cf_vs_cams_global_current_datetime[key]
            dict_global_similarity_all_period["GEOS_CF_vs_CAMS_Global"][key] += dict_geos_cf_vs_cams_global_current_datetime[key]
    
    # ------------- 0,25° x 0,25° --> GEOS CF -------------

    # CAMS EU vs GEOS CF
    if not_available_cams_eu == False and not_available_goes_cf == False:
        dict_cams_eu_vs_geos_cf_current_datetime = compute_global_similarity(   np_current_date_air_pol_cams_eu_geos_cf, \
                                                                                np_current_date_air_pol_geos_cf
                                                                            )

        for key in dict_global_similarity_current_month["CAMS_EU_vs_GEOS_CF"]:
            dict_global_similarity_current_month["CAMS_EU_vs_GEOS_CF"][key] += dict_cams_eu_vs_geos_cf_current_datetime[key]
            dict_global_similarity_all_period["CAMS_EU_vs_GEOS_CF"][key] += dict_cams_eu_vs_geos_cf_current_datetime[key]
    '''
    # ---------------------------------------- Heatmap error ----------------------------------------
    
    # ------------- Abosolute difference 0,75° x 0,75° --> CAMS GLOBAL -------------

    # CAMS EU vs CAMS Global
    if not_available_cams_eu_reanalyses == False and not_available_cams_global == False and ds_current_date_cams_global is not None:
        perc_abs_cams_eu_vs_cams_global = perc_abs_err(np_current_date_air_pol_cams_eu_cams_global, np_current_date_air_pol_cams_global)
    else:
        perc_abs_cams_eu_vs_cams_global = np.zeros_like(np_current_date_air_pol_cams_eu_cams_global)

    '''
    # GEOS CF vs CAMS Global
    if not_available_goes_cf == False and not_available_cams_global== False and ds_current_date_cams_global is not None:
        perc_abs_geos_cf_vs_cams_global = perc_abs_err(np_current_date_air_pol_geos_cf_cams_global, np_current_date_air_pol_cams_global)
    else:
        perc_abs_geos_cf_vs_cams_global = np.zeros_like(np_current_date_air_pol_geos_cf_cams_global)

    # ------------- Abosolute difference percentage error 0,25° x 0,25° --> GEOS CF -------------

    # CAMS EU vs GEOS CF
    if not_available_cams_eu == False and not_available_goes_cf == False:
        perc_abs_cams_eu_vs_geos_cf = perc_abs_err(np_current_date_air_pol_cams_eu_geos_cf, np_current_date_air_pol_geos_cf)
    else:
        perc_abs_cams_eu_vs_geos_cf = np.zeros_like(np_current_date_air_pol_cams_eu_geos_cf)
    '''
    
    # ------------- Plot Abosolute difference -------------
    '''
    title_0_25_cams_eu_abs_perc =   title + " CAMS EU " + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]) + \
                                    "ABS PERC (0,25° x 0,25°)"
    title_0_25_geos_cf_abs_perc =  title + "ABS PERC GEOS CF (0,25° x 0,25°)"
    '''

    title_0_75_cams_eu_abs_perc =   title + " CAMS EU " + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]) + \
                                    "ABS PERC (0,75° x 0,75°)"
    #title_0_75_geos_cf_abs_perc =  title + "ABS PERC GEOS CF (0,75° x 0,75°)"
    title_0_75_cams_global_abs_perc =  title + "ABS PERC CAMS GLOBAL (0,75° x 0,75°)"

    #path_img_0_25_abs_perc = joinpath(PATH_CURRENT_MONTH_DIR_PLOTS, current_date.strftime("%Y-%m-%dT%H-%M-%S") + "_0_25_ABS_PERC.jpg")
    path_img_0_75_abs_perc = joinpath(PATH_CURRENT_MONTH_DIR_PLOTS, current_date.strftime("%Y-%m-%dT%H-%M-%S") + "_0_75_ABS_PERC.jpg")

    '''
    # All methods in 0,25° x 0,25° --> GEOS CF
    list_air_pollutant_0_25_abs_diff_ds = [
                                            np.zeros_like(ds_current_date_geos_cf.to_array()),
                                            perc_abs_cams_eu_vs_geos_cf,
                                        ]
    '''
    
    # All methods in 0,75° x 0,75° --> CAMS GLOBAL
    list_air_pollutant_0_75_abs_diff_ds = [
                                            np.zeros_like(perc_abs_cams_eu_vs_cams_global),
                                            perc_abs_cams_eu_vs_cams_global
                                        ]

    '''
    if not_available_goes_cf == False and not_available_cams_eu == False:
        plot_heatmap_0_25(  lon_italy_bnds, lat_italy_bnds, list_idx_0_25_lat, list_idx_0_25_lon, list_air_pollutant_0_25_abs_diff_ds, 100.0, \
                            title_0_25_cams_eu_abs_perc, title_0_25_geos_cf_abs_perc, savefig = True, path_fig=path_img_0_25_abs_perc, dpi=300)
    '''

    if not_available_cams_global == False and ds_current_date_cams_global is not None:
        plot_heatmap_0_75(  lon_italy_bnds, lat_italy_bnds, list_idx_0_75_lat, list_idx_0_75_lon, list_air_pollutant_0_75_abs_diff_ds, 100.0, \
                            title_0_75_cams_eu_abs_perc, title_0_75_cams_global_abs_perc, savefig = True, path_fig=path_img_0_75_abs_perc, dpi=300)
        
    previous_date = current_date
    current_date += delta

# Save dict_global_similarity_current_month
for key in dict_global_similarity_all_period:
            
    for key_1 in dict_global_similarity_all_period[key]:
        dict_global_similarity_all_period[key][key_1] = np.mean(np.array(dict_global_similarity_all_period[key][key_1]))

with open(path_json_file_global_similarity_all_period, 'w') as json_file:
    json_dumps_str = json.dumps(dict_global_similarity_all_period, indent=4)
    json_file.write(json_dumps_str)

path_latex_file_global_similarity_all_period = joinpath(PATH_DIR_PLOTS, "CAMS_EU_Reanalyses_vs_CAMS_Global.txt")

with open(path_latex_file_global_similarity_all_period, 'a') as latex_txt_file:
    
    string_output = cams_eu + " & "

    for key_1 in dict_global_similarity_all_period["CAMS_EU_Reanalyses_vs_CAMS_Global"][key_1]:
        string_output += " " + str(round(dict_global_similarity_all_period[key][key_1],3)) + " & "

    latex_txt_file.write(string_output)


