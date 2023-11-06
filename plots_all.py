# conda activate Plot_all_air_pol

# Link: https://www.meteoblue.com/en/weather-maps/#map=particulateMatter~pm2.5~CAMSEU~sfc~none&coords=3.09/46.54/26.12

import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from time import sleep
import copy
import shutil
import argparse

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

parser = argparse.ArgumentParser(description='Plots of CAMS Global - GEOS CF - CAMS EU.')
parser.add_argument('-a', '--air_pollutant', help='Air Pollutant CO - NO2 - O3 - PM2p5 - PM10 - SO2', choices=list_air_pollutant, required=True)
parser.add_argument('-cams_eu', '--cams_eu', help='chimere - ensemble - EMEP - LOTOS-EUROS - MATCH - MINNI - MOCAGE - SILAM - EURAD-IM - DEHM - GEM-AQ', \
                     choices=list_numeric_model_cams_eu, required=True)
parser.add_argument('-m_air_pol', '--m_air_pol', help='Model level for air pollution', type=int, required=True)
parser.add_argument('-m_pm', '--m_pm', help='Model level for Particulate', type=int, required=True)
parser.add_argument('-s_date', '--start_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-e_date', '--end_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-delta_h', '--delta_hours', type=int, required=True)
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

fixed_air_density = not bool(args["compute_air_density"])

air_pollutant_selected = list_air_pollutant[idx_air_pollutant_selected]

start_date_time_to_display = args["start_date"]
end_date_time_to_display = args["end_date"]

# If it asked the visualization of CO in ug/m^3
co_in_ug_m3 = args["co_ug_m^3"]

if args["delta_hours"] > 0:
    delta_time_houes = int(args["delta_hours"])

# Limit values WHO: https://www.who.int/news-room/feature-stories/detail/what-are-the-who-air-quality-guidelines

dict_limit_air_pollutants = {}

# NO2 limit value for 24 hours average: 25.0 ug/m3
dict_limit_air_pollutants["NO2"] = 25.0
dict_limit_air_pollutants["NO2"] = 75.0

# CO limit value for 24 hours average: 4.0 mg/m3
dict_limit_air_pollutants["CO"] = 4.0

# Limit CO in ug/m3
dict_limit_air_pollutants["CO_ug_m3"] = dict_limit_air_pollutants["CO"] * 1000

# SO2 limit value for 24 hours average: 40.0 ug/m3
dict_limit_air_pollutants["SO2"] = 40.0
dict_limit_air_pollutants["SO2"] = 120.0

# O3 limit value for 8 hours average: 100.0 ug/m3
dict_limit_air_pollutants["O3"] = 100.0
dict_limit_air_pollutants["O3"] = 250.0

# PM2.5 limit value for 24 hours average: 15.0 ug/m3
dict_limit_air_pollutants["PM2p5"] = 15.0
dict_limit_air_pollutants["PM2p5"] = 60.0

# PM10 limit value for 24 hours average: 15.0 ug/m3
dict_limit_air_pollutants["PM10"] = 15.0
dict_limit_air_pollutants["PM10"] = 60.0

# Italy coordinates
lat_italy_bnds, lon_italy_bnds = [32,50], [5,21]

def joinpath(rootdir, targetdir):
    return os.path.join(os.sep, rootdir + os.sep, targetdir)

def load_ds_datasets(current_date):

    global  dict_start_time_numeric_models_cams_eu, numeric_model_selected, \
            end_time_cams_eu, end_time_cams_global, end_time_geos_cf, air_pollutant_selected, \
            DATADIR_CAMS_EU, DATADIR_CAMS_GLOBAL, DATADIR_CAMS_GEOS_CF, \
            not_available_cams_eu, not_available_cams_global, not_available_goes_cf, \
            list_idx_lat, list_idx_lon, lat_italy_bnds, lon_italy_bnds, co_in_ug_m3
    
    ds_cams_eu = None
    ds_cams_global = None
    ds_geos_cf = None

    list_idx_lat = []
    list_idx_lon = []

    if  current_date >= dict_start_time_numeric_models_cams_eu[numeric_model_selected] and \
        current_date <= end_time_cams_eu:
        DATADIR_CURRENT_MONTH_CAMS_EU = joinpath(DATADIR_CAMS_EU, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_cams_eu = joinpath(DATADIR_CURRENT_MONTH_CAMS_EU, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_cams_eu = xr.open_dataset(file_netcdf_path_cams_eu)
        list_idx_lat.append(ds_cams_eu.indexes["lat"])
        list_idx_lon.append(ds_cams_eu.indexes["lon"])

        if co_in_ug_m3 and air_pollutant_selected == "CO":
            ds_cams_eu["co"] *= 1000

        not_available_cams_eu = False    
    else:
        not_available_cams_eu = True
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
        print("Dataset not available for CAMS GLOABL")

    if current_date >= start_time_geos_cf and current_date <= end_time_geos_cf and air_pollutant_selected != "PM10":
        DATADIR_CURRENT_MONTH_GEOS_CF = joinpath(DATADIR_GEOS_CF, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_geos_cf = joinpath(DATADIR_CURRENT_MONTH_GEOS_CF, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_geos_cf = xr.open_dataset(file_netcdf_path_geos_cf)
        list_idx_lat.append(ds_geos_cf.indexes["latitude"])
        list_idx_lon.append(ds_geos_cf.indexes["longitude"])

        if co_in_ug_m3 and air_pollutant_selected == "CO":
            ds_geos_cf *= 1000

        not_available_goes_cf = False
    else:
        not_available_goes_cf = True
        list_idx_lat.append([*range(lat_italy_bnds[0], lat_italy_bnds[1], 1)])
        list_idx_lon.append([*range(lon_italy_bnds[0], lon_italy_bnds[1], 1)])
        print("Dataset not available for GEOS CF")

    return ds_cams_eu, ds_cams_global, ds_geos_cf

# ------------ Information on CAMS EUROPA ------------
not_available_cams_eu = False

# Path of CAMS Europe
path_main_dir_CAMS_Europe_data = os.environ['CAMS_Europe']

if path_main_dir_CAMS_Europe_data == "":
    print("Error: set the environmental variables of CAMS_Europe")
    exit(-1)

if air_pollutant_selected == "PM2p5" or air_pollutant_selected == "PM10":
    DATADIR_CAMS_EU = joinpath(path_main_dir_CAMS_Europe_data, "model_level_" + str(model_level_pm))
else:
    DATADIR_CAMS_EU = joinpath(path_main_dir_CAMS_Europe_data, "model_level_" + str(model_level_air_pollution))

DATADIR_CAMS_EU = joinpath(DATADIR_CAMS_EU, "italy_ext")

numeric_model_selected = list_numeric_model_cams_eu[idx_numeric_model_cams_eu]
DATADIR_CAMS_EU = joinpath(DATADIR_CAMS_EU, numeric_model_selected)
DATADIR_CAMS_EU = joinpath(DATADIR_CAMS_EU, air_pollutant_selected)

# Time resolution of CAMS Europe
time_res_cams_eu = 1

dict_start_time_numeric_models_cams_eu = {}
dict_start_time_numeric_models_cams_eu["SILAM"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu["MOCAGE"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu["MINNI"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu["MATCH"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu["LOTOS-EUROS"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu["EURAD-IM"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu["ensemble"] = datetime(2016, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu["EMEP"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu["DEHM"] = datetime(2018, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu["chimere"] = datetime(2018, 1, 1, 0, 0)

end_time_cams_eu = datetime(2020, 12, 31, 0, 0)

# ------------ Information on CAMS GLOBAL ------------
not_available_cams_global = False

# Path of CAMS_Global
path_main_dir_CAMS_Global_data = os.environ['CAMS_Global']

if path_main_dir_CAMS_Global_data == "":
    print("Error: set the environmental variables of CAMS_Global")
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

def plot_heatmap(lon_bounds, lat_bounds, list_idx_lat, list_idx_lon, list_air_pollutant_ds, vmax, title, savefig = False, path_fig="", dpi=300):

    global  not_available_cams_eu, not_available_cams_global, not_available_goes_cf, \
            list_numeric_model_cams_eu, idx_numeric_model_cams_eu
    
    fig = plt.figure()

    # ------------ PLOT CAMS GLOBAL ------------
    ax_cams_global = fig.add_subplot(311)
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

    # ------------ PLOT CAMS EUROPA ------------
    ax_cams_eu = fig.add_subplot(313)
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

    if not_available_cams_eu == False:
        c_scheme_cams_eu = mp.pcolor(x_cams_eu, y_cams_eu, np.squeeze(list_air_pollutant_ds[0].to_array()), cmap = 'jet', vmin=0.0, vmax=vmax)
        cbar = mp.colorbar(c_scheme_cams_eu, location='right', pad = '10%') # map information

    # consider this as the outline for the map that is to be created
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawcountries()

    ax_cams_eu.set_title(title + " CAMS EU " + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]) + " (0,1° x 0,1°)")

    if savefig:
        plt.savefig(path_fig, dpi=dpi) #saves the image generated

    plt.close()

# ------------------ PLOT -----------------------

# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "plots_all_air_model_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm) + "_camsEU_" + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]))

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

if fixed_air_density:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "FIXED_AIR_DENSITY")
else:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "FORMULA_AIR_DENSITY")

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

if air_pollutant_selected == "PM2p5":
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, air_pollutant_pm25_geos_cf)
else: 
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, air_pollutant_selected)

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

if air_pollutant_selected == "CO" and co_in_ug_m3:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_ug_m^3")
elif air_pollutant_selected == "CO":
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_mg_m^3")

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

previous_date = start_date_time_to_display
current_date = start_date_time_to_display

ds_cams_eu = None
ds_cams_global = None
ds_geos_cf = None

list_idx_lat = []
list_idx_lon = []

diff_dates = end_date_time_to_display - start_date_time_to_display
diff_dates_hours = int(diff_dates.total_seconds() / (60*60*delta_time_houes))
delta = timedelta(hours=delta_time_houes)

ds_cams_eu, ds_cams_global, ds_geos_cf = load_ds_datasets(current_date)

PATH_CURRENT_MONTH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, str(current_date.year) + "-" + str(current_date.month).zfill(2))

if not os.path.exists(PATH_CURRENT_MONTH_DIR_PLOTS):
  os.mkdir(PATH_CURRENT_MONTH_DIR_PLOTS)

for time in range(diff_dates_hours):
    
    # The month is changed
    if previous_date.year != current_date.year or previous_date.month != current_date.month:
        ds_cams_eu, ds_cams_global, ds_geos_cf = load_ds_datasets(current_date)

        PATH_CURRENT_MONTH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, str(current_date.year) + "-" + str(current_date.month).zfill(2))

        if not os.path.exists(PATH_CURRENT_MONTH_DIR_PLOTS):
            os.mkdir(PATH_CURRENT_MONTH_DIR_PLOTS)

    # Loading datasets
    if not_available_cams_eu == False:
        ds_current_date_cams_eu = ds_cams_eu.sel(time=current_date.isoformat())
    else:
        ds_current_date_cams_eu = None

    if (time*delta_time_houes) % time_res_cams_global == 0 and not_available_cams_global == False:
        ds_current_date_cams_global = ds_cams_global.sel(time=current_date.isoformat())
    else:
        ds_current_date_cams_global = None

    if not_available_goes_cf == False:
        ds_current_date_geos_cf = ds_geos_cf.sel(datetime=current_date.isoformat())
    else:
        ds_current_date_geos_cf = None

    print("Current date (" + air_pollutant_selected + "): " + current_date.isoformat())
    print("CAMS EU: " + str(not not_available_cams_eu))
    print("CAMS GLOBAL: " + str(not not_available_cams_global))
    print("GEOS-CF: " + str(not not_available_goes_cf))

    # -------------------- Visualization of Heatmap --------------------
    title = air_pollutant_selected + " "
    
    max_value_dict = dict_limit_air_pollutants[air_pollutant_selected]

    if air_pollutant_selected == "CO" and co_in_ug_m3 == False:
        title += "(mg/m^3) "
    elif air_pollutant_selected == "CO" and co_in_ug_m3:
        max_value_dict = dict_limit_air_pollutants["CO_ug_m3"]
        title += "(ug/m^3) "
    else:
        title += "(ug/m^3) "
        
    title += current_date.isoformat()
    
    path_img = joinpath(PATH_CURRENT_MONTH_DIR_PLOTS, current_date.isoformat() + ".jpg")

    list_air_pollutant_ds = [ds_current_date_cams_eu, ds_current_date_cams_global, ds_current_date_geos_cf]

    plot_heatmap(lon_italy_bnds, lat_italy_bnds, list_idx_lat, list_idx_lon, list_air_pollutant_ds, max_value_dict, title, savefig = True, path_fig=path_img, dpi=300)

    previous_date = current_date
    current_date += delta