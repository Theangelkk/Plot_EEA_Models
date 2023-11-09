# conda activate Plot_all_air_pol

# Libraries
import os
import sys

import numpy as np
import pandas as pd
import argparse
import math
import airbase
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import gc
import warnings

warnings.filterwarnings("ignore")

def joinpath(rootdir, targetdir):
    return os.path.join(os.sep, rootdir + os.sep, targetdir)

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

list_years = [2013 + idx for idx in range(0,2023-2013)]
list_air_pollutant = ["CO", "NO2", "O3", "PM2.5", "PM10", "SO2"]
list_numeric_model_cams_eu = [  "chimere", "ensemble", "EMEP", "LOTOS-EUROS", "MATCH", \
                                "MINNI", "MOCAGE", "SILAM", "EURAD-IM", "DEHM", "GEM-AQ"]
list_split_trend_seasonlity = ["multiplicative", "additive"]

parser = argparse.ArgumentParser(description='Decompose Trend and Seasonality of EEA - CAMS Europe - GEOS CF - CAMS Global')
parser.add_argument('-a', '--air_pollutant', help='Air Pollutant CO - NO2 - O3 - PM2.5 - PM10 - SO2', choices=list_air_pollutant, required=True)
parser.add_argument('-list_cod_stations', '--list_cod_stations', help='List of code stations EEA', nargs='+', required=True)
parser.add_argument('-m_air_pol', '--m_air_pol', help='Model level for air pollution', type=int, required=True)
parser.add_argument('-m_pm', '--m_pm', help='Model level for Particulate', type=int, required=True)
parser.add_argument('-delta_h', '--delta_hours', type=int, required=True)
parser.add_argument('-s_date', '--start_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-e_date', '--end_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-cams_eu', '--cams_eu', help='chimere - ensemble - EMEP - LOTOS-EUROS - MATCH - MINNI - MOCAGE - SILAM - EURAD-IM - DEHM - GEM-AQ', \
                     choices=list_numeric_model_cams_eu, required=True)
parser.add_argument('-compute_air_dens', '--compute_air_density', help='Compute with formula the air density', action='store_true')
parser.add_argument('-co_ug_m^3', '--co_ug_m^3', help='CO in ug/m^3', action='store_true')
parser.add_argument('-save_plot', '--save_plot', help='Save plot data', action='store_true')
parser.add_argument('-decompose_trend_seas', '--decompose_trend_seas', help='Method for decomposing the time series analysed', \
                        choices=list_split_trend_seasonlity, required=True)
parser.add_argument('-period_decom', '--period_decom', type=int, required=True)
args = vars(parser.parse_args())

air_poll_selected = args["air_pollutant"]
list_cod_stations = args["list_cod_stations"]
start_date_time_to_display = args["start_date"]
end_date_time_to_display = args["end_date"]
decompose_trend_seas = args["decompose_trend_seas"]
period_decomp = int(args["period_decom"])

if period_decomp <= 0:
    print("Period decomposition must be greater then 0")
    exit(-1)

# If it asked the visualization of CO in ug/m^3
co_in_ug_m3 = args["co_ug_m^3"]

# If it is request to save the plot
save_plot = args["save_plot"]

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

idx_numeric_model_cams_eu = list_numeric_model_cams_eu.index(args["cams_eu"])

fixed_air_density = not bool(args["compute_air_density"])

if args["delta_hours"] > 0:
    delta_time_hours = int(args["delta_hours"])

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

# -------------- EEA paths --------------
path_main_dir_EEA = os.environ['EEA_data']

if path_main_dir_EEA == "":
    print("Error: set the environmental variables of EEA data")
    exit(-1)

path_main_dir_EEA_data = joinpath(path_main_dir_EEA, "EEA_data")
path_main_dir_EEA_metainfo = joinpath(path_main_dir_EEA, "New_metainfo")
path_file_air_pol_metainfo = joinpath(path_main_dir_EEA_metainfo, air_poll_selected + "_IT_metainfo.csv")

path_dir_EEA_data = joinpath(path_main_dir_EEA_data, "IT")
path_dir_EEA_data = joinpath(path_dir_EEA_data, air_poll_selected)
path_dir_EEA_data = joinpath(path_dir_EEA_data, "hour")

filename_EEA_csv = "IT_" + air_poll_selected + "_2013_2023_hour.csv"
path_file_data_EEA_csv = joinpath(path_dir_EEA_data, filename_EEA_csv)

# ------------ Information on CAMS EUROPE ------------
not_available_cams_eu = False

# Path of CAMS Europe
path_main_dir_CAMS_Europe_data = os.environ['CAMS_Europe']

if path_main_dir_CAMS_Europe_data == "":
    print("Error: set the environmental variables of CAMS_Europe")
    exit(-1)

if air_poll_selected == "PM2p5" or air_poll_selected == "PM10":
    DATADIR_CAMS_EU = joinpath(path_main_dir_CAMS_Europe_data, "model_level_" + str(model_level_pm))
else:
    DATADIR_CAMS_EU = joinpath(path_main_dir_CAMS_Europe_data, "model_level_" + str(model_level_air_pollution))

DATADIR_CAMS_EU = joinpath(DATADIR_CAMS_EU, "italy_ext")

numeric_model_selected = list_numeric_model_cams_eu[idx_numeric_model_cams_eu]
DATADIR_CAMS_EU = joinpath(DATADIR_CAMS_EU, numeric_model_selected)

if air_poll_selected == "PM2.5":
    DATADIR_CAMS_EU = joinpath(DATADIR_CAMS_EU, "PM2p5")
else:
    DATADIR_CAMS_EU = joinpath(DATADIR_CAMS_EU, air_poll_selected)

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

# ------------ Information on GEOS CF ------------
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
    if air_poll_selected == "PM2.5":
        air_pollutant_pm25_geos_cf = "PM25_RH35_GCC"
        #air_pollutant_pm25_geos_cf = "PM25_RH35_GOCART"

        DATADIR_GEOS_CF = joinpath(DATADIR_GEOS_CF, air_pollutant_pm25_geos_cf)

    elif air_poll_selected == "PM10":
        print("PM10 not present for GEOS CF")

    else:
        DATADIR_GEOS_CF = joinpath(DATADIR_GEOS_CF, air_poll_selected)
else:
    print("Model level " + str(model_level_air_pollution) + " not available for GEOS CF")

# Time resolution of GEOS CF
time_res_geos_cf = 1

start_time_geos_cf = datetime(2018, 1, 1, 0, 0)
end_time_geos_cf = datetime(2023, 9, 30, 0, 0)

# ------------ Information on CAMS GLOBALE ------------
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

if air_poll_selected == "PM2.5":
    DATADIR_CAMS_GLOBAL = joinpath(DATADIR_CAMS_GLOBAL, "PM2p5")
else:
    DATADIR_CAMS_GLOBAL = joinpath(DATADIR_CAMS_GLOBAL, air_poll_selected)

# Time resolution of CAMS Global
time_res_cams_global = 3

start_time_cams_global = datetime(2003, 1, 1, 0, 0)
end_time_cams_global = datetime(2022, 12, 1, 0, 0)

def load_ds_datasets(current_date):

    global  dict_start_time_numeric_models_cams_eu, numeric_model_selected, \
            end_time_cams_eu, end_time_cams_global, end_time_geos_cf, air_poll_selected, \
            DATADIR_CAMS_EU, DATADIR_CAMS_GLOBAL, DATADIR_CAMS_GEOS_CF, \
            not_available_cams_eu, not_available_cams_global, not_available_goes_cf, \
            co_in_ug_m3
    
    ds_cams_eu = None
    ds_cams_global = None
    ds_geos_cf = None

    if  current_date >= dict_start_time_numeric_models_cams_eu[numeric_model_selected] and \
        current_date <= end_time_cams_eu:
        DATADIR_CURRENT_MONTH_CAMS_EU = joinpath(DATADIR_CAMS_EU, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_cams_eu = joinpath(DATADIR_CURRENT_MONTH_CAMS_EU, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_cams_eu = xr.open_dataset(file_netcdf_path_cams_eu)

        if co_in_ug_m3 and air_poll_selected == "CO":
            ds_cams_eu["co"] *= 1000

        not_available_cams_eu = False    
    else:
        not_available_cams_eu = True
        print("Dataset not available for CAMS EU " +  str(numeric_model_selected))

    if current_date >= start_time_cams_global and current_date <= end_time_cams_global:
        DATADIR_CURRENT_MONTH_CAMS_GLOABL = joinpath(DATADIR_CAMS_GLOBAL, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_cams_global = joinpath(DATADIR_CURRENT_MONTH_CAMS_GLOABL, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_cams_global = xr.open_dataset(file_netcdf_path_cams_global)

        if co_in_ug_m3 and air_poll_selected == "CO":
            ds_cams_global["co"] *= 1000

        not_available_cams_global = False
    else:
        not_available_cams_global = True
        print("Dataset not available for CAMS GLOABL")

    if current_date >= start_time_geos_cf and current_date <= end_time_geos_cf and air_poll_selected != "PM10":
        DATADIR_CURRENT_MONTH_GEOS_CF = joinpath(DATADIR_GEOS_CF, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_geos_cf = joinpath(DATADIR_CURRENT_MONTH_GEOS_CF, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_geos_cf = xr.open_dataset(file_netcdf_path_geos_cf)

        if co_in_ug_m3 and air_poll_selected == "CO":
            ds_geos_cf *= 1000

        not_available_goes_cf = False
    else:
        not_available_goes_cf = True
        print("Dataset not available for GEOS CF")

    return ds_cams_eu, ds_cams_global, ds_geos_cf

def load_EEA_station(   
                        cod_station, current_date, end_date,
                        df_air_pol_data, df_air_pol_metainfo
                    ):

    global path_dir_EEA_data, path_file_data_EEA_csv

    end_current_date = datetime(current_date.year, 12, 31, 23, 0)
    
    if end_current_date > end_date:
        end_current_date = end_date

    # Retrive the meta info of current station 
    df_station_info = df_air_pol_metainfo.filter(like=cod_station, axis=0)
    station_region = df_station_info["Regione"].values[0]
    lon_station = float(df_station_info["Longitude"].values[0])
    lat_station = float(df_station_info["Latitude"].values[0])
    
    # Loading file JSON of the current year
    path_json_current_year = joinpath(path_dir_EEA_data, str(current_date.year) + ".json")

    dict_current_year = None

    with open(path_json_current_year, "r") as f:
        dict_current_year = json.load(f)

    # Check if the current station is available for the current year asked
    if cod_station not in dict_current_year[station_region]["stations"]:
        print("Error: Cod_station is not present in the year required.")
        exit(-1)
    
    # Retrive all measures of current station
    df_station_data = df_air_pol_data.filter(like=cod_station, axis=0)

    df_station_date = df_station_data.set_index('DatetimeBegin')
    df_station_date.index = pd.to_datetime(df_station_date.index)

    # Interpolation of measures
    df_station_date['Concentration'] = df_station_date['Concentration'].interpolate(method='linear')

    df_station_date_current_year = df_station_date[ df_station_date.index.to_series().between(current_date.isoformat(), end_current_date.isoformat())]

    return df_station_date_current_year['Concentration'].values, lon_station, lat_station, station_region

# ----------------------- PLOT -----------------------
def plot(   
            cod_station, air_pol_selected, list_values_EEA_station, \
            list_cams_global, list_values_geos_cf, list_values_cams_eu, \
            PATH_DIR_PLOTS, start_time, end_time
        ):

    global  delta, co_in_ug_m3, save_plot, dict_limit_air_pollutants

    # Set up the axes and figure
    fig, ax = plt.subplots()

    dates = mdates.drange(  
                            start_time, end_time, delta
                        )

    ax.plot(dates, list_values_EEA_station, label="EEA", alpha=0.75, linewidth=3)
    ax.plot(dates, list_cams_global, label="CAMS GLOBAL",  alpha=1.0, linewidth=1)
    ax.plot(dates, list_values_geos_cf, label="GEOS CF",  alpha=1.0, linewidth=1)
    ax.plot(dates, list_values_cams_eu, label="CAMS EU", alpha=1.0, linewidth=1)

    ax.axhline( y=dict_limit_air_pollutants[air_pol_selected],
                color='r', linestyle='dotted', label= air_pol_selected + " limit")

    ax.fmt_xdata = mdates.DateFormatter('% Y-% m-% d % H:% M:% S') 

    # Tell matplotlib to interpret the x-axis values as dates
    ax.xaxis_date()

    # Make space for and rotate the x-axis tick labels
    # Consente di definire automaticamente la spaziatura tra le
    # date al fine di ovviare all'overlapping
    fig.autofmt_xdate()

    if air_pol_selected == "CO" and co_in_ug_m3 == False:
        title = air_pol_selected + " mg/m^3 " + cod_station + " " \
                + start_time.isoformat() + " - " + end_time.isoformat()
        unit = air_pol_selected + " mg/m^3"
    else:
        title = air_pol_selected + " μg/m^3 " + cod_station + " " \
                + start_time.isoformat() + " - " + end_time.isoformat()
        unit = air_pol_selected + " μg/m^3"

    plt.legend()
    plt.title(title)
    plt.xlabel("Datetime")
    plt.ylabel(unit)

    if save_plot:
        filename_fig = start_time.date().strftime("%Y-%m-%d") + "_" + end_time.date().strftime("%Y-%m-%d") + ".png"
        path_fig = joinpath(PATH_DIR_PLOTS, filename_fig)
        plt.savefig(path_fig, dpi=300)
    else:
        plt.show()

    plt.close()

previous_date = start_date_time_to_display
current_date = start_date_time_to_display

ds_cams_eu = None
ds_cams_global = None
ds_geos_cf = None

diff_dates = end_date_time_to_display - start_date_time_to_display
diff_dates_hours = int(diff_dates.total_seconds() / (60*60*delta_time_hours))
delta = timedelta(hours=delta_time_hours)

if period_decomp*2 > diff_dates_hours:
    print("The period decomposition must be contained two complete times in the interval time defined")
    exit(-1)

# Reading matainfo file about air pollutant of the country defined
df_air_pol_metainfo = pd.read_table(path_file_air_pol_metainfo, delimiter=',', index_col=0)

# Reading the CSV file about air pollutant defined
df_air_pol_data = pd.read_table(    
                                    path_file_data_EEA_csv, delimiter=',', 
                                    header=[0], index_col=0, low_memory=False
                                )

# Loading the data sets of EEA ground-based stations
dict_code_stations = {}

# Loading CAMS Europe, GEOS CF e CAMS Global data sets
ds_cams_eu, ds_cams_global, ds_geos_cf = load_ds_datasets(current_date)

list_datetime_x = []
dict_values_EEA_station = {}
dict_values_cams_eu = {}
dict_values_geos_cf = {}
dict_values_cams_global = {}

# Initialization of all dictionaries
for cod_station in list_cod_stations:

    df_station_date_current_year_values, lon_station, lat_station, region_station = \
        load_EEA_station(   cod_station, current_date, end_date_time_to_display-delta, \
                            df_air_pol_data, df_air_pol_metainfo
                        )
    
    dict_code_stations[cod_station] = [lon_station, lat_station, region_station]

    dict_values_EEA_station[cod_station] = []
    dict_values_cams_eu[cod_station] = []
    dict_values_geos_cf[cod_station] = []
    dict_values_cams_global[cod_station] = []

    if co_in_ug_m3 and air_poll_selected == "CO":
        dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values * 1000)
    else:
        dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values)

for time in range(diff_dates_hours):

    # The month is changed
    if previous_date.year != current_date.year:
        
        dict_code_stations = {}

        for cod_station in list_cod_stations:
            df_station_date_current_year_values, lon_station, lat_station, region_station = \
                load_EEA_station(   cod_station, current_date, end_date_time_to_display-delta, \
                                    df_air_pol_data, df_air_pol_metainfo
                            )
    
            dict_code_stations[cod_station] = [lon_station, lat_station, region_station]
            
            if co_in_ug_m3 and air_poll_selected == "CO":
                dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values * 1000)
            else:
                dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values)

        ds_cams_eu, ds_cams_global, ds_geos_cf = load_ds_datasets(current_date)
    
    if previous_date.month != current_date.month:
        ds_cams_eu, ds_cams_global, ds_geos_cf = load_ds_datasets(current_date)

    # For each stations
    for cod_station in list_cod_stations:
        
        lon_station = dict_code_stations[cod_station][0]
        lat_station = dict_code_stations[cod_station][1]

        # Loading CAMS Europe - GEOS CF - CAMS Global data sets
        if not_available_cams_eu == False:
            ds_current_date_cams_eu = ds_cams_eu.sel(time=current_date.isoformat())

            if air_poll_selected == "PM2.5":
                cams_eu_delta_time = ds_current_date_cams_eu.sel(lat=lat_station, lon=lat_station, method='nearest')["pm2p5"].values
            else:
                cams_eu_delta_time = ds_current_date_cams_eu.sel(lat=lat_station, lon=lat_station, method='nearest')[air_poll_selected.lower()].values
            
            dict_values_cams_eu[cod_station].append(float(cams_eu_delta_time))
        else:
            ds_current_date_cams_eu = None

        if (time*delta_time_hours) % time_res_cams_global == 0 and not_available_cams_global == False:
            ds_current_date_cams_global = ds_cams_global.sel(time=current_date.isoformat())

            if air_poll_selected == "PM2.5":
                cams_global_delta_time = ds_current_date_cams_global.sel(latitude=lat_station, longitude=lon_station, method='nearest')["pm2p5"].values
            elif air_poll_selected == "O3":
                cams_global_delta_time = ds_current_date_cams_global.sel(latitude=lat_station, longitude=lon_station, method='nearest')["go3"].values
            else:
                cams_global_delta_time = ds_current_date_cams_global.sel(latitude=lat_station, longitude=lon_station, method='nearest')[air_poll_selected.lower()].values
            
            dict_values_cams_global[cod_station].append(float(cams_global_delta_time))
        else:
            ds_current_date_cams_global = None
            dict_values_cams_global[cod_station].append(np.nan)

        if not_available_goes_cf == False:
            ds_current_date_geos_cf = ds_geos_cf.sel(datetime=current_date.isoformat())

            if air_poll_selected == "PM2.5":
                geos_cf_delta_time = ds_current_date_geos_cf.sel(latitude=lat_station, longitude=lat_station, method='nearest')[air_pollutant_pm25_geos_cf].values
            else:
                geos_cf_delta_time = ds_current_date_geos_cf.sel(latitude=lat_station, longitude=lat_station, method='nearest')[air_poll_selected].values
            
            dict_values_geos_cf[cod_station].append(float(geos_cf_delta_time))
        else:
            ds_current_date_geos_cf = None
        
    list_datetime_x.append(current_date.isoformat())
    
    print("Current date (" + air_poll_selected + "): " + current_date.isoformat())
    print("CAMS EU: " + str(not not_available_cams_eu))
    print("CAMS GLOBAL: " + str(not not_available_cams_global))
    print("GEOS-CF: " + str(not not_available_goes_cf))

    previous_date = current_date
    current_date += delta

# ------------------ Interpolation of CAMS GLOBALE ------------------
for cod_station in list_cod_stations:

    dict_cams_global = {'DatetimeBegin': list_datetime_x, 'Concentration': dict_values_cams_global[cod_station]}
    df_cams_global = pd.DataFrame(data=dict_cams_global)

    df_cams_global = df_cams_global.set_index('DatetimeBegin')
    df_cams_global.index = pd.to_datetime(df_cams_global.index)

    # Interpolation
    df_cams_global['Concentration'] = df_cams_global['Concentration'].interpolate(method='linear')

    dict_values_cams_global[cod_station] = df_cams_global['Concentration'].values

# ------------------ Decompose Trend and Sesonality ------------------
# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "Dec_Trend_Seas_EEA_and_all_air_model_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm) + "_camsEU_" + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]))

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

if fixed_air_density:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "FIXED_AIR_DENSITY")
else:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "FORMULA_AIR_DENSITY")

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

if air_poll_selected == "PM2.5":
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, air_pollutant_pm25_geos_cf)
else: 
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, air_poll_selected)

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

if air_poll_selected == "CO" and co_in_ug_m3:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_ug_m^3")
elif air_poll_selected == "CO":
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_mg_m^3")

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

# Multiplicative Decomposition: This technique allws to decompose the product of
# the Trend-Cycle Component T, Seasonal Component S and the noise R:
# Y = T * S * R
if decompose_trend_seas == list_split_trend_seasonlity[0]:
    method_decomposition = list_split_trend_seasonlity[0]

# Additive Decomposition: This technique allws to decompose the sum of
# the Trend-Cycle Component T, Seasonal Component S and the noise R:
# Y = T + S + R
else:
    method_decomposition = list_split_trend_seasonlity[1]

for cod_station in list_cod_stations:

    region_cod_station = dict_code_stations[cod_station][2]
    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS, region_cod_station)

    if not os.path.exists(PATH_DIR_PLOTS_current):
        os.mkdir(PATH_DIR_PLOTS_current)

    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS_current, cod_station)

    if not os.path.exists(PATH_DIR_PLOTS_current):
        os.mkdir(PATH_DIR_PLOTS_current)

    # Decomposition of EEA station
    decomp_EEA = seasonal_decompose(pd.Series(dict_values_EEA_station[cod_station]), model=method_decomposition, period=period_decomp)
    
    plt.rcParams.update({'figure.figsize': (16,12)})
    decomp_EEA.plot().suptitle( method_decomposition + " decomposition period " + str(period_decomp) + \
                                " of " + air_poll_selected + " of EEA - " + str(cod_station), fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_plot:
        filename_fig = method_decomposition + "_decomp_period_" + str(period_decomp) + "_EEA.png"
        path_fig = joinpath(PATH_DIR_PLOTS_current, filename_fig)
        plt.savefig(path_fig, dpi=300)
    else:
        plt.show()

    # Decomposition of CAMS Europe
    if not_available_cams_eu == False:
        decomp_cams_eu = seasonal_decompose(pd.Series(dict_values_cams_eu[cod_station]), model=method_decomposition, period=period_decomp)
        plt.rcParams.update({'figure.figsize': (16,12)})
        decomp_cams_eu.plot().suptitle( method_decomposition + " decomposition period " + str(period_decomp) + \
                                        " of " + air_poll_selected + " of CAMS Europe - " + str(cod_station),  fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plot:
            filename_fig = method_decomposition + "_decomp_period_" + str(period_decomp) + "_CAMS_Europe.png"
            path_fig = joinpath(PATH_DIR_PLOTS_current, filename_fig)
            plt.savefig(path_fig, dpi=300)
        else:
            plt.show()

    # Decomposition of GEOS CF
    if not_available_goes_cf == False:
        decomp_geos_cf = seasonal_decompose(pd.Series(dict_values_geos_cf[cod_station]), model=method_decomposition, period=period_decomp)
        plt.rcParams.update({'figure.figsize': (16,12)})
        decomp_cams_eu.plot().suptitle( method_decomposition + " decomposition period " + str(period_decomp) + \
                                        " of " + air_poll_selected + " of GEOS CF - " + str(cod_station), fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plot:
            filename_fig = method_decomposition + "_decomp_period_" + str(period_decomp) + "_GEOS_CF.png"
            path_fig = joinpath(PATH_DIR_PLOTS_current, filename_fig)
            plt.savefig(path_fig, dpi=300)
        else:
            plt.show()

    # Decomposition of CAMS Global
    if not_available_cams_global == False:
        decomp_cams_global = seasonal_decompose(pd.Series(dict_values_cams_global[cod_station]), model=method_decomposition, period=period_decomp)
        plt.rcParams.update({'figure.figsize': (16,12)})
        decomp_cams_global.plot().suptitle( method_decomposition + " decomposition period " + str(period_decomp) + \
                                        " of " + air_poll_selected + " of CAMS Global - " + str(cod_station), fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plot:
            filename_fig = method_decomposition + "_decomp_period_" + str(period_decomp) + "_CAMS_Global.png"
            path_fig = joinpath(PATH_DIR_PLOTS_current, filename_fig)
            plt.savefig(path_fig, dpi=300)
        else:
            plt.show()