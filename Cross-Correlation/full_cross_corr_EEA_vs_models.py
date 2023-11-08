# Full Cross-correlation EEA vs Models

# Libreries
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
from scipy import signal
from scipy.ndimage import shift
import gc
import warnings

warnings.filterwarnings("ignore")

def joinpath(rootdir, targetdir):
    return os.path.join(os.sep, rootdir + os.sep, targetdir)

list_years = [2013 + idx for idx in range(0,2023-2013)]
list_methods_of_corr = ['pearson', 'kendall', 'spearman']
list_air_pollutant = ["CO", "NO2", "O3", "PM2.5", "PM10", "SO2"]
list_numeric_model_cams_eu = [  "chimere", "ensemble", "EMEP", "LOTOS-EUROS", "MATCH", \
                                "MINNI", "MOCAGE", "SILAM", "EURAD-IM", "DEHM", "GEM-AQ"]

parser = argparse.ArgumentParser(description='Full Cross-Correlation EEA - CAMS EU - GEOS CF - CAMS GLOBALE')
parser.add_argument('-a', '--air_pollutant', help='Air Pollutant CO - NO2 - O3 - PM2.5 - PM10 - SO2', choices=list_air_pollutant, required=True)
parser.add_argument('-list_cod_stations', '--list_cod_stations', help='List of code stations EEA', nargs='+', required=True)
parser.add_argument('-m_air_pol', '--m_air_pol', help='Model level for air pollution', type=int, required=True)
parser.add_argument('-m_pm', '--m_pm', help='Model level for Particulate', type=int, required=True)
parser.add_argument('-delta_h', '--delta_hours', type=int, required=True)
parser.add_argument('-year', '--year', help='Year to consider (2013,2023)', type=int, choices=list_years, required=True)
parser.add_argument('-cams_eu', '--cams_eu', help='chimere - ensemble - EMEP - LOTOS-EUROS - MATCH - MINNI - MOCAGE - SILAM - EURAD-IM - DEHM - GEM-AQ', \
                     choices=list_numeric_model_cams_eu, required=True)
parser.add_argument('-compute_air_dens', '--compute_air_density', help='Compute with formula the air density', action='store_true')
parser.add_argument('-co_ug_m^3', '--co_ug_m^3', help='CO in ug/m^3', action='store_true')
parser.add_argument('-save_plot', '--save_plot', help='Save plot data', action='store_true')
args = vars(parser.parse_args())

air_poll_selected = args["air_pollutant"]
list_cod_stations = args["list_cod_stations"]
year_to_consider = args["year"]

# If it asked the visualization of CO in ug/m^3
co_in_ug_m3 = args["co_ug_m^3"]

# If it is request to save the plot
save_plot = args["save_plot"]

# Model level 55: about 288 meters [CAMS Globale, GEOS CF, CAMS EU]
# Model level 60: about 10 meters [CAMS Globale, CAMS EU]
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
def plot_corr_lags_two_time_series( np_first_ts, np_second_ts, np_corr, np_full_corr, np_lags, \
                                    index_max_value_corr, shifted_corr, title_first_ts, title_second_ts, PATH_DIR_PLOTS):
        
    global start_date_year, end_date_year, delta, air_poll_selected, save_plot

    if air_poll_selected == "CO" and co_in_ug_m3 == False:
        title = air_poll_selected + " mg/m^3 " + cod_station + " " \
                + start_date_year.isoformat() + " - " + end_date_year.isoformat()
        unit = air_poll_selected + " mg/m^3"
    else:
        title = air_poll_selected + " μg/m^3 " + cod_station + " " \
                + start_date_year.isoformat() + " - " + end_date_year.isoformat()
        unit = air_poll_selected + " μg/m^3"

    fig, (ax_first_ts, ax_second_ts, ax_corr, ax_lags, ax_shifted_corr) = plt.subplots(5, 1, figsize=(10, 8))

    plt.title(title)

    dates = mdates.drange(start_date_year, end_date_year, delta)
    
    # Plot of first time series
    ax_first_ts.plot(dates, np_first_ts, label=title_first_ts, linewidth=1)
    ax_first_ts.set_xlabel('Datetime')
    ax_first_ts.set_ylabel(unit)
    ax_first_ts.legend()

    ax_first_ts.fmt_xdata = mdates.DateFormatter('% Y-% m-% d % H:% M:% S') 
    ax_first_ts.xaxis_date()

    fig.tight_layout()
    fig.autofmt_xdate()

    # Plot of second time series
    ax_second_ts.plot(dates, np_second_ts, label=title_second_ts, linewidth=1)
    ax_second_ts.set_xlabel('Datetime')
    ax_second_ts.set_ylabel(unit)
    ax_second_ts.legend()

    ax_second_ts.fmt_xdata = mdates.DateFormatter('% Y-% m-% d % H:% M:% S') 
    ax_second_ts.xaxis_date()

    fig.tight_layout()
    fig.autofmt_xdate()

    # Plot of cross-correlation
    ax_corr.plot(dates, np_corr, label="Cross-correlation", linewidth=1)
    ax_corr.set_xlabel('Datetime')
    ax_corr.legend()

    ax_corr.fmt_xdata = mdates.DateFormatter('% Y-% m-% d % H:% M:% S') 
    ax_corr.xaxis_date()

    fig.tight_layout()
    fig.autofmt_xdate()

    # Plot of cross-correlation lags
    best_lag_value = np_lags[index_max_value_corr]
    ax_lags.plot(np_lags, np_full_corr, label="Full Cross-correlation lags", linewidth=1)
    ax_lags.axvline(x = best_lag_value, color = 'r', label = "Best lag " + str(best_lag_value), linewidth=1)
    ax_lags.set_xlabel('Lag')
    ax_lags.legend()

    # Plot of shifted cross-correlation
    ax_shifted_corr.plot(dates, shifted_corr, label="Shifted best lag " + str(best_lag_value) + " Cross-correlation", linewidth=1)
    ax_shifted_corr.set_xlabel('Datetime')
    ax_shifted_corr.legend()

    ax_shifted_corr.fmt_xdata = mdates.DateFormatter('% Y-% m-% d % H:% M:% S') 
    ax_shifted_corr.xaxis_date()

    fig.tight_layout()
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.autofmt_xdate()

    if save_plot:
        filename_fig = "Cross_corr_" + title_first_ts.replace(" ", "-") + "_" + title_second_ts.replace(" ", "_") + "_" + \
                        start_date_year.date().strftime("%Y-%m-%d") + "_" + end_date_year.date().strftime("%Y-%m-%d") + ".png"
        path_fig = joinpath(PATH_DIR_PLOTS, filename_fig)
        plt.savefig(path_fig, dpi=300)
    else:
        plt.show()

    plt.close()

start_date_year = datetime(year_to_consider, 1, 1, 0, 0)
end_date_year = datetime(year_to_consider+1, 1, 1, 0, 0)

previous_date = start_date_year
current_date = start_date_year

ds_cams_eu = None
ds_cams_global = None
ds_geos_cf = None

diff_dates = end_date_year - start_date_year
diff_dates_hours = int(diff_dates.total_seconds() / (60*60*delta_time_hours))
delta = timedelta(hours=delta_time_hours)

# Reading matainfo file about air pollutant of the country defined
df_air_pol_metainfo = pd.read_table(path_file_air_pol_metainfo, delimiter=',', index_col=0)

# Reading the CSV file about air pollutant defined
df_air_pol_data = pd.read_table(    path_file_data_EEA_csv, delimiter=',', 
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
        load_EEA_station(   cod_station, current_date, end_date_year-delta, \
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
                load_EEA_station(   cod_station, current_date, end_date_year-delta, \
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

# ------------------ Interpolation of CAMS GLOBAL ------------------
for cod_station in list_cod_stations:

    dict_cams_global = {'DatetimeBegin': list_datetime_x, 'Concentration': dict_values_cams_global[cod_station]}
    df_cams_global = pd.DataFrame(data=dict_cams_global)

    df_cams_global = df_cams_global.set_index('DatetimeBegin')
    df_cams_global.index = pd.to_datetime(df_cams_global.index)

    # Interpolation
    df_cams_global['Concentration'] = df_cams_global['Concentration'].interpolate(method='linear')

    dict_values_cams_global[cod_station] = df_cams_global['Concentration'].values

# ------------------ Full Cross-Correlation ------------------
def write_file_corr(    
                        corr, lags, lag_max_value_corr, max_value_corr, \
                        title1, title2, PATH_DIR
                    ):
        
    global start_date_year, end_date_year

    filename =  title1.replace(" ", "_") + "_" + title2.replace(" ", "_") + "_" + \
                start_date_year.date().strftime("%Y-%m-%d") + "_" + end_date_year.date().strftime("%Y-%m-%d") + ".txt"
    path_file_log = joinpath(PATH_DIR, filename)

    with open(path_file_log, "w") as file:
        file.write( "Cross-Corrlation with lags between " + title1 + " and " + title2 + \
                    " from " + start_date_year.isoformat() + " to " + end_date_year.isoformat() + "\n\n")
        
        file.write("Best lag " + str(lag_max_value_corr) + " : " + str(max_value_corr) + "\n\n")

        for i in range(corr.shape[0]):
            string_output = "Lag " + str(lags[i]) + ": " + str(corr[i]) + "\n"
            file.write(string_output)

# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "Full_cross_corr_EEA_vs_all_air_model_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm) + "_camsEU_" + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]))

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

for cod_station in list_cod_stations:

    region_cod_station = dict_code_stations[cod_station][2]
    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS, region_cod_station)

    if not os.path.exists(PATH_DIR_PLOTS_current):
        os.mkdir(PATH_DIR_PLOTS_current)

    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS_current, cod_station)

    if not os.path.exists(PATH_DIR_PLOTS_current):
        os.mkdir(PATH_DIR_PLOTS_current)

    np_values_EEA_station = np.array(dict_values_EEA_station[cod_station])
    np_values_CAMS_eu = np.array(dict_values_cams_eu[cod_station])
    np_values_geos_cf = np.array(dict_values_geos_cf[cod_station])
    np_values_CAMS_global = np.array(dict_values_cams_global[cod_station])

    # NOTE: 
    # Correlate equal to 1 indicates that the two time series observed are correlated, or
    # in other simple words, all samples of two time series are related among them. It is
    # important to note that, the array of correlations is divided respect to the total number
    # of time series samples.
    #
    # Full cross-corration with lag equal to 1 allows to find the peak of the cross-correlation
    # respect to the lag analysed. For sake of simplicity, the best lag of two time series has the 
    # maximum cross-correlation value.  

    # LINK: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
    # missing_value_elem = np.NaA
    missing_value_elem = 0

    # Full Cross-Correlation of EEA station vs CAMS Europe
    if not_available_cams_eu == False:
        corr_EEA_vs_cams_eu = signal.correlate(np_values_EEA_station, np_values_CAMS_eu, mode='same') / np_values_EEA_station.shape[0]
        full_corr_EEA_vs_cams_eu = signal.correlate(np_values_EEA_station, np_values_CAMS_eu)
        lags_EEA_vs_cams_eu = signal.correlation_lags(len(np_values_EEA_station), len(np_values_CAMS_eu))

        index_max_value_corr_EEA_vs_cams_eu = full_corr_EEA_vs_cams_eu.argmax()
        max_value_corr_EEA_vs_cams_eu = full_corr_EEA_vs_cams_eu[index_max_value_corr_EEA_vs_cams_eu]
        lag_max_value_corr_EEA_vs_cams_eu = lags_EEA_vs_cams_eu[index_max_value_corr_EEA_vs_cams_eu]
        
        # Normalization between 0 and 1 of Full Cross-correlation lags
        full_corr_EEA_vs_cams_eu /= max_value_corr_EEA_vs_cams_eu

        # Compute the shifted CAMS Europe due to the best lag computed
        shifted_np_values_CAMS_eu = shift(np_values_CAMS_eu, lag_max_value_corr_EEA_vs_cams_eu, cval=missing_value_elem)
        shifted_corr_EEA_vs_cams_eu = signal.correlate( np_values_EEA_station, shifted_np_values_CAMS_eu, mode='same') / np_values_EEA_station.shape[0]
        
        plot_corr_lags_two_time_series( 
                                        np_values_EEA_station, np_values_CAMS_eu, \
                                        corr_EEA_vs_cams_eu, full_corr_EEA_vs_cams_eu, lags_EEA_vs_cams_eu, \
                                        index_max_value_corr_EEA_vs_cams_eu, shifted_corr_EEA_vs_cams_eu, \
                                        "EEA", "CAMS EU", PATH_DIR_PLOTS_current
                                    )

        write_file_corr(    
                            full_corr_EEA_vs_cams_eu, lags_EEA_vs_cams_eu, \
                            lag_max_value_corr_EEA_vs_cams_eu, max_value_corr_EEA_vs_cams_eu, \
                            "EEA", "CAMS EU", PATH_DIR_PLOTS_current
                        )
    
    # Full Cross-Correlation of EEA station vs GEOS CF
    if not_available_goes_cf == False:
        corr_EEA_vs_geos_cf = signal.correlate(np_values_EEA_station, np_values_geos_cf, mode='same') / np_values_EEA_station.shape[0]
        full_corr_EEA_vs_geos_cf = signal.correlate(np_values_EEA_station, np_values_geos_cf)
        lags_EEA_vs_geos_cf = signal.correlation_lags(len(np_values_EEA_station), len(np_values_geos_cf))

        index_max_value_corr_EEA_vs_geos_cf = full_corr_EEA_vs_geos_cf.argmax()
        max_value_corr_EEA_vs_geos_cf = full_corr_EEA_vs_geos_cf[index_max_value_corr_EEA_vs_geos_cf]
        lag_max_value_corr_EEA_vs_geos_cf = lags_EEA_vs_geos_cf[index_max_value_corr_EEA_vs_geos_cf]
        
        # Normalization between 0 and 1 of Full Cross-correlation lags
        full_corr_EEA_vs_geos_cf /= max_value_corr_EEA_vs_geos_cf

        # Compute the shifted GEOS CF due to the best lag computed
        shifted_np_values_geos_cf = shift(np_values_geos_cf, lag_max_value_corr_EEA_vs_geos_cf, cval=missing_value_elem)
        shifted_corr_EEA_vs_geos_cf = signal.correlate( np_values_EEA_station, shifted_np_values_geos_cf, mode='same') / np_values_EEA_station.shape[0]
        
        plot_corr_lags_two_time_series( 
                                        np_values_EEA_station, np_values_geos_cf, \
                                        corr_EEA_vs_geos_cf, full_corr_EEA_vs_geos_cf, lags_EEA_vs_geos_cf, \
                                        index_max_value_corr_EEA_vs_geos_cf, shifted_corr_EEA_vs_geos_cf, \
                                        "EEA", "GEOS CF", PATH_DIR_PLOTS_current
                                    )
        
        write_file_corr(    
                            full_corr_EEA_vs_geos_cf, lags_EEA_vs_geos_cf, \
                            lag_max_value_corr_EEA_vs_geos_cf, max_value_corr_EEA_vs_geos_cf, \
                            "EEA", "GEOS CF", PATH_DIR_PLOTS_current
                        )

    # Full Cross-Correlation of EEA station vs CAMS Global
    if not_available_cams_global == False:
        corr_EEA_vs_cams_global = signal.correlate(np_values_EEA_station, np_values_CAMS_global, mode='same') / np_values_EEA_station.shape[0]
        full_corr_EEA_vs_cams_global = signal.correlate(np_values_EEA_station, np_values_CAMS_global)
        lags_EEA_vs_cams_global = signal.correlation_lags(len(np_values_EEA_station), len(np_values_CAMS_global))

        index_max_value_corr_EEA_vs_cams_global = full_corr_EEA_vs_cams_global.argmax()
        max_value_corr_EEA_vs_cams_global = full_corr_EEA_vs_cams_global[index_max_value_corr_EEA_vs_cams_global]
        lag_max_value_corr_EEA_vs_cams_global = lags_EEA_vs_cams_global[index_max_value_corr_EEA_vs_cams_global]
        
        # Normalization between 0 and 1 of Full Cross-correlation lags
        full_corr_EEA_vs_cams_global /= max_value_corr_EEA_vs_cams_global

        # Compute the shifted CAMS Global due to the best lag computed
        shifted_np_values_cams_global = shift(np_values_CAMS_global, lag_max_value_corr_EEA_vs_cams_global, cval=missing_value_elem)
        shifted_corr_EEA_vs_cams_global = signal.correlate( np_values_EEA_station, shifted_np_values_cams_global, mode='same') / np_values_EEA_station.shape[0]
        
        plot_corr_lags_two_time_series( 
                                        np_values_EEA_station, np_values_CAMS_global, \
                                        corr_EEA_vs_cams_global, full_corr_EEA_vs_cams_global, lags_EEA_vs_cams_global, \
                                        index_max_value_corr_EEA_vs_cams_global, shifted_corr_EEA_vs_cams_global, \
                                        "EEA", "CAMS Global", PATH_DIR_PLOTS_current
                                    )
        
        write_file_corr(    
                            full_corr_EEA_vs_cams_global, lags_EEA_vs_cams_global, \
                            lag_max_value_corr_EEA_vs_cams_global, max_value_corr_EEA_vs_cams_global, \
                            "EEA", "CAMS Global", PATH_DIR_PLOTS_current
                        )

    # Full Cross-Correlation of CAMS EU vs GEOS CF
    if not_available_cams_eu == False and not_available_goes_cf == False:
        corr_cams_eu_vs_geos_cf = signal.correlate(np_values_CAMS_eu, np_values_geos_cf, mode='same') / np_values_CAMS_eu.shape[0]
        full_corr_cams_eu_vs_geos_cf = signal.correlate(np_values_CAMS_eu, np_values_geos_cf)
        lags_cams_eu_vs_geos_cf = signal.correlation_lags(len(np_values_CAMS_eu), len(np_values_geos_cf))

        index_max_value_corr_cams_eu_vs_geos_cf = full_corr_cams_eu_vs_geos_cf.argmax()
        max_value_corr_cams_eu_vs_geos_cf = full_corr_cams_eu_vs_geos_cf[index_max_value_corr_cams_eu_vs_geos_cf]
        lag_max_value_corr_cams_eu_vs_geos_cf = lags_cams_eu_vs_geos_cf[index_max_value_corr_cams_eu_vs_geos_cf]
        
        # Normalization between 0 and 1 of Full Cross-correlation lags
        full_corr_cams_eu_vs_geos_cf /= max_value_corr_cams_eu_vs_geos_cf

        # Compute the shifted GEOS CF due to the best lag computed
        shifted_np_values_geos_cf = shift(np_values_geos_cf, lag_max_value_corr_cams_eu_vs_geos_cf, cval=missing_value_elem)
        shifted_corr_cams_eu_vs_geos_cf = signal.correlate(np_values_CAMS_eu, shifted_np_values_geos_cf, mode='same') / np_values_CAMS_eu.shape[0]
        
        plot_corr_lags_two_time_series( 
                                        np_values_CAMS_eu, np_values_geos_cf, \
                                        corr_cams_eu_vs_geos_cf, full_corr_cams_eu_vs_geos_cf, lags_cams_eu_vs_geos_cf, \
                                        index_max_value_corr_cams_eu_vs_geos_cf, shifted_corr_cams_eu_vs_geos_cf, \
                                        "CAMS EU", "GEOS CF", PATH_DIR_PLOTS_current
                                    )

        write_file_corr(    
                            full_corr_cams_eu_vs_geos_cf, lags_cams_eu_vs_geos_cf, \
                            lag_max_value_corr_cams_eu_vs_geos_cf, max_value_corr_cams_eu_vs_geos_cf, \
                            "CAMS EU", "GEOS CF", PATH_DIR_PLOTS_current
                        )

    # Full Cross-Correlation of CAMS EU vs CAMS Global
    if not_available_cams_eu == False and not_available_cams_global == False:
        corr_cams_eu_vs_cams_global = signal.correlate(np_values_CAMS_eu, np_values_CAMS_global, mode='same') / np_values_CAMS_eu.shape[0]
        full_corr_cams_eu_vs_cams_global = signal.correlate(np_values_CAMS_eu, np_values_CAMS_global)
        lags_cams_eu_vs_cams_global = signal.correlation_lags(len(np_values_CAMS_eu), len(np_values_CAMS_global))

        index_max_value_corr_cams_eu_vs_cams_global = full_corr_cams_eu_vs_cams_global.argmax()
        max_value_corr_cams_eu_vs_cams_global = full_corr_cams_eu_vs_cams_global[index_max_value_corr_cams_eu_vs_cams_global]
        lag_max_value_corr_cams_eu_vs_cams_global = lags_cams_eu_vs_cams_global[index_max_value_corr_cams_eu_vs_cams_global]
        
        # Normalization between 0 and 1 of Full Cross-correlation lags
        full_corr_cams_eu_vs_cams_global /= max_value_corr_cams_eu_vs_cams_global

        # Compute the shifted CAMS Global due to the best lag computed
        shifted_np_values_cams_global = shift(np_values_CAMS_global, lag_max_value_corr_cams_eu_vs_cams_global, cval=missing_value_elem)
        shifted_corr_cams_eu_vs_cams_global = signal.correlate(np_values_CAMS_eu, shifted_np_values_cams_global, mode='same') / np_values_CAMS_eu.shape[0]
        
        plot_corr_lags_two_time_series( 
                                        np_values_CAMS_eu, np_values_CAMS_global, \
                                        corr_cams_eu_vs_cams_global, full_corr_cams_eu_vs_cams_global, lags_cams_eu_vs_cams_global, \
                                        index_max_value_corr_cams_eu_vs_cams_global, shifted_corr_cams_eu_vs_cams_global, \
                                        "CAMS EU", "CAMS Global", PATH_DIR_PLOTS_current
                                    )
        
        write_file_corr(    full_corr_cams_eu_vs_cams_global, lags_cams_eu_vs_cams_global, \
                            lag_max_value_corr_cams_eu_vs_cams_global, max_value_corr_cams_eu_vs_cams_global, \
                            "CAMS EU", "CAMS Global", PATH_DIR_PLOTS_current)
    
    # Full Cross-Correlation of GEOS CF vs CAMS Global
    if not_available_cams_eu == False and not_available_goes_cf == False:
        corr_geos_cf_vs_cams_global = signal.correlate(np_values_geos_cf, np_values_CAMS_global, mode='same') / np_values_geos_cf.shape[0]
        full_corr_geos_cf_vs_cams_global = signal.correlate(np_values_geos_cf, np_values_CAMS_global)
        lags_geos_cf_vs_cams_global = signal.correlation_lags(len(np_values_geos_cf), len(np_values_CAMS_global))

        index_max_value_corr_geos_cf_vs_cams_global = full_corr_geos_cf_vs_cams_global.argmax()
        max_value_corr_geos_cf_vs_cams_global = full_corr_geos_cf_vs_cams_global[index_max_value_corr_geos_cf_vs_cams_global]
        lag_max_value_corr_geos_cf_vs_cams_global = lags_geos_cf_vs_cams_global[index_max_value_corr_geos_cf_vs_cams_global]
        
        # Normalization between 0 and 1 of Full Cross-correlation lags
        full_corr_geos_cf_vs_cams_global /= max_value_corr_geos_cf_vs_cams_global

        # Compute the shifted CAMS Global due to the best lag computed
        shifted_np_values_cams_global = shift(np_values_CAMS_global, lag_max_value_corr_geos_cf_vs_cams_global, cval=missing_value_elem)
        shifted_corr_geos_cf_vs_cams_global = signal.correlate(np_values_geos_cf, shifted_np_values_cams_global, mode='same') / np_values_CAMS_eu.shape[0]
        
        plot_corr_lags_two_time_series( 
                                        np_values_geos_cf, np_values_CAMS_global, \
                                        corr_geos_cf_vs_cams_global, full_corr_geos_cf_vs_cams_global, lags_geos_cf_vs_cams_global, \
                                        index_max_value_corr_geos_cf_vs_cams_global, shifted_corr_geos_cf_vs_cams_global, \
                                        "GEOS CF", "CAMS Global", PATH_DIR_PLOTS_current
                                    )
        
        write_file_corr(    
                            full_corr_geos_cf_vs_cams_global, lags_geos_cf_vs_cams_global, \
                            lag_max_value_corr_geos_cf_vs_cams_global, max_value_corr_geos_cf_vs_cams_global, \
                            "GEOS CF", "CAMS Global", PATH_DIR_PLOTS_current
                        )