# Auto-correlation function (ACF) among EEA, CAMS Europe and Global Reanalyses

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
from pandas.plotting import autocorrelation_plot
import gc
import warnings

warnings.filterwarnings("ignore")

def joinpath(rootdir, targetdir):
    return os.path.join(os.sep, rootdir + os.sep, targetdir)

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

list_methods_of_corr = ['pearson', 'kendall', 'spearman']
list_air_pollutant = ["CO", "NO2", "O3", "PM2.5", "PM10", "SO2"]
list_numeric_model_cams_eu = [  "chimere", "ensemble", "EMEP", "LOTOS-EUROS", "MATCH", \
                                "MINNI", "MOCAGE", "SILAM", "EURAD-IM", "DEHM", "GEM-AQ"]
list_freq_mode = ["hour", "day"]

parser = argparse.ArgumentParser(description='Auto-Correlation EEA - CAMS EU - GEOS CF - CAMS GLOBALE')
parser.add_argument('-a', '--air_pollutant', help='Air Pollutant CO - NO2 - O3 - PM2.5 - PM10 - SO2', choices=list_air_pollutant, required=True)
parser.add_argument('-list_cod_stations', '--list_cod_stations', help='List of code stations EEA', nargs='+', required=True)
parser.add_argument('-m_air_pol', '--m_air_pol', help='Model level for air pollution', type=int, required=True)
parser.add_argument('-m_pm', '--m_pm', help='Model level for Particulate', type=int, required=True)
parser.add_argument('-freq_mode', '--freq_mode', help='Frequency mode: hour or day', choices=list_freq_mode, required=True)
parser.add_argument('-s_date', '--start_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-e_date', '--end_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-cams_eu', '--cams_eu', help='chimere - ensemble - EMEP - LOTOS-EUROS - MATCH - MINNI - MOCAGE - SILAM - EURAD-IM - DEHM - GEM-AQ', \
                     choices=list_numeric_model_cams_eu, required=True)
parser.add_argument('-compute_air_dens', '--compute_air_density', help='Compute with formula the air density', action='store_true')
parser.add_argument('-co_ug_m^3', '--co_ug_m^3', help='CO in ug/m^3', action='store_true')
parser.add_argument('-save_plot', '--save_plot', help='Save plot data', action='store_true')
args = vars(parser.parse_args())

air_poll_selected = args["air_pollutant"]
list_cod_stations = args["list_cod_stations"]
start_date_time_to_display = args["start_date"]
end_date_time_to_display = args["end_date"]

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

freq_mode = args["freq_mode"]

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
path_dir_EEA_data = joinpath(path_dir_EEA_data, freq_mode)

filename_EEA_csv = "IT_" + air_poll_selected + "_2013_2023_" + freq_mode + ".csv"
path_file_data_EEA_csv = joinpath(path_dir_EEA_data, filename_EEA_csv)

# ------------ Information on CAMS EUROPE Reanalyses ------------
not_available_cams_eu_reanalyses = False

# Path of CAMS Europe
path_main_dir_CAMS_Europe_data_reanalyses = os.environ['CAMS_Europe_Reanalyses']

if path_main_dir_CAMS_Europe_data_reanalyses == "":
    print("Error: set the environmental variables of CAMS_Europe_Reanalyses")
    exit(-1)

if air_poll_selected == "PM2p5" or air_poll_selected == "PM10":
    DATADIR_CAMS_EU_reanalyses = joinpath(path_main_dir_CAMS_Europe_data_reanalyses, "model_level_" + str(model_level_pm))
else:
    DATADIR_CAMS_EU_reanalyses = joinpath(path_main_dir_CAMS_Europe_data_reanalyses, "model_level_" + str(model_level_air_pollution))

DATADIR_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, "italy_ext")

numeric_model_selected = list_numeric_model_cams_eu[idx_numeric_model_cams_eu]
DATADIR_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, numeric_model_selected)

if air_poll_selected == "PM2.5":
    DATADIR_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, "PM2p5")
else:
    DATADIR_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, air_poll_selected)

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

end_time_cams_eu_reanalyses = datetime(2020, 12, 31, 0, 0)

# ------------ Information on CAMS GLOBAL Reanalyses ------------
not_available_cams_global_reanalyses = False

# Path of CAMS_Global
path_main_dir_CAMS_Global_reanalyses_data = os.environ['CAMS_Global_Reanalyses']

if path_main_dir_CAMS_Global_reanalyses_data == "":
    print("Error: set the environmental variables of CAMS_Global_Reanalyses")
    exit(-1)

DATADIR_CAMS_GLOBAL_renalyses = joinpath(path_main_dir_CAMS_Global_reanalyses_data, "datasets_model_level_" + str(model_level_air_pollution))

if fixed_air_density:
    DATADIR_CAMS_GLOBAL_reanalyses = joinpath(DATADIR_CAMS_GLOBAL_renalyses, "italy_ext_fixed")
else:
    DATADIR_CAMS_GLOBAL_reanalyses = joinpath(DATADIR_CAMS_GLOBAL_renalyses, "italy_ext_formula")

if air_poll_selected == "PM2.5":
    DATADIR_CAMS_GLOBAL_reanalyses = joinpath(DATADIR_CAMS_GLOBAL_reanalyses, "PM2p5")
else:
    DATADIR_CAMS_GLOBAL_reanalyses = joinpath(DATADIR_CAMS_GLOBAL_reanalyses, air_poll_selected)

# Time resolution of CAMS Global Reanalyses
time_res_cams_global_reanalyses = 3

start_time_cams_global_reanalyses = datetime(2003, 1, 1, 0, 0)
end_time_cams_global_reanalyses = datetime(2022, 12, 1, 0, 0)

def load_ds_datasets(current_date):

    global  dict_start_time_numeric_models_cams_eu_reanalyses, numeric_model_selected, \
            end_time_cams_eu_reanalyses, end_time_cams_global_reanalyses, air_poll_selected, \
            DATADIR_CAMS_EU_reanalyses, DATADIR_CAMS_GLOBAL_reanalyses, \
            not_available_cams_eu_reanalyses, not_available_cams_global_reanalyses, \
            co_in_ug_m3
    
    ds_cams_eu_reanalyses = None
    ds_cams_global_reanalyses = None

    if  current_date >= dict_start_time_numeric_models_cams_eu_reanalyses[numeric_model_selected] and \
        current_date <= end_time_cams_eu_reanalyses:
        DATADIR_CURRENT_MONTH_CAMS_EU_reanalyses = joinpath(DATADIR_CAMS_EU_reanalyses, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_cams_eu_reanalyses = joinpath(DATADIR_CURRENT_MONTH_CAMS_EU_reanalyses, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_cams_eu_reanalyses = xr.open_dataset(file_netcdf_path_cams_eu_reanalyses)

        if co_in_ug_m3 and air_poll_selected == "CO":
            ds_cams_eu_reanalyses["co"] *= 1000

        not_available_cams_eu_reanalyses = False    
    else:
        not_available_cams_eu_reanalyses = True
        print("Dataset not available for CAMS EU Reanalyses" +  str(numeric_model_selected))

    if current_date >= start_time_cams_global_reanalyses and current_date <= end_time_cams_global_reanalyses:
        DATADIR_CURRENT_MONTH_CAMS_GLOBAL__reanalyses = joinpath(DATADIR_CAMS_GLOBAL_reanalyses, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_cams_global_reanalyses = joinpath(DATADIR_CURRENT_MONTH_CAMS_GLOBAL__reanalyses, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_cams_global_reanalyses = xr.open_dataset(file_netcdf_path_cams_global_reanalyses)

        if co_in_ug_m3 and air_poll_selected == "CO":
            ds_cams_global_reanalyses["co"] *= 1000

        not_available_cams_global_reanalyses = False
    else:
        not_available_cams_global_reanalyses = True
        print("Dataset not available for CAMS GLOBAL Reanalyses")

    return ds_cams_eu_reanalyses, ds_cams_global_reanalyses

def load_EEA_station(   
                        cod_station, current_date, end_date,
                        df_air_pol_data, df_air_pol_metainfo
                    ):

    global path_dir_EEA_data, path_file_data_EEA_csv, freq_mode, air_poll_selected

    end_current_date = datetime(current_date.year, 12, 31, 23, 0)
    
    if end_current_date > end_date:
        end_current_date = end_date
    
    # Retrive the meta info of current station 
    df_station_info = df_air_pol_metainfo.filter(like=cod_station, axis=0)
    station_region = df_station_info["Regione"].values[0]
    lon_station = float(df_station_info["Longitude"].values[0])
    lat_station = float(df_station_info["Latitude"].values[0])
    
    area_station = df_station_info["AirQualityStationArea"].values[0]
    type_station = df_station_info["AirQualityStationType"].values[0]                      
    
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

    df_station_date.sort_values(by='DatetimeBegin', ascending = True, inplace = True)

    df_station_date_current_year = df_station_date[df_station_date.index.to_series().between(current_date.isoformat(), end_current_date.isoformat())]

    if freq_mode == "hour":
        dates = pd.date_range(start=current_date.isoformat(), end=end_current_date.isoformat(), freq="H", tz='UTC+01:00').tolist()
    else:
        dates = pd.date_range(start=current_date.isoformat(), end=end_current_date.isoformat(), freq="D", tz='UTC+01:00').tolist()
    
    conc_dates_nan = [np.nan for i in range(len(dates))]

    df_all_datetime = pd.DataFrame(list(zip(dates, conc_dates_nan)), columns=['DatetimeBegin', 'Concentration'])
    df_all_datetime = df_all_datetime.set_index('DatetimeBegin')

    # Dictionary of limit threshoulds of air pollutants, expressed
    # in μg/m^3, used for checking if the measurements of EEA station are valid
    dict_local_limit_threshould_air_pollutant = {}

    dict_local_limit_threshould_air_pollutant["CO"] = 15000
    dict_local_limit_threshould_air_pollutant["NO2"] = 700
    dict_local_limit_threshould_air_pollutant["O3"] = 500
    dict_local_limit_threshould_air_pollutant["PM10"] = 1000
    dict_local_limit_threshould_air_pollutant["PM2.5"] = 700
    dict_local_limit_threshould_air_pollutant["SO2"] = 1200

    for index, row in df_station_date_current_year.iterrows():
        try:
            value_concentration = df_station_date_current_year.loc[index]["Concentration"]

            if value_concentration != -9999.0 and value_concentration <= dict_local_limit_threshould_air_pollutant[air_poll_selected]:
                if value_concentration < 0.0:
                    df_all_datetime.loc[index]["Concentration"] = 0.0
                else:
                    df_all_datetime.loc[index]["Concentration"] = df_station_date_current_year.loc[index]["Concentration"]
        except:
            value_concentration = df_station_date_current_year.loc[index]["Concentration"].values[0]

            if value_concentration != -9999.0 and value_concentration <= dict_local_limit_threshould_air_pollutant[air_poll_selected]:
                if value_concentration < 0.0:
                    df_all_datetime.loc[index]["Concentration"] = 0.0
                else:
                    df_all_datetime.loc[index]["Concentration"] = df_station_date_current_year.loc[index]["Concentration"].values[0]
                    
    # Interpolation of measures
    df_all_datetime['Concentration'].interpolate(method='linear', inplace=True, limit_direction='both')
    
    return df_all_datetime['Concentration'].values, lon_station, lat_station, station_region, area_station, type_station

previous_date = start_date_time_to_display
current_date = start_date_time_to_display

ds_cams_eu_reanalyses = None
ds_cams_global_reanalyses = None
delta_time_hours_cams_eu = 1

diff_dates = end_date_time_to_display - start_date_time_to_display
diff_dates_hours = int(diff_dates.total_seconds() / (60*60*delta_time_hours_cams_eu))
delta = timedelta(hours=delta_time_hours_cams_eu)

# Reading matainfo file about air pollutant of the country defined
df_air_pol_metainfo = pd.read_table(path_file_air_pol_metainfo, delimiter=',', index_col=0)

# Reading the CSV file about air pollutant defined
df_air_pol_data = pd.read_table(    
                                    path_file_data_EEA_csv, delimiter=',', 
                                    header=[0], index_col=0, low_memory=False
                                )

# Loading the data sets of EEA ground-based stations
dict_code_stations = {}

# Loading CAMS Europe Reanalyses and Analyses data sets
ds_cams_eu_reanalyses, ds_cams_global_reanalyses = load_ds_datasets(current_date)

list_datetime_x = []
dict_values_EEA_station = {}
dict_values_cams_eu_reanalyses = {}
dict_values_cams_global_reanalyses = {}

dict_hours_of_current_day_cams_eu_reanalyses = {}
dict_hours_of_current_day_cams_global_reanalyses = {}

# Initialization of all dictionaries
for cod_station in list_cod_stations:

    df_station_date_current_year_values, lon_station, lat_station, region_station, area_station, type_station = \
        load_EEA_station(   cod_station, current_date, end_date_time_to_display-delta, \
                            df_air_pol_data, df_air_pol_metainfo
                        )
    
    dict_code_stations[cod_station] = [lon_station, lat_station, region_station, area_station, type_station]

    dict_values_EEA_station[cod_station] = []
    dict_values_cams_eu_reanalyses[cod_station] = []
    dict_values_cams_global_reanalyses[cod_station] = []

    dict_hours_of_current_day_cams_eu_reanalyses[cod_station] = []
    dict_hours_of_current_day_cams_global_reanalyses[cod_station] = []

    if co_in_ug_m3 and air_poll_selected == "CO":
        dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values * 1000)
    else:
        dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values)

for time in range(diff_dates_hours):

    if previous_date.day != current_date.day and freq_mode == "day":
        for cod_station in list_cod_stations:
            dict_values_cams_eu_reanalyses[cod_station].append(float(np.nanmean(dict_hours_of_current_day_cams_eu_reanalyses[cod_station])))
            dict_values_cams_global_reanalyses[cod_station].append(float(np.nanmean(dict_hours_of_current_day_cams_global_reanalyses[cod_station])))

            dict_hours_of_current_day_cams_eu_reanalyses[cod_station] = []
            dict_hours_of_current_day_cams_global_reanalyses[cod_station] = []

    # The month is changed
    if previous_date.year != current_date.year:
        
        dict_code_stations = {}

        for cod_station in list_cod_stations:
            df_station_date_current_year_values, lon_station, lat_station, region_station, area_station, type_station = \
                load_EEA_station(   cod_station, current_date, end_date_time_to_display-delta, \
                                    df_air_pol_data, df_air_pol_metainfo
                            )
    
            dict_code_stations[cod_station] = [lon_station, lat_station, region_station, area_station, type_station]
            
            if co_in_ug_m3 and air_poll_selected == "CO":
                dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values * 1000)
            else:
                dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values)

        ds_cams_eu_reanalyses, ds_cams_global_reanalyses = load_ds_datasets(current_date)
    
    if previous_date.month != current_date.month:
        ds_cams_eu_reanalyses, ds_cams_global_reanalyses = load_ds_datasets(current_date)

    # For each stations
    for cod_station in list_cod_stations:
        
        lon_station = dict_code_stations[cod_station][0]
        lat_station = dict_code_stations[cod_station][1]

        # Loading CAMS Europe Reanalyses and CAMS Global Reanalyses data sets

        # CAMS Europe Reanalyses
        if not_available_cams_eu_reanalyses == False:

            ds_current_date_cams_eu_reanalyses = ds_cams_eu_reanalyses.sel(time=current_date.isoformat())

            if air_poll_selected == "PM2.5":
                cams_eu_delta_time_reanalyses = ds_current_date_cams_eu_reanalyses.sel(lat=lat_station, lon=lat_station, method='nearest')["pm2p5"].values
            else:
                cams_eu_delta_time_reanalyses = ds_current_date_cams_eu_reanalyses.sel(lat=lat_station, lon=lat_station, method='nearest')[air_poll_selected.lower()].values
            
            if freq_mode == "day":
                dict_hours_of_current_day_cams_eu_reanalyses[cod_station].append(float(cams_eu_delta_time_reanalyses))
            else:
                dict_values_cams_eu_reanalyses[cod_station].append(float(cams_eu_delta_time_reanalyses))
        else:
            ds_current_date_cams_eu = None

        # CAMS Global Reanalyses
        if (time*delta_time_hours_cams_eu) % time_res_cams_global_reanalyses == 0 and not_available_cams_global_reanalyses == False:
            
            ds_current_date_cams_global_reanalyses = ds_cams_global_reanalyses.sel(time=current_date.isoformat())

            if air_poll_selected == "PM2.5":
                cams_global_delta_time_reanalyses = ds_cams_global_reanalyses.sel(latitude=lat_station, longitude=lon_station, method='nearest')["pm2p5"].values
            elif air_poll_selected == "O3":
                cams_global_delta_time_reanalyses = ds_cams_global_reanalyses.sel(latitude=lat_station, longitude=lon_station, method='nearest')["go3"].values
            else:
                cams_global_delta_time_reanalyses = ds_cams_global_reanalyses.sel(latitude=lat_station, longitude=lon_station, method='nearest')[air_poll_selected.lower()].values
            
            if freq_mode == "day":
                dict_hours_of_current_day_cams_global_reanalyses[cod_station].append(float(cams_global_delta_time_reanalyses))
            else:
                dict_values_cams_global_reanalyses[cod_station].append(float(cams_global_delta_time_reanalyses))

        else:
            ds_current_date_cams_global_reanalyses = None

            if freq_mode == "day":
                dict_hours_of_current_day_cams_global_reanalyses[cod_station].append(np.nan)
            else:
                dict_values_cams_global_reanalyses[cod_station].append(np.nan)
        
    if freq_mode == "day":
        if previous_date.day != current_date.day:
            list_datetime_x.append(current_date.isoformat())
    else:
        list_datetime_x.append(current_date.isoformat())
    
    print("Current date (" + air_poll_selected + "): " + current_date.isoformat())
    print("CAMS EU Reanalyses: " + str(not not_available_cams_eu_reanalyses))
    print("CAMS GLOBAL Reanalyses: " + str(not not_available_cams_global_reanalyses))

    previous_date = current_date
    current_date += delta

# Last day
if freq_mode == "day":
    list_datetime_x.append(current_date.isoformat())
    for cod_station in list_cod_stations:
        dict_values_cams_eu_reanalyses[cod_station].append(float(np.nanmean(dict_hours_of_current_day_cams_eu_reanalyses[cod_station])))
        dict_values_cams_global_reanalyses[cod_station].append(float(np.nanmean(dict_hours_of_current_day_cams_global_reanalyses[cod_station])))

# ------------------ Interpolation of CAMS Global Reanalyses ------------------
for cod_station in list_cod_stations:

    dict_cams_global_renanalyses = {'DatetimeBegin': list_datetime_x, 'Concentration': dict_values_cams_global_reanalyses[cod_station]}
    df_cams_global_renanalyses = pd.DataFrame(data=dict_cams_global_renanalyses)

    df_cams_global_renanalyses = df_cams_global_renanalyses.set_index('DatetimeBegin')
    df_cams_global_renanalyses.index = pd.to_datetime(df_cams_global_renanalyses.index)

    # Interpolation
    df_cams_global_renanalyses['Concentration'].interpolate(method='linear', inplace=True, limit_direction='both')

    dict_values_cams_global_reanalyses[cod_station] = df_cams_global_renanalyses['Concentration'].values

# ------------------ Auto-Correlation ------------------
# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "Auto_corr_EEA_and_CAMS_Europe_and_CAMS_Global_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm) + "_camsEU_" + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]))

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, air_poll_selected)

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

if air_poll_selected == "CO" and co_in_ug_m3:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_ug_m^3")
elif air_poll_selected == "CO":
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_mg_m^3")

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, freq_mode)

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, freq_mode)

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

for cod_station in list_cod_stations:

    area_cod_station = dict_code_stations[cod_station][3]
    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS, area_cod_station)

    type_cod_station = dict_code_stations[cod_station][4]
    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS, type_station)

    region_cod_station = dict_code_stations[cod_station][2]
    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS, region_cod_station)

    if not os.path.exists(PATH_DIR_PLOTS_current):
        os.mkdir(PATH_DIR_PLOTS_current)

    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS_current, cod_station)

    if not os.path.exists(PATH_DIR_PLOTS_current):
        os.mkdir(PATH_DIR_PLOTS_current)

    list_all_station = []

    # Autocorrelation of EEA station
    plt.figure(figsize=(10,6))
    autocorrelation_plot(pd.Series(dict_values_EEA_station[cod_station]))
    plt.title("Autocorrelation of " + air_poll_selected + " of EEA - " + str(cod_station))

    if save_plot:
        filename_fig = "Autocorrelation_EEA.png"
        path_fig = joinpath(PATH_DIR_PLOTS_current, filename_fig)
        plt.savefig(path_fig, dpi=300)
    else:
        plt.show()

    list_all_station.append([dict_values_EEA_station[cod_station], "EEA"])

    # Autocorrelation of CAMS Europe Reanalyses
    plt.figure(figsize=(10,6))
    autocorrelation_plot(pd.Series(dict_values_cams_eu_reanalyses[cod_station]))
    plt.title("Autocorrelation of " + air_poll_selected + " of CAMS Europe Reanalyses - " + str(cod_station))

    if save_plot:
        filename_fig = "Autocorrelation_CAMS_Europe_Reanalyses.png"
        path_fig = joinpath(PATH_DIR_PLOTS_current, filename_fig)
        plt.savefig(path_fig, dpi=300)
    else:
        plt.show()

    list_all_station.append([dict_values_cams_eu_reanalyses[cod_station], "CAMS EU Reanalyses"])

    # Autocorrelation of CAMS Global Reanalyses
    plt.figure(figsize=(10,6))
    autocorrelation_plot(pd.Series(dict_values_cams_global_reanalyses[cod_station]))
    plt.title("Autocorrelation of " + air_poll_selected + " of CAMS Global Reanalyses - " + str(cod_station))

    if save_plot:
        filename_fig = "Autocorrelation_CAMS_Global_Reanalyses.png"
        path_fig = joinpath(PATH_DIR_PLOTS_current, filename_fig)
        plt.savefig(path_fig, dpi=300)
    else:
        plt.show()
    
    list_all_station.append([dict_values_cams_global_reanalyses[cod_station], "CAMS Global Reanalyses"])

    # Autocorrelation of EEA, CAMS Europe and Global Reanalyses
    plt.figure(figsize=(10,6))
    for i in range(len(list_all_station)):
        autocorrelation_plot(pd.Series(list_all_station[i][0]), label = list_all_station[i][1])
    
    plt.title("Autocorrelation of EEA / CAMS Europe and Global Reanalyses - " + str(cod_station))
    plt.legend()

    if save_plot:
        filename_fig = "Autocorrelation_EEA_and_CAMS_Europe_and_Global_Reanalyses.png"
        path_fig = joinpath(PATH_DIR_PLOTS_current, filename_fig)
        plt.savefig(path_fig, dpi=300)
    else:
        plt.show() 