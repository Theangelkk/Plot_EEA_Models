# Taylor and Target diagram EEA, CAMS Europe Reanalyses and Analyses

# Libreries
from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
from inspect import CORO_RUNNING
import os
from shlex import join
import sys

import numpy as np
import pandas as pd
import argparse
import math
import airbase
import json
from datetime import datetime, timedelta
from matplotlib import rcParams
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
import skill_metrics as sm
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

list_methods_of_corr = ['pearson', 'kendall', 'spearman']
list_air_pollutant = ["CO", "NO2", "O3", "PM2.5", "PM10", "SO2"]
list_numeric_model_cams_eu = ["chimere", "ensemble", "LOTOS-EUROS", "MOCAGE", "SILAM", "EURAD-IM", "DEHM"]
list_freq_mode = ["hour", "day"]
list_type_stations = ["all", "background", "industrial", "traffic", "rural", "suburban", "urban"]
list_italy_region = [   "All_regions", "Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna", \
                        "Friuli Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche", \
                        "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia", "Toscana", \
                        "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto"
                    ]

parser = argparse.ArgumentParser(description='Taylor Diagram EEA - CAMS EU Reanalyses and Analyses')
parser.add_argument('-a', '--air_pollutant', help='Air Pollutant CO - NO2 - O3 - PM2.5 - PM10 - SO2', choices=list_air_pollutant, required=True)
parser.add_argument('-m_air_pol', '--m_air_pol', help='Model level for air pollution', type=int, required=True)
parser.add_argument('-m_pm', '--m_pm', help='Model level for Particulate', type=int, required=True)
parser.add_argument('-freq_mode', '--freq_mode', help='Frequency mode: hour or day', choices=list_freq_mode, required=True)
parser.add_argument('-co_ug_m^3', '--co_ug_m^3', help='CO in ug/m^3', action='store_true')
parser.add_argument('-save_plot', '--save_plot', help='Save plot data', action='store_true')
parser.add_argument('-list_cod_stations', '--list_cod_stations', help='List of code stations EEA', nargs='+', required=True)
parser.add_argument('-s_date', '--start_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-e_date', '--end_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
args = vars(parser.parse_args())

air_poll_selected = args["air_pollutant"]

list_cod_stations = args["list_cod_stations"]
start_date_time_to_display = args["start_date"]
end_date_time_to_display = args["end_date"]

filename_output = start_date_time_to_display.date().strftime("%Y-%m-%d") + "_" + end_date_time_to_display.date().strftime("%Y-%m-%d")

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

freq_mode = args["freq_mode"]

# Limit values Directive 2008/50/EC
dict_limit_air_pollutants = {}

# NO2 limit value 1 hour: 18.0 ug/m3
dict_limit_air_pollutants["NO2"] = 18.0

# CO limit value for daily max 8 hours: 10.0 mg/m3
dict_limit_air_pollutants["CO"] = 10.0

# Limit CO in ug/m3
dict_limit_air_pollutants["CO_ug_m3"] = dict_limit_air_pollutants["CO"] * 1000

# SO2 limit value for 1 hour: 350.0 ug/m3
dict_limit_air_pollutants["SO2"] = 350.0

# O3 limit value for daily max 8 hours: 120.0 ug/m3
dict_limit_air_pollutants["O3"] = 120.0

# PM2.5 limit value for 1 year: 25.0 ug/m3
dict_limit_air_pollutants["PM2.5"] = 15.0

# PM10 limit value for 1 year: 50.0 ug/m3
dict_limit_air_pollutants["PM10"] = 50.0

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
dict_not_available_CAMS_EU_reanalyses = {}

for current_cams_eu in list_numeric_model_cams_eu:
    dict_not_available_CAMS_EU_reanalyses[current_cams_eu] = False

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

dict_DATADIR_CAMS_EU_reanalyses = {}

for current_cams_eu in list_numeric_model_cams_eu:
    dict_DATADIR_CAMS_EU_reanalyses[current_cams_eu] = joinpath(DATADIR_CAMS_EU_reanalyses, current_cams_eu)

    if air_poll_selected == "PM2.5":
        dict_DATADIR_CAMS_EU_reanalyses[current_cams_eu] = joinpath(dict_DATADIR_CAMS_EU_reanalyses[current_cams_eu], "PM2p5")
    else:
        dict_DATADIR_CAMS_EU_reanalyses[current_cams_eu] = joinpath(dict_DATADIR_CAMS_EU_reanalyses[current_cams_eu], air_poll_selected)

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

# ------------ Information on CAMS EUROPE Analyses ------------
dict_not_available_CAMS_EU_analyses = {}

for current_cams_eu in list_numeric_model_cams_eu:
    dict_not_available_CAMS_EU_analyses[current_cams_eu] = False

# Path of CAMS Europe
path_main_dir_CAMS_Europe_analyses_data = os.environ['CAMS_Europe_Analyses']

if path_main_dir_CAMS_Europe_analyses_data == "":
    print("Error: set the environmental variables of CAMS_Europe_Analyses")
    exit(-1)

DATADIR_CAMS_EU_Analyses = joinpath(path_main_dir_CAMS_Europe_analyses_data, "model_level_" + str(model_level_air_pollution))
DATADIR_CAMS_EU_Analyses = joinpath(DATADIR_CAMS_EU_Analyses, "italy_ext")

dict_DATADIR_CAMS_EU_analyses = {}

for current_cams_eu in list_numeric_model_cams_eu:

    dict_DATADIR_CAMS_EU_analyses[current_cams_eu] = joinpath(DATADIR_CAMS_EU_Analyses, current_cams_eu)

    if air_poll_selected == "PM2.5":
        dict_DATADIR_CAMS_EU_analyses[current_cams_eu] = joinpath(dict_DATADIR_CAMS_EU_analyses[current_cams_eu], "PM2p5")
    else:
        dict_DATADIR_CAMS_EU_analyses[current_cams_eu] = joinpath(dict_DATADIR_CAMS_EU_analyses[current_cams_eu], air_poll_selected)

# Time resolution of CAMS Europe Analyses
time_res_cams_eu_analyses = 1

dict_start_time_numeric_models_cams_eu_analyses = {}
dict_start_time_numeric_models_cams_eu_analyses["SILAM"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analyses["MOCAGE"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analyses["MINNI"] = datetime(2022, 6, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analyses["MATCH"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analyses["LOTOS-EUROS"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analyses["EURAD-IM"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analyses["ensemble"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analyses["EMEP"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analyses["DEHM"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analyses["chimere"] = datetime(2020, 1, 1, 0, 0)

end_time_cams_eu_analyses = datetime(2022, 12, 31, 0, 0)

def load_ds_datasets(current_date, cams_eu):

    global  dict_start_time_numeric_models_cams_eu_reanalyses, dict_start_time_numeric_models_cams_eu_analyses, \
            end_time_cams_eu_reanalyses, end_time_cams_eu_analyses, air_poll_selected, \
            dict_DATADIR_CAMS_EU_reanalyses, dict_DATADIR_CAMS_EU_analyses, \
            dict_not_available_CAMS_EU_reanalyses, dict_not_available_CAMS_EU_analyses, \
            co_in_ug_m3
    
    ds_cams_eu_reanalyses = None
    ds_cams_eu_analyses = None

    # CAMS Europe Reanalyses
    if  current_date >= dict_start_time_numeric_models_cams_eu_reanalyses[cams_eu] and \
        current_date <= end_time_cams_eu_reanalyses:
        DATADIR_CURRENT_MONTH_CAMS_EU_reanalyses = joinpath(dict_DATADIR_CAMS_EU_reanalyses[cams_eu], str(current_date.year) + "-" + str(current_date.month).zfill(2))
        file_netcdf_path_cams_eu_reanalyses = joinpath(DATADIR_CURRENT_MONTH_CAMS_EU_reanalyses, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
        ds_cams_eu_reanalyses = xr.open_dataset(file_netcdf_path_cams_eu_reanalyses)

        if co_in_ug_m3 and air_poll_selected == "CO":
            ds_cams_eu_reanalyses["co"] *= 1000

        dict_not_available_CAMS_EU_reanalyses[cams_eu] = False    
    else:
        dict_not_available_CAMS_EU_reanalyses[cams_eu] = True
        print("Dataset not available for CAMS EU " + str(cams_eu) + " Reanalyses")
    
    # CAMS Europe Analyses
    if  current_date >= dict_start_time_numeric_models_cams_eu_analyses[cams_eu] and \
        current_date <= end_time_cams_eu_analyses:
        DATADIR_CURRENT_MONTH_CAMS_EU_Analyses = joinpath(dict_DATADIR_CAMS_EU_analyses[cams_eu], str(current_date.year) + "-" + str(current_date.month).zfill(2))
        
        if os.path.exists(DATADIR_CURRENT_MONTH_CAMS_EU_Analyses):
            file_netcdf_path_cams_eu_analyses = joinpath(DATADIR_CURRENT_MONTH_CAMS_EU_Analyses, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
            ds_cams_eu_analyses = xr.open_dataset(file_netcdf_path_cams_eu_analyses)

            if co_in_ug_m3 and air_poll_selected == "CO":
                ds_cams_eu_analyses["CO"] *= 1000

            dict_not_available_CAMS_EU_analyses[cams_eu] = False  
        else:
            dict_not_available_CAMS_EU_analyses[cams_eu] = True
            print("Dataset not available for CAMS EU " + str(cams_eu) + " Analyses")
    else:
        dict_not_available_CAMS_EU_analyses[cams_eu] = True
        print("Dataset not available for CAMS EU " + str(cams_eu) + " Aanalyses")

    return ds_cams_eu_reanalyses, ds_cams_eu_analyses

def initial_check(df_air_pol_metainfo, cod_station):

    global path_dir_EEA_data, start_date_time_to_display, end_date_time_to_display

    start_year = start_date_time_to_display.year
    end_year = end_date_time_to_display.year

    df_station_info = df_air_pol_metainfo.filter(like=cod_station, axis=0)

    try:
        station_region = df_station_info["Regione"].values[0]
    except:
        print("Code station " + cod_station + " not found")
        exit(-2)

    for current_year in range(start_year, end_year):
        
        # Loading file JSON of the current year
        path_json_current_year = joinpath(path_dir_EEA_data, str(current_year) + ".json")

        dict_current_year = None

        with open(path_json_current_year, "r") as f:
            dict_current_year = json.load(f)

        # Check if the current station is available for the current year asked
        if cod_station not in dict_current_year[station_region]["stations"]:
            print("Error: " + str(cod_station) + " is not present in the year " + str(current_date.year))
            exit(-1)

def load_EEA_station(   
                        cod_station, current_date, end_date,
                        df_air_pol_data, df_air_pol_metainfo
                    ):

    global path_dir_EEA_data, path_file_data_EEA_csv, freq_mode

    end_current_date = datetime(current_date.year, 12, 31, 23, 0)
    
    if end_current_date > end_date:
        end_current_date = end_date
    
    # Retrive the meta info of current station 
    df_station_info = df_air_pol_metainfo.filter(like=cod_station, axis=0)

    try:
        station_region = df_station_info["Regione"].values[0]
    except:
        print("Code station " + cod_station + " not found")
        exit(-2)

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

    for index, row in df_station_date_current_year.iterrows():
        try:
            df_all_datetime.loc[index]["Concentration"] = df_station_date_current_year.loc[index]["Concentration"]
        except:
            df_all_datetime.loc[index]["Concentration"] = df_station_date_current_year.loc[index]["Concentration"].values[0]
    
    # Interpolation of measures
    df_all_datetime['Concentration'].interpolate(method='linear', inplace=True, limit_direction='both')

    if df_all_datetime['Concentration'].isnull().values.any():
        print("Error: NaN values present in cod_station: " + cod_station)
        exit(-1)
    
    return df_all_datetime['Concentration'].values, lon_station, lat_station, station_region, area_station, type_station

# ----------------------- PLOT -----------------------
previous_date = start_date_time_to_display
current_date = start_date_time_to_display

dict_ds_cams_eu_reanalyses = {}
dict_ds_cams_eu_analyses = {}

for current_cams_eu in list_numeric_model_cams_eu:
    dict_ds_cams_eu_reanalyses[current_cams_eu] = None
    dict_ds_cams_eu_analyses[current_cams_eu] = None

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
for current_cams_eu in list_numeric_model_cams_eu:
    ds_cams_eu_reanalyses, ds_cams_eu_analyses = load_ds_datasets(current_date, current_cams_eu)

    dict_ds_cams_eu_reanalyses[current_cams_eu] = ds_cams_eu_reanalyses
    dict_ds_cams_eu_analyses[current_cams_eu] = ds_cams_eu_analyses

list_datetime_x = []
dict_values_EEA_station = {}

dict_all_cams_eu_reanalyses = {}
dict_all_cams_eu_analyses = {}

dict_all_hours_of_current_day_cams_eu_reanalyses = {}
dict_all_hours_of_current_day_cams_eu_analyses = {}

# Initial check all stations
for cod_station in list_cod_stations:
    initial_check(df_air_pol_metainfo, cod_station)

for current_cams_eu in list_numeric_model_cams_eu:

    dict_all_cams_eu_reanalyses[current_cams_eu] = {}
    dict_all_cams_eu_analyses[current_cams_eu] = {}

    dict_all_hours_of_current_day_cams_eu_reanalyses[current_cams_eu] = {}
    dict_all_hours_of_current_day_cams_eu_analyses[current_cams_eu] = {}

# Initialization of all dictionaries
for cod_station in list_cod_stations:

    df_station_date_current_year_values, lon_station, lat_station, region_station, area_station, type_station = \
        load_EEA_station(   cod_station, current_date, end_date_time_to_display-delta, \
                            df_air_pol_data, df_air_pol_metainfo
                        )
    
    dict_code_stations[cod_station] = [lon_station, lat_station, region_station, area_station, type_station]

    dict_values_EEA_station[cod_station] = []

    if co_in_ug_m3 and air_poll_selected == "CO":
        dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values * 1000)
    else:
        dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values)

    for current_cams_eu in list_numeric_model_cams_eu:

        dict_all_cams_eu_reanalyses[current_cams_eu][cod_station] = []
        dict_all_cams_eu_analyses[current_cams_eu][cod_station] = []

        dict_all_hours_of_current_day_cams_eu_reanalyses[current_cams_eu][cod_station] = []
        dict_all_hours_of_current_day_cams_eu_analyses[current_cams_eu][cod_station] = []

for time in range(diff_dates_hours):

    if previous_date.day != current_date.day and freq_mode == "day":
        for cod_station in list_cod_stations:

            for current_cams_eu in list_numeric_model_cams_eu:
                dict_all_cams_eu_reanalyses[current_cams_eu][cod_station].append(float(np.mean(dict_all_hours_of_current_day_cams_eu_reanalyses[current_cams_eu][cod_station])))
                dict_all_cams_eu_analyses[current_cams_eu][cod_station].append(float(np.mean(dict_all_hours_of_current_day_cams_eu_analyses[current_cams_eu][cod_station])))

            dict_all_hours_of_current_day_cams_eu_reanalyses[current_cams_eu][cod_station] = []
            dict_all_hours_of_current_day_cams_eu_analyses[current_cams_eu][cod_station] = []

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

        for current_cams_eu in list_numeric_model_cams_eu:
            ds_cams_eu_reanalyses, ds_cams_eu_analyses = load_ds_datasets(current_date, current_cams_eu)

            dict_ds_cams_eu_reanalyses[current_cams_eu] = ds_cams_eu_reanalyses
            dict_ds_cams_eu_analyses[current_cams_eu] = ds_cams_eu_analyses
    
    if previous_date.month != current_date.month:
        for current_cams_eu in list_numeric_model_cams_eu:
            ds_cams_eu_reanalyses, ds_cams_eu_analyses = load_ds_datasets(current_date, current_cams_eu)

            dict_ds_cams_eu_reanalyses[current_cams_eu] = ds_cams_eu_reanalyses
            dict_ds_cams_eu_analyses[current_cams_eu] = ds_cams_eu_analyses

    # For each stations
    for cod_station in list_cod_stations:
        
        lon_station = dict_code_stations[cod_station][0]
        lat_station = dict_code_stations[cod_station][1]

        # Loading CAMS Europe Reanalyses and Analyses data sets
        for current_cams_eu in list_numeric_model_cams_eu:

            ds_cams_eu_reanalyses = dict_ds_cams_eu_reanalyses[current_cams_eu]
            ds_cams_eu_analyses = dict_ds_cams_eu_analyses[current_cams_eu]

            # CAMS Europe Reanalyses
            if dict_not_available_CAMS_EU_reanalyses[current_cams_eu] == False:

                ds_current_date_cams_eu_reanalyses = ds_cams_eu_reanalyses.sel(time=current_date.isoformat())

                if air_poll_selected == "PM2.5":
                    cams_eu_delta_time_reanalyses = ds_current_date_cams_eu_reanalyses.sel(lat=lat_station, lon=lat_station, method='nearest')["pm2p5"].values
                else:
                    cams_eu_delta_time_reanalyses = ds_current_date_cams_eu_reanalyses.sel(lat=lat_station, lon=lat_station, method='nearest')[air_poll_selected.lower()].values
                
                if freq_mode == "day":
                    dict_all_hours_of_current_day_cams_eu_reanalyses[current_cams_eu][cod_station].append(float(cams_eu_delta_time_reanalyses))
                else:
                    dict_all_cams_eu_reanalyses[current_cams_eu][cod_station].append(float(cams_eu_delta_time_reanalyses))
            else:
                ds_current_date_cams_eu = None

            # CAMS Europe Analyses
            if dict_not_available_CAMS_EU_analyses[current_cams_eu] == False:
                
                # Block try - except to catch the missing data presents in the analyses data of CAMS Europe
                try:
                    ds_current_date_cams_eu_analyses = ds_cams_eu_analyses.sel(time=current_date.isoformat())

                    if ds_current_date_cams_eu_analyses['time'].size > 0:

                        if air_poll_selected == "PM2.5":
                            cams_eu_analyses_delta_time = ds_current_date_cams_eu_analyses.sel(latitude=lat_station, longitude=lat_station, method='nearest')["PM2p5"].values
                        else:
                            cams_eu_analyses_delta_time = ds_current_date_cams_eu_analyses.sel(latitude=lat_station, longitude=lat_station, method='nearest')[air_poll_selected].values
                except:
                    cams_eu_analyses_delta_time = np.nan

                if freq_mode == "day":
                    dict_all_hours_of_current_day_cams_eu_analyses[current_cams_eu][cod_station].append(float(cams_eu_analyses_delta_time))
                else:
                    dict_all_cams_eu_analyses[current_cams_eu][cod_station].append(float(cams_eu_analyses_delta_time))
            
                ds_current_date_cams_eu_analyses = None

            else:
                ds_current_date_cams_eu_analyses = None

                if freq_mode == "day":
                    dict_all_hours_of_current_day_cams_eu_analyses[current_cams_eu][cod_station].append(np.nan)
                else:
                    dict_all_cams_eu_analyses[current_cams_eu][cod_station].append(np.nan)
            
    print("Current date (" + air_poll_selected + "): " + current_date.isoformat())

    if freq_mode == "day":
        if previous_date.day != current_date.day:
            list_datetime_x.append(current_date.isoformat())
    else:
        list_datetime_x.append(current_date.isoformat())

    previous_date = current_date
    current_date += delta

# Last day
if freq_mode == "day":
    list_datetime_x.append(current_date.isoformat())
    for cod_station in list_cod_stations:
        for current_cams_eu in list_numeric_model_cams_eu:
            dict_all_cams_eu_reanalyses[current_cams_eu][cod_station].append(float(np.mean(dict_all_hours_of_current_day_cams_eu_reanalyses[current_cams_eu][cod_station])))
            dict_all_cams_eu_analyses[current_cams_eu][cod_station].append(float(np.mean(dict_all_hours_of_current_day_cams_eu_analyses[current_cams_eu][cod_station])))

# ------------------ Interpolation of CAMS Europe Analyses ------------------
for cod_station in list_cod_stations:
    for current_cams_eu in list_numeric_model_cams_eu:
        dict_cams_eu_analyses = {'DatetimeBegin': list_datetime_x, 'Concentration': dict_all_cams_eu_analyses[current_cams_eu][cod_station]}
        df_cams_eu_analyses = pd.DataFrame(data=dict_cams_eu_analyses)

        df_cams_eu_analyses = df_cams_eu_analyses.set_index('DatetimeBegin')
        df_cams_eu_analyses.index = pd.to_datetime(df_cams_eu_analyses.index)

        # Interpolation
        df_cams_eu_analyses['Concentration'].interpolate(method='linear', inplace=True, limit_direction='both')

        dict_all_cams_eu_analyses[current_cams_eu][cod_station] = df_cams_eu_analyses['Concentration'].values

        if df_cams_eu_analyses['Concentration'].isnull().values.any():
            print("Error: NaN values present in CAMS Analyses: " + current_cams_eu)
            exit(-1)

# ------------------ Full Cross-Correlation ------------------
def write_file_corr(    
                        list_bias, list_std, list_rmse, list_crmse, list_corr, \
                        path_file_txt
                    ):
    
    list_measures = ["Bias", "STD", "RMSE", "CRMSE", "Correlation"]

    list_bias.pop(0)
    list_std.pop(0)
    list_rmse.pop(0)
    list_corr.pop(0)

    dict_values = {
        "Bias": list_bias,
        "STD": list_std,
        "RMSE": list_rmse,
        "CRMSE": list_crmse,
        "Correlation": list_corr
    }

    # For each row of the table: 
    # Measure & chimere R & chiemere A & ensemble R & ensemble A & LOTOS R & LOTOS A & MOCAGE R & MOCAGE A & SILAM R 
    # & SILAM A & EURAD R % EURAD A & DEHM R & DEHM A 
    string_txt = ""

    with open(path_file_txt, "a") as file:

        for measure in list_measures:
            string_txt += measure + " & "

            if measure == "Correlation":
               best_idx = list(map(abs,dict_values[measure])).index(max(list(map(abs,dict_values[measure]))))
            else:
                best_idx = list(map(abs,dict_values[measure])).index(min(list(map(abs,dict_values[measure]))))

            idx = 0

            for elem in dict_values[measure]:

                if idx == best_idx:
                    string_txt += "\\textbf{" + str(round(elem,3)) + "}"
                else:
                    string_txt += str(round(elem,3))

                if idx < len(dict_values[measure]) - 1:
                    string_txt += " & "
                else:
                    string_txt += " "
                
                idx += 1

            string_txt += " \\\\ \n"
            string_txt += " \hline \n"

        file.write(string_txt)

# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

PATH_DIR_LATEX = joinpath(PATH_DIR_PLOTS, "Latex")
PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, air_poll_selected)
PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, freq_mode)

os.makedirs(PATH_DIR_LATEX, exist_ok=True)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "Taylor_and_Target_diagram_EEA_vs_CAMS_Reanalyses_Analyses_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm))
PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, air_poll_selected)

if air_poll_selected == "CO" and co_in_ug_m3:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_ug_m^3")
elif air_poll_selected == "CO":
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_mg_m^3")

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, freq_mode)
PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, filename_output)

os.makedirs(PATH_DIR_PLOTS, exist_ok=True)

filename_latex_file = filename_output + ".txt"
path_latex_file = joinpath(PATH_DIR_LATEX, filename_latex_file)

# [Mean_BIAS_EEA = 0, CAMS_EU_Reanalyses, CAMS_EU_Analyses]
list_all_bias = [0.0]

# [Mean_STD_EEA, CAMS_EU_Reanalyses, CAMS_EU_Analyses]
list_all_std = []

list_EEA_std = []
for cod_station in list_cod_stations:
    list_EEA_std.append(np.std(np.array(dict_values_EEA_station[cod_station])))

list_all_std.append(np.mean(np.array(list_EEA_std)))

# [Mean_RMSE_EEA = 0, CAMS_EU_Reanalyses, CAMS_EU_Analyses] or Root-Mean Square Deviation
list_all_rmse = [0.0]

# [Mean_Corr_EEA = 1, CAMS_EU_Reanalyses, CAMS_EU_Analyses]
list_all_corr = [1.0]

# Centered Root Mean Standard Deviation
# [CAMS_EU_Reanalyses, CAMS_EU_Analyses]
list_all_crmsd = []

list_labels = []

for current_cams_eu in list_numeric_model_cams_eu:

    list_current_cams_eu_reanalyses_bias = []
    list_current_cams_eu_reanalyses_std = []
    list_current_cams_eu_reanalyses_rmse = []
    list_current_cams_eu_reanalyses_corr = []
    list_current_cams_eu_reanalyses_crmsd = []

    list_current_cams_eu_analyses_bias = []
    list_current_cams_eu_analyses_std = []
    list_current_cams_eu_analyses_rmse = []
    list_current_cams_eu_analyses_corr = []
    list_current_cams_eu_analyses_crmsd = []

    for cod_station in list_cod_stations:
        
        np_diff_EEA_cams_eu_reanalyses = np.array(dict_all_cams_eu_reanalyses[current_cams_eu][cod_station]) - \
                                            np.array(dict_values_EEA_station[cod_station])
        
        np_diff_EEA_cams_eu_analyses = np.array(dict_all_cams_eu_analyses[current_cams_eu][cod_station]) - \
                                            np.array(dict_values_EEA_station[cod_station])
        
    
        bias_EEA_cams_eu_reanalyses = np.mean(np_diff_EEA_cams_eu_reanalyses) 
        bias_EEA_cams_eu_analyses = np.mean(np_diff_EEA_cams_eu_analyses) 

        list_current_cams_eu_reanalyses_bias.append(bias_EEA_cams_eu_reanalyses)
        list_current_cams_eu_analyses_bias.append(bias_EEA_cams_eu_analyses)

        list_current_cams_eu_reanalyses_std.append(np.std(np.array(dict_all_cams_eu_reanalyses[current_cams_eu][cod_station])))
        list_current_cams_eu_analyses_std.append(np.std(np.array(dict_all_cams_eu_analyses[current_cams_eu][cod_station])))

        rmse_EEA_cams_eu_reanalyses = np.sqrt(np.mean((np_diff_EEA_cams_eu_reanalyses)**2))
        rmse_EEA_cams_eu_analyses = np.sqrt(np.mean((np_diff_EEA_cams_eu_analyses)**2))

        list_current_cams_eu_reanalyses_rmse.append(rmse_EEA_cams_eu_reanalyses)
        list_current_cams_eu_analyses_rmse.append(rmse_EEA_cams_eu_analyses)
        
        # Pearson correlation
        if np.std(np.array(dict_values_EEA_station[cod_station])) == 0:
            print("Standard deviation of EEA is equal to 0 --> Pearson correlation of Reanalyses and Analyses cannot be computed")
        else:
            if np.std(np.array(dict_all_cams_eu_reanalyses[current_cams_eu][cod_station])) == 0:
                print("Standard deviation of CAMS EU Reanalyses is equal to 0 --> Pearson correlation cannot be computed")
            else:
                corr_EEA_cams_eu_reanalyses = abs(np.corrcoef(np.array(dict_all_cams_eu_reanalyses[current_cams_eu][cod_station]), np.array(dict_values_EEA_station[cod_station])))
                list_current_cams_eu_reanalyses_corr.append(corr_EEA_cams_eu_reanalyses)

            if np.std(np.array(dict_all_cams_eu_analyses[current_cams_eu][cod_station])) == 0:
                print("Standard deviation of CAMS EU Analyses is equal to 0 --> Pearson correlation cannot be computed")
            else:
                corr_EEA_cams_eu_analyses = abs(np.corrcoef(np.array(dict_all_cams_eu_analyses[current_cams_eu][cod_station]), np.array(dict_values_EEA_station[cod_station])))
                list_current_cams_eu_analyses_corr.append(corr_EEA_cams_eu_analyses)

        crmsd_EEA_cams_eu_reanalyses = sm.centered_rms_dev(np.array(dict_all_cams_eu_reanalyses[current_cams_eu][cod_station]), np.array(dict_values_EEA_station[cod_station]))
        crmsd_EEA_cams_eu_analyses = sm.centered_rms_dev(np.array(dict_all_cams_eu_analyses[current_cams_eu][cod_station]), np.array(dict_values_EEA_station[cod_station]))

        list_current_cams_eu_reanalyses_crmsd.append(crmsd_EEA_cams_eu_reanalyses)
        list_current_cams_eu_analyses_crmsd.append(crmsd_EEA_cams_eu_analyses)

    list_all_bias.append(np.mean(np.array(list_current_cams_eu_reanalyses_bias)))
    list_all_bias.append(np.mean(np.array(list_current_cams_eu_analyses_bias)))

    list_all_std.append(np.mean(np.array(list_current_cams_eu_reanalyses_std)))
    list_all_std.append(np.mean(np.array(list_current_cams_eu_analyses_std)))

    list_all_rmse.append(np.mean(np.array(list_current_cams_eu_reanalyses_rmse)))
    list_all_rmse.append(np.mean(np.array(list_current_cams_eu_analyses_rmse)))

    list_all_corr.append(np.mean(np.array(list_current_cams_eu_reanalyses_corr)))
    list_all_corr.append(np.mean(np.array(list_current_cams_eu_analyses_corr)))

    list_all_crmsd.append(np.mean(np.array(list_current_cams_eu_reanalyses_crmsd)))
    list_all_crmsd.append(np.mean(np.array(list_current_cams_eu_analyses_crmsd)))

    list_labels.append(current_cams_eu + " Reanalyses")
    list_labels.append(current_cams_eu + " Analyses")

# Set the figure properties (optional)
rcParams["figure.figsize"] = [12.0, 8.0]
rcParams['lines.linewidth'] = 1     # line width for plots
rcParams.update({'font.size': 10})  # font size of axes text

# Close any previously open graphics windows
# ToDo: fails to work within Eclipse
plt.close('all')

MARKERS = {
    list_labels[0]: {
        "labelColor": "k",
        "symbol": "+",
        "size": 9,
        "faceColor": "r",
        "edgeColor": "r",
    },
    list_labels[1]: {
        "labelColor": "k",
        "symbol": ".",
        "size": 9,
        "faceColor": "b",
        "edgeColor": "b",
    },
    list_labels[2]: {
        "labelColor": "k",
        "symbol": "x",
        "size": 9,
        "faceColor": "g",
        "edgeColor": "g",
    },
    list_labels[3]: {
        "labelColor": "k",
        "symbol": "s",
        "size": 9,
        "faceColor": "c",
        "edgeColor": "c",
    },
    list_labels[4]: {
        "labelColor": "k",
        "symbol": "d",
        "size": 9,
        "faceColor": "m",
        "edgeColor": "m",
    },
    list_labels[5]: {
        "labelColor": "k",
        "symbol": "^",
        "size": 9,
        "faceColor": "y",
        "edgeColor": "y",
    },
    list_labels[6]: {
        "labelColor": "k",
        "symbol": "v",
        "size": 9,
        "faceColor": "r",
        "edgeColor": "r",
    },
    list_labels[7]: {
        "labelColor": "k",
        "symbol": "p",
        "size": 9,
        "faceColor": "b",
        "edgeColor": "b",
    },
    list_labels[8]: {
        "labelColor": "k",
        "symbol": "h",
        "size": 9,
        "faceColor": "g",
        "edgeColor": "g",
    },
    list_labels[9]: {
        "labelColor": "k",
        "symbol": "*",
        "size": 9,
        "faceColor": "c",
        "edgeColor": "c",
    },
    list_labels[10]: {
        "labelColor": "k",
        "symbol": "+",
        "size": 9,
        "faceColor": "m",
        "edgeColor": "m",
    },
    list_labels[11]: {
        "labelColor": "k",
        "symbol": ".",
        "size": 9,
        "faceColor": "y",
        "edgeColor": "y",
    },
    list_labels[12]: {
        "labelColor": "k",
        "symbol": "x",
        "size": 9,
        "faceColor": "r",
        "edgeColor": "r",
    },
    list_labels[13]: {
        "labelColor": "k",
        "symbol": "s",
        "size": 9,
        "faceColor": "b",
        "edgeColor": "b",
    },
}

sm.taylor_diagram(  np.array(list_all_std), np.array(list_all_rmse), np.array(list_all_corr), \
                    markers=MARKERS, markerLegend='on', titleOBS='EEA', \
                    styleOBS='-', colOBS = 'r', markerobs = 'o',
                    titleRMS='off', titleSTD ='on', titleCOR='on', tickRMSangle = 115,
                ) 

if save_plot:
    filename_png = filename_output + "_taylor_diagram.png"
    plt.savefig(joinpath(PATH_DIR_PLOTS, filename_png))
else:
    plt.show()

plt.close('all')

value_range = np.max(np.abs(np.array(list_all_bias))) * 2

if value_range < 1.0:
    range_ticks = np.arange(-1.0, 1.0, 0.10)

    step_value_circle = 0.20
    circle_list = np.arange(step_value_circle, 1.0, step_value_circle).tolist()
else:
    range_ticks = []
    current_ticks = -value_range

    n_step = 10

    for i in range((2*n_step)-2):
        range_ticks.append(round(current_ticks,2))
        current_ticks += float(value_range/n_step)

    range_ticks = np.array(range_ticks)

    step_value_circle = round(float( (value_range/2.0) / 10), 2)
    circle_list = np.arange(step_value_circle, (value_range/2.0), step_value_circle).tolist()

sm.target_diagram(  np.array(list_all_bias), np.array(list_all_crmsd), np.array(list_all_rmse), 
                    markers = MARKERS, markerLegend = 'on', 
                    ticks=range_ticks, axismax=value_range*2, 
                    circles=circle_list, circleLineSpec = 'b-.', circleLineWidth = 1.0,
                    equalAxes='off'
                )
if save_plot:
    filename_png = filename_output + "_target_diagram.png"
    plt.savefig(joinpath(PATH_DIR_PLOTS, filename_png))
else:
    plt.show()
            
plt.close('all')

write_file_corr(    
                    list_all_bias, list_all_std, list_all_rmse, list_all_crmsd, \
                    list_all_corr, path_latex_file
                )   