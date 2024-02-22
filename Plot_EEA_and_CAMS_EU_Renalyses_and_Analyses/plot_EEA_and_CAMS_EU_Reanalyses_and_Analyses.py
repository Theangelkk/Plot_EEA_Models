# Scripy of plot of EEA stations and CAMS Europe Reanalyses and Analyses
# conda activate Plot_all_air_pol

# Libraries
from ast import Tuple
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
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
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

list_air_pollutant = ["CO", "NO2", "O3", "PM2.5", "PM10", "SO2"]
list_numeric_model_cams_eu = [  "chimere", "ensemble", "EMEP", "LOTOS-EUROS", "MATCH", \
                                "MINNI", "MOCAGE", "SILAM", "EURAD-IM", "DEHM", "GEM-AQ"]
list_freq_mode = ["hour", "day"]
list_type_stations = ["all", "background", "industrial", "traffic", "rural", "suburban", "urban"]

parser = argparse.ArgumentParser(description='Plot EEA - CAMS Europe Reanalyses and Analyses')
parser.add_argument('-a', '--air_pollutant', help='Air Pollutant CO - NO2 - O3 - PM2.5 - PM10 - SO2', choices=list_air_pollutant, required=True)
parser.add_argument('-list_cod_stations', '--list_cod_stations', help='List of code stations EEA', nargs='+', required=True)
parser.add_argument('-m_air_pol', '--m_air_pol', help='Model level for air pollution', type=int, required=True)
parser.add_argument('-m_pm', '--m_pm', help='Model level for Particulate', type=int, required=True)
parser.add_argument('-freq_mode', '--freq_mode', help='Frequency mode: hour or day', choices=list_freq_mode, required=True)
parser.add_argument('-s_date', '--start_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-e_date', '--end_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-cams_eu', '--cams_eu', help='chimere - ensemble - EMEP - LOTOS-EUROS - MATCH - MINNI - MOCAGE - SILAM - EURAD-IM - DEHM - GEM-AQ', \
                     choices=list_numeric_model_cams_eu, required=True)
parser.add_argument('-co_ug_m^3', '--co_ug_m^3', help='CO in ug/m^3', action='store_true')
parser.add_argument('-save_plot', '--save_plot', help='Save plot data', action='store_true')
parser.add_argument('-split_days_plot', '--split_days_plot', help='How many days to visualize in one plot', type=int, required=True)
parser.add_argument('-type_stations', '--type_stations', help='Type of stations to consider ("all" for all stations)', choices=list_type_stations, required=True)
args = vars(parser.parse_args())

air_poll_selected = args["air_pollutant"]
list_cod_stations = args["list_cod_stations"]
start_date_time_to_display = args["start_date"]
end_date_time_to_display = args["end_date"]
split_days_plot = int(args["split_days_plot"])
type_stations = args["type_stations"]

# If it asked the visualization of CO in ug/m^3
co_in_ug_m3 = args["co_ug_m^3"]

# If it is request to save the plot
save_plot = args["save_plot"]

# Model level 55: about 288 meters
# Model level 60: about 10 meters
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

end_time_cams_eu_reanalyses = datetime(2022, 12, 31, 0, 0)

# ------------ Information on CAMS EUROPE Analyses ------------
not_available_cams_eu_analyses = False

# Path of CAMS Europe
path_main_dir_CAMS_Europe_analyses_data = os.environ['CAMS_Europe_Analyses']

if path_main_dir_CAMS_Europe_analyses_data == "":
    print("Error: set the environmental variables of CAMS_Europe_Analyses")
    exit(-1)

DATADIR_CAMS_EU_Analyses = joinpath(path_main_dir_CAMS_Europe_analyses_data, "model_level_" + str(model_level_air_pollution))
DATADIR_CAMS_EU_Analyses = joinpath(DATADIR_CAMS_EU_Analyses, "italy_ext")

numeric_model_selected = list_numeric_model_cams_eu[idx_numeric_model_cams_eu]
DATADIR_CAMS_EU_Analyses = joinpath(DATADIR_CAMS_EU_Analyses, numeric_model_selected)

if air_poll_selected == "PM2.5":
    DATADIR_CAMS_EU_Analyses = joinpath(DATADIR_CAMS_EU_Analyses, "PM2p5")
else:
    DATADIR_CAMS_EU_Analyses = joinpath(DATADIR_CAMS_EU_Analyses, air_poll_selected)

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

def load_ds_datasets(current_date):

    global  dict_start_time_numeric_models_cams_eu_reanalyses, numeric_model_selected, \
            end_time_cams_eu_reanalyses, end_time_cams_eu_analyses, air_poll_selected, \
            DATADIR_CAMS_EU_reanalyses, DATADIR_CAMS_EU_Analyses, \
            not_available_cams_eu_reanalyses, not_available_cams_eu_analyses, \
            co_in_ug_m3
    
    ds_cams_eu_reanalyses = None
    ds_cams_eu_analyses = None

    # CAMS Europe Reanalyses
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
        print("Dataset not available for CAMS EU Reanalyses " +  str(numeric_model_selected))
    
    # CAMS Europe Analyses
    if  current_date >= dict_start_time_numeric_models_cams_eu_analyses[numeric_model_selected] and \
        current_date <= end_time_cams_eu_analyses:
        DATADIR_CURRENT_MONTH_CAMS_EU_Analyses = joinpath(DATADIR_CAMS_EU_Analyses, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        
        if os.path.exists(DATADIR_CURRENT_MONTH_CAMS_EU_Analyses):
            file_netcdf_path_cams_eu_analyses = joinpath(DATADIR_CURRENT_MONTH_CAMS_EU_Analyses, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
            ds_cams_eu_analyses = xr.open_dataset(file_netcdf_path_cams_eu_analyses)

            if co_in_ug_m3 and air_poll_selected == "CO":
                ds_cams_eu_analyses["CO"] *= 1000

            not_available_cams_eu_analyses = False
        else:
            not_available_cams_eu_analyses = True
            print("Dataset not available for CAMS EU Analyses " +  str(numeric_model_selected))
    else:
        not_available_cams_eu_analyses = True
        print("Dataset not available for CAMS EU Analyses " +  str(numeric_model_selected))

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
            print("Error: " + str(cod_station) + " is not present in the year " + str(current_year))
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
    
    area_station_current_station = df_station_info["AirQualityStationArea"].values[0]
    type_station_current_station = df_station_info["AirQualityStationType"].values[0]                      
    
    # Loading file JSON of the current year
    path_json_current_year = joinpath(path_dir_EEA_data, str(current_date.year) + ".json")

    dict_current_year = None

    with open(path_json_current_year, "r") as f:
        dict_current_year = json.load(f)

    # Check if the current station is available for the current year asked
    if cod_station not in dict_current_year[station_region]["stations"]:
        print("Error: " + str(cod_station) + " is not present in the year " + str(current_date.year))
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
            value_concentration = df_station_date_current_year.loc[index]["Concentration"]

            if value_concentration != -9999.0:
                if value_concentration < 0.0:
                    df_all_datetime.loc[index]["Concentration"] = 0.0
                else:
                    df_all_datetime.loc[index]["Concentration"] = df_station_date_current_year.loc[index]["Concentration"]
        except:
            value_concentration = df_station_date_current_year.loc[index]["Concentration"].values[0]

            if value_concentration != -9999.0:
                if value_concentration < 0.0:
                    df_all_datetime.loc[index]["Concentration"] = 0.0
                else:
                    df_all_datetime.loc[index]["Concentration"] = df_station_date_current_year.loc[index]["Concentration"].values[0]
                    
    # Interpolation of measures
    df_all_datetime['Concentration'].interpolate(method='linear', inplace=True, limit_direction='both')
    
    return df_all_datetime['Concentration'].values, lon_station, lat_station, station_region, area_station_current_station, type_station_current_station

# ----------------------- PLOT -----------------------
def plot(   
            cod_station, air_pol_selected, list_values_EEA_station, \
            list_values_cams_eu_reanalyses, list_values_cams_eu_analyses, \
            PATH_DIR_PLOTS, start_time, end_time, area_station, type_station
        ):

    global  freq_mode, co_in_ug_m3, save_plot, dict_limit_air_pollutants

    # Set up the axes and figure
    fig, ax = plt.subplots()

    if freq_mode == "hour":
        delta_time = timedelta(hours=1)
    else:
        delta_time = timedelta(days=1)

    dates = mdates.drange(start_time, end_time, delta_time)

    ax.plot(dates, list_values_EEA_station, label="EEA", alpha=0.75, linewidth=2, linestyle='dashed')
    
    # CAMS Europe Reanalyses
    ax.plot(dates, list_values_cams_eu_reanalyses, label="CAMS EU Reanalyses",  alpha=1.0, linewidth=1)
    
    ax.plot(dates, list_values_cams_eu_analyses, label="CAMS EU Analyses",  alpha=1.0, linewidth=1)
    
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
        title = air_pol_selected + " mg/m^3 " + cod_station + " " + area_station + " " + type_station + " " \
                + start_time.isoformat() + " - " + end_time.isoformat()
        unit = air_pol_selected + " mg/m^3"
    else:
        title = air_pol_selected + " μg/m^3 " + cod_station + " " + area_station + " " + type_station + " " \
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

ds_cams_eu_reanalyses = None
ds_cams_eu_analyses = None
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
ds_cams_eu_reanalyses, ds_cams_eu_analyses = load_ds_datasets(current_date)

list_datetime_x = []
dict_values_EEA_station = {}
dict_values_cams_eu_reanalyses = {}
dict_values_cams_eu_analyses = {}

dict_hours_of_current_day_cams_eu_reanalyses = {}
dict_hours_of_current_day_cams_eu_analyses = {}

# Initial check all stations
for cod_station in list_cod_stations:
    initial_check(df_air_pol_metainfo, cod_station)

# Initialization of all dictionaries
for cod_station in list_cod_stations:

    df_station_date_current_year_values, lon_station, lat_station, region_station, area_station_current_station, type_station_current_station = \
        load_EEA_station(   cod_station, current_date, end_date_time_to_display-delta, \
                            df_air_pol_data, df_air_pol_metainfo
                        )
    
    dict_code_stations[cod_station] = [lon_station, lat_station, region_station, area_station_current_station, type_station_current_station]

    dict_values_EEA_station[cod_station] = []
    dict_values_cams_eu_reanalyses[cod_station] = []
    dict_values_cams_eu_analyses[cod_station] = []

    dict_hours_of_current_day_cams_eu_reanalyses[cod_station] = []
    dict_hours_of_current_day_cams_eu_analyses[cod_station] = []

    if co_in_ug_m3 and air_poll_selected == "CO":
        dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values * 1000)
    else:
        dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values)

for time in range(diff_dates_hours):

    if previous_date.day != current_date.day and freq_mode == "day":
        for cod_station in list_cod_stations:
            dict_values_cams_eu_reanalyses[cod_station].append(float(np.nanmean(dict_hours_of_current_day_cams_eu_reanalyses[cod_station])))
            dict_values_cams_eu_analyses[cod_station].append(float(np.nanmean(dict_hours_of_current_day_cams_eu_analyses[cod_station])))

            dict_hours_of_current_day_cams_eu_reanalyses[cod_station] = []
            dict_hours_of_current_day_cams_eu_analyses[cod_station] = []

    # The month is changed
    if previous_date.year != current_date.year:
        
        dict_code_stations = {}

        for cod_station in list_cod_stations:
            df_station_date_current_year_values, lon_station, lat_station, region_station, area_station_current_station, type_station_current_station = \
                load_EEA_station(   cod_station, current_date, end_date_time_to_display-delta, \
                                    df_air_pol_data, df_air_pol_metainfo
                            )
    
            dict_code_stations[cod_station] = [lon_station, lat_station, region_station, area_station_current_station, type_station_current_station]
            
            if co_in_ug_m3 and air_poll_selected == "CO":
                dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values * 1000)
            else:
                dict_values_EEA_station[cod_station].extend(df_station_date_current_year_values)

        ds_cams_eu_reanalyses, ds_cams_eu_analyses = load_ds_datasets(current_date)
    
    if previous_date.month != current_date.month:
        ds_cams_eu_reanalyses, ds_cams_eu_analyses = load_ds_datasets(current_date)

    # For each stations
    for cod_station in list_cod_stations:
        
        lon_station = dict_code_stations[cod_station][0]
        lat_station = dict_code_stations[cod_station][1]

        # Loading CAMS Europe Reanalyses and Analyses data sets

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

        # CAMS Europe Analyses
        if not_available_cams_eu_analyses == False:
            
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
                dict_hours_of_current_day_cams_eu_analyses[cod_station].append(float(cams_eu_analyses_delta_time))
            else:
                dict_values_cams_eu_analyses[cod_station].append(float(cams_eu_analyses_delta_time))
          
            ds_current_date_cams_eu_analyses = None

        else:
            ds_current_date_cams_eu_analyses = None

            if freq_mode == "day":
                dict_hours_of_current_day_cams_eu_analyses[cod_station].append(np.nan)
            else:
                dict_values_cams_eu_analyses[cod_station].append(np.nan)
    
    if freq_mode == "day":
        if previous_date.day != current_date.day:
            list_datetime_x.append(current_date.isoformat())
    else:
        list_datetime_x.append(current_date.isoformat())
    
    print("Current date (" + air_poll_selected + "): " + current_date.isoformat())
    print("CAMS EU Reanalyses: " + str(not not_available_cams_eu_reanalyses))
    print("CAMS EU Analyses: " + str(not not_available_cams_eu_analyses))

    previous_date = current_date
    current_date += delta

# Last day
if freq_mode == "day":
    list_datetime_x.append(current_date.isoformat())
    for cod_station in list_cod_stations:
        dict_values_cams_eu_reanalyses[cod_station].append(float(np.nanmean(dict_hours_of_current_day_cams_eu_reanalyses[cod_station])))
        dict_values_cams_eu_analyses[cod_station].append(float(np.nanmean(dict_hours_of_current_day_cams_eu_analyses[cod_station])))

# ------------------ PLOT -----------------------
# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "EEA_plots_CAMS_Reanalyses_Analyses_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm) + "_camsEU_" + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]))
PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, air_poll_selected)

if air_poll_selected == "CO" and co_in_ug_m3:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_ug_m^3")
elif air_poll_selected == "CO":
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_mg_m^3")

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, freq_mode)

os.makedirs(PATH_DIR_PLOTS, exist_ok=True)

for cod_station in list_cod_stations:
    
    area_cod_station = dict_code_stations[cod_station][3]
    type_cod_station = dict_code_stations[cod_station][4]

    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS, type_stations)

    region_cod_station = dict_code_stations[cod_station][2]
    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS_current, region_cod_station)

    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS_current, cod_station)

    os.makedirs(PATH_DIR_PLOTS_current, exist_ok=True)

    plot(   
            cod_station, air_poll_selected, dict_values_EEA_station[cod_station], \
            dict_values_cams_eu_reanalyses[cod_station], dict_values_cams_eu_analyses[cod_station], \
            PATH_DIR_PLOTS_current, start_date_time_to_display, end_date_time_to_display, \
            area_cod_station, type_cod_station
    )

    # Split the time series in "split_days_plot" days defined
    if split_days_plot > 0:
        
        delta_days = timedelta(days=split_days_plot)
        n_hours = delta_days.total_seconds()//3600

        steps_dict = int(n_hours / delta_time_hours_cams_eu)

        current_start_time_to_display = start_date_time_to_display
        current_end_time_to_display = start_date_time_to_display + delta_days

        idx_dict = 0

        res = False

        while res == False:
            if current_end_time_to_display >= end_date_time_to_display:
                res = True
                current_end_time_to_display = end_date_time_to_display
                idx_end = len(dict_values_EEA_station[cod_station])
            else:
                idx_end = idx_dict + steps_dict
    
            current_dict_values_EEA_station = \
                        dict_values_EEA_station[cod_station][idx_dict : idx_end]

            current_dict_values_cams_eu_reanalyses = []

            current_dict_values_cams_eu_reanalyses = \
                            dict_values_cams_eu_reanalyses[cod_station][idx_dict : idx_end]
            
            current_dict_values_cams_eu_analyses = []

            current_dict_values_cams_eu_analyses = \
                            dict_values_cams_eu_analyses[cod_station][idx_dict : idx_end]
            plot(   
                    cod_station, air_poll_selected, current_dict_values_EEA_station, \
                    current_dict_values_cams_eu_reanalyses, current_dict_values_cams_eu_analyses, \
                    PATH_DIR_PLOTS_current, current_start_time_to_display, \
                    current_end_time_to_display, area_cod_station, type_cod_station
            )

            current_start_time_to_display = current_end_time_to_display
            current_end_time_to_display = current_end_time_to_display + delta_days
            idx_dict = idx_dict + steps_dict