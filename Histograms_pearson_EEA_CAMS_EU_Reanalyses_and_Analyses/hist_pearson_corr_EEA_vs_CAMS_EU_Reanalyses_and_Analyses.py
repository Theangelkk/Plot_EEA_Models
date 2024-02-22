# Pearson coefficient histograms [EEA vs CAMS EU Reanalyses and Analyses]

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
import gc
import warnings
import matplotlib.gridspec as gridspec

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

parser = argparse.ArgumentParser(description='Histgrams of Pearson Correlations EEA - CAMS Europe Reanalyses and Analyses')
parser.add_argument('-a', '--air_pollutant', help='Air Pollutant CO - NO2 - O3 - PM2.5 - PM10 - SO2', choices=list_air_pollutant, required=True)
parser.add_argument('-m_air_pol', '--m_air_pol', help='Model level for air pollution', type=int, required=True)
parser.add_argument('-m_pm', '--m_pm', help='Model level for Particulate', type=int, required=True)
parser.add_argument('-list_window_lenght', '--list_window_lenght', help='List of Window lenght', nargs="+", type=int, required=True)
parser.add_argument('-previous_elems', '--previous_elems', help='Subseries with previous samples', action='store_true')
parser.add_argument('-method_corr', '--method_corr', help='Method for computing correlation [pearson, kendall, spearman]', default='pearson', choices=list_methods_of_corr)
parser.add_argument('-save_plot', '--save_plot', help='Save plot data', action='store_true')
parser.add_argument('-no_overlap', '--no_overlap', help='No overlap window lenght', action='store_true')
parser.add_argument('-freq_mode', '--freq_mode', help='Frequency mode: hour or day', choices=list_freq_mode, required=True)
parser.add_argument('-co_ug_m^3', '--co_ug_m^3', help='CO in ug/m^3', action='store_true')
parser.add_argument('-type_stations', '--type_stations', help='Type of stations to consider ("all" for all stations)', choices=list_type_stations, required=True)
parser.add_argument('-italy_region', '--italy_region', help='Italy region to analyses ("All_regions" for all Italy)', choices=list_italy_region, required=True)
parser.add_argument('-filename_json_file', '--filename_json_file', help='Filename json file of EEA code stations to consider', type=str, required=True)
args = vars(parser.parse_args())

air_poll_selected = args["air_pollutant"]
type_stations = args["type_stations"]
filename_json_file = args["filename_json_file"]
italy_region = args["italy_region"]
list_windows_lenght = args["list_window_lenght"]
method_corr = args["method_corr"]

if filename_json_file.endswith(".json") == False:
    print("ERROR: filename_json_file is not a JSON file")
    exit(-1)

# If it is necessary to split the time series considering the
# previous elements of i-th sample.
# Example: window_lenght=3 --> [ABC] --> [[00A], [0AB], [ABC]]
is_previous_elems = args["previous_elems"]

if is_previous_elems == False:
    for windows_lenght in list_windows_lenght:
        if windows_lenght % 2 == 0:
            print("Window lenght must be odd!")
            exit(-1)

# If it asked the visualization of CO in ug/m^3
co_in_ug_m3 = args["co_ug_m^3"]

# If it is request to save the plot
save_plot = args["save_plot"]

# No overlap among time series partitions
no_overlap = args["no_overlap"]

start_date_time_to_display = datetime(int(filename_json_file.split("_")[0]), 1, 1, 0, 0)

if filename_json_file.split("_")[1].isnumeric():
    end_date_time_to_display = datetime(int(filename_json_file.split("_")[1]) + 1, 1, 1, 0, 0)
else:
    end_date_time_to_display = datetime(int(filename_json_file.split("_")[0]) + 1, 1, 1, 0, 0)

filename_output = filename_json_file.split(".")[0]

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

path_dir_json_files = joinpath(path_dir_EEA_data, "merged_json")
filename_json_file = joinpath(path_dir_json_files, filename_json_file)

if os.path.exists(filename_json_file) == False:
    print("ERROR: Json file not exist")
    exit(-1)

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

def load_cod_stations(file_json, italy_region):

    list_italy_region = [   "Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna", \
                            "Friuli Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche", \
                            "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia", "Toscana", \
                            "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto"
                    ]

    list_cod_stations = []

    json_dict = None

    with open(file_json) as json_file:
        
        json_dict = json.load(json_file)

        if italy_region == "All_regions":
            for region in list_italy_region:
                for current_station in json_dict[region]["sorted_station_MDTs_min"]:

                    if json_dict[region]["stations"][current_station]["Percentage_of_valid_data"] >= 75.0:
                        list_cod_stations.append(current_station)
        else:
            for current_station in json_dict[italy_region]["sorted_station_MDTs_min"]:
                if json_dict[italy_region]["stations"][current_station]["Percentage_of_valid_data"] >= 75.0:
                        list_cod_stations.append(current_station)

    if len(list_cod_stations) == 0:
        print("ERROR: There are not EEA station for Italy region: " + italy_region)
        exit(-1)

    return list_cod_stations

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

    global path_dir_EEA_data, path_file_data_EEA_csv, freq_mode, air_poll_selected

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

    if df_all_datetime['Concentration'].isnull().values.any():
        print("Error: NaN values present in cod_station: " + cod_station)
        exit(-1)
    
    return df_all_datetime['Concentration'].values, lon_station, lat_station, station_region, area_station, type_station

# ----------------------- PLOT -----------------------
def plot_hist(  
                    cams_eu, windows_lenght, method_corr, is_previous_elems, \
                    list_corr_EEA_vs_cams_eu_reanalyses, list_corr_EEA_vs_cams_eu_analyses, \
                    list_corr_cams_eu_reanalyses_vs_analyses, air_pol_selected, fig, inner
    ):

    global  start_date_time_to_display, end_date_time_to_display

    bins_range = np.arange(0.0, 1.0, 0.05).tolist()

    count_inner_plot = 0

    # ------------ PLOT EEA vs CAMS Europe Reanalyses ------------
    ax_EEA_vs_cams_eu_reanalyses = plt.Subplot(fig, inner[count_inner_plot])

    ax_EEA_vs_cams_eu_reanalyses.hist(list_corr_EEA_vs_cams_eu_reanalyses, bins = bins_range)
    ax_EEA_vs_cams_eu_reanalyses.set_title("EEA vs " + cams_eu + " R")
    fig.add_subplot(ax_EEA_vs_cams_eu_reanalyses)

    # ------------ PLOT EEA vs CAMS Europe Analyses ------------
    count_inner_plot += 1
    ax_EEA_vs_cams_eu_analyses = plt.Subplot(fig, inner[count_inner_plot])

    ax_EEA_vs_cams_eu_analyses.hist(list_corr_EEA_vs_cams_eu_analyses, bins = bins_range)
    ax_EEA_vs_cams_eu_analyses.set_title("EEA vs " + cams_eu + " A")
    fig.add_subplot(ax_EEA_vs_cams_eu_analyses)

    # ------------ PLOT EEA vs CAMS EUROPA ------------
    count_inner_plot += 1
    ax_cams_eu_reanalyses_vs_analyses = plt.Subplot(fig, inner[count_inner_plot])

    ax_cams_eu_reanalyses_vs_analyses.hist(list_corr_cams_eu_reanalyses_vs_analyses, bins = bins_range)
    ax_cams_eu_reanalyses_vs_analyses.set_title(cams_eu + " R vs A")
    fig.add_subplot(ax_cams_eu_reanalyses_vs_analyses)

list_cod_stations = load_cod_stations(filename_json_file, italy_region)

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
                dict_all_cams_eu_reanalyses[current_cams_eu][cod_station].append(float(np.nanmean(dict_all_hours_of_current_day_cams_eu_reanalyses[current_cams_eu][cod_station])))
                dict_all_cams_eu_analyses[current_cams_eu][cod_station].append(float(np.nanmean(dict_all_hours_of_current_day_cams_eu_analyses[current_cams_eu][cod_station])))

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
            dict_all_cams_eu_reanalyses[current_cams_eu][cod_station].append(float(np.nanmean(dict_all_hours_of_current_day_cams_eu_reanalyses[current_cams_eu][cod_station])))
            dict_all_cams_eu_analyses[current_cams_eu][cod_station].append(float(np.nanmean(dict_all_hours_of_current_day_cams_eu_analyses[current_cams_eu][cod_station])))

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

# -------------- Functions of Correlation --------------
def split_centered_list_sublists(n, iterable, fillvalue=0.0):

    global no_overlap

    sublists = []

    if no_overlap:
        range_for = range(0, len(iterable), int(n/2))
    else:
        range_for = range(len(iterable))

    for i in range_for:
        
        current_sublist = []

        # Example: n=3 [ABC] --> [[0AB], [ABC], [BC0]]

        # If we are in the head of the list:
        # It is necessary to apply padding of (n-1)/2 elements

        if i < (n-1)/2:
            for k in range(int((n-1)/2)):
                current_sublist.append(fillvalue)

            for k in range(int((n-1)/2 + 1)):
                current_sublist.append(iterable[i+k])
        
        # If we are in the last part of the list
        elif int(i + (n-1)/2) >= len(iterable):
            
            for k in reversed(range(int((n-1)/2 + 1))):
                current_sublist.append(iterable[i-k])

            for k in range(i+1, len(iterable)):
                current_sublist.append(iterable[k])

            for k in range(n - len(current_sublist)):
                current_sublist.append(fillvalue)
        
        # If we are in the middle part of the list
        else:
            for k in range(i-int((n-1)/2), i+int((n-1)/2) + 1):
                current_sublist.append(iterable[k])
        
        sublists.append(current_sublist)
    
    return sublists

def split_previous_list_sublists(n, iterable, fillvalue=0.0):

    global no_overlap

    sublists = []

    if no_overlap:
        range_for = range(0, len(iterable), int(n))
    else:
        range_for = range(len(iterable))

    if n % 2 == 0:
        print("n must be odd!")
        exit(-1)

    for i in range_for:
        
        current_sublist = []

        # Example: n=3 [ABC] --> [[00A], [0AB], [ABC]]

        # If we are in the head of the list:
        # It is necessary to apply padding of (n-1) elements
        if i < int(n-1):
            for k in range(int((n-1)) - i):
                current_sublist.append(fillvalue)

            for k in range(0, i + 1):
                current_sublist.append(iterable[k])
        
        # If we are in the middle or last part of the list
        else:
            for k in range(i - int((n-1)), i + 1):
                current_sublist.append(iterable[k])
        
        sublists.append(current_sublist)
    
    return sublists

def compute_correlation(x, y, method_corr):

    list_corr = []

    count_nan_values = 0

    # For each partition of time series
    for i in range(len(x)):

        current_sublist_x = x[i]
        current_sublist_y = y[i]

        if np.std(np.array(current_sublist_x)) == 0 or np.std(np.array(current_sublist_y)) == 0:
            count_nan_values += 1
        else:
            x_pd = pd.Series(current_sublist_x)
            y_pd = pd.Series(current_sublist_y)

            corr_current_sublist = x_pd.corr(y_pd, method=method_corr)
        
            list_corr.append(abs(corr_current_sublist))
    
    return list_corr, count_nan_values

# ------------------ Pearson Correlation ------------------
# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

PATH_DIR_LATEX = joinpath(PATH_DIR_PLOTS, "Latex")
PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, air_poll_selected)
PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, freq_mode)
PATH_DIR_LATEX = joinpath(PATH_DIR_LATEX, type_stations)
PATH_DIR_LATEX = joinPATH_DIR_PLOTSpath(PATH_DIR_LATEX, italy_region)

os.makedirs(PATH_DIR_LATEX, exist_ok=True)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, method_corr + "_EEA_hist_corr_CAMS_EU_Reanalyses_Analyses_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm))
PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, air_poll_selected)

if air_poll_selected == "CO" and co_in_ug_m3:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_ug_m^3")
elif air_poll_selected == "CO":
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "CO_mg_m^3")

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, freq_mode)
PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, type_stations)
PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, italy_region)

if no_overlap:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "No_overlap")
else:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "overlap")

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, filename_output)

os.makedirs(PATH_DIR_PLOTS, exist_ok=True)

filename_latex_file = filename_output + ".txt"
path_latex_file = joinpath(PATH_DIR_LATEX, filename_latex_file)

for windows_lenght in list_windows_lenght:

    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS, "WL_" + str(windows_lenght))

    if not os.path.exists(PATH_DIR_PLOTS_current):
        os.mkdir(PATH_DIR_PLOTS_current)

    if is_previous_elems == False:
        sublists_EEA_station = split_centered_list_sublists(windows_lenght, dict_values_EEA_station[cod_station], 0.0)
    else:
        sublists_EEA_station = split_previous_list_sublists(windows_lenght, dict_values_EEA_station[cod_station], 0.0)
    
    fig = plt.figure(figsize=(20, 15))
    outer = gridspec.GridSpec(2, 4, wspace=0.3, hspace=0.3)

    count_outer_plot = 0

    for current_cams_eu in list_numeric_model_cams_eu:

        list_count_nan_values_corr = [-1,-1,-1]

        list_corr_EEA_vs_cams_eu_reanalyses = []
        list_corr_EEA_vs_cams_eu_analyses = []
        list_corr_cams_eu_reanalyses_vs_analyses = []

        for cod_station in list_cod_stations:

            if is_previous_elems == False:
                sublists_current_cams_eu_reanalyses = split_centered_list_sublists(windows_lenght, dict_all_cams_eu_reanalyses[current_cams_eu][cod_station], 0.0)
                sublists_current_cams_eu_analyses = split_centered_list_sublists(windows_lenght, dict_all_cams_eu_analyses[current_cams_eu][cod_station], 0.0)
            else:
                sublists_current_cams_eu_reanalyses = split_previous_list_sublists(windows_lenght, dict_all_cams_eu_reanalyses[current_cams_eu][cod_station], 0.0)
                sublists_current_cams_eu_analyses = split_previous_list_sublists(windows_lenght, dict_all_cams_eu_analyses[current_cams_eu][cod_station], 0.0)
    
            # Compute correlation between EEA vs CAMS Europe Reanalyses
            list_corr_EEA_vs_cams_eu_reanalyses = []

            
            list_corr_EEA_vs_cams_eu_reanalyses, count_nan_values_EEA_vs_cams_eu_reanalyses = compute_correlation(    
                                                                                                    sublists_EEA_station, 
                                                                                                    sublists_current_cams_eu_reanalyses, 
                                                                                                    method_corr
                                                                                                )
            
            list_count_nan_values_corr[0] = count_nan_values_EEA_vs_cams_eu_reanalyses

            # Compute correlation between EEA vs CAMS Europe Analyses
            list_corr_EEA_vs_cams_eu_analyses = []


            list_corr_EEA_vs_cams_eu_analyses, count_nan_values_EEA_vs_cams_eu_analyses = compute_correlation( 
                                                                                                sublists_EEA_station, 
                                                                                                sublists_current_cams_eu_analyses, 
                                                                                                method_corr
                                                                                            )

            list_count_nan_values_corr[1] = count_nan_values_EEA_vs_cams_eu_analyses

            # Compute correlation between CAMS Europa Reanalyses vs Analyses
            list_corr_cams_eu_reanalyses_vs_analyses = []

            list_corr_cams_eu_reanalyses_vs_analyses, count_nan_values_cams_eu_reanalyses_vs_analyses = compute_correlation(    
                                                                                                            sublists_current_cams_eu_reanalyses, 
                                                                                                            sublists_current_cams_eu_analyses, 
                                                                                                            method_corr
                                                                                                        )
            
            list_count_nan_values_corr[2] = count_nan_values_cams_eu_reanalyses_vs_analyses
        
        # --------------- SUBPLOT Current CAMS Europe ---------------
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[count_outer_plot], wspace=0.1, hspace=0.1)
        
        plot_hist(  
                        current_cams_eu, windows_lenght, method_corr, is_previous_elems, \
                        list_corr_EEA_vs_cams_eu_reanalyses, list_corr_EEA_vs_cams_eu_analyses, list_corr_cams_eu_reanalyses_vs_analyses, \
                        air_poll_selected, fig, inner
                )

        count_outer_plot += 1

        PATH_DIR_PLOTS_current_log_files = joinpath(PATH_DIR_PLOTS_current, "log")

        if not os.path.exists(PATH_DIR_PLOTS_current_log_files):
            os.mkdir(PATH_DIR_PLOTS_current_log_files)
        
        with open(joinpath(PATH_DIR_PLOTS_current_log_files, "log_nan.txt"), "w") as file:
            file.write("NaN cross-correlations values of EEA vs CAMS Europe Reanalyses (" + str(current_cams_eu) + "):" + str(list_count_nan_values_corr[0]) + "\n")
            file.write("NaN cross-correlations values of EEA vs CAMS Europe Analyses (" + str(current_cams_eu) + "):" + str(list_count_nan_values_corr[1]) + "\n")
            file.write("NaN cross-correlations values of CAMS Europe Reanalyses vs Analyses (" + str(current_cams_eu) + "):" + str(list_count_nan_values_corr[2]) + "\n")
    
    if is_previous_elems: 
        plt.title( 
                air_poll_selected + " " + method_corr.upper() + \
                    " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " + \
                    end_date_time_to_display.strftime('%Y/%m/%d') + \
                    " - WL: " + str(windows_lenght) + \
                    " - Previous", y=-0.01
                )
    else:
        plt.title( 
                    air_poll_selected + " " + method_corr.upper() + \
                    " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " + \
                    end_date_time_to_display.strftime('%Y/%m/%d') + \
                    " - WL: " + str(windows_lenght) + \
                    " - Centered", y=-0.01
                )
    
    fig.tight_layout()

    if save_plot:
        # ------------------ PLOT -----------------------
        if is_previous_elems: 
            filename_fig =  filename_output + "_" + method_corr.upper() + "_WL_" + str(windows_lenght) + "_previous.png"
        else:
            filename_fig =  filename_output + "_" + method_corr.upper() + "_WL_" + str(windows_lenght) + "_centered.png"

            filename_fig =  "EEA_" + start_date_time_to_display.strftime('%Y-%m-%d') + "_" + end_date_time_to_display.strftime('%Y-%m-%d') + \
                            method_corr.upper() + "_WL_" + str(windows_lenght) + "_centered.png"

        path_fig = joinpath(PATH_DIR_PLOTS_current, filename_fig)
        plt.savefig(path_fig, dpi=300)
    else:
        plt.show()

    plt.close('all')

    