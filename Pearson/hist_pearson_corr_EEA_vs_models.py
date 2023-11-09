# Pearson coefficient cross-correlation histograms
# [EEA vs CAMS EU - GEOS CF - CAMS GLOBAL]
# [CAMS EU vs GEOS CF - CAMS EU vs CAMS GLOBAL - GEOS CF vs CAMS GLOBAL]

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
list_numeric_model_cams_eu = [  "chimere", "ensemble", "EMEP", "LOTOS-EUROS", "MATCH", \
                                "MINNI", "MOCAGE", "SILAM", "EURAD-IM", "DEHM", "GEM-AQ"]

parser = argparse.ArgumentParser(description='Correlations EEA - CAMS Europe - GEOS CF - CAMS Global')
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
parser.add_argument('-list_window_lenght', '--list_window_lenght', help='List of Window lenght', nargs="+", type=int, required=True)
parser.add_argument('-previous_elems', '--previous_elems', help='Subseries with previous samples', action='store_true')
parser.add_argument('-method_corr', '--method_corr', help='Method for computing correlation [pearson, kendall, spearman]', default='pearson', choices=list_methods_of_corr)
parser.add_argument('-save_plot', '--save_plot', help='Save plot data', action='store_true')
parser.add_argument('-no_overlap', '--no_overlap', help='No overlap window lenght', action='store_true')
args = vars(parser.parse_args())

cams_eu = args["cams_eu"]
air_poll_selected = args["air_pollutant"]
list_cod_stations = args["list_cod_stations"]
start_date_time_to_display = args["start_date"]
end_date_time_to_display = args["end_date"]
list_windows_lenght = args["list_window_lenght"]
method_corr = args["method_corr"]

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
def plot_hist_EEA(  
                    cod_station, windows_lenght, method_corr, is_previous_elems, \
                    list_corr_EEA_vs_cams_global, list_corr_EEA_vs_geos_cf, \
                    list_corr_EEA_vs_cams_eu, air_pol_selected, PATH_DIR_PLOTS
    ):

    global  start_date_time_to_display, end_date_time_to_display, \
            not_available_cams_eu, not_available_goes_cf, not_available_cams_global

    bins_range = np.arange(0.0, 1.0, 0.05).tolist()

    # Set up the axes and figure
    fig = plt.figure()

    # ------------ PLOT EEA vs CAMS GLOBAL ------------
    ax_EEA_vs_cams_global = fig.add_subplot(311)

    if not_available_cams_global == False:
        ax_EEA_vs_cams_global.hist(list_corr_EEA_vs_cams_global, bins = bins_range)

        if is_previous_elems: 
            ax_EEA_vs_cams_global.set_title( air_pol_selected + " " + method_corr.upper() + \
                                            " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                            + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                            " EEA vs CAMS GLOBAL " + str(cod_station) + \
                                            " - WL: " + str(windows_lenght) + \
                                            " - Previous")
        else:
            ax_EEA_vs_cams_global.set_title( air_pol_selected + " " + method_corr.upper() + \
                                            " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                            + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                            " EEA vs CAMS GLOBAL " + str(cod_station) + \
                                            " - WL: " + str(windows_lenght) + \
                                            " - Centered")


    # ------------ PLOT EEA vs GEOS CF ------------
    ax_EEA_vs_geos_cf = fig.add_subplot(312)

    if not_available_goes_cf == False:
        ax_EEA_vs_geos_cf.hist(list_corr_EEA_vs_geos_cf, bins = bins_range)

        if is_previous_elems: 
            ax_EEA_vs_geos_cf.set_title( air_pol_selected + " " + method_corr.upper() + \
                                            " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                            + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                            " EEA vs GEOS CF " + str(cod_station) + \
                                            " - WL: " + str(windows_lenght) + \
                                            " - Previous")
        else:
            ax_EEA_vs_geos_cf.set_title( air_pol_selected + " " + method_corr.upper() + \
                                            " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                            + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                            " EEA vs GEOS CF " + str(cod_station) + \
                                            " - WL: " + str(windows_lenght) + \
                                            " - Centered")

    # ------------ PLOT EEA vs CAMS EUROPA ------------
    ax_EEA_vs_cams_eu = fig.add_subplot(313)

    if not_available_cams_eu == False:
        ax_EEA_vs_cams_eu.hist(list_corr_EEA_vs_cams_eu, bins = bins_range)

        if is_previous_elems: 
            ax_EEA_vs_cams_eu.set_title( air_pol_selected + " " + method_corr.upper() + \
                                            " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                            + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                            " EEA vs CAMS EU " + str(cod_station) + \
                                            " - WL: " + str(windows_lenght) + \
                                            " - Previous")
        else:
            ax_EEA_vs_cams_eu.set_title( air_pol_selected + " " + method_corr.upper() + \
                                            " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                            + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                            " EEA vs CAMS EU " + str(cod_station) + \
                                            " - WL: " + str(windows_lenght) + \
                                            " - Centered")

    if save_plot:
        if is_previous_elems: 
            filename_fig =  "EEA_" + start_date_time_to_display.strftime('%Y-%m-%d') + "_" + end_date_time_to_display.strftime('%Y-%m-%d') + \
                            method_corr.upper() + "_WL_" + str(windows_lenght) + "_previous.png"
        else:
            filename_fig =  "EEA_" + start_date_time_to_display.strftime('%Y-%m-%d') + "_" + end_date_time_to_display.strftime('%Y-%m-%d') + \
                            method_corr.upper() + "_WL_" + str(windows_lenght) + "_centered.png"

        path_fig = joinpath(PATH_DIR_PLOTS, filename_fig)
        plt.savefig(path_fig, dpi=300)
    else:
        plt.show()

    plt.close()

def plot_hist_models(  
                    cod_station, windows_lenght, method_corr, is_previous_elems, \
                    list_corr_cams_eu_vs_geos_cf, list_corr_cams_eu_vs_cams_global, \
                    list_corr_geos_cf_vs_cams_global, air_pol_selected, PATH_DIR_PLOTS
    ):

    global  start_date_time_to_display, end_date_time_to_display, \
            not_available_cams_eu, not_available_goes_cf, not_available_cams_global

    bins_range = np.arange(0.0, 1.0, 0.05).tolist()

    # Set up the axes and figure
    fig = plt.figure()

    # ------------ PLOT CAMS EU vs GEOS CF ------------
    ax_cams_eu_vs_geos_cf = fig.add_subplot(311)

    if not_available_cams_eu == False and not_available_goes_cf == False:
        ax_cams_eu_vs_geos_cf.hist(list_corr_cams_eu_vs_geos_cf, bins = bins_range)

        if is_previous_elems: 
            ax_cams_eu_vs_geos_cf.set_title( air_pol_selected + " " + method_corr.upper() + \
                                             " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                             + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                             " CAMS EU vs GEOS CF " + str(cod_station) + \
                                             " - WL: " + str(windows_lenght) + \
                                             " - Previous")
        else:
            ax_cams_eu_vs_geos_cf.set_title( air_pol_selected + " " + method_corr.upper() + \
                                             " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                             + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                             " CAMS EU vs GEOS CF " + str(cod_station) + \
                                             " - WL: " + str(windows_lenght) + \
                                             " - Centered")

    # ------------ PLOT CAMS EU vs CAMS GLOBAL ------------
    ax_cams_eu_vs_cams_global = fig.add_subplot(312)

    if not_available_cams_eu == False and not_available_cams_global == False:
        ax_cams_eu_vs_cams_global.hist(list_corr_cams_eu_vs_cams_global, bins = bins_range)

        if is_previous_elems: 
            ax_cams_eu_vs_cams_global.set_title( air_pol_selected + " " + method_corr.upper() + \
                                                 " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                                 + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                                 " CAMS EU vs CAMS GLOBAL " + str(cod_station) + \
                                                 " - WL: " + str(windows_lenght) + \
                                                 " - Previous")
        else:
            ax_cams_eu_vs_cams_global.set_title( air_pol_selected + " " + method_corr.upper() + \
                                                 " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                                 + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                                 " CAMS EU vs CAMS GLOBAL " + str(cod_station) + \
                                                 " - WL: " + str(windows_lenght) + \
                                                 " - Centered")

    # ------------ PLOT GEOS CF vs CAMS GLOBAL ------------
    ax_geos_cf_vs_cams_global = fig.add_subplot(313)
    
    if not_available_goes_cf == False and not_available_cams_global == False:
        ax_geos_cf_vs_cams_global.hist(list_corr_geos_cf_vs_cams_global, bins = bins_range)

        if is_previous_elems: 
            ax_geos_cf_vs_cams_global.set_title( air_pol_selected + " " + method_corr.upper() + \
                                                 " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                                 + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                                 " GEOS CF vs CAMS GLOBAL " + str(cod_station) + \
                                                 " - WL: " + str(windows_lenght) + \
                                                 " - Previous")
        else:
            ax_geos_cf_vs_cams_global.set_title( air_pol_selected + " " + method_corr.upper() + \
                                                 " Correlation " + start_date_time_to_display.strftime('%Y/%m/%d') + " - " \
                                                 + end_date_time_to_display.strftime('%Y/%m/%d') + \
                                                 " GEOS CF vs CAMS GLOBAL " + str(cod_station) + \
                                                 " - WL: " + str(windows_lenght) + \
                                                 " - Centered")

    if save_plot:
        if is_previous_elems: 
            filename_fig =  "MODELS_" + start_date_time_to_display.strftime('%Y-%m-%d') + "_" + end_date_time_to_display.strftime('%Y-%m-%d') + \
                            method_corr.upper() + "_WL_" + str(windows_lenght) + "_previous.png"
        else:
            filename_fig =  "MODELS_" + start_date_time_to_display.strftime('%Y-%m-%d') + "_" + end_date_time_to_display.strftime('%Y-%m-%d') + \
                            method_corr.upper() + "_WL_" + str(windows_lenght) + "_centered.png"

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

    # Vuol dire che è cambiato il mese
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

# ------------------ Interpolation of CAMS GLOBAL ------------------
for cod_station in list_cod_stations:

    dict_cams_global = {'DatetimeBegin': list_datetime_x, 'Concentration': dict_values_cams_global[cod_station]}
    df_cams_global = pd.DataFrame(data=dict_cams_global)

    df_cams_global = df_cams_global.set_index('DatetimeBegin')
    df_cams_global.index = pd.to_datetime(df_cams_global.index)

    # Interpolation
    df_cams_global['Concentration'] = df_cams_global['Concentration'].interpolate(method='linear')

    dict_values_cams_global[cod_station] = df_cams_global['Concentration'].values

# -------------- Functions of Correlations --------------
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

        x_pd = pd.Series(current_sublist_x)
        y_pd = pd.Series(current_sublist_y)

        corr_current_sublist = x_pd.corr(y_pd, method=method_corr)
        
        # LINK: https://www.mathworks.com/matlabcentral/answers/506464-getting-a-nan-in-correlation-coefficient

        # Viene restituto un valore di correlazione tra due variabili
        # uguale a NaN, quando uno dei due vettori dispone di elementi
        # tutti uguali --> quindi la deviazione standard è uguale a 0 e quindi
        # la frazione della correlazione dispone di denominatore uguale a 0
        # ECCO perchè corr = np.nan
        # In questo caso può essere considerata l'indipendenza tra due vettori
        # quindi correlazione = 0
        if math.isnan(corr_current_sublist):
            count_nan_values += 1

        list_corr.append(abs(corr_current_sublist))
    
    return list_corr, count_nan_values

# ------------------ Correlation ------------------
# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, method_corr + "_EEA_hist_corr_all_air_model_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm) + "_camsEU_" + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]))

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

if no_overlap:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "No_overlap")
else:
    PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "overlap")

if not os.path.exists(PATH_DIR_PLOTS):
    os.mkdir(PATH_DIR_PLOTS)

for cod_station in list_cod_stations:

    for windows_lenght in list_windows_lenght:
        
        list_count_nan_values_corr = [-1,-1,-1,-1,-1,-1]

        if is_previous_elems == False:
            sublists_EEA_station = split_centered_list_sublists(windows_lenght, dict_values_EEA_station[cod_station], 0.0)
            
            if not_available_cams_global == False:
                sublists_cams_global = split_centered_list_sublists(windows_lenght, dict_values_cams_global[cod_station], 0.0)
            
            if not_available_goes_cf == False:
                sublists_geos_cf = split_centered_list_sublists(windows_lenght, dict_values_geos_cf[cod_station], 0.0)
            
            if not_available_cams_eu == False:
                sublists_cams_eu = split_centered_list_sublists(windows_lenght, dict_values_cams_eu[cod_station], 0.0)
        else:
            sublists_EEA_station = split_previous_list_sublists(windows_lenght, dict_values_EEA_station[cod_station], 0.0)
            
            if not_available_cams_global == False:
                sublists_cams_global = split_previous_list_sublists(windows_lenght, dict_values_cams_global[cod_station], 0.0)
            
            if not_available_goes_cf == False:
                sublists_geos_cf = split_previous_list_sublists(windows_lenght, dict_values_geos_cf[cod_station], 0.0)
            
            if not_available_cams_eu == False:
                sublists_cams_eu = split_previous_list_sublists(windows_lenght, dict_values_cams_eu[cod_station], 0.0)

        # Compute correlation between EEA vs CAMS Global
        list_corr_EEA_vs_cams_global = []

        if not_available_cams_global == False:
            list_corr_EEA_vs_cams_global, count_nan_values_EEA_vs_cams_global = compute_correlation(    
                                                                                            sublists_EEA_station, 
                                                                                            sublists_cams_global, 
                                                                                            method_corr
                                                                                        )
            
            list_count_nan_values_corr[0] = count_nan_values_EEA_vs_cams_global

        # Compute correlation between EEA vs Geos CF
        list_corr_EEA_vs_geos_cf = []

        if not_available_goes_cf == False:
            list_corr_EEA_vs_geos_cf, count_nan_values_EEA_vs_geos_cf = compute_correlation( 
                                                                                        sublists_EEA_station, 
                                                                                        sublists_geos_cf, 
                                                                                        method_corr
                                                                                    )

            list_count_nan_values_corr[1] = count_nan_values_EEA_vs_geos_cf

        # Compute correlation between EEA vs CAMS Europe
        list_corr_EEA_vs_cams_eu = []

        if not_available_cams_eu == False:
            list_corr_EEA_vs_cams_eu, count_nan_values_EEA_vs_cams_eu = compute_correlation(    
                                                                                        sublists_EEA_station, 
                                                                                        sublists_cams_eu, 
                                                                                        method_corr
                                                                                    )
            
            list_count_nan_values_corr[2] = count_nan_values_EEA_vs_cams_eu

        # Compute correlation between CAMS Europa vs GEOS CF
        list_corr_cams_eu_vs_geos_cf = []

        if not_available_cams_eu == False and not_available_goes_cf == False:
            list_corr_cams_eu_vs_geos_cf, count_nan_values_cams_eu_vs_geos_cf = compute_correlation(    
                                                                                                sublists_cams_eu, 
                                                                                                sublists_geos_cf, 
                                                                                                method_corr
                                                                                            )
            
            list_count_nan_values_corr[3] = count_nan_values_cams_eu_vs_geos_cf
            
        # Compute correlation between CAMS Europe vs CAMS Global
        list_corr_cams_eu_vs_cams_global = []

        if not_available_cams_eu == False and not_available_cams_global == False:
            list_corr_cams_eu_vs_cams_global, count_nan_values_cams_eu_vs_cams_global = compute_correlation(    
                                                                                                sublists_cams_eu, 
                                                                                                sublists_cams_global, 
                                                                                                method_corr
                                                                                            )
            
            list_count_nan_values_corr[4] = count_nan_values_cams_eu_vs_cams_global
            
        # Compute correlation between GEOS CF vs CAMS Global
        list_corr_geos_cf_vs_cams_global = []

        if not_available_goes_cf == False and not_available_cams_global == False:
            list_corr_geos_cf_vs_cams_global, count_nan_values_geos_cf_vs_cams_global = compute_correlation(    
                                                                                                sublists_geos_cf, 
                                                                                                sublists_cams_global, 
                                                                                                method_corr
                                                                                            )
            
            list_count_nan_values_corr[5] = count_nan_values_geos_cf_vs_cams_global
        
        # ------------------ PLOT -----------------------
        region_cod_station = dict_code_stations[cod_station][2]
        PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS, region_cod_station)

        if not os.path.exists(PATH_DIR_PLOTS_current):
            os.mkdir(PATH_DIR_PLOTS_current)

        PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS_current, cod_station)

        if not os.path.exists(PATH_DIR_PLOTS_current):
            os.mkdir(PATH_DIR_PLOTS_current)

        PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS_current, "WL_" + str(windows_lenght))

        if not os.path.exists(PATH_DIR_PLOTS_current):
            os.mkdir(PATH_DIR_PLOTS_current)

        plot_hist_EEA(  
                        cod_station, windows_lenght, method_corr, is_previous_elems, \
                        list_corr_EEA_vs_cams_global, list_corr_cams_eu_vs_cams_global, list_corr_EEA_vs_cams_eu, \
                        air_poll_selected, PATH_DIR_PLOTS_current
                )
        
        plot_hist_models(  
                        cod_station, windows_lenght, method_corr, is_previous_elems, \
                        list_corr_cams_eu_vs_geos_cf, list_corr_EEA_vs_geos_cf, \
                        list_corr_geos_cf_vs_cams_global, air_poll_selected, PATH_DIR_PLOTS_current
                )
        
        PATH_DIR_PLOTS_current_log_files = joinpath(PATH_DIR_PLOTS_current, "log")

        if not os.path.exists(PATH_DIR_PLOTS_current_log_files):
            os.mkdir(PATH_DIR_PLOTS_current_log_files)
        
        with open(joinpath(PATH_DIR_PLOTS_current_log_files, "log_nan.txt")) as file:
            file.write("NaN cross-correlations values of EEA vs CAMS GLOBAL:" + str(list_count_nan_values_corr[0]) + "\n")
            file.write("NaN cross-correlations values of EEA vs GEOS CF:" + str(list_count_nan_values_corr[1]) + "\n")
            file.write("NaN cross-correlations values of EEA vs CAMS EU " + cams_eu + ": " + str(list_count_nan_values_corr[2]) + "\n")
            file.write("NaN cross-correlations values of CAMS EU vs GEOS CF:" + str(list_count_nan_values_corr[3]) + "\n")
            file.write("NaN cross-correlations values of CAMS EU vs CAMS GLOBAL:" + str(list_count_nan_values_corr[4]) + "\n")
            file.write("NaN cross-correlations values of GEOS CF vs CAMS GLOBAL:" + str(list_count_nan_values_corr[5]) + "\n")