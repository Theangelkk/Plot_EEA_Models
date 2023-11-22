# Scripy of plot of EEA stations and CAMS Europe Reanalysis and Analysis
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

parser = argparse.ArgumentParser(description='Plot EEA - CAMS Europe - GEOS CF - CAMS Global')
parser.add_argument('-a', '--air_pollutant', help='Air Pollutant CO - NO2 - O3 - PM2.5 - PM10 - SO2', choices=list_air_pollutant, required=True)
parser.add_argument('-list_cod_stations', '--list_cod_stations', help='List of code stations EEA', nargs='+', required=True)
parser.add_argument('-m_air_pol', '--m_air_pol', help='Model level for air pollution', type=int, required=True)
parser.add_argument('-m_pm', '--m_pm', help='Model level for Particulate', type=int, required=True)
parser.add_argument('-delta_h', '--delta_hours', type=int, required=True)
parser.add_argument('-s_date', '--start_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-e_date', '--end_date', metavar='YYYY-MM-DD HH:MM:SS', type=valid_datetime, required=True)
parser.add_argument('-cams_eu', '--cams_eu', help='chimere - ensemble - EMEP - LOTOS-EUROS - MATCH - MINNI - MOCAGE - SILAM - EURAD-IM - DEHM - GEM-AQ', \
                     choices=list_numeric_model_cams_eu, required=True)
parser.add_argument('-co_ug_m^3', '--co_ug_m^3', help='CO in ug/m^3', action='store_true')
parser.add_argument('-save_plot', '--save_plot', help='Save plot data', action='store_true')
parser.add_argument('-split_days_plot', '--split_days_plot', help='How many days to visualize in one plot', type=int, required=True)
args = vars(parser.parse_args())

air_poll_selected = args["air_pollutant"]
list_cod_stations = args["list_cod_stations"]
start_date_time_to_display = args["start_date"]
end_date_time_to_display = args["end_date"]
split_days_plot = int(args["split_days_plot"])

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

if args["delta_hours"] > 0:
    delta_time_hours = int(args["delta_hours"])

# Limit values Directive 2008/50/EC

dict_range_air_pollutants = {}
dict_limit_air_pollutants = {}

# NO2 limit value 1 hour: 18.0 ug/m3
dict_limit_air_pollutants["NO2"] = 18.0
dict_range_air_pollutants["NO2"] = 75.0

# CO limit value for daily max 8 hours: 10.0 mg/m3
dict_limit_air_pollutants["CO"] = 10.0
dict_range_air_pollutants["CO"] = 15.0

# Limit CO in ug/m3
dict_limit_air_pollutants["CO_ug_m3"] = dict_limit_air_pollutants["CO"] * 1000
dict_range_air_pollutants["CO_ug_m3"] = dict_range_air_pollutants["CO"] * 1000

# SO2 limit value for 1 hour: 350.0 ug/m3
dict_limit_air_pollutants["SO2"] = 350.0
dict_range_air_pollutants["SO2"] = 500.0

# O3 limit value for daily max 8 hours: 120.0 ug/m3
dict_limit_air_pollutants["O3"] = 120.0
dict_range_air_pollutants["O3"] = 250.0

# PM2.5 limit value for 1 year: 25.0 ug/m3
dict_limit_air_pollutants["PM2.5"] = 15.0
dict_range_air_pollutants["PM2.5"] = 60.0

# PM10 limit value for 1 year: 50.0 ug/m3
dict_limit_air_pollutants["PM10"] = 50.0
dict_limit_air_pollutants["PM10"] = 100.0

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

# ------------ Information on CAMS EUROPE Reanalysis ------------
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

# ------------ Information on CAMS EUROPE Analysis ------------
not_available_cams_eu_analysis = False

# Path of CAMS Europe
path_main_dir_CAMS_Europe_analysis_data = os.environ['CAMS_Europe_Analysis']

if path_main_dir_CAMS_Europe_analysis_data == "":
    print("Error: set the environmental variables of CAMS_Europe_Analysis")
    exit(-1)

DATADIR_CAMS_EU_Analysis = joinpath(path_main_dir_CAMS_Europe_analysis_data, "model_level_" + str(model_level_air_pollution))
DATADIR_CAMS_EU_Analysis = joinpath(DATADIR_CAMS_EU_Analysis, "italy_ext")

numeric_model_selected = list_numeric_model_cams_eu[idx_numeric_model_cams_eu]
DATADIR_CAMS_EU_Analysis = joinpath(DATADIR_CAMS_EU_Analysis, numeric_model_selected)

if air_poll_selected == "PM2.5":
    DATADIR_CAMS_EU_Analysis = joinpath(DATADIR_CAMS_EU_Analysis, "PM2p5")
else:
    DATADIR_CAMS_EU_Analysis = joinpath(DATADIR_CAMS_EU_Analysis, air_poll_selected)

# Time resolution of CAMS Europe
time_res_cams_eu = 1

dict_start_time_numeric_models_cams_eu_analysis = {}
dict_start_time_numeric_models_cams_eu_analysis["SILAM"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analysis["MOCAGE"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analysis["MINNI"] = datetime(2022, 6, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analysis["MATCH"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analysis["LOTOS-EUROS"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analysis["EURAD-IM"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analysis["ensemble"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analysis["EMEP"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analysis["DEHM"] = datetime(2020, 1, 1, 0, 0)
dict_start_time_numeric_models_cams_eu_analysis["chimere"] = datetime(2020, 1, 1, 0, 0)

end_time_cams_eu_analysis = datetime(2022, 12, 31, 0, 0)

def load_ds_datasets(current_date):

    global  dict_start_time_numeric_models_cams_eu, numeric_model_selected, \
            end_time_cams_eu, end_time_cams_eu_analysis, air_poll_selected, \
            DATADIR_CAMS_EU, DATADIR_CAMS_EU_Analysis, \
            not_available_cams_eu, not_available_cams_eu_analysis, \
            co_in_ug_m3
    
    ds_cams_eu = None
    ds_cams_eu_analysis = None

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
    
    if  current_date >= dict_start_time_numeric_models_cams_eu_analysis[numeric_model_selected] and \
        current_date <= end_time_cams_eu_analysis:
        DATADIR_CURRENT_MONTH_CAMS_EU_Analysis = joinpath(DATADIR_CAMS_EU_Analysis, str(current_date.year) + "-" + str(current_date.month).zfill(2))
        
        if os.path.exists(DATADIR_CURRENT_MONTH_CAMS_EU_Analysis):
            file_netcdf_path_cams_eu_analysis = joinpath(DATADIR_CURRENT_MONTH_CAMS_EU_Analysis, str(current_date.year) + "-" + str(current_date.month).zfill(2) + ".nc")
            ds_cams_eu_analysis = xr.open_dataset(file_netcdf_path_cams_eu_analysis)
        else:
            not_available_cams_eu_analysis = True
            print("Dataset not available for CAMS EU " +  str(numeric_model_selected))

        if co_in_ug_m3 and air_poll_selected == "CO":
            ds_cams_eu["CO"] *= 1000

        not_available_cams_eu = False    
    else:
        not_available_cams_eu = True
        print("Dataset not available for CAMS EU " +  str(numeric_model_selected))

    return ds_cams_eu, ds_cams_eu_analysis

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

    df_station_date.sort_values(by='DatetimeBegin', ascending = True, inplace = True)

    df_station_date_current_year = df_station_date[ df_station_date.index.to_series().between(current_date.isoformat(), end_current_date.isoformat())]

    dates = pd.date_range(start=current_date.isoformat(), end=end_current_date.isoformat(), freq="H", tz='UTC+01:00').tolist()
    conc_dates_nan = [np.nan for i in range(len(dates))]

    df_all_datetime = pd.DataFrame(list(zip(dates, conc_dates_nan)), columns=['DatetimeBegin', 'Concentration'])
    df_all_datetime = df_all_datetime.set_index('DatetimeBegin')

    for index, row in df_station_date_current_year.iterrows():
        df_all_datetime.loc[index]["Concentration"] = df_station_date_current_year.loc[index]["Concentration"] 
    
    # Interpolation of measures
    df_all_datetime['Concentration'] = df_all_datetime['Concentration'].interpolate(method='linear')
    
    return df_all_datetime['Concentration'].values, lon_station, lat_station, station_region

# ----------------------- PLOT -----------------------
def plot(   
            cod_station, air_pol_selected, list_values_EEA_station, \
            list_values_cams_eu_reanalysis, list_values_cams_eu_analysis, \
            PATH_DIR_PLOTS, start_time, end_time
        ):

    global  delta, co_in_ug_m3, save_plot, dict_limit_air_pollutants, dict_range_air_pollutants, \
            not_available_cams_eu, not_available_cams_eu_analysis

    # Set up the axes and figure
    fig, ax = plt.subplots()

    dates = mdates.drange(start_time, end_time, delta)

    ax.plot(dates, list_values_EEA_station, label="EEA", alpha=0.75, linewidth=3)
    
    if not_available_cams_eu == False:
        ax.plot(dates, list_values_cams_eu_reanalysis, label="CAMS EU Reanalysis",  alpha=1.0, linewidth=1)
    
    if not_available_cams_eu_analysis == False:
        ax.plot(dates, list_values_cams_eu_analysis, label="CAMS EU Analysis",  alpha=1.0, linewidth=1)
    
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
ds_cams_eu_analysis = None

diff_dates = end_date_time_to_display - start_date_time_to_display
diff_dates_hours = int(diff_dates.total_seconds() / (60*60*delta_time_hours))
delta = timedelta(hours=delta_time_hours)

# Reading matainfo file about air pollutant of the country defined
df_air_pol_metainfo = pd.read_table(path_file_air_pol_metainfo, delimiter=',', index_col=0)

# Reading the CSV file about air pollutant defined
df_air_pol_data = pd.read_table(    
                                    path_file_data_EEA_csv, delimiter=',', 
                                    header=[0], index_col=0, low_memory=False
                                )

# Loading the data sets of EEA ground-based stations
dict_code_stations = {}

# Loading CAMS Europe Reanalysis and Analysis data sets
ds_cams_eu, ds_cams_analysis = load_ds_datasets(current_date)

list_datetime_x = []
dict_values_EEA_station = {}
dict_values_cams_eu = {}
dict_values_cams_eu_analysis = {}

# Initialization of all dictionaries
for cod_station in list_cod_stations:

    df_station_date_current_year_values, lon_station, lat_station, region_station = \
        load_EEA_station(   cod_station, current_date, end_date_time_to_display-delta, \
                            df_air_pol_data, df_air_pol_metainfo
                        )
    
    dict_code_stations[cod_station] = [lon_station, lat_station, region_station]

    dict_values_EEA_station[cod_station] = []
    dict_values_cams_eu[cod_station] = []
    dict_values_cams_eu_analysis[cod_station] = []

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

        ds_cams_eu, ds_cams_analysis = load_ds_datasets(current_date)
    
    if previous_date.month != current_date.month:
        ds_cams_eu, ds_cams_analysis = load_ds_datasets(current_date)

    # For each stations
    for cod_station in list_cod_stations:
        
        lon_station = dict_code_stations[cod_station][0]
        lat_station = dict_code_stations[cod_station][1]

        # Loading CAMS Europe Reanalysis and Analysis data sets
        if not_available_cams_eu == False:
            ds_current_date_cams_eu = ds_cams_eu.sel(time=current_date.isoformat())

            if air_poll_selected == "PM2.5":
                cams_eu_delta_time = ds_current_date_cams_eu.sel(lat=lat_station, lon=lat_station, method='nearest')["pm2p5"].values
            else:
                cams_eu_delta_time = ds_current_date_cams_eu.sel(lat=lat_station, lon=lat_station, method='nearest')[air_poll_selected.lower()].values
            
            dict_values_cams_eu[cod_station].append(float(cams_eu_delta_time))
        else:
            ds_current_date_cams_eu = None

        if not_available_cams_eu_analysis == False:
            ds_current_date_cams_eu_analysis = ds_cams_eu_analysis.sel(time=current_date.isoformat())

            if air_poll_selected == "PM2.5":
                cams_eu_analysis_delta_time = ds_current_date_cams_eu_analysis.sel(lat=lat_station, lon=lat_station, method='nearest')["PM25"].values
            else:
                cams_eu_analysis_delta_time = ds_current_date_cams_eu_analysis.sel(lat=lat_station, lon=lat_station, method='nearest')[air_poll_selected].values
            
            dict_values_cams_eu_analysis[cod_station].append(float(cams_eu_analysis_delta_time))
        else:
            ds_current_date_cams_eu_analysis = None
        
    list_datetime_x.append(current_date.isoformat())
    
    print("Current date (" + air_poll_selected + "): " + current_date.isoformat())
    print("CAMS EU Reanalysis: " + str(not not_available_cams_eu))
    print("CAMS EU Analysis: " + str(not not_available_cams_eu_analysis))

    previous_date = current_date
    current_date += delta

# ------------------ PLOT -----------------------
# Path of Plots
PATH_DIR_PLOTS = os.environ['Plot_dir']

if PATH_DIR_PLOTS == "":
    print("Error: set the environmental variables of Plot_dir")
    exit(-1)

if not os.path.exists(PATH_DIR_PLOTS):
  os.mkdir(PATH_DIR_PLOTS)

PATH_DIR_PLOTS = joinpath(PATH_DIR_PLOTS, "EEA_plots_CAMS_Reanalysis_Analysis_" + str(model_level_air_pollution) + "_pm_" + str(model_level_pm) + "_camsEU_" + str(list_numeric_model_cams_eu[idx_numeric_model_cams_eu]))

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

for cod_station in list_cod_stations:
    
    region_cod_station = dict_code_stations[cod_station][2]
    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS, region_cod_station)

    if not os.path.exists(PATH_DIR_PLOTS_current):
        os.mkdir(PATH_DIR_PLOTS_current)

    PATH_DIR_PLOTS_current = joinpath(PATH_DIR_PLOTS_current, cod_station)

    if not os.path.exists(PATH_DIR_PLOTS_current):
        os.mkdir(PATH_DIR_PLOTS_current)

    plot(   
            cod_station, air_poll_selected, dict_values_EEA_station[cod_station], \
            dict_values_cams_eu[cod_station], dict_values_cams_eu_analysis[cod_station], \
            PATH_DIR_PLOTS_current, start_date_time_to_display, end_date_time_to_display
    )

    # Split the time series in "split_days_plot" days defined
    if split_days_plot > 0:
        
        delta_days = timedelta(days=split_days_plot)
        n_hours = delta_days.total_seconds()//3600

        steps_dict = int(n_hours / delta_time_hours)

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

            current_dict_values_cams_eu = []

            if not_available_cams_eu == False:
                current_dict_values_cams_eu = \
                            dict_values_cams_eu[cod_station][idx_dict : idx_end]
            
            current_dict_values_cams_eu_analysis = []

            if not_available_cams_eu_analysis == False:
                current_dict_values_cams_eu_analysis = \
                            dict_values_cams_eu_analysis[cod_station][idx_dict : idx_end]
            plot(   
                    cod_station, air_poll_selected, current_dict_values_EEA_station, \
                    current_dict_values_cams_eu, current_dict_values_cams_eu_analysis, \
                    PATH_DIR_PLOTS_current, current_start_time_to_display, \
                    current_end_time_to_display
            )

            current_start_time_to_display = current_end_time_to_display
            current_end_time_to_display = current_end_time_to_display + delta_days
            idx_dict = idx_dict + steps_dict