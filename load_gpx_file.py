

import extract_column_data
import pandas as pd
from os import listdir
from os.path import isfile, join
import os
import glob
import gpxpy
import gpxpy.gpx

stored_data_path = "C:\John_folder\Pycharm_blog\Data_raw\John_Run_Data"

col_name = ['Run_date', 'Run_year', 'Run_month', 'Total_time', 'Total_distance', 'Total_run_time', 'Total_run_distance', 'Start_time', 'End_time', 'Total_pause_time',
                'Total_pause_distance', 'Total_uphill', 'Total_downhill'] # column names for my running data
run_dataframe = pd.DataFrame(columns=col_name) # Initilise the DataFrame


for l in os.listdir(stored_data_path):
    # print(l)
    gpx_file = open(l, 'r', encoding='utf-8')
    gpx = gpxpy.parse(gpx_file)
# # print(gpx)
#
# run_dataframe = get_run_data(run_dataframe, gpx, col_name)
# print(run_dataframe)
#
# ## Open file