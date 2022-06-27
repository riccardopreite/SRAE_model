import os, sys
import numpy as np
import pandas as pd
from utils import *
DATASET_OUTPUT_DIRECTORY = 'datasets'
MODEL_DIR_NAME = "SRAE_model"
SUSTAINABLE_PATH = os.path.join(os.getcwd(), "..",MODEL_DIR_NAME)
# SUSTAINABLE_PATH = os.path.join(os.getcwd(), "..","sustenaibility_machine_learning")
PREPROCESS_PATH = os.path.join(SUSTAINABLE_PATH, "preprocess")
FILLED_PATH = os.path.join(PREPROCESS_PATH, "filled_datasets")

DATASET_DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data','unsdg_dataset')
FILTERED_COLUMN = ["Goal","Indicator","SeriesCode","SeriesDescription","GeoAreaCode","GeoAreaName","TimePeriod","Value"]
OUTPUT_COLUMN = ["GeoAreaCode","GeoAreaName"] 

def setup_env(output_dir):
    global DATASET_DIRECTORY
    global DATASET_OUTPUT_DIRECTORY
    if output_dir != '':
        DATASET_OUTPUT_DIRECTORY = output_dir
    if not os.path.exists(DATASET_OUTPUT_DIRECTORY):
        os.makedirs(DATASET_OUTPUT_DIRECTORY)

def create_bidimensional_data(tasks, indicators, output_dir, data_type):
    user_df = {}
    country = set()
    matrix_year = []
    for task in tasks:
        task_dir = os.path.join(FILLED_PATH, str(task))
        df_paths = []
        for name in os.listdir(task_dir):
            indicator = name.replace(".csv","")
            if indicator in indicators:
                df_paths.append(os.path.join(task_dir, name))
        
        for df_path in df_paths:
            unique_code = os.path.split(df_path)[1].replace('.csv','')
            tmp_df = pd.read_csv(df_path)
            user_df[unique_code] = tmp_df
            country.update(list(tmp_df["GeoAreaName"]))
            matrix_year.append(tmp_df.columns)
            
    years = list(set.intersection(*map(set,matrix_year)))
    years.remove("GeoAreaName")
    years.remove("GeoAreaCode")
    if np.nan in country:
        country.remove(np.nan)
    merge_user_df(user_df, list(country), years, output_dir, data_type)

def merge_user_df(user_df, country, years, DATASET_OUTPUT_DIRECTORY, data_type):
    index_sdi = pd.read_csv(os.path.join(SUSTAINABLE_PATH,'data','index',data_type+'sdi.csv'))
    
    for year in years:
        year_df = {}
        country_index = 0
        for c in country:
            row = {}
            if index_sdi['Country'].eq(c).any():
                row["GeoAreaName"]  = c
                for code in user_df:
                    tmp_df = user_df[code]
                    if c not in list(tmp_df["GeoAreaName"]):
                        
                        row[code] = tmp_df[year].mean()
                    else:
                        index = list(tmp_df["GeoAreaName"]).index(c)
                        row[code] = tmp_df[year].iloc[index]
                        
                year_df[country_index] = row
                country_index+=1
        tmp = pd.DataFrame.from_dict(year_df, orient="index").sort_values("GeoAreaName")
        years_dir = os.path.join(DATASET_OUTPUT_DIRECTORY,"years")
        if not os.path.exists(years_dir):
            os.makedirs(years_dir)
        path = os.path.join(years_dir,str(year)+'.csv')
        tmp.to_csv(path,index=False)

def split_indicators(output_dir='',indicators=[], tasks=[], data_type="imputer"):
    global FILLED_PATH
    FILLED_PATH = os.path.join(PREPROCESS_PATH, data_type+"filled_datasets")
    setup_env(output_dir)
    create_bidimensional_data(tasks, indicators, output_dir, data_type)
