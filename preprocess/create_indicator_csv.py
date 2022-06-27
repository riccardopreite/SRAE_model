from multiprocessing import Pool, cpu_count
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *

'''
    This file is called once to create indicator dataset from Goal csv.
    When all indicators are ready `fillna.py` is executed to fill NaN value
'''
MODEL_DIR_NAME = "SRAE_model"
# SUSTAINABLE_PATH = os.path.join(os.getcwd(), "..","sustenaibility_machine_learning")
SUSTAINABLE_PATH = os.path.join(os.getcwd(), "..",MODEL_DIR_NAME)
DATASET_OUTPUT_DIRECTORY = os.path.join("preprocess", 'datasets')

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
def separate_dataset(data_dict):
    goal_df = data_dict["goal_df"]
    task_name = data_dict["task_name"]
    series_valid = data_dict["series_valid"]
    DATASET_OUTPUT_DIRECTORY = data_dict["DATASET_OUTPUT_DIRECTORY"]
    dir_path = os.path.join(DATASET_OUTPUT_DIRECTORY, task_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    country_indicator = ""
    cols = set(OUTPUT_COLUMN)
    
    for choosen in series_valid:
        series_code = choosen
        sub_df = goal_df[goal_df["SeriesCode"] == choosen]

        new_df = []
        if not sub_df.empty:
            cols = set_sdi_year(sub_df)
            sub_df_cols = set_year(sub_df)
            if len(sub_df_cols) > (len(cols)*0.6):
                new_row = {key:np.nan for key in cols}

                for _, row in (pbar := tqdm(sub_df.iterrows())):
                    pbar.set_description("Creating sub dataset %s code: %s" % (str(row["Indicator"]), str(row["SeriesCode"]))) 
                    if country_indicator == "":
                        country_indicator = row["GeoAreaCode"]
                        new_row["GeoAreaCode"] = int(country_indicator)
                        new_row["GeoAreaName"] = row["GeoAreaName"]

                    elif country_indicator != row["GeoAreaCode"]:
                        for col in cols:
                            if col not in new_row.keys():
                                new_row[col] = np.nan

                        new_df.append(new_row)
                        new_row = {}
                        new_row = {key:np.nan for key in cols}

                        country_indicator = row["GeoAreaCode"]
                        new_row["GeoAreaCode"] = int(country_indicator)
                        new_row["GeoAreaName"] = row["GeoAreaName"]
                    
                    new_row[row["TimePeriod"]] = row["Value"]
                path = os.path.join(dir_path, series_code+".csv")
                columns = list(cols)
                df = pd.DataFrame(new_df, columns=columns)
                df.to_csv(path, index=False, header=True)
            else:
                print("Skipped",task_name,"since his cols are:",sub_df_cols)
    return task_name

def split_big_csv_file_with_taked_tasks(goals_to_open, choosen_indicators):
    to_open = [task for task in os.listdir(DATASET_DIRECTORY) if task in goals_to_open]

    with Pool(processes=cpu_count()) as p:
        splitted_data = []
        for task in to_open:
            task_path = os.path.join(DATASET_DIRECTORY, task)
            print("opening dataset:",task)
            goal_df = pd.read_csv(task_path, low_memory=False)
            name = get_name(task)
            goal_df = goal_df[FILTERED_COLUMN]
            splitted_data.append({"goal_df":goal_df,"task_name":name,'series_valid':choosen_indicators, 'DATASET_OUTPUT_DIRECTORY':'preprocess/datasets'})
        
        with tqdm(total=len(to_open)) as pbar:
            for i, _ in enumerate(p.map(separate_dataset, splitted_data)):
                pbar.set_description("Created dataset for %s" % str(_))
                pbar.update()
            
'''
@param output_dir: path to user save splitted data
'''
def split_unsdg_dataset(output_dir=''):
    
    setup_env(output_dir)
    
    task = [item.replace('.csv', '') for item in os.listdir('data/unsdg_dataset')]
    task = [item.replace('Goal', '') for item in task]
    goals = ['Goal'+str(task)+'.csv' for task in task]
    inds = []
    for goal in goals:
        task_path = os.path.join(DATASET_DIRECTORY, goal)
        goal_df = pd.read_csv(task_path, low_memory=False)
        inds.extend(np.unique(goal_df["SeriesCode"]).tolist())
    print("updated indicators")
    split_big_csv_file_with_taked_tasks(goals, inds)
    
    
if __name__ == "__main__":
    split_unsdg_dataset()