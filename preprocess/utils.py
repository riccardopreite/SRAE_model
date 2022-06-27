import os
OUTPUT_COLUMN = ["GeoAreaCode","GeoAreaName"] 

def get_name(file_name):
    name = file_name.replace("Goal","")
    name = name.replace(".csv","")
    return name
    
def set_sdi_year(goal_df):
    cols = set(OUTPUT_COLUMN)
    max_year = int(goal_df["TimePeriod"].max())
    min_year = int(goal_df["TimePeriod"].min())
    for year in range(1990, 2020):
        cols.add(year)
    return cols

def set_year(goal_df):
    cols = set(OUTPUT_COLUMN)
    max_year = int(goal_df["TimePeriod"].max())
    min_year = int(goal_df["TimePeriod"].min())
    for year in range(min_year, max_year+1):
        cols.add(year)
    return cols
def return_path(dataset_indicator, series_indicator, dir_path):
    char = ['<','>',':','/','|','?','*']

    file_name = dataset_indicator.replace('.','_')
    description = series_indicator.replace(" ", "_")
    description = description.replace("/", "_")
    for ch in char:
        description = description.replace(ch, "_")
    c = "\\"
    description = description.replace(c, "_")
    path = os.path.join(dir_path, file_name+"_"+description+".csv")
    return path 
