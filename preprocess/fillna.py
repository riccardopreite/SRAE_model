from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
import os

path = "datasets"
path = "no_all_nan"
output = "imputerfilled_datasets"
os.makedirs(output, exist_ok=True)
tasks = os.listdir(path)
indicators = []
for task in tasks:
    os.makedirs(os.path.join(output, task), exist_ok=True)
    paths = [os.path.join(path, task, ind) for ind in os.listdir(os.path.join(path, task))]
    indicators.extend(paths)

i = 0

for to_fill in indicators:

    tmp_df = pd.read_csv(to_fill)
    nan_occ = tmp_df.isna().sum().sum()
    threshold = (tmp_df.shape[0] * tmp_df.shape[1])*0.9
    if len(tmp_df) > 7 and nan_occ < threshold:
        
        try:
            tmp_df = tmp_df.replace({'N': np.nan}, regex=True)
            tmp_df = tmp_df.replace({'<': ''}, regex=True)
            tmp_df = tmp_df.replace({'>': ''}, regex=True)
            GeoAreaName = tmp_df.pop('GeoAreaName')
            GeoAreaCode = tmp_df.pop('GeoAreaCode')
            tmp_df = tmp_df.astype(float)
            tmp_df.dropna(axis=0, how="all", inplace=True)
            tmp_df.dropna(axis=1, how="all", inplace=True)
            imp = IterativeImputer(max_iter=4, random_state=0, tol=0.1, min_value=tmp_df.min().min(), max_value=tmp_df.max().max())
            
            imp.fit(tmp_df)
            new = np.round(imp.transform(tmp_df), decimals=3)
            new_goal = pd.DataFrame(new, columns=tmp_df.columns)
            new_goal['GeoAreaName'] = GeoAreaName
            new_goal['GeoAreaCode'] = GeoAreaCode
            out_path = to_fill.replace(path, output)
            new_goal.to_csv(out_path, index=False, header=True)
        except Exception as e:
            print("error in",to_fill,e)
            i+=1
        

    else:
        i+=1

print("Skipped",i,"on:",len(indicators))
