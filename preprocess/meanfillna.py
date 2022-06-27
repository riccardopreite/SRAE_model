from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import os

path = "datasets"
output = "meanfilled_datasets"
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
            
            GeoAreaName = tmp_df.pop('GeoAreaName')
            GeoAreaCode = tmp_df.pop('GeoAreaCode')
            for col in tmp_df.columns:
                if tmp_df[col].isna().sum() == len(tmp_df[col]):
                    tmp_df.pop(col)
            

            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer = imputer.fit(tmp_df)
            
            new = imputer.transform(tmp_df)
            new_goal = pd.DataFrame(new, columns=tmp_df.columns)
            new_goal['GeoAreaName'] = GeoAreaName
            new_goal['GeoAreaCode'] = GeoAreaCode
            out_path = to_fill.replace(path, output)
            new_goal.to_csv(out_path, index=False, header=True)
        except Exception as e:
            print("(mean_file) error in",to_fill,e)
            i+=1
        

    else:
        i+=1

print("Skipped",i,"on:",len(indicators))
