import os
import pandas as pd
output="no_all_nan"
os.makedirs(output, exist_ok=True)
path = "datasets"
tasks = os.listdir(path)
indicators = []
for task in tasks:
    os.makedirs(os.path.join(output, task), exist_ok=True)
    paths = [os.path.join(path, task, ind) for ind in os.listdir(os.path.join(path, task))]
    indicators.extend(paths)

for to_fill in indicators:
    tmp_df = pd.read_csv(to_fill)
    GeoAreaName = tmp_df.pop('GeoAreaName')
    GeoAreaCode = tmp_df.pop('GeoAreaCode')
    tmp_df.dropna(axis=0, how="all", inplace=True)
    tmp_df.dropna(axis=1, how="all", inplace=True)
    tmp_df['GeoAreaName'] = GeoAreaName
    tmp_df['GeoAreaCode'] = GeoAreaCode
    out_path = to_fill.replace(path, output)
    tmp_df.to_csv(out_path, index=False, header=True)