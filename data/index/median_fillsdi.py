from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os

tmp_df = pd.read_csv('sdi.csv')
Country = tmp_df.pop('Country')
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(tmp_df)

new = imputer.transform(tmp_df)
new_goal = pd.DataFrame(new, columns=tmp_df.columns)
new_goal['Country'] = Country
new_goal.to_csv('mediansdi.csv', index=False, header=True)