from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
import os

tmp_df = pd.read_csv('sdi_not_fill.csv')

Country = tmp_df.pop('Country')
imp = IterativeImputer(max_iter=4, random_state=0, tol=0.001, min_value=tmp_df.min().min(), max_value=tmp_df.max().max())

imp.fit(tmp_df)
new = np.round(imp.transform(tmp_df), decimals=3)
new_goal = pd.DataFrame(new, columns=tmp_df.columns)
new_goal['Country'] = Country
new_goal.to_csv('imputersdi.csv', index=False, header=True)