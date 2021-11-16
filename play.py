from dsmlbc6.hi import hello
hello()

from dsmlbc6.hi import hello4
hello4()

import seaborn as sns
df = sns.load_dataset("tips")

from dsmlbc6.eda.eda import check_df
check_df(df)

from dsmlbc6.eda.eda import check_df, grab_col_names
cat_cols, num_cols, cat_but_car = grab_col_names(df)

from dsmlbc6.eda.eda import *

import dsmlbc6
help(dsmlbc6)

help(dsmlbc6.eda)

help(dsmlbc6.eda.eda)

import dsmlbc6.eda.eda
help(dsmlbc6.eda.eda)

import pandas as pd
help(pd)
help(pd.tseries)

help(dsmlbc6.eda.eda.check_df)

