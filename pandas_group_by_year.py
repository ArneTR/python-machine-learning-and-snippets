import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("../data/sample_data_1.csv", index_col="date", parse_dates=True)

df.groupby([(df.index.year),(df.index.month)]).sales.sum() # Sales by year, month

# if the column is not directly readable as pd.DatetimeIndex we can create it
# ex: separate date and time columns
#
df_demo.mydate # date
df_demo.mytime # time

df_demo["mydatetime"] = df_demo.mydate + " " + df_demo.mytime
df_demo = df_demo.set_index(pd.DatetimeIndex(df_demo.mydatetime))

df_demo.groupby([(df_demo.index.year),(df_demo.index.month)]).size() # amounts by year, month
