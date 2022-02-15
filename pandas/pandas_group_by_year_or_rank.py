import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import numpy as np
import seaborn as sns

df = pd.read_csv("~/Code/data/sample_data_1.csv", index_col="date", parse_dates=True)

df.groupby([(df.index.year),(df.index.month)]).sales.sum() # Sales by year, month

# if the column is not directly readable as pd.DatetimeIndex we can create it
# ex: separate date and time columns
#
df_demo.mydate # date
df_demo.mytime # time

df_demo["mydatetime"] = df_demo.mydate + " " + df_demo.mytime
df_demo = df_demo.set_index(pd.DatetimeIndex(df_demo.mydatetime))

df_demo.groupby([(df_demo.index.year),(df_demo.index.month)]).size() # amounts by year, month


## You do not always have to use aggregate functions. You can also use in-group functions like .rank() 
## This keeps the full size of the dataframe, but adds a new column, rank

df = pd.read_csv("~/Code/data/autos.csv")
df = df[["make", "fuel_type", "price", "city_mpg"]]

df["mpg_rank_in_group"] = df.groupby("make").city_mpg.rank(method="dense")
df # you can see that ranks are per make and not global

df.city_mpg.rank() # this would be a global rank