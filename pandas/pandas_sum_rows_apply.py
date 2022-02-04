import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import numpy as np
import seaborn as sns

df = pd.read_csv("../data/sample_data_1.csv", index_col="date", parse_dates=True)

df.apply(lambda x: x.max(), axis="columns") # geht the highest value in all columns per row

df.apply(lambda x: x.idxmax(), axis="columns") # geht the column name of the highest value in all columns per row

# Beware as always of NA values!