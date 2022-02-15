#Install the below libaries before importing
import pandas as pd
from pandas_profiling import ProfileReport

#EDA using pandas-profiling
profile = ProfileReport(pd.read_csv('./../../data/SpotifyDataset160k.csv'), explorative=True)

#Saving results to a HTML file
profile.to_file("output.html")
