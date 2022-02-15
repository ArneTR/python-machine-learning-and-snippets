#import pandas as pd
import vaex

df_file = '/Users/light/Code/data/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography.csv.hdf5' # 6 GB file

print("Load OK")




#df = pd.read_csv(df_file) # Pandas load time is ~8 Min and uses the full 6GB for the file

## In case conversion is necessary do this:
#df = vaex.from_csv(df_file, , convert=True, chunk_size=5_000_000)


df = vaex.open(df_file)


print(df.mean())