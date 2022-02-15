import pandas as pd; pd.set_option('mode.chained_assignment','raise');
df = pd.DataFrame({"name": ["Arne", "Tobias", "Claudia", "Ilyana"], "age": [22,23,21,35]})

df.loc[2].name = "Neu" # will not work
print(df)
df.loc[2]["name"] = "Neu" # will not work
print(df)
df.loc[2, "name"] = "Neu" # will work
print(df)
