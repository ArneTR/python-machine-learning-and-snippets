import pandas as pd
df = pd.DataFrame({"names": ["Arne", "Tobias", "Claudia", "Ilyana"], "age": [22,23,21,35]})

df.names.str.len() # access len through .str

df.names.str[0:2] # Substring

# same goes if you have a datetime column. Then access with df.column.dt.month .... .dt is the accessor
