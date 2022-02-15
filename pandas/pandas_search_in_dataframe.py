import pandas as pd; pd.set_option('mode.chained_assignment','raise');
df = pd.DataFrame({"names": ["Arne", "Tobias", "Claudia", "Ilyana"], "age": [22,23,21,35]})

(df == "Tobias").any()
