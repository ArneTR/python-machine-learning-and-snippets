import pandas as pd
df = pd.DataFrame({"names": ["Arne", "Tobias", "Claudia", "Ilyana"], "age": [22,23,21,35]})

(df == "Tobias").any()
