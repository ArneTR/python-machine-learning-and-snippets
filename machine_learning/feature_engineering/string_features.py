
# String features can be parsed to extract common data 

# Phone numbers: '(999) 555-0123' 
#   - Extract area codes in separate row

# Street addresses: '8241 Kaggle Ln., Goose City, NV'
#   Extract state to separate row

# Internet addresses: 'http://www.kaggle.com
    # extract URL-Scheme to separate row
    # extract if www. present or not
    # extract if path after / present or not

# Product codes: '0 36000 29145 2'
    # if specification known, extract recurring parts.
    # or extract sum
    # or extract if > some threshold value     

# Dates and times: 'Mon Sep 30 07:06:05 2013'
    # extract year
    # extract timezone


#     ....

# example: Splitting string and expanding directly to new columns
import pandas as pd
customer = pd.read_csv("~/Code/data/customer.csv")
customer
customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)
customer


