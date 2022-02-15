
# In python there are often no .toX methods.
# Instead you use list comprehension
lst = ["1900", "2000", "2021"]

[float(i) for i in lst]

# or
#
[int(i) - 10 for i in lst]
