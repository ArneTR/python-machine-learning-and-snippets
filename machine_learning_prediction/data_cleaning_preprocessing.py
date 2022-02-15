# Familiar imports

# Please see jupyter notebook for generalt methods like imputing, NA-checks, outliers etc.

# Here we only use machine learning specific methods


# Prebalancing - Only needed for Tree methods
# This step is optional and should only be considered, if the target has many categories that
# are outbalanced. Say 200000x A + 1000x B + 1000x C.
# The tree might focus too much on A and neglect the other cases. This however may be bad for
# future classification if C gets more important in production
#
# Therefore it must be determined how often the classes should be present as target variable
# Important: Note that purging too many rows may make the tree bad cause of low sample  size


# here we downsample all rows where Purchased == 0 and merge with Purchased == 1 rows. THis code asssumes a 1:2 ratio and must be adapated if ratio is different
data = data[data.Purchased == 0][::2].append(data[data.Purchased == 1])
 



## Colinearity checks - Still moved from R: must be adapted to python
#If the data was normal, we continue with the Person correlation coefficient
cor.test(X,Y,method="spearman")
