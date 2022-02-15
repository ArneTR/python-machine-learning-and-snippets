# Familiar imports
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("~/Code/data/autos.csv")

features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]


# Step 1: standardize! 
# 
# Otherwise PCA will not work properly
######################################
X = df.copy()
y = X.pop('price')
X = X.loc[:, features]
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)


# Step 2: Create principal components
#
#####################################
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca # These are all the new values for the matrix with the rotated axed.
# In the following analysis we will now determine which of the Principal-Components we will keep (dimensionality reduction .. which is the main use-case) 

# Step 3: Analysis
# 
# we examine the loading matrix to name the features. The features should be named according to the features the load on in the loading matrix
# We examine the explained variance ratio to examine what to keep
##################################################################

pd.Series(pca.explained_variance_ratio_, index=component_names) # Explained variance is backfilled to the original pca object. Very strange
## => Here we see, that PC1 and PC2 and PC3 have relevant variances. So we might either drop PC2-4 if we really need computational speed.
## Or we at least drop PC4


pd.DataFrame(pca.components_.T, columns=component_names, index=features) # Important: components are in shape of: (rows x columns) = (PCAs x features) 
## => The loadings tell us which feature is fusioned in the new component. 
##    PC1 consists of a mix of all. Specifically in the contrast of decreasing miles per gallon and increasing and engine_size,horsepower, curb_weight and vice-versa
##      a good name would be luxury cars vs economy/mini cars
##    PC2 is the fusion of highway_mpg and engine_size. Note that the high variance value does not mean, that these are BIG engines, or big values for highway_mpg
##      rather this means that the variance of this variable is fully explained by the component  and they have the same magnitude when they occur. So that a high value
##      value for highway_mpg always takes place with a high value for engine_size, and little values for highway_mpg with little values for engine_size.
##      So a good name would be "Efficiency", as it reflects engines with increasing size can do more miles per gallon
##    PC3 is the contrast of curb_weight and horsepower. Meaning that little values for horsepower combine with high values for curb_weight and vice-versa. This works for 
##      "Kombis" vs. "sports-cars". As "Kombis" have high weight, but little horsepower and sports cars the opposite.
##    PC4 is again a mix of all and therefore exactly the same as PC1. 
##
##    Conclusion: Now we now, that PC4 is reduandant AND does not have high variance. So we can safely drop it. PC2 and PC3 however seem to single out some cars and may
##      be useful for classification


# Step 4: Dropping columns
#
# This always happens in the PCA matrix. In the example we drop PC4
# We also rename the column according to our analysis. This step is optional however
###################################################################

X_pca = X_pca.drop("PC4", axis="columns")
X_pca = X_pca.rename(
    {
        "PC1":"Luxury/Economy",
        "PC2":"Efficiency",
        "PC3":"kombis-vs-sportscars" 
    }, axis="columns"
)
X_pca # Note that the matrix has always the same rows, but less or same features