# To make an intersection is easiest in python by using a set

# Often we do not only have one element and want to find if it is in a another group_
"x" in ["a","b", "c", "x"]

# we rather have this:
["a","b"] in ["a","b","c", "d", "x"]

# here we use sets:
set(["a", "b", "c"]) & set(["a", "b", "c", "d"])

# this tells us which SINGLE elements from the first set also occur in the other one. 
# By definition it cannot tell the amount, if there ar multiple occurences, cause the set() command removes duplicates.
