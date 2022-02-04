print("Sorted vector test!")

x = [1,2,3,4,4,5,6,7,9,10]
target = 0
pos_goal = None

for pos in range(0, len(x)):
    if x[pos] >= target:
        pos_goal = pos
        break

if(pos_goal == None):
    print("Could not found any matching element")
else:
    print("Found ", x[pos_goal], "at position", pos_goal, "which is just >= than ", target)


