from pdb import set_trace
def binary_search_buggy(v,x):
    set_trace()
    start,end = 0,len(v)
    while start<end:
        mid = (start+end) // 2
        if v[mid]>=x:
            end = mid
        else:
            start = mid+1
    return start

v=[1,2,5,7,10]
print(binary_search_buggy(v,11))