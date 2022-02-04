import timeit
import random
import numpy as np
import base64

v=list(range(1,10001))

def test_for_loop(n):
    random.seed(2019)
    for _ in range(n):
        np.sin(random.randint(1,10000))
def test_bs(n):
    random.seed(2019)
    for _ in range(n):
        base64.standard_b64encode(b"asdj84htgrneuskr89ewugir")


x= 30
print("np.sin:", timeit.timeit('test_for_loop(1000)',setup='from __main__ import test_for_loop ' ,number=1))


print("base64:",timeit.timeit('test_bs(1000)',setup='from __main__ import test_bs', number=1))

print(f"base64: {timeit.timeit('test_bs(1000)',setup='from __main__ import test_bs', number=1)}")