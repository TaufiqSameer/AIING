from numba import jit;
import numpy as np;
import time;

x = np.arange(100).reshape(10,10)
@jit(nopython=True)
def done(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i,i])
    return a + trace
start = time.time()
done(x)
end = time.time()
print("Time completed : ", (end-start))
start = time.time()
done(x)
end = time.time()
print("Time completed : ", (end-start))

