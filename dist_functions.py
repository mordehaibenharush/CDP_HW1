import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def dist_cpu(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    res = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res += abs(A[i][j] - B[i][j])**p
    res = res ** (1/p)
    return res



@njit(parallel=True)
def dist_numba(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    res = 0.0
    for i in prange(A.shape[0]):
        for j in prange(A.shape[1]):
            res += abs(A[i][j] - B[i][j])**p
    res = res ** (1/p)
    return res
	

def dist_gpu(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    C = np.arange(1)
    C[0] = 0.0
    res = 0.0
    C = cuda.to_device(C)
    A = cuda.to_device(A)
    B = cuda.to_device(B)
    dist_kernel[1000, 1000](A, B, p, C)
    C = C.copy_to_host()
    res = C[0] ** (1/p)
    return res

@cuda.jit
def dist_kernel(A, B, p, C):
    i = cuda.threadIdx.x
    j = cuda.blockIdx.x
    if i < A.shape[0] and j < A.shape[1]:
        res = abs(A[i][j] - B[i][j])**p
        cuda.atomic.add(C, 0, res)  # need for every thread to sum near threads
    
   
#this is the comparison function - keep it as it is.
def dist_comparison():
    A = np.random.randint(0,256,(1000, 1000))
    B = np.random.randint(0,256,(1000, 1000))
#    p = [1, 2]
    p = range(76,80)

    def timer(f, q):
        return min(timeit.Timer(lambda: f(A, B, q)).repeat(3, 20))

    for power in p:
        print('p=' + str(power))
        #print('     [*] CPU:', timer(dist_cpu,power))
        print('     [*] Numba:', timer(dist_numba,power))
        print('     [*] CUDA:', timer(dist_gpu, power))

if __name__ == '__main__':
    dist_comparison()
