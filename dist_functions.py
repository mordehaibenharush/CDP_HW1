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
    res = res ** (1/float(p))
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
            #calculate difference for the i,j element, power by p and sum up
            res += abs(A[i][j] - B[i][j])**p
    
    res = res ** (1/float(p))
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
    #copy memory arrays to GPU kernel
    C = cuda.to_device(C)
    A = cuda.to_device(A)
    B = cuda.to_device(B)
    #run kernel
    dist_kernel[1000, 1000](A, B, p, C)
    #copy back
    C = C.copy_to_host()
    #calculate root
    res = C[0] ** (1/float(p))

@cuda.jit
def dist_kernel(A, B, p, C):
    #get indices
    i = cuda.threadIdx.x
    j = cuda.blockIdx.x
    #if thread is not out-of-bound calculate differences for indices j,i and store in global memory
    if i < A.shape[0] and j < A.shape[1]:
        res = abs(A[j][i] - B[j][i])**p 
        cuda.atomic.add(C, 0, res) 
   
#this is the comparison function - keep it as it is.
def dist_comparison():
    A = np.random.randint(0,256,(1000, 1000))
    B = np.random.randint(0,256,(1000, 1000))
    p = [1, 2]

    def timer(f, q):
        return min(timeit.Timer(lambda: f(A, B, q)).repeat(3, 20))
    
    
    
    
    for power in p:
        print('p=' + str(power))
        print('     [*] CPU:', timer(dist_cpu,power))
        print('     [*] Numba:', timer(dist_numba,power))
        print('     [*] CUDA:', timer(dist_gpu, power))

if __name__ == '__main__':
    dist_comparison()
