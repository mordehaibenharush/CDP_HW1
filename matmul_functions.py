import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    #print("***************** trivial ********************")
    result = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            for k in range(len(X.T)):
                #print("[", i, "] [", j, "]")
                result[i][j] = X[i][k] * X.T[k][j]
    return result

@njit(parallel=True)
def matmul_transpose_numba(X):
    #print("***************** numba ********************")
    result = np.zeros((len(X), len(X)))
    for i in prange(len(X)):
        for j in prange(len(X)):
            for k in prange(len(X[0])):
                #print("[", i, "] [", j, "]")
                result[i][j] = X[i][k] * X.T[k][j]
    return result


def matmul_transpose_gpu(X):
    #print("***************** gpu ********************")
    res_arr = np.zeros((len(X), len(X)))
    cdata = cuda.to_device(X)
    cresult = cuda.to_device(res_arr)
    matmul_kernel[1, 1024](cdata, cresult)
    result = cresult.copy_to_host()


@cuda.jit
def matmul_kernel(A, C):
    result_size = len(A)*len(A)
    batch_size = result_size//1024 + 1
    pos = cuda.grid(1)*batch_size
    limit = pos+batch_size
    while pos < limit:
        itr_sum = 0
        i = pos//len(A)
        j = pos % len(A)
        for k in range(len(A[0])):
            itr_sum += A[i][k] * A.T[k][j]
        C[i][j] = itr_sum
        pos += 1


#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X,Xt)).repeat(3, 100))

    #print('Python:', timer(matmul_transpose_trivial, 1)) #we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()
