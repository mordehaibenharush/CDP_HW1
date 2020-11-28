import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    #print("***************** trivial ********************")
    Xt = X.copy().transpose()
    result = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i + 1):
            itr_sum = 0
            for k in range(len(X[0])):
                itr_sum += X[i][k] * Xt[k][j]
            result[i][j] = itr_sum
            result[j][i] = itr_sum
    return result


@njit(parallel=True)
def matmul_transpose_numba(X):
    #print("***************** numba ********************")
    Xt = X.copy().transpose()
    result = np.zeros((len(X), len(X)))
    for i in prange(len(X)):
        for j in prange(i+1):
            itr_sum = 0
            for k in prange(len(X[0])):
                itr_sum += X[i][k] * Xt[k][j]
            result[i][j] = itr_sum
            result[j][i] = itr_sum
    return result





def matmul_transpose_gpu_nosym(X):
    #print("***************** gpu ********************")
    res_arr = np.zeros((len(X), len(X)))
    cdata = cuda.to_device(X)
    cresult = cuda.to_device(res_arr)
    matmul_kernel[1, 1024](cdata, cresult)
    result = cresult.copy_to_host()
    return result

@cuda.jit
def matmul_nosym_kernel(A, C):
    result_size = len(A)*len(A)
    batch_size = result_size//1024 + 1
    pos = cuda.grid(1)*batch_size
    limit = pos+batch_size
    while pos < limit and pos < result_size:
        itr_sum = 0
        i = pos//len(A)
        j = pos % len(A)
        for k in range(len(A[0])):
            itr_sum += A[i][k] * A.T[k][j]
        C[i][j] = itr_sum
        pos += 1


def matmul_transpose_gpu(X):
    #print("***************** gpu ********************")
    #rows = np.zeros(len(X))
    #rows[0] = 1
    #for i in range(1, len(X)):
    #    rows[i] = rows[i-1]+(i+1)
    res_arr = np.zeros((len(X), len(X)))
    cdata = cuda.to_device(X)
    cresult = cuda.to_device(res_arr)
    matmul_kernel[1, 1024](cdata, cresult)
    result = cresult.copy_to_host()
    return result


@cuda.jit
def matmul_kernel(A, C):
    batch_size = (len(A)//1024) + 1
    thread = cuda.grid(1)
    counter = 0
    i = thread
    while counter < batch_size:
        if i < len(A):
            for j in range(i+1):
                itr_sum = 0
                for k in range(len(A[0])):
                    itr_sum += A[i][k] * A.T[k][j]
                C[i][j] = itr_sum
                C[j][i] = itr_sum
        i += 1024
        counter += 1


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


def testGPUmatmul():
    X = np.random.randn(3000, 1)
    Xt = X.copy().transpose()
    res = matmul_transpose_gpu(X)
    comp = np.matmul(X, Xt)
    for i in range(len(X)):
        for j in range(len(X)):
            if res[i][j] != comp[i][j]:
                print("res[", i, "] [", j, "] = ", res[i][j])
                print("comp[", i, "] [", j, "] = ", comp[i][j])
    if np.allclose(res, comp, 1.e-5):
        print("pass")
    else:
        print("fail")

if __name__ == '__main__':
    #matmul_comparison()
    testGPUmatmul()