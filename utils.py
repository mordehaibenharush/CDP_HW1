import numpy as np
import math as math


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


def random_weights(sizes):
    return [xavier_initialization(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    # weights_list = []
    # for i in range(len(sizes)-2):
    #   weights_list.append(xavier_initialization(sizes[i], sizes[i+1]))
    #return weights_list


def zeros_weights(sizes):
    return [np.zeros((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 2)]
    # zeros_list = []
    # for i in range(len(sizes) - 2):
    #   zeros_list.append(np.zeros((sizes[i], sizes[i + 1])))
    # return zeros_list


def zeros_biases(list):
    return [np.zeros(list[i]) for i in range(len(list))]
    # biases_list = []
    # for i in range(len(list)-1):
    #    biases_list.append(np.zeros(list[i]))
    #return biases_list


def create_batches(data, labels, batch_size):
    return [(data[i:i+batch_size], labels[i:i+batch_size]) for i in range(0, len(data), batch_size)]


def add_elementwise(list1, list2):
    return [list1[i]+list2[i] for i in range(len(list1))]


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))


def testSigmoid():
    x = np.arange(3)
    y = sigmoid(x)
    for i in range(len(x)):
        if 1/(1+math.exp(-x[i])) != y[i]:
            print("fail ", i)
        else:
            print("pass ", i)


def testZero():
    sizes = [2, 4, 2, 3, 3, 2]
    arrays = zeros_biases(sizes)
    for a in arrays:
        print(a, a.shape)


def testBatch():
    data = [i for i in range(10)]
    labels = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    batches = create_batches(data, labels, 3)
    for a in batches:
        print(a, len(a[0]))


def testAdd():
    l1 = [i for i in range(10)]
    l2 = [3 for i in range(10)]
    res = add_elementwise(l1, l2)
    print(res)


if __name__ == "__main__":
    testAdd()