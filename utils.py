import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


def random_weights(sizes):
    return [xavier_initialization((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 2)]
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
    return [np.zeros(list[i]) for i in range(len(list) - 1)]
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
