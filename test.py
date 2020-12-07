from matmul_functions import matmul_transpose_numba, matmul_transpose_gpu
from dist_functions import dist_cpu, dist_numba, dist_gpu
import numpy as np
from numba import njit, cuda
import random

@njit
def compare(res, result):
    diff = 0
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            diff += abs(res[i][j] - result[i][j])
    return diff

def Jit_tester(f):
    #Test1
    print("Running test1... lets see how smart are you :)")
    X = np.random.randn(4, 4) # Lets start with small numbers
    res = np.dot(X, np.transpose(X))
    result = f(X)
    if res.shape != result.shape:
        print("Size incorrect you idiot... And we just started!")
        return -1
    diff = compare(res, result)
    if diff > 0.001:
        print("Wrong calculation... No way in hell you are Faculta Nehshevet")
        return -1

    #print("Test 1 failed... You should quit the course (And your degree)") # Just kidding you past it... remove this line
    print("Test 1 passed")

    #Test2
    print("Test2 Fun Fun Fun")
    X = np.random.randn(5, 3)# Lets start with small numbers
    res = np.dot(X, np.transpose(X))
    result = f(X)
    if res.shape != result.shape:
        print("Size incorrect... How many losers does it take to switch a lamp?")
        return -1
    diff = compare(res, result)
    if diff > 0.001:
        print("Wrong calculation... Corona Virus does less damage than you")
        return -1

    print("Test 2 passed")

    print("Test3 running now: (I really hope you fail and get to see my prints)")
    for i in range(10):
        dim1 = random.randint(1, 1000)
        dim2 = random.randint(1, 1000)
        # Test2
        X = np.random.randn(dim1, dim2)  # Lets start with small numbers
        res = np.dot(X, np.transpose(X))
        result = f(X)
        if res.shape != result.shape:
            print("Size incorrect... Something here stinks. Take a shower smelly!")
            return -1
        diff = compare(res, result)
        if diff > 0.001:
            print("Wrong calculation... But don't worry! You can still be a shepherd for ISIS with your skills!")
            print("There are people in the bible who programed better than you...")
            return -1

    very_scret_cber = random.randint(1, 1000)
    if very_scret_cber % 3 != 0 and f == matmul_transpose_numba:
        # Randomly prints a failure... Sorry I couldn't resist it :) Just remove this if you want, you passed
        print("Test3 failed... fix your code and try not to be an imbecile next time")
    else:
        print("Test3 passed! You passed all test!")
    return very_scret_cber % 3

def test3():
    """
        Found a problem in the test? Did you Like it and want to tell us? please don't waste a second and inform
        us on our hot line via email:
        cyberWeDontGiveAShit@gmail.com (Its a real email)
        or on call of duty vip_CsTech (a hero user)
    """
    types = [matmul_transpose_numba, matmul_transpose_gpu]
    print("Now testing matmul_transpose_numba:")
    if Jit_tester(types[0]) < 0:
        print("Failed numba... such a bronze... just like Yaron."
              "I cant handle this shit anymore I am now stopping to run")
        # If you failed numba you should fix it before you continue to cude
        return

    print("passed all tests of matmul_transpose_numba\n")
    print("Now testing cude:")
    res = Jit_tester(types[1])
    if res < 0:
        print("Failed cuda... You should really go and see the lectures. Lets just say you will probably need the "
              "Factor....")
    else:
        print("Passed all tests")


def test2():
    types = [dist_cpu, dist_numba, dist_gpu] 
    names = {}; names[dist_cpu] = "dist_cpu"; names[dist_numba] = "dist_numba"; names[dist_gpu] = "dist_gpu"

    p = [1, 2, 3]
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))
    check = (A - B).reshape((A.shape[0] * A.shape[1],))
    check = check.astype('float64')
    for i in p:
        print("Test1 with p = ", i)
        good_res = np.linalg.norm(check, i)
        for t in types:
            my_res = t(A, B, i)
            if abs(good_res-my_res) > 1:
                print("Failed test1!!! you have a problem in ", names[t],
                      "\nYou ans = ", my_res, " Good ans = ", good_res,
                      "\nYou should start learning how to clean toilets because you are a shitty programmer")
                return -1

    print("Passed Test1")
    for x in range(5):
        A = np.random.randint(0, 256, (1000, 1000))
        B = np.random.randint(0, 256, (1000, 1000))
        check = (A - B).reshape((A.shape[0] * A.shape[1],))
        check = check.astype('float64')
        p = [1, 2]
        for i in p:
            print("Test2 with p = ", i)
            good_res = np.linalg.norm(check, i)
            # good_res = np.sum(np.abs(check) ** i) ** (1. / i)
            for t in types:
                my_res = t(A, B, i)
                if abs(good_res - my_res) > 1:
                    print("Failed test2!!! you have a problem in ", names[t],
                          "\nYou ans = ", my_res, " Good ans = ", good_res,
                          "\nYou should really start selling fertilizer because you are amazing at creating Bullshit")
                    return -1

    print("Passed Test2")
    return 0


#You should be selling fertilizer because you are amazing at creating Bullshit


if __name__ == '__main__':
    print("Hello useless earth creatures who did not write tests on their own.\n This test includes test for part 2"
          "and part 3 of HW1. \nThis test does *not* test your running time, but only the correctness of your "
          "calculations. \n If you fail one of the tests, go to the code in test and check why you failed. \n"
          "*** Common test failures are at part3: test1 (line 29 in this code), test3 (line 64) *** "
          "\nBefore we start lets have a fun fact- Did you know Adolf Hitler had only one testicle?")

    print("\n\nTesting part 2:")
    if test2() == 0:
        print("\n\nPassed tests on part 2\n")
        print("Now before we continue, did you know that driving a roller coaster might help people get rid of kidney stone? "
              "\nIn a little research I had I found the in the long term, drinking high amounts of rat poison, "
              "driving of a cliff or poking your finger in an electrical outlet might solve the kidney problem to.")
        print("\n\n Now lets run tests on part 3 \n\n")
        test3()
        print("\n\nPassed tests on part 3\n")
        print("Great! you passed all tests. \nFor grand finale, did you know that pigeons are monogamous creatures "
              "and stay loyal to their life partner until he/she dies? \nSome might say that Atudaim are loyal"
              "to their girlfriends to, but so far it has been proven only in an empty way")
