import numpy as np
from numpy.random import laplace as lap
from numpy.random import normal as nor
from numpy.random import uniform as uni
from numpy.random import zipf as zipf
from scipy.stats import truncnorm
np.set_printoptions(suppress=True)


def lapLM(data, eps, m):
    lapv = lap(scale=(m/eps), size=len(data))
    loc_data = data + lapv
    loc_mean = np.mean(loc_data)
    return loc_mean

def fullLM_err(eps, m, n):
    err = (2*m**2)/(n*eps**2)
    return err

def experiment_n(start, end, step, trials, eps, c, m, sigma):

    upper_bound = end + step
    error = np.zeros(shape=(int((end - start)/step) + 1, 4, trials))

    indices = np.zeros(shape=((upper_bound - start)/step, 1))

    for n in np.arange(start, upper_bound, step):
        print n
        indices[(n - start)/step] = n
        row = int((n - start)/step)
        n0 = int(n*c)
        n1 = int(n*(1-c))

        for i in range(trials):
            l = uni(high=m)
            data = nor(loc=l, scale=m/2.0, size=n)
            tcm_data = data[:n0]
            loc_data = data[n0:]
            mean = np.mean(data)
            full_loc_err1 = (mean - lapLM(data, eps, m))**2
            error[row][0][i] += full_loc_err1

            l = uni(high=m)
            data = nor(loc=l, scale=m/6.0, size=n)
            tcm_data = data[:n0]
            loc_data = data[n0:]
            mean = np.mean(data)
            full_loc_err2 = (mean - lapLM(data, eps, m))**2
            error[row][1][i] += full_loc_err2

            l = uni(high=m)
            data = nor(loc=l, scale=m/10.0, size=n)
            tcm_data = data[:n0]
            loc_data = data[n0:]
            mean = np.mean(data)
            full_loc_err3 = (mean - lapLM(data, eps, m))**2
            error[row][2][i] += full_loc_err3

            full_loc_ana = fullLM_err(eps, m, n)
            error[row][3][i] += full_loc_ana


    error = np.mean(error, axis=2)
    error = np.sqrt(error)
    error = np.hstack((indices, error))
    # error = np.log10(error)

    print error
    header = "n, k = 2, k = 6, k = 10, analyt"
    np.savetxt("fullLM_plot.csv", error, header=header, delimiter=",")

eps = 0.01
c = 0.01
m = 1.0
sigma = m/6.0
experiment_n(10000, 100000, 10000, 2000, eps, c, m, sigma)



