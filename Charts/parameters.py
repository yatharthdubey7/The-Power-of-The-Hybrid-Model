import numpy as np
from numpy.random import laplace as lap
from numpy.random import normal as nor
from numpy.random import uniform as uni
from numpy.random import zipf as zipf
np.set_printoptions(suppress=True)

def truncate(data, low, high):
    data = np.minimum(np.full(data.size, high), data)
    data = np.maximum(np.full(data.size, low), data)
    return data

def tcm(data, eps, m):
    tcm_mean = np.mean(data) + lap(scale=m/(len(data)*eps))
    # tcm_mean = truncate(tcm_mean, 0.0, 1.0)
    return tcm_mean

def lapLM(data, eps, m):
    lapv = lap(scale=(m/eps), size=len(data))
    loc_data = data + lapv
    loc_mean = np.mean(loc_data)
    # loc_mean = truncate(loc_mean, 0.0, 1.0)
    return loc_mean

def hybrid(tcm_data, loc_data, eps, m, c, w):
    tcm_mean = tcm(tcm_data, eps, m)
    loc_mean = lapLM(loc_data, eps, m)
    hmean = w*tcm_mean + (1-w)*loc_mean
    return hmean

def hybrid_online(tcm_data, loc_data, n, eps, m, c):
    tcm_mean = tcm(tcm_data, c*n, eps, m)
    run_mean = tcm_mean
    ptrue = np.exp(eps)/(np.exp(eps) + 1)
    probs = uni(size=(1-c)*n)
    for i in range(len(loc_data)):
        if loc_data[i] < run_mean:
            if probs[i] < ptrue:
                run_mean += m/(c*n + i)
            else:
                run_mean -= m/(c*n + i)
        if loc_data[i] > run_mean:
            if probs[i] < ptrue:
                run_mean -= m/(c*n + i)
            else:
                run_mean += m/(c*n + i)
        # run_mean = truncate(run_mean, 0.0, 1.0)
    return run_mean

def experiment_n(start, end, step, trials, eps, c, m, sigma):

    upper_bound = end + step
    error = np.zeros(shape=(int((end - start)/step) + 1, 3, trials))

    indices = np.zeros(shape=((upper_bound - start)/step, 1))

    for n in np.arange(start, upper_bound, step):
        print n
        c = np.log(n)/n
        indices[(n - start)/step] = n
        row = int((n - start)/step)
        n0 = int(n*c)
        n1 = int(n*(1-c))

        for i in range(trials):
            l = uni(high=m)
            data = nor(loc=l, scale=sigma, size=n)
            data = m*(data/float(max(data)))
            # data = truncate(data, 0.0, 1.0)
            tcm_data = data[:n0]
            loc_data = data[n0:]
            mean = np.mean(data)

            only_tcm_err = (mean - tcm(tcm_data, eps, m))**2
            error[row][0][i] = only_tcm_err

            full_loc_err = (mean - lapLM(data, eps, m))**2
            error[row][1][i] += full_loc_err

            top_coeff = (c**2)*n
            top_par_first = (eps**2)*(sigma**2)
            top_par_sec = 2*(m**2)
            bot_first = c*n*(eps**2)*(sigma**2)
            bot_coeff = 2*(m**2)
            bot_par = ((c**2)*n) + 1 - c
            top = top_coeff*(top_par_first + top_par_sec)
            bot = bot_first + bot_coeff*bot_par
            w1 = top/bot

            hybrid_err2 = (mean - hybrid(tcm_data, loc_data, eps, m, c, w1))**2
            error[row][2][i] += hybrid_err2



    error = np.mean(error, axis=2)
    error = np.hstack((indices, error))
    # error = np.log10(error)

    print error
    header = "n, OnlyTCM, FullLM, Hybrid"
    np.savetxt("squared_error_"+str(eps)+"_"+str(c)+".csv", error, header=header, delimiter=",")

eps = 0.1
c = 0.025
m = 1.0
sigma = m/6.0
experiment_n(10000, 100000, 5000, 1000, eps, c, m, sigma)
