import numpy as np
np.set_printoptions(precision=4, suppress=True)

eps = 0.1
m = 1.0
sig = m/6.0

target = 1.25

clow = []
chigh = []

for n in np.arange(1000, 41000, 1000):
    # print n
    c = 0.0001
    flag = True
    while flag:
        left = (2*(m**2))/(n*(eps**2))
        right_top = 2*(m**2) + (1-c)*c*(eps**2)*n*(sig**2)
        right_bot = (c**2)*(eps**2)*(n**2)
        right = right_top/right_bot
        gam = min(left, right)
        top = n*(eps**2)*(2*(m**2)*((c**2)*n - c + 1) + c*(eps**2)*n*(sig**2))
        bot = 2*c*(eps**2)*(m**2)*(sig**2)*(-c*n + n + 1) + 4*(m**4)
        r = gam*(top/bot)

        if r < target:
            if c < 1:
                c += 0.0001
            else:
                flag = False
        else:
            flag = False

    # print c
    clow.append(c)

    flag = True
    while flag:
        left = (2*(m**2))/(n*(eps**2))
        right_top = 2*(m**2) + (1-c)*c*(eps**2)*n*(sig**2)
        right_bot = (c**2)*(eps**2)*(n**2)
        right = right_top/right_bot
        gam = min(left, right)
        top = n*(eps**2)*(2*(m**2)*((c**2)*n - c + 1) + c*(eps**2)*n*(sig**2))
        bot = 2*c*(eps**2)*(m**2)*(sig**2)*(-c*n + n + 1) + 4*(m**4)
        r = gam*(top/bot)

        if r > target:
            if c < 1:
                c += 0.0001
            else:
                flag = False
        else:
            flag = False

    chigh.append(c)

clow = np.array(clow)
chigh = np.array(chigh)
indices = np.arange(1000, 41000, 1000)

final = (np.vstack((indices, clow, chigh))).T
print final

header = "n, c"
np.savetxt("graph_b_r_"+str(target)+".csv", final, header=header, delimiter=",")
