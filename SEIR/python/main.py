import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import csv


x0 = np.array([7000, 0, 35, .0001, .4, .3])
lb = np.array([100, 0, 0, .000000007, 1 / 3, .2])
ub = np.array([7000000, 100, 500, .7, 1, 1])
sigma1 = np.repeat(1e-4, 24)

ADDRESS = '/home/rasoul/Dropbox/Programming/SEIR/Matlab/data/new/2010_11.csv'

with open(ADDRESS) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data_str = []
    for row in readCSV:
        data_str.append(row[2])
data_str.pop(0)
ydata = list(map(int, data_str))
xdata = np.array(
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])
ydata = np.array(ydata[9:35])


def sir_model(y, x, beta, sigma, gamma):
    S = -beta * y[0] * y[2]
    E = beta * y[0] * y[2] - sigma * y[1]
    I = sigma * y[1] - gamma * y[2]
    return S, E, I


def fit_odeint(x, p):
    z = integrate.odeint(sir_model, (p[0], p[1], p[2]), x, args=(p[3], p[4], p[5]))
    Y1 = -z[1:25, 0] - z[1:25, 1] + z[0:24, 0] + z[0:24, 1]
    return Y1


def r_resid(p):
    T = 0
    Y2 = fit_odeint(xdata, p)
    a = np.array(Y2)
    b = np.array(ydata[1:25])
    dist = np.linalg.norm(Y2 - ydata[1:25])
    dist = np.linalg.norm((a - b), ord=1)
    print(T)
    if T == 0:
        return dist
    else:
        return 1e20


xx = optimize.least_squares(r_resid, x0, method="trf",
                       bounds=bounds,
                       options={'maxiter': 100000, 'xtol': 1e-12, 'disp': True}
                       )

# fitted = fit_odeint(xdata, *popt)

print(xx)

Y1 = ydata
Y2 = fit_odeint(xdata, xx.x)
print(Y2)

# z= fit_odeint(xdata,.0004,.3, .6)
# print(z[:,1])
plt.plot(xdata[1:25], Y1[1:25], 'o')
plt.plot(xdata[1:25], Y2)
plt.show()




# plt.plot(xdata, ydata, 'o')
# plt.plot(xdata, fitted)
# plt.show()
