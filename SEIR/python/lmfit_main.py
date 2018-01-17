import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import csv

# tial Conditions
INTERVAL = range(1, 36)
WEEK = np.array(INTERVAL)
m = len(WEEK)
x0 = np.array([200.00, 5.00, 1.0, 0.0006, 0.37, 0.50])
lb = np.array([100.00, 0.00, 0.0, .000000007, .333, 0.20])
ub = np.array([700000.00, 100.00, 100.00, 0.7, 1.00, 1.00])
sigma1 = np.repeat(1e-4, 24)

# Read Data
ADDRESS = '/home/rasoul/Dropbox/Programming/SEIR/Matlab/data/new/2011_12.csv'

with open(ADDRESS) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data_str = []
    for row in readCSV:
        data_str.append(row[2])

data_str.pop(0)
ydata = np.array(list(map(int, data_str)))
xdata = WEEK
ydata = ydata[WEEK - 1]


# Define SEIR Equation
def sir_model(y, t, beta, sigma, gamma):
    # y=np.array([0,0,0])
    q = 0
    S = -beta * y[0] * y[2] * (1 / (1 - q * y[2]))
    E = beta * y[0] * y[2] * (1 / (1 - q * y[2])) - sigma * y[1]
    I = sigma * y[1] - gamma * y[2]
    return S, E, I


def fit_odeint(t, p):
    print(p)
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    z = integrate.odeint(sir_model, (p[0], p[1], p[2]), t, args=(p[3], p[4], p[5]))
    S1 = np.zeros(m)
    E1 = np.zeros(m)
    I1 = np.zeros(m)

    dt = (t.max() - t.min()) / (len(t))
    for i in range(m):
        if i == 0:
            S1[i] = z[i, 0];
            E1[i] = z[i, 1];
            I1[i] = z[i, 2]
        else:
            S1[i] = z[1000 * i - 1, 0]
            E1[i] = z[1000 * i - 1, 1]
            I1[i] = z[1000 * i - 1, 2]
    Y1 = -S1[1:(m)] - E1[1:(m)] + S1[0:(m - 1)] + E1[0:(m - 1)]
    return Y1


def r_resid(p):
    T = 0
    # for i in range(len(lb)):
    #    if p[i] < lb[i] or p[i] > ub[i]:
    #        T = 1
    t = np.linspace(WEEK.min(), WEEK.max(), 1000 * (m - 1))
    Y2 = fit_odeint(t, p)
    a = Y2
    b = np.array(ydata[1:m])
    dist = np.linalg.norm((a - b), ord=1)
    print(T)
    if T == 0:
        return dist
    else:
        return 1e20


def f(p):
    return lb - p


def g(p):
    return p - ub


cons = ({'type': 'ineq', 'fun': f},
        {'type': 'ineq', 'fun': g})


class MyBounds(object):
    def __init__(self, xmax=lb, xmin=ub):
        self.xmax = xmax
        self.xmin = xmin

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


class MyTakeStep(object):
    def __init__(self, xmin=lb, xmax=ub, stepsize=np.array([10, 2, 2, .0001, .01, .01])):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        s = self.stepsize
        xnew = np.zeros(len(x))
        for i in range(len(x)):
            xnew[i] = x[i] + np.random.uniform(-s[i], s[i], np.shape(x[i]))
        if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
            xnew = x
        return xnew


bounds = [(low, high) for low, high in zip(lb, ub)]
minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)

take_step = MyTakeStep(xmin=lb, xmax=ub)
# con1 = {'type': 'ineq', 'fun': }
# res = optimize.minimize(r_resid, x0, method="TNC",tol=1e-6,bounds=bnds)

res = optimize.basinhopping(r_resid, x0, niter=100,
                            minimizer_kwargs=minimizer_kwargs, take_step=take_step)
# res = optimize.fmin(r_resid, x0)
# fitted = fit_odeint(xdata, *popt)

print(res)
t = np.linspace(WEEK.min(), WEEK.max(), 1000 * (m - 1))
Y1 = ydata
Y2 = fit_odeint(t, res.x)

print(Y2)
# z= fit_odeint(xdata,.0004,.3, .6)
# print(z[:,1])
plt.plot(xdata[1:m], Y1[1:m], 'o')
plt.plot(xdata[1:m], Y2)
plt.show()


