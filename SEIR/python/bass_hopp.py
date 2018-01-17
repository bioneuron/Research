import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import csv

# tial Conditions
INTERVAL = range(10, 36)
WEEK = np.array(INTERVAL)
m = len(WEEK)
#x0 = np.array([200.00, 0.00, 1.0, 0.0006, 0.37, 0.50])
lb = np.array([100.00, 0.00, 0.0, .000000007, .333, 0.20, -0.2, -1.00])
ub = np.array([10000.00, 100.00, 100.00, 0.7, 1.00, 1.00, 0.00, 1,00])
x0 = [  2.19624294e+03,   9.21314575e+01,   9.74936296e+01,
         4.87744177e-04,   4.00953905e-01,   4.31897600e-01, -.1, .02]

#sigma1 = np.repeat(1e-4, 24)
refine_factor = 100


# Read Influenza Data
ADDRESS = '/home/rasoul/Dropbox/Programming/SEIR/Matlab/data/new/2010_11.csv'

with open(ADDRESS) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data_str = []
    for row in readCSV:
        data_str.append(row[2])

data_str.pop(0)
ydata = np.array(list(map(int, data_str)))
xdata = WEEK
ydata = ydata[WEEK - 1]

# Read Humidity Data
hum1 =   np.array([0.0031,0.0031,0.0029,0.0030,0.0030,0.0029,0.0024,0.0023,0.0026,0.0025,
                  0.0025,0.0029,0.0030,0.0033,0.0043,0.0045,0.0047,0.0048,0.0054,
    0.0060,0.0063,0.0069,0.0076,0.0081,0.0079,0.0102])

with open(ADDRESS) as csvfile1:
    readCSV_AH = csv.reader(csvfile1, delimiter=',')
    data_str_AH = []
    for row in readCSV_AH:
        data_str_AH.append(row[4])

data_str_AH.pop(0)
ydata_AH = np.array(list(map(float, data_str_AH)))
xdata_AH = WEEK
ydata_AH = ydata_AH[WEEK - 1]

hum = ydata_AH * 1e-5 + .3277

# Normalize and Refine Humidity

AH_main = []
for i in range(len(hum)-1):
    x1 = list(np.linspace(hum[i],hum[i+1],refine_factor))
    AH_main += x1

hum1 = np.array(AH_main)

m1 = 2.00 / (hum1.max()- hum1.min())
AH = m1 * (hum1 - hum1.min()) - 1



# Define SEIR Equation
def sir_model(y, t, beta, sigma, gamma,q1,q2):
    # y=np.array([0,0,0])
    f_AH = 0
    if t >= 10 and t < 35:
        f_AH = AH[int((t - 10) * refine_factor)]


    S = -beta * y[0] * y[2] * (1 + q1 * f_AH) / (1 - q2 * y[2])
    E = beta * y[0] * y[2] * (1 + q1 * f_AH) / (1 - q2 * y[2]) - sigma * y[1]
    I = sigma * y[1] - gamma * y[2]

    return S, E, I


def fit_odeint(t, p):
    print(p)
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    z = integrate.odeint(sir_model, (p[0], p[1], p[2]), t, args=(p[3], p[4], p[5], p[6], p[7]))
    S1 = np.zeros(m)
    E1 = np.zeros(m)
    I1 = np.zeros(m)
    Y1 = np.zeros(m - 1)
    dt = (t.max() - t.min()) / (len(t))
    for i in range(m-1):
        alpha = i * refine_factor
        beta = (i + 1) * refine_factor - 1
        Y1[i] = p[4] * np.array(z[alpha:beta, 1]).sum() * dt
    #Y1 = -S1[1:(m)] - E1[1:(m)] + S1[0:(m - 1)] + E1[0:(m - 1)]
    return Y1


def r_resid(p):
    T = 0
    # for i in range(len(lb)):
    #    if p[i] < lb[i] or p[i] > ub[i]:
    #        T = 1
    t = np.linspace(WEEK.min(), WEEK.max(), refine_factor * (m - 1))
    Y2 = fit_odeint(t, p)
    a = Y2
    b = np.array(ydata[1:m])
    dist = np.linalg.norm((a - b), ord=1) #+ .5 * np.linalg.norm(p, ord = 1)
    print(T)
    if T == 0:
        return dist
    else:
        return 1e20


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



result = optimize.basinhopping(r_resid, x0, niter=300,
                            minimizer_kwargs=minimizer_kwargs)


print(result)
t = np.linspace(WEEK.min(), WEEK.max(), refine_factor * (m - 1))
Y1 = ydata
Y2 = fit_odeint(t, result.x)

print(Y2)

plt.plot(xdata[1:m], Y1[1:m], 'o')
plt.plot(xdata[1:m], Y2)
plt.show()
