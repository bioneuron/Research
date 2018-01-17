import numpy as np
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from scipy.interpolate import interp1d
import csv


# tial Conditions
INTERVAL = range(1, 36)
WEEK = np.array(INTERVAL)
m = len(WEEK)
x0 = np.array([200.00, 5.00, 1.0, 0.0006, 0.37, 0.50, .1, 0.0])
lb = np.array([100.00, 0.00, 0.0, .000000007, .333, 0.20, -0.20, -0.00])
ub = np.array([700000.00, 100.00, 100.00, 0.7, 1.00, 1.00, 0.00, 0.00])
sigma1 = np.repeat(1e-4, 24)
refine_factor = 10

# Read Data
ADDRESS = '/home/rasoul/Dropbox/Programming/SEIR/Matlab/data/new/2007_08.csv'

ADDRESS_AH = '/home/rasoul/Dropbox/Programming/SEIR/Matlab/data/AH.csv'

with open(ADDRESS) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data_str = []
    for row in readCSV:
        data_str.append(row[2])

data_str.pop(0)
ydata = np.array(list(map(float, data_str)))
xdata = WEEK
ydata = ydata[WEEK - 1]

#################################
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

AH_main = []
for i in range(len(hum)-1):
    x1 = list(np.linspace(hum[i],hum[i+1],refine_factor))
    AH_main += x1

hum1 = np.array(AH_main)

m1 = 2.00 / (hum1.max()- hum1.min())
AH = m1 * (hum1 - hum1.min()) - 1

t = np.linspace(WEEK.min(),WEEK.max(), refine_factor*(m-1))

tmin = t[0]
tmax = t[-1]
def f(x):
    #ind1 = [i for i, y in enumerate(t) if y > x and y < x+.001]
    ind1 = int((x - tmin) * refine_factor)
    if ind1 < 0 or ind1>= 25 * refine_factor:
        ind1 = 0
    return 0



# Define SEIR Equation
def sir_model(y, t, beta, sigma, gamma, q1, q2):
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

    dt = (t.max() - t.min()) / (len(t))
    for i in range(m):
        if i == 0:
            S1[i] = z[i, 0];
            E1[i] = z[i, 1];
            I1[i] = z[i, 2]
        else:
            S1[i] = z[refine_factor * i - 1, 0]
            E1[i] = z[refine_factor * i - 1, 1]
            I1[i] = z[refine_factor * i - 1, 2]
    Y1 = -S1[1:(m)] - E1[1:(m)] + S1[0:(m - 1)] + E1[0:(m - 1)]
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
    dist = np.linalg.norm((a - b), ord=1)
    print(T)
    if T == 0:
        return dist
    else:
        return 1e20
data = ydata[1:m]
x = WEEK[1:m]
bounds = [(low, high) for low, high in zip(lb, ub)]

result = optimize.differential_evolution(r_resid, bounds)


# res = optimize.fmin(r_resid, x0)
# fitted = fit_odeint(xdata, *popt)
t = np.linspace(WEEK.min(),WEEK.max(), refine_factor*(m-1))
Y1 = ydata
Y2  = fit_odeint(t, result.x)

print(Y2)
#z= fit_odeint(xdata,.0004,.3, .6)
#print(z[:,1])
plt.plot(xdata[1:m], Y1[1:m], '+k')
plt.plot(xdata[1:m], Y2)
plt.show()



