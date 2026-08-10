import numpy as np
import scipy.optimize as sciop
import matplotlib.pyplot as plt


def func(x, a, b):
    return a*x + b

x = np.linspace(-10, 10, 1000)

a = 2
b = 1

y = func(x, a, b) + np.random.normal(0, 0.1, len(x))

plt.scatter(x, y)

const, cov = sciop.curve_fit(func, x, y)

string = str(round(const[0], 2)) + "* x + " + str(round(const[1], 2))

plt.plot(x, func(x, *const), linewidth = 3)
plt.text(5, min(y), string)

print(cov)