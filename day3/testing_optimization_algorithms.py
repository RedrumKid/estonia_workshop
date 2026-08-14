import numpy as np
import matplotlib.pyplot as plt
import cma
import scipy.optimize as sciop

# check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

def fun(x):
    return x**4 - 4*x**2 - x -1 

x = np.linspace(-2, 2, 100)

# plt.plot(x, fun(x))

# sol = sciop.minimize(fun, -1, method = "BFGS")

# print(sol.x)

es = cma.CMAEvolutionStrategy([-1], 10)
while not es.stop():
    X = es.ask()
    F = [fun(x) for x in X]
    es.tell(X, F)
    es.disp()

es.result_pretty()

