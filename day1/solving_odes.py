import numpy as np
import matplotlib.pyplot as plt

x0 = 1
k = 0.1

xs = [x0]

dt = 30

t = np.arange(0, 1000, dt)

for i in range(len(t)-1):

    old_x = xs[-1]

    new_x = old_x - dt*k*old_x

    # new_x = old_x/(1 + dt*k)

    xs.append(new_x)

plt.plot(t, xs)
plt.plot(t, np.exp(-k*t))