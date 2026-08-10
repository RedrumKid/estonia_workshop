import numpy as np
import matplotlib.pyplot as plt

coeficient = 0.45

D = 0.1
c0 = 1
nx = 100

x = np.linspace(0,10,nx)
dx = 10/nx

tmax = 100

dt = coeficient*dx**2/D
t = np.arange(0, tmax, dt)

nt = len(t)

c = np.zeros(nx)
# c[35:66] = c0
c[:] = 1
cn = ()

for tt in range(0,len(t)):

    # enforce boundary conditions
    c[-1] = 1
    c[0] = 0

    # copy the current concentration profile
    cn = c.copy()

    for xx in range(1,nx-1):

        # update the concentration profile using the explicit scheme
        c[xx] = cn[xx] + D*dt/(dx)**2 * (cn[xx-1]+cn[xx+1]-2*cn[xx])

    i = c[1]/dx - c[0]/dx

    # ploting the results at specific time steps
    if tt == 1 or tt==int(nt/8) or tt==int(nt/4) or tt==int(nt/2) or tt==int(2*nt/3) or tt==int(3*nt/4) or tt==int(nt-5):
        plt.figure(1)
        plt.plot(x,c,label=tt)