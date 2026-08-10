import numpy as np
import matplotlib.pyplot as plt

coeficient = 0.8

D = 0.1
c0 = 1
nx = 10

x = np.linspace(0,10,nx)
dx = 10/nx

tmax = 100

dt = coeficient*dx**2/D
t = np.arange(0, tmax, dt)

nt = len(t)

c = np.zeros(nx)
# c[35:66] = c0
c[3:7] = c0
cn = ()

A = np.eye(nx)*(1 + 2*D*dt/dx**2) - np.eye(nx, k=-1)*(D*dt/dx**2) - np.eye(nx, k=1)*(D*dt/dx**2)

A[0, :] = 0
A[0, 0] = 1
A[0, 1] = -1

A[-1, :] = 0
A[-1, -1] = 1
A[-1, -2] = -1

for tt in range(0, len(t)):

    # enforce boundary conditions
    # c[-1] = c[-2]
    # c[0] = c[1]

    # copy the current concentration profile
    cn = c.copy()
    cn[0] = 0
    cn[-1] = 0
    # update the concentration profile using the implicit scheme
    c = np.linalg.solve(A, cn)

    # ploting the results at specific time steps
    if tt == 1 or tt==int(nt/8) or tt==int(nt/4) or tt==int(nt/2) or tt==int(2*nt/3) or tt==int(3*nt/4) or tt==int(nt-5):
        plt.figure(1)
        plt.plot(x,c,label=tt)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('c')
        plt.title('Implicit Scheme for Diffusion')
