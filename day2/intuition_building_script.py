import numpy as np
import matplotlib.pyplot as plt
from electrokitty import ElectroKitty

%matplotlib Qt

mechanism = "E(1): a* = b*"

kin = [[0.5, 10**1, 0.0]]

D = []

iso = [0, 0]

intial_conditions = [[10**-4, 0], []]

Ru = 1 * 1
Cdl = 50 * 0

cell_const = [293, Ru, Cdl, 10**-4]

sim = ElectroKitty(mechanism)
sim.create_simulation(kin, cell_const, D, iso, [0.0001, 10, 10**-5, 0], intial_conditions)

#defining ACV experiment
J = 100
dt = 0.002

Ei = 0.5
Ef = -0.5
# v = 0.1
nt = 2**14
amp = 0.1
freq = 9

v = abs(Ei - Ef)*freq/J
print(v)

sim.V_potential(Ei, Ef, v, amp, freq, nt)

sim.simulate()

sim.Plot_simulation()
sim.FFT_analyze_sim(freq, 7, 0.15*freq*np.ones(10))

sim.Harmonic_plots(plot_sim=True, w = 1)
sim.FT_plot()