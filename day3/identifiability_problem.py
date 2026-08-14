import numpy as np
import matplotlib.pyplot as plt

import json

from electrokitty import ElectroKitty

## creating data

sim = ElectroKitty("E(1): a* = b*")

sim.create_simulation([[0.5, 10, 0.0]], [293, 0, 0, 10**-4], [], [0, 0], [0.001, 10, 10**-5, 0], [[10**-4, 0], []])

sim.V_potential(0.5, -0.5, 0.001, 0, 0, 1000)

e, i, t = sim.simulate()

sim.Plot_simulation()

np.savetxt("data.txt", np.array([e, i + np.random.normal(0, 0.01 * max(i), len(t)), t]).T)



## fitting the data

sim = ElectroKitty("E(1): a* = b*")

sim.create_simulation([[0.1, 0.1, 0.0]], [293, 0, 0, 10**-4], [], [0, 0], [0.001, 10, 10**-5, 0], [[10**-4, 0], []])

sim.load_data_from_txt("data.txt")

sim.fit_to_data(algorithm="CMA-ES")

sim.Plot_data()
sim.Plot_simulation()
sim.print_fitting_parameters()

# sim.sample_parameter_distribution(n_samples = 30000)