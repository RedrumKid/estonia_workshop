import numpy as np
import matplotlib.pyplot as plt

import cma

from electrokitty import ElectroKitty

from multiprocessing import Pool 

# create a function that wraps around ElectroKitty to compute the loss function

def wrap_around(x):
    ### !!!!!!!!!!!! please NOTE this line
    x = x.tolist() # very important to add, this converts numpy array into a list with standard floats and intigers

    kin_guess = [[x[0], x[1], x[2]]] # we create a new kinetic list, based on the suggestion

    sim = ElectroKitty("E(1): a* = b*")

    sim.create_simulation(kin_guess, [293, 0, 0, 10**-4], [], [0, 0], [0.001, 10, 10**-5, 0], [[10**-4, 0], []])

    sim.load_data_from_txt("data.txt") # we can load the data like so

    e, i, t = sim.simulate() # simulate based on the data

    L = np.sum((i - sim.I_data)**2) # the loss function is a simple difference squared, feel free to use other loss functions

    return L # output the loss value

# NOTE, when multiprocessing/multithreading, very important to include the line below.
# If you do not include it, Python will run indefinitly 

if __name__ == "__main__":

    ## creating data, just a simple system

    sim = ElectroKitty("E(1): a* = b*")

    sim.create_simulation([[0.5, 10, 0.0]], [293, 0, 0, 10**-4], [], [0, 0], [0.001, 10, 10**-5, 0], [[10**-4, 0], []])

    sim.V_potential(0.5, -0.5, 0.001, 0, 0, 1000)

    e, i, t = sim.simulate()

    # sim.Plot_simulation()

    # adding some noise to make it interesting
    np.savetxt("data.txt", np.array([e, i + np.random.normal(0, 0.01 * max(i), len(t)), t]).T) # save to data.txt



    ## fitting the data using electrokitty

    sim = ElectroKitty("E(1): a* = b*")

    sim.create_simulation([[0.1, 0.1, 0.0]], [293, 0, 0, 10**-4], [], [0, 0], [0.001, 10, 10**-5, 0], [[10**-4, 0], []])

    sim.load_data_from_txt("data.txt") # loading directly

    sim.fit_to_data(algorithm = "CMA-ES")

    # sim.Plot_data()
    # sim.Plot_simulation()
    sim.print_fitting_parameters() # printing the final parameters


    ### fitting with CMA, the basic way

    # call CMA to minimize the loss function
    #                         fun              x0           sigma0
    x, es = cma.fmin2(wrap_around, [0.5, 0.1, 0.05], 0.01) # note the changes 

    ### multiprocessing with ElectroKitty and CMA

    # NOTE how fast it is with only 4 processes

    n_proc = 4 # the number of cores to be used

    # the fit is now a bit more manual, but that is good, since we now have more options to play around with
    es = cma.CMAEvolutionStrategy([0.5, 1, 0], 0.1, {"bounds": [[0.4, 0.01, -0.05], [0.6, 20, 0.05]], "tolfun": 10**-9})
    # note the bounds here. ElectroKitty does this automatically, basedon what I consider to make sense, but now you can set them maunally
    # This opens up options and power to have the fit go in a better direction

    pool = Pool(n_proc) # think of this as creating a pool of resources that are allocated for this script.
    # there are other options, but the most important one is to include more than 1 process

    while not es.stop(): # have the cma class start a loop, until convergance
        sols = es.ask() # we ask the class to generate proposed solutions
        
        f_evals = pool.map(wrap_around, sols) # have Python use multiple resources to compute loss values based on the propositions
        
        es.tell(sols, f_evals) # give the solutions back to CMA
        es.disp() # display how good its going
        es.logger.add() # this will create the outcmaes folder, where the class logs inbetween soultions
        
    pool.close() # important to say to Python to stop using the allocated resources
    pool.join()
    es.result_pretty() # display the final result