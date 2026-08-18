import numpy as np
import matplotlib.pyplot as plt
from electrokitty import ElectroKitty
import scipy.signal as sig

%matplotlib Qt

mechanism = "E(1): a* = b*"

kin = [[0.5, 10**3, 0.0]]

D = []

iso = [0, 0]

intial_conditions = [[10**-4, 0], []]

Ru = 0
Cdl =  0

cell_const = [293, Ru, Cdl, 10**-4]

sim = ElectroKitty(mechanism)
sim.create_simulation(kin, cell_const, D, iso, [0.0001, 10, 10**-5, 0], intial_conditions)

#defining ACV experiment
J = 1000
dt = 0.005
fs=1/dt

Ei = 0.5
Ef = -0.5

nt = 2**14
amp = 0.1
freq = J*fs/nt
v = 2*abs(Ei - Ef)*fs/nt
print(v)
print(freq)

sim.V_potential(Ei, Ef, v, amp, freq, nt)

e,j,t=sim.simulate()

sim.Plot_simulation()


#Harmonic extraction


######------This is Electrokitty's functions------#########
#sim.FFT_analyze_sim(freq, 10, band*np.ones(12))
#sim.Harmonic_plots(plot_sim=True,w=1)
#sim.FT_plot()

#####------Let's start by doing the FFT----####

j_fft = np.fft.fft(j) # FT the current

l = len(t)

fs = len(t) / t[-1] # our sampling frequency 

x_f = np.fft.fftfreq(len(t), d=t[1] - t[0]) # the frequencies

plt.plot(x_f,np.abs(j_fft))


#----Parameters for the harmonic extraction------
band = 5 
harmonic = np.zeros((l, n_h)) #Pre-alocate space
N_harmonics = 7 # Define the number of harmonics to extract
#std = 7
#g = sig.windows.gaussian(l, std)


#----This loop extraction the harmonics------
for i in range(N_harmonics):
    h = i + 1 # Starts from fundamental up to 8

    # We defind the upper and lower bounds of the bandpass filter
    f_low = h * freq - band / 2  
    f_high = h * freq + band / 2  
    
   
    #mask = (x_f >= f_low) & (x_f <= f_high)
    
    # This is a rectangular bandbass
    mask = (
        ((x_f >= f_low) & (x_f <= f_high)) |
        ((x_f >= -f_high) & (x_f <= -f_low))
    )

    # This convolutes a gaussian with the rectangular to reduce side oscillations
    #conv_mask = sig.convolve(mask, g, mode='same')
    #norm_mask = conv_mask / np.max(conv_mask)
    
   
    #j_fft_masked = j_fft * norm_mask
    # This is where the filter is applied
    j_fft_masked = j_fft * mask
    
    # FT back to the time domain
    j_ifft = np.fft.ifft(j_fft_masked)
    
    # The FT gives us
    harmonic[:, i] = np.abs(sig.hilbert(np.real(j_ifft)))

dc_mask = np.abs(x_f) <= band
j_fft_dc = j_fft * dc_mask
dc = np.fft.ifft(j_fft_dc).real

  

plt.figure("DC")
plt.plot(t, dc)
plt.xlabel("Time / s")
plt.ylabel("Current")
plt.title("DC")


for i in range(n_h):

    plt.figure(f"Harmonic {i+1}")

    plt.plot(t, harmonic[:, i])

    plt.xlabel("Time / s")
    plt.ylabel("Harmonic amplitude")

    plt.title(f"Harmonic {i+1}")
