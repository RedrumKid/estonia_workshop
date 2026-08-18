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

e,j,t=sim.simulate()

sim.Plot_simulation()


#Harmonic extraction

#sim.FFT_analyze_sim(freq, 10, band*np.ones(12))
#sim.Harmonic_plots(plot_sim=True,w=1)
#sim.FT_plot()


n_h = 7

j_fft = np.fft.fft(j)

l = len(t)

harmonic = np.zeros((l, n_h))

fs = len(t) / t[-1]

band = 5


x_f = np.fft.fftfreq(len(t), d=t[1] - t[0])
plt.plot(x_f,np.abs(j_fft))


#std = 7


#g = sig.windows.gaussian(l, std)

for i in range(n_h):
    h = i + 1
    
    f_low = h * freq - band / 2  
    f_high = h * freq + band / 2  
    
   
    #mask = (x_f >= f_low) & (x_f <= f_high)
    mask = (
        ((x_f >= f_low) & (x_f <= f_high)) |
        ((x_f >= -f_high) & (x_f <= -f_low))
    )
    #conv_mask = sig.convolve(mask, g, mode='same')
    #norm_mask = conv_mask / np.max(conv_mask)
    
   
    #j_fft_masked = j_fft * norm_mask
    j_fft_masked = j_fft * mask
    
 
    j_ifft = np.fft.ifft(j_fft_masked)
    
    
    
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
