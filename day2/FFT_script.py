import numpy as np
import matplotlib.pyplot as plt
%matplotlib Qt

def fun(x):
    return np.sin(x) + np.sin(2*x) + np.sin(4*x)

x = np.linspace(-2*np.pi, 2*np.pi, 1000)

y = fun(x) + np.random.normal(0, 1, len(x))

plt.figure("actual data")

plt.plot(x, y)

sp = np.fft.fft(y)

freq = np.fft.fftfreq(len(x), d = x[1] - x[0])

plt.figure("FT")
plt.plot(freq, np.abs(sp))

sp1 = sp.copy()
sp1[3 : -3] = 0

plt.plot(freq, np.abs(sp1))

sp2 = sp.copy()
sp2[:4] = 0
sp2[5:-4] = 0
sp2[-3:] = 0

plt.plot(freq, np.abs(sp2))

sp3 = sp.copy()
sp3[:8] = 0
sp3[9:-8] = 0
sp3[-7:] = 0

plt.plot(freq, np.abs(sp3))

plt.figure("actual data")

plt.plot(x, np.fft.ifft(sp1))
plt.plot(x, np.fft.ifft(sp2))
plt.plot(x, np.fft.ifft(sp3))