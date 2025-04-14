import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from fft_pink_noise import *
import colorednoise
import json


# Create a voltage signal from a PHE sensor
#    CONSTS      #
rho_perp = 2.7e-7 #[ohm * m]
AMRR = 0.02 #AMR ratio (rho_par-rho_perp)/rho_perp
B_k = 10e-4 #[T] Difference of resistence for parallel B vs Perp B
#Isotopy Field
thickness = 200e-9 #[m] of the actual magnetic layer
LL = 10e-6 #The misalignment[m]
WW = 600e-6 #The width of the sensor [m]
F_c = 2000 #The frequency of the current in the sensor [Hz]
dt = 1e-4 #[sec]
pi = np.pi


#   GENERATING VOLTAGE TIME DOMAIN    #
Time = np.arange(0,1,dt) # Starts at t=0 and ends at t=0.999s so it is repeating itself
n = np.size(Time)
delta_rho = rho_perp * AMRR #[ohm * m]

I0 = 50e-3 #The current amplitude in the sensor[A]
B0 = 1e-4 #The magnetic field on the sensor [T]
F_B = 10 #The magnetic field frequency [Hz]
noise_typical_range = 2.5e-5
Noise_var = noise_variance(noise_typical_range)

BB = B0 * np.sin(2*pi*F_B*Time)
II = I0 * np.sin(2*pi*F_c*Time)
hh = BB / B_k

Voltage = (II / thickness) * ( (rho_perp + delta_rho - delta_rho*(hh**2))*(LL/WW) + delta_rho * hh)

freq, X = make_fft(Voltage, Time)


#   CREATING TRAIN NOISE + SIGNAL  #
sig_time = []
sig_f = []

for _ in range(2):
    wNoise = np.random.normal(0, Noise_var, n)
    _, fwNoise = make_fft(wNoise, Time)
    _, S_wNoise = welch(wNoise, 1/dt, nperseg=n)

    # Using filtering in DFT Domain
    pNoise = make_pink_noise(Time, Noise_var)
    _, S_pNoise = welch(pNoise, 1/dt, nperseg=n)
    _, fpNoise = make_fft(pNoise, Time)

    # Pink Using a library
    # pNoise = colorednoise.powerlaw_psd_gaussian(1, n) * Noise_var

    Noise = 1/2*(pNoise + wNoise)
    fNoise = S_pNoise


    #   COMBINED SIGNAL   #
    Signal = Noise + Voltage
    _, fSignal = make_fft(Signal, Time)

    sig_time.append(Signal.tolist())
    sig_f.append(fSignal.tolist())

train = {
    "signals": sig_time,
    "f_signals": sig_f,
}

#   CREATING TEST NOISE AND SIGNAL    #
sig_time = []
sig_f = []

for _ in range(2):
    wNoise = np.random.normal(0, Noise_var, n)
    _, fwNoise = make_fft(wNoise, Time)
    _, S_wNoise = welch(wNoise, 1/dt, nperseg=n)

    # Using filtering in DFT Domain
    pNoise = make_pink_noise(Time, Noise_var)
    _, S_pNoise = welch(pNoise, 1/dt, nperseg=n)
    _, fpNoise = make_fft(pNoise, Time)

    # Pink Using a library
    # pNoise = colorednoise.powerlaw_psd_gaussian(1, n) * Noise_var

    Noise = 1/2*(pNoise + wNoise)
    _, fNoise = welch(Noise, 1/dt, nperseg=n)


    #   COMBINED SIGNAL   #
    Signal = Noise + Voltage
    _, fSignal = make_fft(Signal, Time)

    sig_time.append(Signal.tolist())
    sig_f.append(fSignal.tolist())


test = {
    "signals": sig_time,
    "f_signals": sig_f,
}

data = {
    "train": train,
    "test": test,
    "t": Time.tolist(),
    "f": freq.tolist()
}

#Writting to the data:
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

fig, pureSig = plt.subplots(nrows=2, ncols=2)

pureSig[0, 0].plot(Time, wNoise, 'r-')
S_wNoisedB = 20 * np.log10(S_wNoise)
pureSig[1, 0].semilogy(freq, S_wNoise, '.-')
pureSig[0, 1].plot(Time, pNoise, 'm-')
pureSig[1, 1].loglog(freq[1:], S_pNoise[1:], 'm')

#   Add reference 1/f line for pink noise  #
ref = S_pNoise[50] * (freq[50] / freq[1:])  # reference line ~1/f
pureSig[1,1].loglog(freq[1:], ref, 'k--', label='~1/f')

plt.tight_layout()

plt.show()



''' PLOTTING
#   PLOTTING Volt+fVolt, Noise+fNoise   #
fig, pureSig = plt.subplots(nrows=2, ncols=2)

pureSig[0, 0].plot(Time, Voltage, 'r-')
pureSig[1, 0].semilogy(freq, X, '.-')
pureSig[0, 1].plot(Time, Noise, 'm-')
pureSig[1, 1].loglog(freq[1:], fNoise[1:], 'm')

#   Add reference 1/f line for pink noise  #
ref = fNoise[50] * (freq[50] / freq[1:])  # reference line ~1/f
pureSig[1,1].loglog(freq[1:], ref, 'k--', label='~1/f')

plt.tight_layout()

#    PLOTTING THE COMBINED SIGNAL   #
fig2, sigPlot = plt.subplots(nrows=2, ncols=1)

sigPlot[0].plot(Time, Signal, 'g')
sigPlot[1].loglog(freq, fSignal, "g--")
plt.show()
'''
