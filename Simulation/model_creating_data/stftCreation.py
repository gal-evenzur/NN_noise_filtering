import torch

from fft_pink_noise import *
import json

add_signals = False
testing = True
# Stft parameters
fs = 10000
total_cycles = 100
overlap_perc = 0.85
cycles_per_window = 5  # Number of cycles in one window




if testing:
    n_trains = 10
    n_validate = 1
    n_tests = 1
else:
    n_trains = 64 * 16  # 1024
    n_validate = 64 * 3  # 192
    n_tests = 64 * 3

# Finding out the size of stft #

stft_size = dummy_stft_size(fs, total_cycles, overlap_perc, cycles_per_window, Tperiod=1, only_center=True)
print("Dummy stft size:", stft_size)

#   CREATING TRAIN NOISE + SIGNAL  #
clear_stft = torch.empty(n_trains, stft_size[0], stft_size[1])  # Shape: (n_trains, n_freqs, n_time_bins)
noisy_stft = torch.empty(n_trains, stft_size[0], stft_size[1])
sig_f = []
sig_t = []
Corresponding_F_B = []

# printing the size of the clear_stft and noisy_stft tensors
print("Size of clear_stft:", clear_stft.shape)

for i in range(n_trains):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    # Create a tuple with all the randomized parameters
    I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage = rand_train(I0, B0, F_B,noise_strength)


    clean_magnitude, _, _ = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                                               fs=fs,
                                               total_cycles=total_cycles,
                                               overlap=overlap_perc,
                                               cycles_per_window=cycles_per_window,
                                               Tperiod=1,
                                               is_clean=True,
                                               only_center=True)

    magnitude, freqs_stft, time_bins = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                                               fs=fs,
                                               total_cycles=total_cycles,
                                               overlap=overlap_perc,
                                               cycles_per_window=cycles_per_window,
                                               Tperiod=1,
                                               only_center=True)



    clear_stft[i] = clean_magnitude
    noisy_stft[i] = magnitude
    sig_f.append(freqs_stft.tolist())
    sig_t.append(time_bins.tolist())

    Corresponding_F_B.append(F_B_r)


train = {
    "f_cSignal": clear_stft.tolist(),
    "f_signals": noisy_stft.tolist(),
    "F_B": Corresponding_F_B,
}

clear_stft = torch.zeros(n_validate, stft_size[0], stft_size[1])
noisy_stft = torch.zeros(n_validate, stft_size[0], stft_size[1])
sig_f = []
sig_t = []
Corresponding_F_B = []

for i in range(n_validate):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    # Create a tuple with all the randomized parameters
    I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage = rand_train(I0, B0, F_B, noise_strength)

    clean_magnitude, _, _ = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                                    fs=fs,
                                    total_cycles=total_cycles,
                                    overlap=overlap_perc,
                                    cycles_per_window=cycles_per_window,
                                    Tperiod=1,
                                    is_clean=True,
                                    only_center=True)

    magnitude, freqs_stft, time_bins = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                                               fs=fs,
                                               total_cycles=total_cycles,
                                               overlap=overlap_perc,
                                               cycles_per_window=cycles_per_window,
                                               Tperiod=1,
                                               only_center=True)

    clear_stft[i] = clean_magnitude
    noisy_stft[i] = magnitude
    sig_f.append(freqs_stft.tolist())
    sig_t.append(time_bins.tolist())

    Corresponding_F_B.append(F_B_r)

validate = {
    "f_cSignal": clear_stft.tolist(),
    "f_signals": noisy_stft.tolist(),
    "F_B": Corresponding_F_B,
}


#   CREATING TEST NOISE AND SIGNAL    #
clear_stft = torch.zeros(n_tests, stft_size[0], stft_size[1])
noisy_stft = torch.zeros(n_tests, stft_size[0], stft_size[1])
sig_f = []
sig_t = []
Corresponding_F_B = []

for i in range(n_tests):

    # The base parameters:
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10

    I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage = rand_test(I0, B0, F_B, noise_strength)

    clean_magnitude, _, _ = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                                    fs=fs,
                                    total_cycles=total_cycles,
                                    overlap=overlap_perc,
                                    cycles_per_window=cycles_per_window,
                                    Tperiod=1,
                                    is_clean=True,
                                    only_center=True)

    magnitude, freqs_stft, time_bins = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                                               fs=fs,
                                               total_cycles=total_cycles,
                                               overlap=overlap_perc,
                                               cycles_per_window=cycles_per_window,
                                               Tperiod=1,
                                               only_center=True)

    clear_stft[i] = clean_magnitude
    noisy_stft[i] = magnitude
    sig_f.append(freqs_stft.tolist())
    sig_t.append(time_bins.tolist())

    Corresponding_F_B.append(F_B_r)

test = {
    "f_cSignal": clear_stft.tolist(),
    "f_signals": noisy_stft.tolist(),
    "F_B": Corresponding_F_B,
}

#   CREATING ONLY NOISE WITH SPECS OF TESTING #
'''
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []

n_noise = 10
for _ in range(10):

    params = rand_test(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params, only_noise=True)

    real_F_B = params[2]

    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())


noise_for_test = {
    "f_cSignal": clear_sig_f,
    "f_noise": sig_f,
}
'''

data = {
    "train": train,
    "validate": validate,
    "test": test,
    "t": sig_t,
    "f": sig_f
}


#Writting to the data:
with open("data_stft.json", "w") as f:
    json.dump(data, f, indent=4)