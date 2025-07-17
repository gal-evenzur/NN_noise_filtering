import torch
import time  # Add time module for performance measurement

from torch_stft_pink_noise import *
import json
import numpy as np
from tqdm import trange

add_signals = False
testing = True
# Stft parameters
fs = 10000
total_cycles = 100
overlap_perc = 0.85
cycles_per_window = 5  # Number of cycles in one window

# Small for cpu, large for gpu
batch_size = 64 if torch.cuda.is_available() else 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if testing:
    n_trains = 10
    n_validate = 10
    n_tests = 10
else:
    n_trains = 64 * 12  # 768
    n_validate = 64 * 2  # 128
    n_tests = 64 * 2

# Finding out the size of stft #

stft_size = dummy_stft_size(fs, total_cycles, overlap_perc, cycles_per_window, Tperiod=1, only_center=True)
print("Dummy stft size:", stft_size)

# Get STFT axes once, as they are the same for all samples
_, sig_f, sig_t = my_stft(50e-3, 4e-12, 15, 0.5e-10,
                                fs=fs,
                                total_cycles=total_cycles,
                                overlap=overlap_perc,
                                cycles_per_window=cycles_per_window,
                                Tperiod=1,
                                only_center=True,
                                device=device)

# Performance tracking variables
total_start_time = time.time()
stft_timings = {'train': [], 'validate': [], 'test': []}
data_creation_times = {'train': 0, 'validate': 0, 'test': 0}

#   CREATING TRAIN NOISE + SIGNAL  #
train_start_time = time.time()
n = n_trains
num_batches = (n + batch_size - 1) // batch_size
clear_stft = torch.empty(n, stft_size[0], stft_size[1], device=device)  # Shape: (n_trains, n_freqs, n_time_bins)
noisy_stft = torch.empty(n, stft_size[0], stft_size[1], device=device)
Corresponding_F_B = []

# printing the size of the clear_stft and noisy_stft tensors
print("Size of clear_stft:", clear_stft.shape)

for batch_idx in trange(0, n_trains, batch_size, desc="Creating training data batches"):
    # Calculate actual batch size (may be smaller for last batch)
    current_batch_size = min(batch_size, n_trains - batch_idx)
    
    # Base parameters
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    
    # Create randomized parameters in batch
    rand_start = time.time()
    I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage = rand_train(
        I0, B0, F_B, noise_strength, device=device, batch_size=current_batch_size
    )
    rand_time = time.time() - rand_start
    
    # Generate clean STFTs in batch
    stft_start = time.time()
    clean_magnitude, _, _ = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                                   fs=fs,
                                   total_cycles=total_cycles,
                                   overlap=overlap_perc,
                                   cycles_per_window=cycles_per_window,
                                   Tperiod=1,
                                   is_clean=True,
                                   only_center=True,
                                   device=device)
    clean_stft_time = time.time() - stft_start
    
    # Generate noisy STFTs in batch
    stft_start = time.time()
    magnitude, _, _ = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                             fs=fs,
                             total_cycles=total_cycles,
                             overlap=overlap_perc,
                             cycles_per_window=cycles_per_window,
                             Tperiod=1,
                             only_center=True,
                             device=device)
    noisy_stft_time = time.time() - stft_start
    
    # Record timing information for each sample in the batch
    for i in range(current_batch_size):
        stft_timings['train'].append({
            'rand_time': rand_time / current_batch_size,  # Approximate per-sample time
            'clean_stft_time': clean_stft_time / current_batch_size,
            'noisy_stft_time': noisy_stft_time / current_batch_size,
            'total_iteration_time': (rand_time + clean_stft_time + noisy_stft_time) / current_batch_size
        })
    
    # Store results
    clear_stft[batch_idx:batch_idx+current_batch_size] = clean_magnitude
    noisy_stft[batch_idx:batch_idx+current_batch_size] = magnitude
    Corresponding_F_B.extend(F_B_r.tolist())


train = {
    "f_cSignal": clear_stft.cpu().tolist(),
    "f_signals": noisy_stft.cpu().tolist(),
    "F_B": Corresponding_F_B,  # Already a list of integers, no need for conversion
}

data_creation_times['train'] = time.time() - train_start_time

print("Training data created. sizes:")
for key, value in train.items():
    print(f"{key}: {np.shape(value)}")
    
# Print time statistics for training data
print("\nTraining data creation time statistics:")
print(f"Total time: {data_creation_times['train']:.2f} seconds")
print(f"Average time per sample: {data_creation_times['train']/n_trains:.4f} seconds")
avg_times = {k: sum(item[k] for item in stft_timings['train'])/n_trains for k in stft_timings['train'][0].keys()}
print(f"Average randomization time: {avg_times['rand_time']:.4f} seconds")
print(f"Average clean STFT generation time: {avg_times['clean_stft_time']:.4f} seconds")
print(f"Average noisy STFT generation time: {avg_times['noisy_stft_time']:.4f} seconds")
print("")

#  CREATING VALIDATION NOISE + SIGNAL  #

n = n_validate
num_batches = (n + batch_size - 1) // batch_size
clear_stft = torch.empty(n, stft_size[0], stft_size[1], device=device)  # Shape: (n_trains, n_freqs, n_time_bins)
noisy_stft = torch.empty(n, stft_size[0], stft_size[1], device=device)
Corresponding_F_B = []

validate_start_time = time.time()
for batch_idx in trange(0, n_validate, batch_size, desc="Creating validation data batches"):
    # Calculate actual batch size (may be smaller for last batch)
    current_batch_size = min(batch_size, n_validate - batch_idx)
    
    # Base parameters
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    
    # Create randomized parameters in batch
    rand_start = time.time()
    I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage = rand_train(
        I0, B0, F_B, noise_strength, device=device, batch_size=current_batch_size
    )
    rand_time = time.time() - rand_start
    
    # Generate clean STFTs in batch
    stft_start = time.time()
    clean_magnitude, _, _ = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                                   fs=fs,
                                   total_cycles=total_cycles,
                                   overlap=overlap_perc,
                                   cycles_per_window=cycles_per_window,
                                   Tperiod=1,
                                   is_clean=True,
                                   only_center=True,
                                   device=device)
    clean_stft_time = time.time() - stft_start
    
    # Generate noisy STFTs in batch
    stft_start = time.time()
    magnitude, _, _ = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                             fs=fs,
                             total_cycles=total_cycles,
                             overlap=overlap_perc,
                             cycles_per_window=cycles_per_window,
                             Tperiod=1,
                             only_center=True,
                             device=device)
    noisy_stft_time = time.time() - stft_start
    
    # Store results
    clear_stft[batch_idx:batch_idx+current_batch_size] = clean_magnitude
    noisy_stft[batch_idx:batch_idx+current_batch_size] = magnitude
    Corresponding_F_B.extend(F_B_r.tolist())

validate = {
    "f_cSignal": clear_stft.cpu().tolist(),
    "f_signals": noisy_stft.cpu().tolist(),
    "F_B": Corresponding_F_B,  # Already a list of integers, no need for conversion
}

data_creation_times['validate'] = time.time() - validate_start_time

print("Validation data created. sizes:")
for key, value in validate.items():
    print(f"{key}: {np.shape(value)}")
print("")

#   CREATING TEST NOISE AND SIGNAL    #
clear_stft = torch.empty(n_tests, stft_size[0], stft_size[1], device=device)
noisy_stft = torch.empty(n_tests, stft_size[0], stft_size[1], device=device)
Corresponding_F_B = []

test_start_time = time.time()
for batch_idx in trange(0, n_tests, batch_size, desc="Creating testing data batches"):
    # Calculate actual batch size (may be smaller for last batch)
    current_batch_size = min(batch_size, n_tests - batch_idx)
    
    # Base parameters
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    
    # Create randomized parameters in batch
    rand_start = time.time()
    I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage = rand_test(
        I0, B0, F_B, noise_strength, device=device, batch_size=current_batch_size
    )
    rand_time = time.time() - rand_start
    
    # Generate clean STFTs in batch
    stft_start = time.time()
    clean_magnitude, _, _ = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                                   fs=fs,
                                   total_cycles=total_cycles,
                                   overlap=overlap_perc,
                                   cycles_per_window=cycles_per_window,
                                   Tperiod=1,
                                   is_clean=True,
                                   only_center=True,
                                   device=device)
    clean_stft_time = time.time() - stft_start
    
    # Generate noisy STFTs in batch
    stft_start = time.time()
    magnitude, _, _ = my_stft(I0_r, B0_r, F_B_r, noise_strength_r,
                             fs=fs,
                             total_cycles=total_cycles,
                             overlap=overlap_perc,
                             cycles_per_window=cycles_per_window,
                             Tperiod=1,
                             only_center=True,
                             device=device)
    noisy_stft_time = time.time() - stft_start
    
    # Record timing information for each sample in the batch
    for i in range(current_batch_size):
        stft_timings['test'].append({
            'rand_time': rand_time / current_batch_size,
            'clean_stft_time': clean_stft_time / current_batch_size,
            'noisy_stft_time': noisy_stft_time / current_batch_size,
            'total_iteration_time': (rand_time + clean_stft_time + noisy_stft_time) / current_batch_size
        })
    
    # Store results
    clear_stft[batch_idx:batch_idx+current_batch_size] = clean_magnitude
    noisy_stft[batch_idx:batch_idx+current_batch_size] = magnitude
    Corresponding_F_B.extend(F_B_r.tolist())


test = {
    "f_cSignal": clear_stft.cpu().tolist(),
    "f_signals": noisy_stft.cpu().tolist(),
    "F_B": Corresponding_F_B,  # Already a list of integers, no need for conversion
}

data_creation_times['test'] = time.time() - test_start_time

print("Test data created. sizes:")
for key, value in test.items():
    print(f"{key}: {np.shape(value)}")


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
    "t": sig_t.cpu().tolist(),
    "f": sig_f.cpu().tolist()
}


#Writting to the data:
with open("Data/data_stft.json", "w") as f:
    json.dump(data, f, indent=4)

total_execution_time = time.time() - total_start_time
print("\nOverall Performance Statistics:")
print(f"Total script execution time: {total_execution_time:.2f} seconds")
print(f"Data creation times: Train={data_creation_times['train']:.2f}s, Validate={data_creation_times['validate']:.2f}s, Test={data_creation_times['test']:.2f}s")
print(f"Percentage of time spent: Train={100*data_creation_times['train']/total_execution_time:.1f}%, Validate={100*data_creation_times['validate']/total_execution_time:.1f}%, Test={100*data_creation_times['test']/total_execution_time:.1f}%")