# %% Importing libraries
import torch
import time  # Add time module for performance measurement

from torch_stft_pink_noise import *
import h5py
import numpy as np
from tqdm import trange

# %% Parameters
add_signals = False
testing = False
# Stft parameters
fs = 10000
total_cycles = 100
overlap_perc = 0.85
cycles_per_window = 5  # Number of cycles in one window

B0 = 5e-12  # The magnetic field on the sensor [T]
F_B = 15  # The magnetic field frequency [Hz]
I0 = 50e-3  # The current amplitude in the sensor[A]
noise_strength = 1e-10  # The noise strength [V]

# Small for cpu, large for gpu
batch_size = 16 if torch.cuda.is_available() else 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if testing:
    n_trains = 20
    n_validate = 10
    n_tests = 10
else:
    n_trains = 64 * 1  # = 1920
    n_validate = 64 * 3  # = 192
    n_tests = 64 * 3

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

# %% **************TRAINING*************** # 

total_start_time = time.time()
stft_timings = {'train': [], 'validate': [], 'test': []}
data_creation_times = {'train': 0, 'validate': 0, 'test': 0}

#   CREATING TRAIN NOISE + SIGNAL  #
train_start_time = time.time()
n = n_trains
clear_stft = torch.empty(n, stft_size[1], stft_size[2], device=device)  # Shape: (n_trains, n_freqs, n_time_bins)
noisy_stft = torch.empty(n, stft_size[1], stft_size[2], device=device)
Corresponding_F_B = []
Corresponding_B0 = []

# printing the size of the clear_stft and noisy_stft tensors
print("Size of clear_stft:", clear_stft.shape)

for batch_idx in trange(0, n_trains, batch_size, desc="Creating training data batches"):
    # Calculate actual batch size (may be smaller for last batch)
    current_batch_size = min(batch_size, n_trains - batch_idx)
    
    
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
    Corresponding_B0.extend(B0_r.tolist())


# Store data directly as numpy arrays for better efficiency
train_f_cSignal = clear_stft.cpu().numpy()  # Direct numpy conversion
train_f_signals = noisy_stft.cpu().numpy()
train_F_B = np.array(Corresponding_F_B)
train_B0 = np.array(Corresponding_B0)

data_creation_times['train'] = time.time() - train_start_time

print("Training data created. sizes:")
print(f"f_cSignal: {train_f_cSignal.shape}")
print(f"f_signals: {train_f_signals.shape}")
print(f"F_B: {train_F_B.shape}")
print(f"B0: {train_B0.shape}")

# Print time statistics for training data
print("\nTraining data creation time statistics:")
print(f"Total time: {data_creation_times['train']:.2f} seconds")
print(f"Average time per sample: {data_creation_times['train']/n_trains:.4f} seconds")
avg_times = {k: sum(item[k] for item in stft_timings['train'])/n_trains for k in stft_timings['train'][0].keys()}
print(f"Average randomization time: {avg_times['rand_time']:.4f} seconds")
print(f"Average clean STFT generation time: {avg_times['clean_stft_time']:.4f} seconds")
print(f"Average noisy STFT generation time: {avg_times['noisy_stft_time']:.4f} seconds")
print("")

# %% **************VALIDATION*************** #

n = n_validate
clear_stft = torch.empty(n, stft_size[1], stft_size[2], device=device)  # Shape: (n_trains, n_freqs, n_time_bins)
noisy_stft = torch.empty(n, stft_size[1], stft_size[2], device=device)
Corresponding_F_B = []
Corresponding_B0 = []

validate_start_time = time.time()
for batch_idx in trange(0, n_validate, batch_size, desc="Creating validation data batches"):
    # Calculate actual batch size (may be smaller for last batch)
    current_batch_size = min(batch_size, n_validate - batch_idx)
    
    
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
    Corresponding_B0.extend(B0_r.tolist())

# Store data directly as numpy arrays for better efficiency
validate_f_cSignal = clear_stft.cpu().numpy()
validate_f_signals = noisy_stft.cpu().numpy()
validate_F_B = np.array(Corresponding_F_B)
validate_B0 = np.array(Corresponding_B0)

data_creation_times['validate'] = time.time() - validate_start_time

print("Validation data created. sizes:")
print(f"f_cSignal: {validate_f_cSignal.shape}")
print(f"f_signals: {validate_f_signals.shape}")
print(f"F_B: {validate_F_B.shape}")
print(f"B0: {validate_B0.shape}")

# %% # **************TESTING*************** #
n = n_tests
clear_stft = torch.empty(n, stft_size[1], stft_size[2], device=device)  # Shape: (n_trains, n_freqs, n_time_bins)
noisy_stft = torch.empty(n, stft_size[1], stft_size[2], device=device)
Corresponding_F_B = []
Corresponding_B0 = []

test_start_time = time.time()
for batch_idx in trange(0, n_tests, batch_size, desc="Creating testing data batches"):
    # Calculate actual batch size (may be smaller for last batch)
    current_batch_size = min(batch_size, n_tests - batch_idx)
    
    
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
    Corresponding_B0.extend(B0_r.tolist())


# Store data directly as numpy arrays for better efficiency
test_f_cSignal = clear_stft.cpu().numpy()
test_f_signals = noisy_stft.cpu().numpy()
test_F_B = np.array(Corresponding_F_B)
test_B0 = np.array(Corresponding_B0)

data_creation_times['test'] = time.time() - test_start_time

print("Test data created. sizes:")
print(f"f_cSignal: {test_f_cSignal.shape}")
print(f"f_signals: {test_f_signals.shape}")
print(f"F_B: {test_F_B.shape}")
print(f"B0: {test_B0.shape}")


# %% Noise for testing
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

# %% File saving
path = "C:\\Users\\galev\\Documents\\Research\\Data\\data_stft.h5"
with h5py.File(path, "w") as f:
    # Determine optimal chunk sizes based on data dimensions and typical access patterns
    # For STFT data: chunk by whole samples to maintain data locality
    # Typical chunk size: 8-32 samples at a time for efficient I/O
    
    # Training data chunking
    train_samples = train_f_cSignal.shape[0]
    train_chunk_size = min(16, train_samples)  # Chunk 16 samples or all if fewer
    train_stft_chunks = (train_chunk_size, train_f_cSignal.shape[1], train_f_cSignal.shape[2])
    train_fb_chunks = (min(64, train_samples),)  # For 1D F_B array
    
    train_group = f.create_group("train")
    train_group.create_dataset("f_cSignal", data=train_f_cSignal, 
                              chunks=train_stft_chunks,
                              compression=None)  # No compression as requested
    train_group.create_dataset("f_signals", data=train_f_signals, 
                              chunks=train_stft_chunks,
                              compression=None)
    train_group.create_dataset("F_B", data=train_F_B, 
                              chunks=train_fb_chunks,
                              compression=None)
    train_group.create_dataset("B0", data=train_B0, 
                              chunks=train_fb_chunks,
                              compression=None)

    # Validation data chunking
    validate_samples = validate_f_cSignal.shape[0]
    validate_chunk_size = min(16, validate_samples)
    validate_stft_chunks = (validate_chunk_size, validate_f_cSignal.shape[1], validate_f_cSignal.shape[2])
    validate_fb_chunks = (min(64, validate_samples),)
    
    validate_group = f.create_group("validate")
    validate_group.create_dataset("f_cSignal", data=validate_f_cSignal, 
                                 chunks=validate_stft_chunks,
                                 compression=None)
    validate_group.create_dataset("f_signals", data=validate_f_signals, 
                                 chunks=validate_stft_chunks,
                                 compression=None)
    validate_group.create_dataset("F_B", data=validate_F_B, 
                                 chunks=validate_fb_chunks,
                                 compression=None)
    validate_group.create_dataset("B0", data=validate_B0, 
                                 chunks=validate_fb_chunks,
                                 compression=None)

    # Test data chunking
    test_samples = test_f_cSignal.shape[0]
    test_chunk_size = min(16, test_samples)
    test_stft_chunks = (test_chunk_size, test_f_cSignal.shape[1], test_f_cSignal.shape[2])
    test_fb_chunks = (min(64, test_samples),)
    
    test_group = f.create_group("test")
    test_group.create_dataset("f_cSignal", data=test_f_cSignal, 
                             chunks=test_stft_chunks,
                             compression=None)
    test_group.create_dataset("f_signals", data=test_f_signals, 
                             chunks=test_stft_chunks,
                             compression=None)
    test_group.create_dataset("F_B", data=test_F_B, 
                             chunks=test_fb_chunks,
                             compression=None)
    test_group.create_dataset("B0", data=test_B0, 
                             chunks=test_fb_chunks,
                             compression=None)

    # Time and frequency axes (no chunking needed for small 1D arrays)
    f.create_dataset("t", data=sig_t.cpu().numpy(), compression=None)
    f.create_dataset("f", data=sig_f.cpu().numpy(), compression=None)
    
    print(f"\nHDF5 file created with chunking:")
    print(f"Training STFT chunks: {train_stft_chunks}")
    print(f"Validation STFT chunks: {validate_stft_chunks}")
    print(f"Test STFT chunks: {test_stft_chunks}")
    print(f"F_B/B0 chunks: {train_fb_chunks[0]} samples")

total_execution_time = time.time() - total_start_time
print("\nOverall Performance Statistics:")
print(f"Total script execution time: {total_execution_time:.2f} seconds")
print(f"Data creation times: Train={data_creation_times['train']:.2f}s, Validate={data_creation_times['validate']:.2f}s, Test={data_creation_times['test']:.2f}s")
print(f"Percentage of time spent: Train={100*data_creation_times['train']/total_execution_time:.1f}%, Validate={100*data_creation_times['validate']/total_execution_time:.1f}%, Test={100*data_creation_times['test']/total_execution_time:.1f}%")

# %%
