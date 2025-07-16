from fft_pink_noise import *
import json
from tqdm import tqdm

add_signals = False
testing = True

#   CREATING TRAIN NOISE + SIGNAL  #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []
Corresponding_B_strength = []
Corresponding_Main_Peak_strength = []
Corresponding_F_B = []
Corresponding_B_strength = []


if testing:
    n_trains = 10
    n_validate = 0
    n_tests = 0
else:
    n_trains = 64 * 16  # 1024
    n_validate = 64 * 3  # 192
    n_tests = 64 * 3


for _ in tqdm(range(n_trains), desc="Generating training data"):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    # Create a tuple with all the randomized parameters
    params = rand_train(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)

    real_F_B = params[2]
    real_B = params[1]

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())
    Corresponding_F_B.append(real_F_B)
    Corresponding_B_strength.append(real_B)

    # Corresponding_B_strength.append(fVolt[2000-real_F_B])
    # Corresponding_Main_Peak_strength.append(np.max(fVolt))

train = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
    "signals": sig_time,
    "f_signals": sig_f,
    "F_B": Corresponding_F_B,
    "B_strength": Corresponding_B_strength,
}
print(f"Finished generating training dataset with {len(train['f_cSignal'])} samples")
for key, val in train.items():
    print(f"Training {key} samples: {len(val)}")
print()

clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []
Corresponding_B_strength = []
Corresponding_Main_Peak_strength = []
Corresponding_F_B = []
Corresponding_B_strength = []



for _ in tqdm(range(n_validate), desc="Generating validation data"):
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    # Create a tuple with all the randomized parameters
    params = rand_train(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)

    real_F_B = params[2]
    real_B = params[1]

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())
    Corresponding_F_B.append(real_F_B)
    Corresponding_B_strength.append(real_B)

    # Corresponding_B_strength.append(fVolt[2000-real_F_B])
    # Corresponding_Main_Peak_strength.append(np.max(fVolt))

validate = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
    "signals": sig_time,
    "f_signals": sig_f,
    "F_B": Corresponding_F_B,
    "B_strength": Corresponding_B_strength,
}
print(f"Finished generating validation dataset with {len(validate['f_cSignal'])} samples")
for key, val in validate.items():
    print(f"Validation {key} samples: {len(val)}")
print()


#   CREATING TEST NOISE AND SIGNAL    #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []
Corresponding_B_strength = []
Corresponding_Main_Peak_strength = []
Corresponding_F_B = []
Corresponding_B_strength = []

for _ in tqdm(range(n_tests), desc="Generating test data"):

    # The base parameters:
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    # Create a tuple with all the randomized parameters
    params = rand_test(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params)

    real_F_B = params[2]
    real_B = params[1]

    if add_signals:
        clear_sig.append(Volt.tolist())
        sig_time.append(Signal.tolist())
    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())
    Corresponding_F_B.append(real_F_B)
    Corresponding_B_strength.append(real_B)

    # Corresponding_B_strength.append(fVolt[2000-real_F_B])
    # Corresponding_Main_Peak_strength.append(np.max(fVolt))

test = {
    "cSignal": clear_sig,
    "f_cSignal": clear_sig_f,
    "signals": sig_time,
    "f_signals": sig_f,
    "F_B": Corresponding_F_B,
    "B_strength": Corresponding_B_strength,
}

print(f"Finished generating test dataset with {len(test['f_cSignal'])} samples")
for key, val in test.items():
    print(f"Test {key} samples: {len(val)}")
print()

#   CREATING ONLY NOISE WITH SPECS OF TESTING #
clear_sig = []
clear_sig_f = []
sig_time = []
sig_f = []

n_noise = 0
for _ in tqdm(range(10), desc="Generating noise data"):

    params = rand_test(I0, B0, F_B,noise_strength)
    Volt, fVolt, Signal, fSignal, Time, freq = Signal_Noise_FFts(*params, only_noise=True)

    real_F_B = params[2]

    clear_sig_f.append(fVolt.tolist())
    sig_f.append(fSignal.tolist())


noise_for_test = {
    "f_cSignal": clear_sig_f,
    "f_noise": sig_f,
}
print(f"Finished generating noise dataset with {len(noise_for_test['f_cSignal'])} samples")
for key, val in noise_for_test.items():
    print(f"Noise {key} samples: {len(val)}")
print()

data = {
    "train": train,
    "validate": validate,
    "test": test,
    "noise_for_test": noise_for_test,
    "t": Time.tolist(),
    "f": freq.tolist()
}


#Writting to the data:
with open("Data/data.json", "w") as f:
    json.dump(data, f, indent=4)