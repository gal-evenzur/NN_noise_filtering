import random
from math import gcd
from functools import reduce
import numpy as np
import torch
from numpy.random import normal as normal

def var(reducer):
    # White noise will fall 95% of the time between +- 1.96 * sigma
    # Returns Variance for which White noise will fall 95% between +- reducer
    return reducer/1.96

def make_fft(data, rate, num):
    Fs = rate  # Sampling frequency
    L = num  # Length of signal

    # Compute the Fourier transform of the signal
    Y = np.fft.fft(data)

    # Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1
    # based on P2 and the even-valued signal length L
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]  # Double the amplitudes except DC and Nyquist

    # Define the frequency domain f
    f = Fs * np.arange(0, L // 2 + 1) / L

    return f, P1

def make_pink_noise(t, sigma):
    L = len(t)
    d = t[1] - t[0]
    rate = 1/d
    white_noise = np.random.normal(0, sigma, L)

    fwNoise = np.fft.rfft(white_noise)
    freq = rate/L * np.arange(0, int(L/2) + 1)


    freq[0] = freq[1]  # Avoid division by zero at DC
    fwNoise = fwNoise / (freq)**0.5

    pink_noise = np.fft.irfft(fwNoise)

    # 6. Normalize to zero mean and unit variance (optional but useful)
    pink_noise -= np.mean(pink_noise)

    return pink_noise

def make_noise(Time, n_power, p_perc):
    n = np.size(Time)

    #   CREATING NOISES     #
    Noise_variance = var(n_power)

    wNoise = np.random.normal(0, Noise_variance, n)
    # _, fwNoise = make_fft(wNoise, Time)
    # _, S_wNoise = welch(wNoise, 1/dt, nperseg=n)

    # Using filtering in DFT Domain
    pNoise = make_pink_noise(Time, Noise_variance)
    # _, S_pNoise = welch(pNoise, 1/dt, nperseg=n)
    # _, fpNoise = make_fft(pNoise, Time)

    # Pink Using a library
    # pNoise = colorednoise.powerlaw_psd_gaussian(1, n) * Noise_var

    Noise = p_perc * pNoise + (1 - p_perc) * wNoise
    # _, fNoise = make_fft(Noise, Time)

    return Noise, pNoise, wNoise

def rand_train(I0, B0, F_B, noise_strength):
    I0_r = I0 * random.uniform(0.1, 5)
    B0_r = B0 * random.uniform(1, 20)
    F_B_r = random.randint(F_B - 5, F_B + 20)
    noise_strength_r = noise_strength*random.uniform(0.5, 1.4)
    pink_percentage = 0

    return I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage


def rand_test(I0, B0, F_B, noise_strength):
    I0_r = I0 + normal(0, var(0.1*I0))
    B0_r = B0 * random.uniform(0.25, 3)
    F_B_r = random.randint(F_B - 5, F_B + 5)
    noise_strength_r = noise_strength
    pink_percentage = 0

    return I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage


def generate_voltage_signal(I0, B0, F_B, dt, start_time, end_time):
    rho_perp = 2.7e-7  # [ohm * m]
    AMRR = 0.02  # AMR ratio (rho_par-rho_perp)/rho_perp
    B_k = 10e-4  # [T] Difference of resistance for parallel B vs Perp B

    # Isotropy Field
    thickness = 200e-9  # [m] Thickness of magnetic layer
    LL = 10e-6  # The misalignment [m]
    WW = 600e-6  # The width of the sensor [m]
    F_c = 2000  # The frequency of the current in the sensor [Hz]

    # Time vector
    Time = np.arange(start_time, end_time, dt)

    # Calculate voltage signal
    delta_rho = rho_perp * AMRR  # [ohm * m]
    BB = B0 * np.sin(2 * np.pi * F_B * Time)
    II = I0 * np.sin(2 * np.pi * F_c * Time)
    hh = BB / B_k

    Voltage = (II / thickness) * ((rho_perp + delta_rho - delta_rho * (hh ** 2)) * (LL / WW) + delta_rho * hh)
    return Voltage, Time


def Signal_Noise_FFts(I0, B0, F_B, noise_strength,
                        pink_percentage=0,
                        dt=1e-4,
                        start_time=0,
                        end_time=1,
                        is_padding=False,
                        is_window=False,
                        only_noise=False):
    #I0 The current amplitude in the sensor[A]
    #B0 = The magnetic field on the sensor [T]
    #F_B  The magnetic field frequency [Hz]
    #noise_strength = 2.5e-5  # Noise will be 95% of times in this +-range

    #    CONSTS      #
    # Create a voltage signal from a PHE sensor

    Voltage, Time = generate_voltage_signal(I0, B0, F_B, dt, start_time, end_time)

    # If only noise is needed, return it without FFT
    if only_noise:
        Voltage = np.zeros_like(Voltage)

    if is_window:
        # Window the signal to prevent spectal leakage when padding
        window = np.hanning(len(Voltage))
        Voltage = window * Voltage

    if is_padding:
        # Padding so the peaks aren't as sharp
        n = np.ceil(np.log2(len(Voltage)))
        Voltage = np.pad(Voltage, (0,len(Voltage)*4))


    f, P1 = make_fft(Voltage, 1 / dt, len(Voltage))

    #   CREATING NOISES     #
    Noise, *_ = make_noise(Time, noise_strength, pink_percentage)


    #   COMBINED SIGNAL   #
    Signal = Noise + Voltage

    if is_window:
        # Window the signal to prevent spectal leakage when padding
        window = np.hanning(len(Signal))
        Signal = window * Signal

    if is_padding:
        # Padding so the peaks aren't as sharp
        n = np.ceil(np.log2(len(Signal)))
        Signal = np.pad(Signal, (0,len(Signal)*4))

    _, fSignal = make_fft(Signal, 1/dt, len(Signal))

    return Voltage, P1, Signal, fSignal, Time, f


def my_stft(I0, B0, F_B, noise_strength,

            fs, total_cycles, overlap=0.75, cycles_per_window=4,

            Tperiod=None,
            only_center=False,
            is_clean=False,
            only_noise=False):

    dt = 1 / fs
    freqs = [2000 - F_B, 2000, 2000 + F_B]  # Frequencies of interest

    # Period is 1 / GCD of all frequency components
    if Tperiod is None:
        # Compute GCD
        freq_gcd = reduce(gcd, freqs)
        Tperiod = 1 / freq_gcd


    end_time = Tperiod * total_cycles

    Voltage, _, Signal, _, Time, _ = Signal_Noise_FFts(I0, B0, F_B, noise_strength,
                               dt=dt,
                               start_time=0,
                               end_time=end_time)

    Signal = torch.from_numpy(Signal)

    if is_clean:
        # If only the clean signal is needed, return it without noise
        Signal = torch.from_numpy(Voltage)

    N_samples_period = int(fs * Tperiod)  # Number of samples in one period
    n_fft = N_samples_period * cycles_per_window

    # calculate hop length based on overlap percentage
    hop_length = int(n_fft * (1 - overlap))

    stft_result = torch.stft(Signal, n_fft=n_fft, hop_length=hop_length,
                             return_complex=True,
                             window=torch.ones(n_fft),
                             center=False)

    mag = stft_result.abs()

    # Frequency axis for STFT
    freqs_stft = np.linspace(0, fs // 2, n_fft // 2 + 1)
    time_bins = np.linspace(0, total_cycles * Tperiod, mag.shape[1])

    if only_center:
        # Return only around the center frequency (2000 Hz)

        freq_mask = (freqs_stft >= 1950) & (freqs_stft <= 2050)
        freqs_zoom = freqs_stft[freq_mask]
        mag_zoom = mag[freq_mask, :]

        return mag_zoom, freqs_zoom, time_bins

    return mag, freqs_stft, time_bins

def dummy_stft_size(fs, total_cycles, overlap, cycles_per_window, Tperiod, only_center=True):

    # Dummy values
    I0 = 50e-3  # The current amplitude in the sensor[A]
    B0 = 4e-12  # The magnetic field on the sensor [T]
    F_B = 15  # The magnetic field frequency [Hz]
    noise_strength = 0.5e-10
    dummy, _, _ = my_stft(I0, B0, F_B, noise_strength,
           fs=fs,
           total_cycles=total_cycles,
           overlap=overlap,
           cycles_per_window=cycles_per_window,
           Tperiod=Tperiod,
           only_center=only_center)
    return dummy.shape


def peak_heights(clear_signal, freqs, f_b, f_center, dir=False):
    # Receives a Tensor, and returns the height of each of the peaks in the signal
    # Needs to receive the clean signals in a dataset form (n_samps x L)
    if type(clear_signal) == np.ndarray:
        freqs = [f_center - f_b, f_center, f_center + f_b]
        return clear_signal[freqs]

    n_samp = clear_signal.shape[0]
    if not isinstance(freqs, torch.Tensor):
        freqs = torch.from_numpy(freqs).to(clear_signal.device)
    # Ensure f_b and f_center are tensors
    if not isinstance(f_b, torch.Tensor):
        f_b = torch.tensor(f_b, device=clear_signal.device)
    if not isinstance(f_center, torch.Tensor):
        f_center = torch.tensor(f_center, device=clear_signal.device)


    # Calculate target frequencies for each sample in the batch
    l_freqs = (f_center - f_b)
    # Broadcast f_center to match the batch size of f_b
    m_freqs = f_center.expand_as(f_b)
    r_freqs = (f_center + f_b)

    target_freqs = torch.cat([l_freqs, m_freqs, r_freqs], dim=1) # (n_samp, 3)



    # Find the closest indices in the frequency axis for each target frequency
    # The shape of indices will be (n_samp, 3)
    indices = torch.argmin(torch.abs(freqs.unsqueeze(0) - target_freqs.unsqueeze(2)), dim=2)

    # Create row indices for advanced indexing
    row_indices = torch.arange(n_samp, device=clear_signal.device).unsqueeze(1).expand(-1, 3)

    # Use advanced indexing to extract the peak values
    if clear_signal.dim() == 3:  # STFT case: (n_samples, freq_bins, time_bins)
        result = clear_signal[row_indices, indices, 0]
    else:  # FFT case: (n_samples, freq_bins)
        # For FFT, we directly index into the 2D tensor
        result = torch.stack([clear_signal[i, indices[i]] for i in range(n_samp)])


    if dir:
        peaks = {
            'left': result[:, 0],
            'center': result[:, 1],
            'right': result[:, 2],
        }
        return peaks
    else:
        return result.squeeze()
