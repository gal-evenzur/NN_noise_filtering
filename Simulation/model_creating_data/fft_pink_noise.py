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

def make_fft(data, rate, num, use_torch=False):
    Fs = rate  # Sampling frequency
    L = num  # Length of signal

    if use_torch:
        fft_fn = torch.fft.fft
        backend = torch
        # PyTorch requires explicit dtype for arange
        arange_fn = lambda start, stop: torch.arange(start, stop, dtype=torch.float32)
    else:
        fft_fn = np.fft.fft
        backend = np
        arange_fn = np.arange

    # Compute the Fourier transform of the signal
    Y = fft_fn(data)

    # Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1
    # based on P2 and the even-valued signal length L
    P2 = backend.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]  # Double the amplitudes except DC and Nyquist

    # Define the frequency domain f
    f = Fs * arange_fn(0, L // 2 + 1) / L

    return f, P1

def make_pink_noise(t, sigma, use_torch=False):
    L = len(t)
    d = t[1] - t[0]
    rate = 1/d

    if use_torch:
        random_normal = lambda _, std, size: torch.randn(size) * std
        rfft = torch.fft.rfft
        irfft = torch.fft.irfft
        arange = lambda start, stop: torch.arange(start, stop, dtype=torch.float32)
        mean = torch.mean
    else:
        random_normal = np.random.normal
        rfft = np.fft.rfft
        irfft = np.fft.irfft
        arange = np.arange
        mean = np.mean

    white_noise = random_normal(0, sigma, L)

    fwNoise = rfft(white_noise)
    freq = rate/L * arange(0, int(L/2) + 1)


    freq[0] = freq[1]  # Avoid division by zero at DC
    fwNoise = fwNoise / (freq)**0.5

    pink_noise = irfft(fwNoise)

    # 6. Normalize to zero mean and unit variance (optional but useful)
    pink_noise -= mean(pink_noise)

    return pink_noise

def make_noise(Time, n_power, p_perc, use_torch=False):
    n = len(Time)

    #   CREATING NOISES     #
    Noise_variance = var(n_power)

    if use_torch:
        wNoise = torch.randn(n) * Noise_variance
    else:
        wNoise = np.random.normal(0, Noise_variance, n)
        wNoise = np.random.randn(n) * Noise_variance

    # Using filtering in DFT Domain
    pNoise = make_pink_noise(Time, Noise_variance, use_torch=use_torch)

    Noise = p_perc * pNoise + (1 - p_perc) * wNoise

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


def generate_voltage_signal(I0, B0, F_B, dt, start_time, end_time, use_torch=False):
    rho_perp = 2.7e-7  # [ohm * m]
    AMRR = 0.02  # AMR ratio (rho_par-rho_perp)/rho_perp
    B_k = 10e-4  # [T] Difference of resistance for parallel B vs Perp B

    # Isotropy Field
    thickness = 200e-9  # [m] Thickness of magnetic layer
    LL = 10e-6  # The misalignment [m]
    WW = 600e-6  # The width of the sensor [m]
    F_c = 2000  # The frequency of the current in the sensor [Hz]

    backend = torch if use_torch else np

    # Time vector
    Time = backend.arange(start_time, end_time, dt)

    # Calculate voltage signal
    delta_rho = rho_perp * AMRR  # [ohm * m]
    BB = B0 * backend.sin(2 * np.pi * F_B * Time)
    II = I0 * backend.sin(2 * np.pi * F_c * Time)
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
                        only_noise=False,
                        use_torch=False):
    #I0 The current amplitude in the sensor[A]
    #B0 = The magnetic field on the sensor [T]
    #F_B  The magnetic field frequency [Hz]
    #noise_strength = 2.5e-5  # Noise will be 95% of times in this +-range

    if use_torch:
        zeros_like = torch.zeros_like
        hanning = torch.hann_window
        pad = lambda data, pad_width: torch.nn.functional.pad(data, (0, pad_width))
    else:
        zeros_like = np.zeros_like
        hanning = np.hanning
        pad = lambda data, pad_width: np.pad(data, (0, pad_width))

    #    CONSTS      #
    # Create a voltage signal from a PHE sensor

    Voltage, Time = generate_voltage_signal(I0, B0, F_B, dt, start_time, end_time, use_torch=use_torch)

    # If only noise is needed, return it without FFT
    if only_noise:
        Voltage = zeros_like(Voltage)

    if is_window:
        # Window the signal to prevent spectal leakage when padding
        window = hanning(len(Voltage))
        Voltage = window * Voltage

    if is_padding:
        # Padding so the peaks aren't as sharp
        pad_amount = len(Voltage) * 4
        Voltage = pad(Voltage, pad_amount)


    f, P1 = make_fft(Voltage, 1 / dt, len(Voltage), use_torch=use_torch)

    #   CREATING NOISES     #
    Noise, *_ = make_noise(Time, noise_strength, pink_percentage, use_torch=use_torch)


    #   COMBINED SIGNAL   #
    Signal = Noise + Voltage

    if is_window:
        # Window the signal to prevent spectal leakage when padding
        window = hanning(len(Signal))
        Signal = window * Signal

    if is_padding:
        # Padding so the peaks aren't as sharp
        pad_amount = len(Signal) * 4
        Signal = pad(Signal, pad_amount)

    _, fSignal = make_fft(Signal, 1/dt, len(Signal), use_torch=use_torch)

    return Voltage, P1, Signal, fSignal, Time, f


def my_stft(I0, B0, F_B, noise_strength,
            fs, total_cycles, overlap=0.75, cycles_per_window=4,
            Tperiod=None,
            only_center=False,
            only_noise=False,
            use_torch=True):

    dt = 1 / fs
    freqs = [2000 - F_B, 2000, 2000 + F_B]  # Frequencies of interest

    # Period is 1 / GCD of all frequency components
    if Tperiod is None:
        # Compute GCD
        freq_gcd = reduce(gcd, freqs)
        Tperiod = 1 / freq_gcd


    end_time = Tperiod * total_cycles

    _, _, Signal, _, Time, _ = Signal_Noise_FFts(I0, B0, F_B, noise_strength,
                               dt=dt,
                               start_time=0,
                               end_time=end_time,
                               use_torch=use_torch)
    if not use_torch:
        Signal = torch.from_numpy(Signal).to(torch.float32)

    N_samples_period = int(fs * Tperiod)  # Number of samples in one period
    n_fft = N_samples_period * cycles_per_window

    # calculate hop length based on overlap percentage
    hop_length = int(n_fft * (1 - overlap))

    stft_result = torch.stft(Signal, n_fft=n_fft, hop_length=hop_length,
                             return_complex=True,
                             center=False)

    mag = stft_result.abs()

    # Frequency axis for STFT
    freqs_stft = np.linspace(0, fs // 2, n_fft // 2 + 1)
    time_bins = np.linspace(0, total_cycles * Tperiod, mag.shape[1])

    if only_center:
        # Return only around the center frequency (2000 Hz)

        freq_mask = (freqs_stft >= 1900) & (freqs_stft <= 2100)
        freqs_zoom = freqs_stft[freq_mask]
        mag_zoom = mag[freq_mask, :].numpy()

        return mag_zoom, freqs_zoom, time_bins

    return mag.numpy(), freqs_stft, time_bins


def peak_heights(clear_signal, f_b, f_center, dir=False):
    # Receives a Tensor, and returns the height of each of the peaks in the signal
    # Needs to receive the clean signals in a dataset form (n_samps x L)
    if type(clear_signal) == np.ndarray:
        freqs = [f_center - f_b, f_center, f_center + f_b]
        return clear_signal[freqs]

    n_samp = len(clear_signal)

    l = f_center - f_b
    m = torch.full_like(l, 2000)
    r = f_center + f_b

    # Concatenate indices: shape (n_samp, 3)
    indices = torch.cat([l, m, r], dim=1)  # (n_samp, 3)

    # Create row indices: [0, 1, ..., n_samp-1] → shape (n_samp, 1), then expand to (n_samp, 3)
    row_indices = torch.arange(n_samp).unsqueeze(1).expand(-1, 3)

    # Use advanced indexing to extract the values
    result = clear_signal[row_indices, indices]  # shape: (n_samp, 3)

    peaks = {
        'left': result[:, 0],
        'center': result[:, 1],
        'right': result[:, 2],
    }

    if dir:
        return peaks
    else:
        return result
