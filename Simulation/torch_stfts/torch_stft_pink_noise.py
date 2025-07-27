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

def make_fft(data, rate, num, device='cpu'):
    Fs = rate  # Sampling frequency
    L = num  # Length of signal

    # Convert to tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float64, device=device)
    elif data.device.type != device:
        data = data.to(device)

    # Ensure data has batch dimension if it doesn't already
    if data.dim() == 1:
        data = data.unsqueeze(0)
        
    # Compute the Fourier transform of the signal
    Y = torch.fft.fft(data)
    
    # Compute the two-sided spectrum P2
    P2 = torch.abs(Y / L)
    
    # Compute the single-sided spectrum P1
    P1 = P2[:, :L // 2 + 1]
    
    # Double the amplitudes except DC and Nyquist
    P1[:, 1:-1] = 2 * P1[:, 1:-1]

    # Define the frequency domain f with batch dimension
    batch_size = data.shape[0]
    freq_bins = L // 2 + 1
    f = Fs * torch.arange(0, freq_bins, device=device) / L
    
    # Expand frequency array to match batch dimension [batch_size, freq_bins]
    f = f.unsqueeze(0).expand(batch_size, -1)

    # Remove batch dimension if input was not batched
    if data.shape[0] == 1 and not isinstance(rate, torch.Tensor):
        P1 = P1.squeeze(0)
        f = f.squeeze(0)
        
    return f, P1

def make_pink_noise(t, sigma, device='cpu'):
    L = len(t)
    d = t[1] - t[0]
    rate = 1/d
    
    # Create white noise directly as a tensor on the specified device
    white_noise = torch.normal(0, sigma, size=(L,), device=device)

    # Perform FFT using PyTorch
    fwNoise = torch.fft.rfft(white_noise)
    freq = torch.arange(0, int(L/2) + 1, device=device) * (rate/L)

    # Avoid division by zero at DC
    freq[0] = freq[1]
    
    # Apply pink noise filter in frequency domain
    fwNoise = fwNoise / torch.sqrt(freq)

    # Inverse FFT to get back to time domain
    pink_noise = torch.fft.irfft(fwNoise, n=L)

    # Normalize to zero mean
    pink_noise -= pink_noise.mean()

    return pink_noise
def make_noise(Time, n_power, p_perc, device='cpu'):
    n = len(Time)
    
    # Convert Time to tensor if it's not already
    if not isinstance(Time, torch.Tensor):
        Time = torch.tensor(Time, dtype=torch.float32, device=device)
    elif Time.device.type != device:
        Time = Time.to(device)
    
    # Convert inputs to tensors and handle batch dimension
    if not isinstance(n_power, torch.Tensor):
        n_power = torch.tensor(n_power, device=device)
    
    if not isinstance(p_perc, torch.Tensor):
        p_perc = torch.tensor(p_perc, device=device)
    
    # Get batch size if any
    batch_size = max(n_power.shape[0] if n_power.dim() > 0 else 1, 
                        p_perc.shape[0] if p_perc.dim() > 0 else 1)
    
    # Ensure both have batch dimension
    if n_power.dim() == 0:
        n_power = n_power.expand(batch_size)
    if p_perc.dim() == 0:
        p_perc = p_perc.expand(batch_size)
    
    # Calculate variance for each item in the batch
    Noise_variance = var(n_power)
    
    # Create white noise for each batch item
    wNoise = torch.stack([torch.normal(0.0, np, size=(n,), device=device) 
                            for np in n_power])
    
    # Create pink noise for each batch item
    pNoise = torch.stack([make_pink_noise(Time, nv, device=device) 
                            for nv in Noise_variance])
    
    # Combine white and pink noise based on percentage for each batch
    p_perc = p_perc.view(-1, 1)  # Reshape for broadcasting
    Noise = p_perc * pNoise + (1 - p_perc) * wNoise
    
    return Noise, pNoise, wNoise

def rand_train(I0, B0, F_B, noise_strength, device='cpu', batch_size=1):
    # Create random multipliers as tensors directly on the specified device
    I0_mult = torch.rand(batch_size, device=device) * 2 + 0.5  # uniform between 0.5 and 2.5
    B0_mult = torch.rand(batch_size, device=device) * 6 + 1     # uniform between 1 and 7
    F_B_offsets = torch.randint(F_B - 5, F_B + 21, (batch_size,), device=device)  # inclusive lower, exclusive upper bound
    noise_mult = torch.rand(batch_size, device=device) * 5  # uniform between 0 and 5

    # Apply the multipliers
    I0_r = I0 * I0_mult
    B0_r = B0 * B0_mult
    F_B_r = F_B_offsets
    noise_strength_r = noise_strength * noise_mult
    pink_percentage = torch.zeros(batch_size, device=device)
    
    return I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage


def rand_test(I0, B0, F_B, noise_strength, device='cpu', batch_size=1):
    # Create random values as tensors directly on the specified device
    I0_noise = torch.normal(0, var(0.1*I0), size=(batch_size,), device=device)
    B0_mult = torch.rand(batch_size, device=device) * 4 + 1  # uniform between 1 and 5
    F_B_offsets = torch.randint(F_B - 5, F_B + 6, (batch_size,), device=device)  # inclusive lower, exclusive upper bound
    
    # Apply the values
    I0_r = I0 + I0_noise
    B0_r = B0 * B0_mult
    F_B_r = F_B_offsets
    noise_strength_r = torch.full((batch_size,), noise_strength, device=device)
    pink_percentage = torch.zeros(batch_size, device=device)
    
    return I0_r, B0_r, F_B_r, noise_strength_r, pink_percentage

def generate_voltage_signal(I0, B0, F_B, dt, start_time, end_time, device='cpu'):
    rho_perp = 2.7e-7  # [ohm * m]
    AMRR = 0.02  # AMR ratio (rho_par-rho_perp)/rho_perp
    B_k = 10e-4  # [T] Difference of resistance for parallel B vs Perp B

    # Isotropy Field
    thickness = 200e-9  # [m] Thickness of magnetic layer
    LL = 10e-6  # The misalignment [m]
    WW = 600e-6  # The width of the sensor [m]
    F_c = 2000  # The frequency of the current in the sensor [Hz]

    # Create time vector as tensor directly on the specified device
    Time = torch.arange(start_time, end_time, dt, dtype=torch.float64, device=device)

    # Calculate voltage signal using tensors
    delta_rho = rho_perp * AMRR  # [ohm * m]
    
    # Convert to tensors if not already
    if not isinstance(I0, torch.Tensor):
        I0 = torch.tensor(I0, dtype=torch.float64, device=device)
    elif I0.device.type != device:
        I0 = I0.to(device)
        
    if not isinstance(B0, torch.Tensor):
        B0 = torch.tensor(B0, dtype=torch.float64, device=device)
    elif B0.device.type != device:
        B0 = B0.to(device)
        
    if not isinstance(F_B, torch.Tensor):
        F_B = torch.tensor(F_B, dtype=torch.float64, device=device)
    elif F_B.device.type != device:
        F_B = F_B.to(device)
    
    # Ensure all inputs have batch dimension
    if I0.dim() == 0:
        I0 = I0.unsqueeze(0)
    if B0.dim() == 0:
        B0 = B0.unsqueeze(0)
    if F_B.dim() == 0:
        F_B = F_B.unsqueeze(0)
    
    # Get batch size
    batch_size = max(I0.shape[0], B0.shape[0], F_B.shape[0])
    
    # Broadcast to same batch size if needed
    if I0.shape[0] != batch_size:
        I0 = I0.expand(batch_size)
    if B0.shape[0] != batch_size:
        B0 = B0.expand(batch_size)
    if F_B.shape[0] != batch_size:
        F_B = F_B.expand(batch_size)
    
    # Calculate the signal components with proper broadcasting
    II = I0.unsqueeze(-1) * torch.sin(2 * torch.pi * F_c * Time)  # Current signal
    BB = B0.unsqueeze(-1) * torch.sin(2 * torch.pi * F_B.unsqueeze(-1) * Time)  # Magnetic field signal
    hh = BB / B_k
    
    # Calculate voltage with batch as first dimension
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
                        device='cpu'):
    # I0 The current amplitude in the sensor[A]
    # B0 = The magnetic field on the sensor [T]
    # F_B  The magnetic field frequency [Hz]
    # noise_strength = 2.5e-5  # Noise will be 95% of times in this +-range

    # Create a voltage signal from a PHE sensor
    Voltage, Time = generate_voltage_signal(I0, B0, F_B, dt, start_time, end_time, device=device)

    # If only noise is needed, return it without FFT
    if only_noise:
        Voltage = torch.zeros_like(Voltage)

    if is_window:
        # Window the signal to prevent spectral leakage when padding
        window = torch.hann_window(Voltage.shape[-1], device=device)
        Voltage = window * Voltage

    if is_padding:
        # Padding so the peaks aren't as sharp
        n = torch.ceil(torch.log2(torch.tensor(Voltage.shape[-1], device=device)))
        padding_size = Voltage.shape[-1] * 4
        Voltage = torch.nn.functional.pad(Voltage, (0, padding_size))

    f, P1 = make_fft(Voltage, 1 / dt, Voltage.shape[-1], device=device)

    # CREATING NOISES
    Noise, _, _= make_noise(Time, noise_strength, pink_percentage, device=device)

    # COMBINED SIGNAL
    # Handle batched tensors properly
    if Noise.dim() > Voltage.dim():
        # If Noise has more dimensions than Voltage, ensure Voltage has batch dimension
        Signal = Noise + Voltage.unsqueeze(0)
    elif Noise.dim() < Voltage.dim():
        # If Voltage has more dimensions than Noise, ensure Noise is expanded properly
        # Squeeze only if Noise has unnecessary dimensions
        if Noise.dim() == 2:
            Signal = Noise + Voltage
        else:
            Signal = Noise.squeeze(0) + Voltage
    else:
        # Same number of dimensions, simple addition
        Signal = Noise + Voltage

    if is_window:
        # Window the signal to prevent spectral leakage when padding
        window = torch.hann_window(Signal.shape[-1], device=device)
        Signal = window * Signal

    if is_padding:
        # Padding so the peaks aren't as sharp
        n = torch.ceil(torch.log2(torch.tensor(Signal.shape[-1], device=device)))
        padding_size = Signal.shape[-1] * 4
        Signal = torch.nn.functional.pad(Signal, (0, padding_size))

    _, fSignal = make_fft(Signal, 1/dt, Signal.shape[-1], device=device)

    return Voltage, P1, Signal, fSignal, Time, f


def my_stft(I0, B0, F_B, noise_strength,
            fs, total_cycles, overlap=0.75, cycles_per_window=4,
            Tperiod=None,
            only_center=False,
            is_clean=False,
            only_noise=False,
            device='cpu'):

    dt = 1 / fs
    freqs = [2000 - F_B, 2000, 2000 + F_B]  # Frequencies of interest

    # Period is 1 / GCD of all frequency components
    if Tperiod is None:
        # Compute GCD - for batch processing, we'll use a common period
        # This is a simplification - ideally each batch item would have its own period
        if isinstance(F_B, torch.Tensor):
            if F_B.dim() > 0:
                # For batch, use the first item's F_B value for period calculation
                # This is a compromise - a more accurate approach would calculate different 
                # periods for each batch item, but that would complicate the implementation
                F_B_val = F_B[0].item()
            else:
                F_B_val = F_B.item()
        else:
            F_B_val = F_B
        freq_gcd = reduce(gcd, [2000 - F_B_val, 2000, 2000 + F_B_val])
        Tperiod = 1 / freq_gcd

    end_time = Tperiod * total_cycles

    Voltage, _, Signal, _, Time, freqs_stft = Signal_Noise_FFts(
        I0, B0, F_B, noise_strength,
        dt=dt,
        start_time=0,
        end_time=end_time,
        only_noise=only_noise,
        device=device
    )

    if is_clean:
        # If only the clean signal is needed, return it without noise
        Signal = Voltage

    N_samples_period = int(fs * Tperiod)  # Number of samples in one period
    n_fft = N_samples_period * cycles_per_window

    # calculate hop length based on overlap percentage
    hop_length = int(n_fft * (1 - overlap))

    # Create window tensor on the specified device
    window = torch.ones(n_fft, device=device)
    
    # Check if Signal has batch dimension
    if Signal.dim() == 1:
        Signal = Signal.unsqueeze(0)  # Add batch dimension if missing
    
    stft_result = torch.stft(Signal, n_fft=n_fft, hop_length=hop_length,
                             return_complex=True,
                             window=window,
                             center=False)

    mag = stft_result.abs() / n_fft

    # Frequency axis for STFT as tensor
    freqs_stft = torch.linspace(0, fs // 2, n_fft // 2 + 1, device=device)
    time_bins = torch.linspace(0, total_cycles * Tperiod, mag.shape[2], device=device)

    if only_center:
        # Return only around the center frequency (2000 Hz)
        freq_mask = (freqs_stft >= 1950) & (freqs_stft <= 2050)
        freqs_zoom = freqs_stft[freq_mask]
        
        # Handle batch dimension properly
        if mag.dim() == 3:  # [batch_size, freq_bins, time_bins]
            mag_zoom = mag[:, freq_mask, :]
        else:
            # Legacy case for single item
            mag_zoom = mag[0, freq_mask, :]

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

def peak_heights_numpy(clear_signal, f_b, f_center, dir=False):
    freqs = [f_center - f_b, f_center, f_center + f_b]
    return clear_signal[freqs]


def peak_heights(clear_signal, freqs, f_b, f_center, dir=False):
    # Receives a Tensor, and returns the height of each of the peaks in the signal
    # Needs to receive the clean signals in a dataset form (n_samps x L)

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
