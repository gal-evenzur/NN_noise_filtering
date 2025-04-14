import numpy as np

def noise_variance(reducer):
    # White noise will fall 95% of the time between +- 1.96 * sigma
    # Returns Variance for which White noise will fall 95% between +- reducer
    return reducer/1.96


def make_fft(data, t):
    L = len(t)
    X_w = abs(np.fft.fft(data)/L)
    X_w2 = X_w[0:int(L/2) + 3]
    X_w2 = 2 * X_w2[1:-1]

    d = t[1] - t[0]
    rate = 1/d
    freq = rate/L * np.arange(0, int(L/2) + 1)

    return freq, X_w2


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
