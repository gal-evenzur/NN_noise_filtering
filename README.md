# Neural Network Noise Filtering for Magnetic Field Sensor Signal Processing

This repository contains a deep learning research project focused on using neural networks to filter noise and detect peak heights in magnetic field sensor signals. The project implements multiple neural network architectures (FCNN, ResNet1D, ResNet2D) to process both FFT and STFT representations of sensor data.

## Project Overview

The project simulates and processes signals from a PHE (Planar Hall Effect) magnetic field sensor. The main goal is to accurately predict peak heights in the frequency domain of noisy sensor signals using various deep learning approaches.

### Key Features
- **Signal Simulation**: Generate realistic magnetic field sensor signals with configurable noise
- **Multiple NN Architectures**: FCNN, 1D ResNet, and 2D ResNet implementations
- **FFT & STFT Processing**: Both frequency domain and time-frequency domain analysis
- **Noise Filtering**: Advanced noise reduction using neural networks
- **Peak Detection**: Accurate detection and height measurement of frequency peaks

## Repository Structure

```
.
├── Data/                           # Datasets
│   ├── data_stft.h5               # STFT dataset for 2D processing
│   ├── data_stft.json             # STFT metadata
│   └── data.json                  # FFT dataset for 1D processing
├── Simulation/                     # Signal generation and simulation
│   ├── numpy_ffts/                # NumPy-based FFT signal generation
│   │   ├── dataCreation.py        # Dataset creation script
│   │   └── fft_pink_noise.py      # Core signal generation functions
│   └── torch_stfts/               # PyTorch-based STFT signal generation
│       ├── stftCreation.py        # STFT dataset creation
│       ├── torch_stft_pink_noise.py # STFT signal processing
│       └── stft_Testing_Times.py  # STFT testing and validation
├── NN_structures/                  # Neural network architectures
│   └── resnet1d_master/           # Pre-trained 1D ResNet implementation
├── Piplot/                        # Visualization and plotting utilities
├── ppt/                           # Project presentations
├── FCNN_Binary_MSE.py             # Fully Connected NN with MSE loss
├── FCNN_Simple.py                 # Simple FCNN for noise reduction
├── resnet2D.py                    # 2D ResNet for STFT processing
├── ResnetImp.py                   # 1D ResNet implementation
├── NNfunctions.py                 # Core neural network utilities
└── plot_sim.py                    # Simulation visualization
```

## Neural Network Architectures

### 1. Fully Connected Neural Networks (FCNN)
- **`FCNN_Simple.py`**: Basic noise reduction network
- **`FCNN_Binary_MSE.py`**: Peak height regression with MSE loss
- **Input**: 1D FFT of sensor signals
- **Output**: Either denoised signal or peak height predictions

### 2. 1D ResNet (`ResnetImp.py`)
- **Architecture**: ResNet adapted for 1D signal processing
- **Input**: 1D FFT sequences
- **Output**: Peak height predictions (3 values: left, center, right peaks)
- **Features**: Skip connections, batch normalization, early stopping

### 3. 2D ResNet (`resnet2D.py`)
- **Architecture**: Modified ResNet18 for single-channel STFT data
- **Input**: 2D STFT spectrograms (frequency × time)
- **Output**: Peak height predictions
- **Features**: Pre-trained weights adaptation, frozen/unfrozen training options

## Signal Processing Pipeline

### 1. Signal Generation
The project simulates PHE sensor signals with the following characteristics:
- **Sensor Parameters**: Current amplitude (I₀), magnetic field strength (B₀), frequency (F_B)
- **Noise Types**: Pink noise, white noise, with configurable strength
- **Signal Components**: Three frequency peaks around 2000 Hz (left, center, right)

### 2. Data Processing Options

#### FFT Processing (1D)
```python
# Generate voltage signal
Voltage, Time = generate_voltage_signal(I0, B0, F_B, dt, start_time, end_time)
# Add noise and compute FFT
f, P1 = make_fft(Voltage + Noise, sampling_rate, length)
```

#### STFT Processing (2D)
```python
# Generate STFT for time-frequency analysis
magnitude, freqs, time_bins = my_stft(I0, B0, F_B, noise_strength,
                                     fs=10000, total_cycles=100,
                                     overlap=0.85, cycles_per_window=5)
```

## Key Components

### Core Utilities (`NNfunctions.py`)
- **Data Scaling**: Log scaling, normalization, min-max scaling
- **Dataset Classes**: `SignalDataset` for loading and preprocessing
- **Custom Metrics**: Mean Relative Error, Mean Deviation Error per Intensity
- **Tensor Operations**: Scaling/unscaling for different data types

### Training Infrastructure
- **PyTorch Ignite**: Training engines, validation, early stopping
- **Metrics**: L1 Loss (MAE), custom relative error metrics
- **Visualization**: Loss tracking, prediction comparison plots

## Usage

### 1. Data Generation
```bash
# Generate FFT dataset
python Simulation/numpy_ffts/dataCreation.py

# Generate STFT dataset
python Simulation/torch_stfts/stftCreation.py
```

### 2. Training Models
```bash
# Train FCNN
python FCNN_Binary_MSE.py

# Train 1D ResNet
python ResnetImp.py

# Train 2D ResNet
python resnet2D.py
```

### 3. Visualization
```bash
# Plot simulation results
python Simulation/plot_sim.py

# Visualize STFT processing
python Simulation/torch_stfts/stft_Testing_Times.py
```

## Model Performance Metrics

The project uses several custom metrics to evaluate model performance:

1. **Mean Absolute Error (MAE)**: Primary loss function
2. **Mean Relative Error**: Percentage error relative to true values
3. **Mean Deviation Error per Intensity**: Error analysis across different magnetic field strengths

## Hardware Requirements

- **GPU Support**: CUDA-enabled GPU recommended for faster training
- **Memory**: Sufficient RAM for large datasets (STFT data can be memory-intensive)
- **Storage**: Several GB for generated datasets

## Dependencies

- PyTorch + torchvision
- PyTorch Ignite (training utilities)
- NumPy, SciPy
- Matplotlib (visualization)
- h5py (HDF5 data storage)
- tqdm (progress bars)

## Research Context

This project is part of research into improving magnetic field sensor signal processing using deep learning techniques. The work explores different neural network architectures for noise filtering and peak detection in frequency domain representations of sensor data.

### Key Research Questions
1. Which neural network architecture performs best for magnetic sensor noise filtering?
2. How do FFT vs. STFT representations affect model performance?
3. What is the optimal balance between model complexity and performance?

## Contributing

When extending this project:
1. Follow the existing code structure and naming conventions
2. Add comprehensive documentation for new functions
3. Include visualization capabilities for new models
4. Update this README with any new features or architectures

## File Naming Convention

- `FCNN_*.py`: Fully Connected Neural Network implementations
- `*resnet*.py`: ResNet-based architectures
- `*Creation.py`: Dataset generation scripts
- `*functions.py`: Utility and helper functions
- `plot_*.py`: Visualization scripts
