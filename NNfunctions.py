import torch
import torch.nn as nn
from ignite.metrics.metric import reinit__is_reduced
from torch.utils.data import Dataset

from ignite.metrics import Metric
import json
import h5py
import numpy as np
from Simulation.numpy_ffts.fft_pink_noise import peak_heights

def scale_tensor(dat_raw, log=False, norm=False, minmax=False, tensor=False, resnet=False):
    sc_par = {}
    # print(f"data is {dat_raw.shape} - {min.shape} / ({max.shape} - {min.shape} ")
    if tensor:
        lib = torch
    else:
        lib = np

    if log:
        dat_raw = lib.log10(dat_raw)
        sc_par['log'] = True

    if norm:
        mean = lib.mean(dat_raw)
        std = lib.std(dat_raw)
        dat_raw = (dat_raw - mean)/std
        sc_par['norm'] = (mean, std)


    if minmax:
        min = lib.min(dat_raw)
        max = lib.max(dat_raw)
        dat_raw = (dat_raw - min)/(max - min)
        sc_par['minmax'] = (min, max)

    if resnet:
        #ResNet expects input in shape (batch_size, 1, n_length)
        # Reshape to (batch_size, 1, n_length)
        dat_raw = dat_raw.unsqueeze(1)
        sc_par['resnet'] = True


    return dat_raw, sc_par

def scale_like_tensor(dat_raw, params):
    '''
    This function applies the same scaling parameters as in scale_tensor.
    '''

    if not isinstance(dat_raw, torch.Tensor):
        dat_raw = torch.tensor(dat_raw, dtype=torch.float64)

    if params.get('log'):
        dat_raw = torch.log10(dat_raw)

    if params.get('norm'):
        mean, std = params.get('norm')
        dat_raw = (dat_raw - mean) / std

    if params.get('minmax'):
        min_val, max_val = params.get('minmax')
        dat_raw = (dat_raw - min_val) / (max_val - min_val)

    if params.get('resnet'):
        dat_raw = dat_raw.unsqueeze(1)

    return dat_raw

def unscale_tensor(procss_data, params):
    if params.get('minmax'):
        mn = params.get('minmax')[0]
        mx = params.get('minmax')[1]
        procss_data = procss_data * (mx - mn) + mn

    if params.get('norm'):
        # First multiply by std, then add the mean
        procss_data = (procss_data * params.get('norm')[1]) + params.get('norm')[0]


    if params.get('log'):
        procss_data = 10**procss_data

    if params.get('resnet'):
        # Reshape to (batch_size, n_length)
        procss_data = procss_data.squeeze()

    return procss_data


class SignalDataset(Dataset):
    def __init__(self, data_path, split="train", 
                 transfrom=scale_tensor, resnet=False, same_scale=False):
        # Determine file type based on extension
        if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
            # Load HDF5 data
            with h5py.File(data_path, 'r') as h5_file:
                sig_raw = h5_file[f'{split}/f_signals'][:]
                clean_raw = h5_file[f'{split}/f_cSignal'][:]
                self.F_B = h5_file[f'{split}/F_B'][:]
                self.B0 = h5_file[f'{split}/B0'][:] 
                self.f = torch.tensor(h5_file['f'][:], dtype=torch.float64)
                self.time = torch.tensor(h5_file['t'][:], dtype=torch.float64) if 't' in h5_file else None
            

            self.clean = clean_raw
            sig_raw = torch.tensor(sig_raw, dtype=torch.float64)
            clean_raw = torch.tensor(clean_raw, dtype=torch.float64)
            self.F_B = torch.tensor(self.F_B, dtype=torch.int32).unsqueeze(1)
            
        else:
            # Load JSON data (original implementation)
            with open(data_path, "r") as f:
                dat = json.load(f)
            sig_raw = dat[split]["f_signals"]
            clean_raw = dat[split]["f_cSignal"]
            self.clean = clean_raw
            self.F_B = dat[split]["F_B"]
            self.f = torch.tensor(dat["f"], dtype=torch.float32)

            sig_raw = torch.tensor(sig_raw, dtype=torch.float32)
            clean_raw = torch.tensor(clean_raw, dtype=torch.float32)
            self.F_B = torch.tensor(self.F_B, dtype=torch.int32).unsqueeze(1)

        amps = peak_heights(clean_raw, self.f, f_b=self.F_B, f_center=2000, dir=False)
        self.same_scale = same_scale

        self.X, self.X_scale = transfrom(sig_raw, log=True, norm=True, tensor=True, resnet=resnet)
        if self.same_scale:
            self.Y = scale_like_tensor(amps, self.X_scale)
            self.Y_scale = self.X_scale
        else:
            self.Y, self.Y_scale = transfrom(amps, log=True, tensor=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_freqs(self):
        return self.f
    
    def get_time_bins(self):
        return self.time if self.time is not None else None

    def get_peak_freqs(self, idx):
        return [2000 - self.F_B[idx].item(), 2000, 2000+self.F_B[idx].item()]

    def unscale(self):
        return self.X_scale, self.Y_scale

    def get_clean_sig(self, idx):
        return self.clean[idx]
    
    def get_B0(self, idx):
        return self.B0[idx] if hasattr(self, 'B0') else None
    
    


class MeanRelativeError(Metric):
    """
    Calculates the Mean Relative Error (MRE) between predicted and true values.

    Args:
        output_transform (callable, optional): A function to apply to the
            engine's process_function output to get the `y_pred` and `y` tensors.
            Default is `lambda x: x`, assuming the engine outputs `(y_pred, y)`.
        device (str or torch.device, optional): Specifies the device on which
            the metric's internal state should be kept. Defaults to CPU.
    """
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        super(MeanRelativeError, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        # Resets the metric's internal state.
        self._sum = None
        self._count = 0

    def update(self, output):
        """
                Updates the metric's state with the current batch's predictions and true values.

                Args:
                    output (tuple): A tuple containing `(y_pred, y_true)` from the engine's
                                    `output_transform`.
                """

        y_p, y_t = output[0].detach(), output[1].detach()

        eps = 1e-15
        # Calculate Rel error = Y_pred/Y_th
        # add eps so we don't divide by 0
        rel_error = torch.abs(y_p / (y_t + eps))  # shape: [batch, 3]

        if self._sum is None:
            # rel_error.size(1) = 3
            self._sum = torch.zeros(rel_error.size(1), device=rel_error.device)

        self._sum += rel_error.sum(dim=0)
        self._count += rel_error.size(0)

    def compute(self):
        return ( self._sum / self._count).cpu().numpy()  # shape: (3,)

class PeakComparison(Metric):
    # returns: real_peaks, predicted_peaks, differences
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._device = device
        self._batch_index = 0
        super(PeakComparison, self).__init__(output_transform=output_transform, device=device)


        self._real_peaks = None
        self._predicted_peaks = None
        self._differences = None

    def reset(self):
        # Initialize tensors to store batch results efficiently
        self._real_peaks = []
        self._predicted_peaks = []
        self._differences = []

    def update(self, output):
        """
        Updates the metric's state with the current batch's predictions and true values.

        Args:
            output (tuple): A tuple containing `(y_pred, y_true)` after applying output_transform.
        """
        y_p, y_t = output[0].detach(), output[1].detach()
        # y_t and y_p: [batch_size, 3]
        self._real_peaks.append(y_t)
        self._predicted_peaks.append(y_p)
        self._differences.append(y_p - y_t)

    def compute(self):
        real_peaks = torch.cat(self._real_peaks, dim=0)
        predicted_peaks = torch.cat(self._predicted_peaks, dim=0)
        differences = torch.cat(self._differences, dim=0)
        return real_peaks, predicted_peaks, differences


class MDEPerIntensity(Metric):
    """
    Calculates the Mean Difference Error (MDE) Per Intensity of magnetic field

    Args:
        output_transform (callable, optional): A function to apply to the
            engine's process_function output to get the `y_pred` and `y` tensors.
            Default is `lambda x: x`, assuming the engine outputs `(y_pred, y)`.
        device (str or torch.device, optional): Specifies the device on which
            the metric's internal state should be kept. Defaults to CPU.
    """
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        super(MDEPerIntensity, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        # Store all intensities and differences for plotting
        self._intensities = []
        self._preds = []
        self._diffs = []

    def update(self, output):
        """
        Updates the metric's state with the current batch's predictions and true values.

        Args:
            output (tuple): A tuple containing `(y_pred, y_true)`
            after applying output_transform.
        """
        y_p, y_t = output[0].detach(), output[1].detach()
        # y_t and y_p: [batch_size, 3]
        # Intensity is y_t[:, 0] (or y_t[:, 2])
        intensities = y_t
        preds = y_p
        diffs = (y_p - y_t)  # shape: [batch_size, 3]

        self._intensities.append(intensities)
        self._preds.append(preds)
        self._diffs.append(diffs)

    def compute(self):
        # Concatenate all batches
        intensities = torch.concatenate(self._intensities, dim=0)  # shape: [N, 3]
        preds = torch.concatenate(self._preds, dim=0)              # shape: [N, 3]
        diffs = torch.concatenate(self._diffs, dim=0)              # shape: [N, 3]
        return intensities, preds, diffs


class R2Score(Metric):
    """
    Calculates the R² (coefficient of determination) of the linear fit between predicted and true values.
        
    Args:
        multioutput (str, optional): Defines aggregation in the case of multiple output values.
            Can be 'uniform_average', 'raw_values', or 'variance_weighted'. 
            Default is 'uniform_average'.
    """
    def __init__(self, output_transform=lambda x: x, device="cpu", multioutput='raw_values'):
        super(R2Score, self).__init__(output_transform=output_transform, device=device)
        self.multioutput = multioutput
        
    def reset(self):
        self._y_true = []
        self._y_pred = []
        
    def update(self, output):        
        y_pred, y_true = output[0].detach(), output[1].detach()
        
        self._y_true.append(y_true)
        self._y_pred.append(y_pred)
        
    def compute(self):
        """
        Computes the R² score between the predicted and true values.
        
        Returns:
            torch.Tensor: The R² score. If multioutput is 'raw_values', returns
                         a tensor with R² scores for each output dimension.
        """
        y_true = torch.cat(self._y_true, dim=0)
        y_pred = torch.cat(self._y_pred, dim=0)
        
        # Calculate R²
        y_true_mean = torch.mean(y_true, dim=0)
        
        # Total sum of squares
        ss_tot = torch.sum((y_true - y_true_mean) ** 2, dim=0)
        
        # Residual sum of squares
        ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
        
        r2 = 1 - (ss_res / ss_tot)
        
        # Handle multioutput
        if self.multioutput == 'raw_values':
            return r2
        elif self.multioutput == 'uniform_average':
            return torch.mean(r2)
        elif self.multioutput == 'variance_weighted':
            weights = ss_tot / torch.sum(ss_tot)
            return torch.sum(r2 * weights)
        else:
            raise ValueError(f"Invalid multioutput option: {self.multioutput}")


def score_function(engine):
    # Lower loss is better, so return negative loss
    return -engine.state.metrics['loss']

def r2_score_function(engine):
    # Higher R^2 is better, return R^2 directly
    return engine.state.metrics['r2_score'][0]
