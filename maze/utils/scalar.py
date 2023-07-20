'''
MIT License

Copyright (c) 2022 Yi-hao Sun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/utils/scaler.py

import numpy as np
import os.path as path
import torch


class StandardScaler(object):
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu
    
    def save_scaler(self, save_path):
        mu_path = path.join(save_path, "mu.npy")
        std_path = path.join(save_path, "std.npy")
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)
    
    def load_scaler(self, load_path):
        mu_path = path.join(load_path, "mu.npy")
        std_path = path.join(load_path, "std.npy")
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)

    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data.cpu().numpy())
        data = torch.tensor(data, device=device)
        return data