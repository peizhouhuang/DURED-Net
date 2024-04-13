# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:59:22 2024

@author: peizhouh
"""
import os
import numpy as np
from util import read_training_data, r2c, c2r
import hdf5storage

def load_dataset(data_dir:str, num_images:int = 1000): # , training=False
    """
    Load Data and return image and k-space
    
    Parameters
    ----------
        data_dir (str): Dataset dir, load h5 file data with shape of [num_slice, height, width] or [num_slice, height, width, 2]

    Returns
    -------
        input_img (ndarray): train input [batch, height, width, 2] (seperate real & imag in last dim)
        input_spec (ndarray): k-space of train input [batch, height, width]
        input_mask (ndarray): sampling mask [batch, height, width]
        target_img (ndarray): train target [batch, height, width, 2]
    """
    print ('Loading dataset from', data_dir)
    input_img = read_training_data(data_dir, 'input_img', 'float32')
    input_spec = read_training_data(data_dir, 'input_spec', np.complex64)
    input_mask = read_training_data(data_dir, 'input_mask', 'float32')
    target_img = read_training_data(data_dir, 'target_img', 'float32')

    return input_img, input_spec, input_mask, target_img

def fftshift2d(x, ifft=False):
    #assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

def iterate_minibatches(input_img, input_spec, input_mask, target_img, batch_size, shuffle, args):
    """
    Iterator to generate batch of input data

    Parameters
    ----------
    input_img (ndarray): train input [batch, height, width, 2] (seperate real & imag in last dim)
    input_spec (ndarray): k-space of train input [batch, height, width]
    input_mask (ndarray): sampling mask [batch, height, width]
    target_img (ndarray): train target [batch, height, width, 2]
    batch_size (int): batch size
    shuffle (boolean): shuffle all input slices within one epoch

    Yields
    ----------
        batch size data in list
    """
    assert input_img.shape[0] == input_spec.shape[0]

    num = input_img.shape[0]
    all_indices = np.arange(num)

    if shuffle:
        np.random.shuffle(all_indices)
        
    for start_idx in range(0, num, batch_size):
        if start_idx + batch_size <= num:
            indices = all_indices[start_idx : start_idx + batch_size]
            inputs, targets = [], []
            spec_val, spec_mask = [], []

            for i in indices:
                inp = input_img[i]
                spec = input_spec[i]
                sm = input_mask[i]
                img = target_img[i]
                
                inputs.append((inp))
                spec_val.append(spec)
                spec_mask.append(sm)
                targets.append(img) 

            yield indices, inputs, targets, spec_val, spec_mask