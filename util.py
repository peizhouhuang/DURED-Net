import os
import numpy as np
import pickle
import PIL.Image
import h5py as h5
import tensorflow as tf
import math

def read_training_data(filename, name, dtype):
    f = h5.File(filename, 'r')
    dset = f[name]
    dataset = np.array(dset,dtype=dtype)

    return dataset

def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_snapshot(submit_config, net, fname_postfix):
    dump_fname = os.path.join(submit_config.run_dir, "network_%s.pickle" % fname_postfix)
    with open(dump_fname, "wb") as f:
        pickle.dump(net, f)

def save_all_variables(fn):
    save_pkl({var.name: var.eval() for var in tf.compat.v1.global_variables()}, fn)

def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    return 1.0

def rampdown(epoch, num_epochs, rampdown_length):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    return 1.0

def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex128
    out=np.zeros( inp.shape[0:2],dtype=dtype)
    out=inp[...,0]+1j*inp[...,1]
    return out

def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    out=np.zeros( inp.shape+(2,),dtype=dtype)
    out[...,0]=inp.real
    out[...,1]=inp.imag
    return out

def fftshift3d(x, ifft):
    assert len(x.shape) == 3
    s0 = (x.shape[1] // 2) + (0 if ifft else 1)
    s1 = (x.shape[2] // 2) + (0 if ifft else 1)
    x = tf.concat([x[:, s0:, :], x[:, :s0, :]], axis=1)
    x = tf.concat([x[:, :, s1:], x[:, :, :s1]], axis=2)
    return x

def myPSNR(org,recon):
    mse=np.sum(np.square( np.abs(org-recon)))/org.size
    psnr=20*np.log10(org.max()/(np.sqrt(mse)+1e-10 ))
    return psnr

def calNMSE(org, recon):
    mse = np.linalg.norm(org - recon) ** 2 / np.size(org)
    sigEner = np.linalg.norm(org) ** 2
    return mse/sigEner
    
def normalize(img, Training = True):
    """
    Normalize the image between 0 and 1
    """
    if len(img.shape) == 3 :
        img = r2c(img)
    x, y = np.unravel_index(np.argmin(np.abs(img)),img.shape)
    minimum = img[x,y]
    img = img - minimum
    x, y = np.unravel_index(np.argmax(np.abs(img)),img.shape)
    maximum = img[x,y]
    norm = img / maximum
    if Training:
        norm = c2r(norm)
    return norm