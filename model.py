# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:29:39 2024

@author: peizhouh
"""

import tensorflow as tf
import numpy as np

def autoencoder(input):
    def conv(n, name, n_out, size=3, gain=np.sqrt(2)):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            wshape = [size, size, int(n.get_shape()[-1]), n_out]
            wstd = gain / np.sqrt(np.prod(wshape[:-1])) # He init
            W = tf.compat.v1.get_variable('W', shape=wshape, initializer=tf.initializers.random_normal(0., wstd))
            b = tf.compat.v1.get_variable('b', shape=[n_out], initializer=tf.initializers.zeros())
            n = tf.nn.conv2d(n, W, strides=[1]*4, padding='SAME')
            n = tf.nn.bias_add(n, b)
        return n

    def up(n, name, f=2):
        with tf.name_scope(name):
            s = [-1 if i is None else i for i in n.shape]
            n = tf.reshape(n, [s[0], s[1], 1, s[2], 1, s[3]])
            n = tf.tile(n, [1, 1, f, 1, f, 1])
            n = tf.reshape(n, [s[0], s[1] * f, s[2] * f, s[3]])
        return n

    def down(n, name, f=2):     return tf.nn.max_pool(n, ksize=[1, f, f, 1], strides=[1, f, f, 1], padding='SAME', name=name)
    def concat(name, layers):   return tf.concat(layers, axis=-1, name=name)
    def LR(n):                  return tf.nn.leaky_relu(n, alpha=0.1, name='lrelu')

    # Make even size and add the channel dimension.

    # Encoder part.

    x = input
    x = LR(conv(x, 'enc_conv0', 48))
    x = LR(conv(x, 'enc_conv1', 48))
    x = down(x, 'pool1'); pool1 = x

    x = LR(conv(x, 'enc_conv2', 48))
    x = down(x, 'pool2'); pool2 = x

    x = LR(conv(x, 'enc_conv3', 48))
    x = down(x, 'pool3'); pool3 = x

    x = LR(conv(x, 'enc_conv4', 48))
    x = down(x, 'pool4'); pool4 = x

    x = LR(conv(x, 'enc_conv5', 48))
    x = down(x, 'pool5')

    x = LR(conv(x, 'enc_conv6', 48))

    # Decoder part.

    x = up(x, 'upsample5')
    x = concat('concat5', [x, pool4])
    x = LR(conv(x, 'dec_conv5', 96))
    x = LR(conv(x, 'dec_conv5b', 96))

    x = up(x, 'upsample4')
    x = concat('concat4', [x, pool3])
    x = LR(conv(x, 'dec_conv4', 96))
    x = LR(conv(x, 'dec_conv4b', 96))

    x = up(x, 'upsample3')
    x = concat('concat3', [x, pool2])
    x = LR(conv(x, 'dec_conv3', 96))
    x = LR(conv(x, 'dec_conv3b', 96))

    x = up(x, 'upsample2')
    x = concat('concat2', [x, pool1])
    x = LR(conv(x, 'dec_conv2', 96))
    x = LR(conv(x, 'dec_conv2b', 96))

    x = up(x, 'upsample1')
    x = concat('concat1', [x, input])
    x = LR(conv(x, 'dec_conv1a', 64))
    x = LR(conv(x, 'dec_conv1b', 32))

    x = conv(x, 'dec_conv1', 2, gain=1.0)

    # Remove the channel dimension and crop to odd size.

    return x

def getLambda(name, init=.05):
    """
    create a shared variable called lambda.
    """
    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        lam = tf.compat.v1.get_variable(name=name, dtype=tf.float32, initializer=init)
    return lam

def makeModel(inputs_var, targets_var, spec_value_var, spec_mask_var, args):
    # Settings
    LAMBDA = tf.dtypes.complex(getLambda('lambda', args.lamb),0.)
    BETA = tf.dtypes.complex(getLambda('beta', args.beta),0.)
    #alpha = tf.Variable(args['alpha'], name='alpha')
    TFeps=tf.constant(1e-3, dtype=tf.complex64)
    outer_iters = args.outer_iters
    inner_iters = args.inner_iters
    
    x_est = tf.dtypes.complex(inputs_var[:,:,:,0], inputs_var[:,:,:,1])
    v_est = x_est
    u_est = x_est*0
    Ht_y  = tf.dtypes.complex(inputs_var[:,:,:,0], inputs_var[:,:,:,1])
    
    for k in range(outer_iters):
        ######################################################################
        # print('outer_iters %3d/%d:' %(k+1,outer_iters))                    #   
        # Part1 of the ADMM, approximates the solution of:                   #
        # x = argmin_z 1/(2sigma^2)||Hz-y||_2^2 + 0.5*beta||z - v + u||_2^2  #
        ######################################################################

        for i in range(inner_iters):
            b = Ht_y + BETA*(v_est - u_est)
            A_x_est = tf.signal.ifft2d(tf.signal.fft2d(tf.dtypes.complex(inputs_var[:,:,:,0], inputs_var[:,:,:,1])) * tf.cast(spec_mask_var, tf.complex64)) + BETA*x_est
            res = b - A_x_est
            a_res = tf.signal.ifft2d(tf.signal.fft2d(res * tf.cast(spec_mask_var, tf.complex64))) + BETA*res
            mu_opt = tf.math.reduce_mean(res*res)/(tf.math.reduce_mean(res*a_res) + TFeps)
            x_est = x_est + mu_opt*res
                

        # relaxation          
        x_hat = x_est
                
        ##############################################
        # Part2 of the ADMM, approximates the solution of
        # v = argmin_z lambda*z'*(z-denoiser(z)) +  0.5*beta||z - x - u||_2^2
        # using gradient descent
        ##############################################
                
        for j in range(1):
            f_v_est = autoencoder(tf.concat([tf.expand_dims(tf.math.real(v_est), axis=-1), tf.expand_dims(tf.math.imag(v_est), axis=-1)], axis=-1))
            f_v_est = tf.dtypes.complex(f_v_est[:,:,:,0], f_v_est[:,:,:,1])
                    
            v_est = (BETA*(x_hat + u_est) + LAMBDA*f_v_est)/(LAMBDA + BETA)
              
        ###############################################
        # Part3 of the ADMM, update the dual variable #
        ###############################################
        u_est = u_est + x_hat - v_est
    
    out = tf.concat([tf.expand_dims(tf.math.real(x_est), axis=-1), tf.expand_dims(tf.math.imag(x_est), axis=-1)], axis=-1)
    
    return out