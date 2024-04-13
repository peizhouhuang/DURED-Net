# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:29:00 2024

@author: peizhouh
"""

import os
import math
import time

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # avoid 'tf.placeholder() is not compatible with eager execution'

from util import save_all_variables, load_pkl
import hdf5storage
from mri_util import load_dataset, iterate_minibatches
from model import makeModel
from params import parse_args
import sys

def train(args):
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    
    tf.compat.v1.reset_default_graph()
    config = tf.compat.v1.ConfigProto()
    
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    print('Loading training set.')
    input_img, input_spec, input_mask, target_img = load_dataset(args.data_dir)
    
    tf.compat.v1.reset_default_graph()
    # Construct TF graph.
    input_shape = [None] + list(input_img.shape)[1:]
    spectrum_shape = [None] + list(input_spec.shape)[1:]
    inputs_var = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name='inputs')
    targets_var = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name='targets')
    spec_value_var = tf.compat.v1.placeholder(tf.complex64, shape=spectrum_shape, name='spec_value')
    spec_mask_var  = tf.compat.v1.placeholder(tf.float32, shape=spectrum_shape, name='spec_mask')
    TFeps=tf.constant(1e-3,dtype=tf.complex64)
    
    modelOutput = makeModel(inputs_var, targets_var, spec_value_var, spec_mask_var, args)
    
    with tf.name_scope('loss'):
        diff_expr = targets_var - modelOutput
        loss_train = tf.math.reduce_mean(tf.math.reduce_sum(diff_expr**2)) #tf.reduce_mean(diff_expr**2)

    # Construct optimization function.

    adam_beta1_var = tf.compat.v1.placeholder(tf.float32, shape=[], name='adam_beta1')
    train_updates = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr).minimize(loss_train)

    # Create a log file for Tensorboard
    writer = tf.summary.create_file_writer(args.result_dir)

    # Start training.
    session = tf.compat.v1.get_default_session()
    
    with tf.compat.v1.Session(config=config) as sess:
        for epoch in range(args.num_epochs):
    
            time_start = time.time()
    
            # Calculate epoch parameters.
    
            learning_rate = args.lr
    
            # Epoch initializer.
    
            if epoch == 0:
                sess.run(tf.compat.v1.global_variables_initializer())
                if args.load_network:
                    print('Loading network %s' % args.load_network)
                    var_dict = load_pkl(os.path.join(args.result_dir, args.load_network))
                    for var in tf.compat.v1.global_variables():
                        if var.name in var_dict:
                            tf.assign(var, var_dict[var.name]).eval()
    
            # Train.
            train_loss, train_n = 0., 0.
            batch_size = args.batch_size
            shuffle = True
            
            for (indices, inputs, targets, input_spec_val, input_spec_mask) in iterate_minibatches(input_img, input_spec, input_mask, target_img, batch_size, shuffle, args):
                        
                feed_dict = {inputs_var: inputs, targets_var: targets}
                feed_dict.update({spec_value_var: input_spec_val, spec_mask_var: input_spec_mask})
    
                # Run.
                loss_val, _ = sess.run([loss_train, train_updates], feed_dict=feed_dict)
                # Stats.
                train_loss += loss_val * len(indices)
                train_n += len(indices)
                print('train_n: ', train_n)
    
            train_loss /= train_n
    
            # Export network.
            if epoch % 1 == 0:
                save_all_variables(os.path.join(args.result_dir, 'network-snapshot-%05d.pkl' % epoch))
    
            # Export and print stats, update progress monitor.
    
            time_epoch = time.time() - time_start
    
    
            print('Epoch %3d/%d: time=%7.3f, train_loss=%.7f' % (
                epoch, args.num_epochs, time_epoch, train_loss))
            
            with writer.as_default():
                tf.summary.scalar("Timing/sec_per_epoch", time_epoch, step=epoch)
                tf.summary.scalar("Train/loss", train_loss, step=epoch)
                writer.flush()
            
    
        print("Saving final network weights.")
        save_all_variables(os.path.join(args.result_dir, 'network-final.pkl'))
    
if __name__ == '__main__':
    args = parse_args()
    train(args)
