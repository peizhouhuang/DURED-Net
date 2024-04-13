# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:48:26 2024

@author: peizhouh
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        type=str,
        #default='',
        help="Path to file(s) with training data",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default='result',
        help="Path to saved model file",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Default learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Default num of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Default batch size",
    )
    parser.add_argument(
        "--load_network",
        type=bool,
        default=False,
        help="Load Network",
    )
    
    parser.add_argument(
        "--lamb",
        type=float,
        default=9.0,
        help="Default lambda",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=10.0,
        help="Default beta",
    )
    parser.add_argument(
        "--outer_iters",
        type=int,
        default=2,
        help="Unrolled Step",
    )
    parser.add_argument(
        "--inner_iters",
        type=int,
        default=20,
        help="Default iteration for data consistency",
    )
    
    args = parser.parse_args()
    
    return args