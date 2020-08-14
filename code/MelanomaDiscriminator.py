################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################
# This file serves as the main for the whole module.

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MelanomaDataSet import MelanomaDataSet
from MelanomaModel import Net

import Config

def stdLog(stdwhich, str, DEBUG=True, fd=None):
    """
    Output to std & log file
    """
    if fd:
        fd.write('{}: {}'.format(datetime.now(), str))
    if DEBUG:
        stdwhich.write(str)

def train(learning_rate=Config.LEARNING_RATE_DEFAULT, minibatch_size=Config.BATCH_SIZE_DEFAULT, 
          max_epoch=Config.MAX_EPOCHS_DEFAULT,  eval_freq=Config.EVAL_FREQ_DEFAULT, optimizer=Config.OPTIMIZER_DEFAULT, \
          num_workers=Config.WORKERS_DEFAULT, use_gpu=True, folder=Config.DATA_DIR_DEFAULT, DEBUG=True, fd=None):
    """
    Performs training and evaluation of the CNN model.
    """

    # Load Melanoma Datast
    stdLog(sys.stdout, "Loading Melanoma Dataset...\n", DEBUG, fd)
    dataset = MelanomaDataSet(folder, transform=Config.image_transform)
    trainloader = DataLoader(dataset.trainset, batch_size=minibatch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(dataset.validset, batch_size=int(minibatch_size/2), shuffle=True, num_workers=num_workers)
    testloader = DataLoader(dataset.testset, batch_size=int(minibatch_size/2), shuffle=True, num_workers=num_workers)

    # Initialize the model
    stdLog(sys.stdout, "Initializing the Training Model...\n", DEBUG, fd)
    net = Net()
    criterion = nn.BCEWithLogitsLoss()

    # Select Optimizer
    if optimizer == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=Config.WEIGHT_DECAY_DEFAULT)    # Adam + L2 Norm
    else:
        optimizer = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=Config.WEIGHT_DECAY_DEFAULT)    # Default: Adam + L2 Norm

    # CUDA if possible
    device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
    stdLog(sys.stdout, "device: {}\n".format(device), DEBUG, fd)

    if device:
        net.to(device)
        stdLog(sys.stdout, "Training Model sent to CUDA\n", DEBUG, fd)

    # Start Training
    stdLog(sys.stdout, "Start Training!\n", DEBUG, fd)

    stdLog(sys.stdout, "learning_rate = {}, max_epoch = {}\n".format(learning_rate, max_epoch), DEBUG, fd)
    stdLog(sys.stdout, "eval_freq = {}, minibatch_size = {}, optimizer = {}\n".format(eval_freq, minibatch_size, optimizer), DEBUG, fd)

    stdLog(sys.stdout, "------------------------\n", DEBUG, fd)
    
    for epoch_i in range(max_epoch):
        # SGD_once
        for i, batch in enumerate(trainloader):
            samples, metas, labels = batch['image'], batch['meta'], batch['target']
            if device:
                samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            res = net(samples)
            loss = criterion(res, labels)
            loss.backward()
            print(res, '|', labels, '|', loss)
            optimizer.step()
    
        # evaluate every eval_freq
        if (epoch_i % eval_freq == 0):
            with torch.no_grad():
                #print("epoch = %d,\ttraining_set_accuracy = %.2f%%, training_set_loss = %.2f, test_set_accuracy = %.2f%%, test_set_loss = %.2f" % \
                #      (epoch_i, training_set_accuracy*100, training_set_loss, test_set_accuracy*100, test_set_loss))
                pass
    
if __name__ == '__main__':
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type = float, default = Config.LEARNING_RATE_DEFAULT, \
                        help='Learning rate, default = {}'.format(Config.LEARNING_RATE_DEFAULT))
    parser.add_argument('-me', '--max_steps', type = int, default = Config.MAX_EPOCHS_DEFAULT, \
                        help='Number of epochs to run the trainer, default = {}'.format(Config.MAX_EPOCHS_DEFAULT))
    parser.add_argument('-ef', '--eval_freq', type = int, default = Config.EVAL_FREQ_DEFAULT, \
                        help='Frequency of evaluation on the validation set, default = {}'.format(Config.EVAL_FREQ_DEFAULT))
    parser.add_argument('-bs', '--minibatch_size', type = int, default = Config.BATCH_SIZE_DEFAULT, \
                        help='Size of a minibatch, default = {}'.format(Config.BATCH_SIZE_DEFAULT))
    parser.add_argument('-opt', '--optimizer', type = str, default = Config.OPTIMIZER_DEFAULT, \
                        help='Optimizer to be used, default = {}'.format(Config.OPTIMIZER_DEFAULT))
    parser.add_argument('-dr', '--data_dir', type = str, default = Config.DATA_DIR_DEFAULT, \
                        help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-log', '--log', type = str, default = Config.LOG_DEFAULT, \
                        help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DEFAULT))
    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, \
                        help='number of workers (cores used), default={}'.format(Config.WORKERS_DEFAULT))
    parser.add_argument('-gpu', '--gpu', type=bool, default=Config.USE_GPU_DEFAULT, \
                        help='Specify whether to use GPU, default={}'.format(Config.USE_GPU_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    if FLAGS.log:
        if not os.path.isdir(FLAGS.log):
            os.mkdir(FLAGS.log)
        fd = open(os.path.join(FLAGS.log, '{}.log'.format(datetime.now().strftime('%Y%m%d_%H_%M_%S'))), 'w')
    else:
        fd = None

    train(learning_rate = FLAGS.learning_rate, minibatch_size = FLAGS.minibatch_size, max_epoch = FLAGS.max_steps, \
          eval_freq = FLAGS.eval_freq, optimizer = FLAGS.optimizer, num_workers=FLAGS.cores, use_gpu = FLAGS.gpu, \
          folder = FLAGS.data_dir, DEBUG = True, fd = fd)