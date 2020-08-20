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

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from MelanomaDataSet import MelanomaDataSet
from MelanomaModel import Net, ResNeXt

import Config

def stdLog(stdwhich, str, DEBUG=True, fd=None):
    """
    Output to std & log file
    """
    if fd:
        fd.write('{}: {}'.format(datetime.now(), str))
    if DEBUG:
        stdwhich.write(str)

def train(learning_rate=Config.LEARNING_RATE_DEFAULT, minibatch_size=Config.BATCH_SIZE_DEFAULT, ef_ver=Config.EFNET_VER_DEFAULT, \
          max_epoch=Config.MAX_EPOCHS_DEFAULT,  eval_freq=Config.EVAL_FREQ_DEFAULT, optimizer=Config.OPTIMIZER_DEFAULT, \
          num_workers=Config.WORKERS_DEFAULT, use_gpu=True, folder=Config.DATA_DIR_DEFAULT, DEBUG=True, fd=None, time_tag='WHEN', \
          rs=Config.RESIZE_DEFAULT, dh=Config.DRAW_HAIR_DEFAULT, model = Config.NETWORK_DEFAULT):
    """
    Performs training and evaluation of the CNN model.
    """

    # Load Melanoma Datast
    stdLog(sys.stdout, "Loading Melanoma Dataset...\n", DEBUG, fd)
    dataset = MelanomaDataSet(folder, train_transform=Config.get_train_transform(rs, bool(dh)), eval_transform=Config.get_eval_transform(rs))
    trainloader = DataLoader(dataset.trainset, batch_size=minibatch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(dataset.validset, batch_size=minibatch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(dataset.testset, batch_size=minibatch_size, shuffle=False, num_workers=num_workers)

    # Initialize the model
    stdLog(sys.stdout, "Initializing the Training Model...\n", DEBUG, fd)
    if model == 'EfficientNet':
        net = Net(efnet_version=ef_ver)
        stdLog(sys.stdout, "Using EfficientNet {}, images resized to size = {}\n".format(ef_ver, rs), DEBUG, fd)
    elif model == 'ResNeXt':
        net = ResNeXt()
        stdLog(sys.stdout, "Using ResNeXt\n".format(ef_ver, rs), DEBUG, fd)
    criterion = nn.BCEWithLogitsLoss()

    # Select Optimizer
    if optimizer == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=Config.WEIGHT_DECAY_DEFAULT)    # Adam + L2 Norm
    elif optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(), learning_rate, weight_decay=Config.WEIGHT_DECAY_DEFAULT)
    else:
        optimizer = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=Config.WEIGHT_DECAY_DEFAULT)    # Default: Adam + L2 Norm

    # CUDA if possible
    device = torch.device("cuda:0" if (bool(use_gpu) and torch.cuda.is_available()) else "cpu")
    stdLog(sys.stdout, "device: {}\n".format(device), DEBUG, fd)

    if device:
        net.to(device)
        stdLog(sys.stdout, "Training Model sent to CUDA\n", DEBUG, fd)

    # Start Training
    stdLog(sys.stdout, "learning_rate = {}, max_epoch = {}, num_workers = {}\n".format(learning_rate, max_epoch, num_workers), DEBUG, fd)
    stdLog(sys.stdout, "eval_freq = {}, minibatch_size = {}, optimizer = {}\n".format(eval_freq, minibatch_size, optimizer), DEBUG, fd)
    stdLog(sys.stdout, "train_transform = {}, eval_transform = {}\n".format(dataset.train_transform, dataset.eval_transform), DEBUG, fd)
    stdLog(sys.stdout, "Start Training!\n", DEBUG, fd)

    stdLog(sys.stdout, "------------------------------------------------------------\n", DEBUG, fd)
    
    if not os.path.isdir(Config.MODEL_DEFAULT):
        os.mkdir(Config.MODEL_DEFAULT)
    
    best_auc = 0.0

    for epoch_i in range(max_epoch):
        # train one round
        net.train()
        train_correct = 0
        train_loss = 0
        for i, batch in enumerate(trainloader):
            samples, metas, labels = batch['image'], batch['meta'], batch['target']
            if device:
                samples, labels = samples.to(device), labels.to(device)

            # the following line is to deal with exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Config.MAX_NORM_DEFAULT)

            optimizer.zero_grad()
            res = net(samples)              # [[1], [2], [3]]
            loss = criterion(res.reshape(-1, 1), labels.type_as(res).reshape(-1, 1))  # BCEWithLogitsLoss does not support Long
            loss.backward()
            optimizer.step()

            preds = torch.round(torch.sigmoid(res)) # set threshold to be 0.5 so that values below 0.5 will be considered 0
            train_correct += (preds.reshape(-1, 1) == labels.type_as(res).reshape(-1, 1)).sum().item()
            train_loss += loss.item()
        train_total = len(dataset.trainset)
        train_acc = train_correct / train_total
        stdLog(sys.stdout, 'Training Round %d: acc = %.2f%%, loss = %.2f\n' % (epoch_i, train_acc * 100, loss.item()), DEBUG, fd)
    
        # evaluate every eval_freq
        if ((epoch_i + 1) % eval_freq == 0):
            net.eval()
            with torch.no_grad():
                val_pred_list = torch.zeros((len(dataset.validset), 1)).to(device)
                # Evaluate using the validation set
                for j, val_batch in enumerate(validloader):
                    val_samples, val_metas, val_labels = val_batch['image'], val_batch['meta'], val_batch['target']
                    if device:
                        val_samples, val_labels = val_samples.to(device), val_labels.to(device)
                    val_res = net(val_samples)
                    val_pred = torch.sigmoid(val_res.reshape(-1, 1))
                    val_pred_list[j * validloader.batch_size : j * validloader.batch_size + len(val_samples)] = val_pred
                val_label_list = dataset.validset.label_list.type_as(val_pred_list).reshape(-1, 1)
                val_acc = accuracy_score(val_label_list.cpu(), torch.round(val_pred_list.cpu()))    # accuracy on threshold value = 0.5
                val_roc_auc = roc_auc_score(val_label_list.cpu(), val_pred_list.cpu())               # AUC score
                stdLog(sys.stdout, '!!! Validation : acc = %.2f%%, roc_auc = %.2f%% !!!\n' % (val_acc * 100, val_roc_auc * 100), DEBUG, fd)

                if train_acc >= 0.9 and val_roc_auc > best_auc:
                    best_auc = val_roc_auc
                    model_name = os.path.join(Config.MODEL_DEFAULT, '{}.pth'.format(time_tag))
                    torch.save(net, model_name)
                    stdLog(sys.stdout, 'Model: {} has been saved.\n'.format(model_name), DEBUG, fd)
    
    #net = torch.load(model_name)

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
                        help='Optimizer to be used [Adam, RMSprop], default = {}'.format(Config.OPTIMIZER_DEFAULT))
    parser.add_argument('-dr', '--data_dir', type = str, default = Config.DATA_DIR_DEFAULT, \
                        help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-log', '--log', type = str, default = Config.LOG_DEFAULT, \
                        help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DEFAULT))
    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, \
                        help='number of workers (cores used), default = {}'.format(Config.WORKERS_DEFAULT))
    parser.add_argument('-gpu', '--gpu', type=int, default=Config.USE_GPU_DEFAULT, \
                        help='Specify whether to use GPU, default = {}'.format(Config.USE_GPU_DEFAULT))
    parser.add_argument('-ev', '--efnet_version', type=int, default=Config.EFNET_VER_DEFAULT, \
                        help='The version of EfficientNet to be used, default = {}'.format(Config.EFNET_VER_DEFAULT))
    parser.add_argument('-rs', '--resize', type=int, default=Config.RESIZE_DEFAULT, \
                        help='The resized size of image, default = {}'.format(Config.RESIZE_DEFAULT))
    parser.add_argument('-dh', '--draw_hair', type=int, default=Config.DRAW_HAIR_DEFAULT, \
                        help='Specify whether to draw pseudo-hairs in images, default = {}'.format(Config.DRAW_HAIR_DEFAULT))
    parser.add_argument('-net', '--network', type=str, default=Config.NETWORK_DEFAULT, \
                        help='Specify which model/network to use, default = {}'.format(Config.NETWORK_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    time_tag = datetime.now().strftime('%Y%m%d_%H_%M_%S')

    # Starts a log file in the specified directory
    if FLAGS.log:
        if not os.path.isdir(FLAGS.log):
            os.mkdir(FLAGS.log)
        fd = open(os.path.join(FLAGS.log, '{}.log'.format(time_tag)), 'w')
    else:
        fd = None

    train(learning_rate = FLAGS.learning_rate, minibatch_size = FLAGS.minibatch_size, max_epoch = FLAGS.max_steps, \
          ef_ver=FLAGS.efnet_version, eval_freq = FLAGS.eval_freq, optimizer = FLAGS.optimizer, num_workers=FLAGS.cores, \
          use_gpu = FLAGS.gpu, folder = FLAGS.data_dir, DEBUG = True, fd = fd, time_tag=time_tag, rs = FLAGS.resize, dh = FLAGS.draw_hair, \
          model = FLAGS.network)