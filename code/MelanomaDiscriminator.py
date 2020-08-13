################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################
# This file serves as the main for the whole module.

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from MelanomaDataSet import MelanomaDataSet

import Config


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    preds = torch.argmax(predictions, dim=1)
    accuracy = (preds == targets).float().mean()
    return accuracy

def train(learning_rate=LEARNING_RATE_DEFAULT, minibatch_size= BATCH_SIZE_DEFAULT, 
          max_epoch=MAX_EPOCHS_DEFAULT,  eval_freq=EVAL_FREQ_DEFAULT, optimizer=OPTIMIZER_DEFAULT, \
          use_gpu=True, folder=DATA_DIR_DEFAULT, DEBUG=True):
    """
    Performs training and evaluation of the CNN model.
    """

    # Load Melanoma Datast
    print("Loading Melanoma Dataset...")
    dataset = MelanomaDataSet(folder)

    # Initialize the model
    print("Initializing the Training Model...")
    net = CNN(n_inputs, n_classes)
    loss_func = nn.CrossEntropyLoss()

    # Select Optimizer
    optimizer = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=WEIGHT_DECAY_DEFAULT)    # Adam + L2 Norm

    # CUDA if possible
    device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print("device: {}".format(device))

    if device:
        net.to(device)
        print("Training Model sent to CUDA")

    # Start Training
    print("Start Training!")

    print("learning_rate = {}, max_epoch = {}".format(learning_rate, max_epoch))
    print("eval_freq = {}, minibatch_size = {}, optimizer = {}".format(eval_freq, minibatch_size, optimizer))

    print("------------------------")
    
    for epoch_i in range(max_epoch):
        # SGD_once
        #time0 = time.time()
        for inputs, labels in dataset.trainloader:
            if device:
                inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = net(inputs)
            loss = loss_func(preds, labels)
            loss.backward()
            optimizer.step()
        #time1 = time.time()
        #print("SGD once => {}".format(time1 - time0))
    
        # evaluate every eval_freq
        if (epoch_i % eval_freq == 0):
            #time2 = time.time()
            with torch.no_grad():
                training_set_accuracy = 0
                test_set_accuracy = 0
                sum = 0
                loss = 0
                for xtrain, ytrain in dataset.trainloader:
                    if device:
                        xtrain, ytrain = xtrain.to(device), ytrain.to(device)
                    out_res = net(xtrain)
                    sum += accuracy(out_res, ytrain)
                    loss += loss_func(out_res, ytrain)
                training_set_accuracy = sum / len(dataset.trainloader)
                training_set_loss = loss / len(dataset.trainloader)
                training_set_loss = training_set_loss.item()
                sum = 0
                loss = 0
                for xtest, ytest in dataset.testloader:
                    if device:
                        xtest, ytest = xtest.to(device), ytest.to(device)
                    out_res = net(xtest)
                    sum += accuracy(out_res, ytest)
                    loss += loss_func(out_res, ytest)
                test_set_accuracy = sum / len(dataset.testloader)
                test_set_loss = loss / len(dataset.testloader)
                test_set_loss = test_set_loss.item()
            x_epoch.append(epoch_i)
            y_accuracy_train.append(training_set_accuracy)
            y_accuracy_test.append(test_set_accuracy)
            y_loss_train.append(training_set_loss)
            y_loss_test.append(test_set_loss)
            if DEBUG:
                print("epoch = %d,\ttraining_set_accuracy = %.2f%%, training_set_loss = %.2f, test_set_accuracy = %.2f%%, test_set_loss = %.2f" % (epoch_i, training_set_accuracy*100, training_set_loss, test_set_accuracy*100, test_set_loss))
            #time3 = time.time()
            #print("Evaluate => {}".format(time3 - time2))

            # plot the result
            plt.plot(x_epoch, y_accuracy_train, c="red", label="Training Accuracy", alpha=0.8)
            plt.plot(x_epoch, y_accuracy_test, c="blue", label="Test Accuracy", alpha=0.8)
            plt.plot(x_epoch, y_loss_train, c="orange", label="Train Loss", alpha=0.8)
            plt.plot(x_epoch, y_loss_test, c="green", label="Test Loss", alpha=0.8)
            plt.title("Epoch - Accuracy/Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy/Loss")
            plt.legend(loc="lower right")
            plt.show()

    print("epoch = %d,\ttraining_set_accuracy = %.2f%%, training_set_loss = %.2f, test_set_accuracy = %.2f%%, test_set_loss = %.2f" % (epoch_i, training_set_accuracy*100, training_set_loss, test_set_accuracy*100, test_set_loss))

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

    FLAGS, unparsed = parser.parse_known_args()

    train(learning_rate = FLAGS.learning_rate, minibatch_size = FLAGS.minibatch_size, max_epoch = FLAGS.max_steps, \
          eval_freq = FLAGS.eval_freq, optimizer = FLAGS.optimizer,  use_gpu = True, folder = FLAGS.data_dir, DEBUG = True)