################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################
# This file serves as the main for the whole module.
# Overall framework design & coding: Peter S
# Note: Test API = eval(...)

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

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve

from MelanomaDataSet import MelanomaDataSet
from MelanomaModel import EfNet, ResNeXt, GCNLikeCNN

import Config

def stdLog(stdwhich, str, DEBUG=True, fd=None):
    """
    Output to std & log file
    Args:
        stdwhich: stdout/stdin/stderr
        str: content to be logged
        DEBUG: whether to log the string to std
        fd: the file discriptor of the file to log the string in
    """
    if fd:
        fd.write('{}: {}'.format(datetime.now(), str))
    if DEBUG:
        stdwhich.write(str)

def train(learning_rate=Config.LEARNING_RATE_DEFAULT, minibatch_size=Config.BATCH_SIZE_DEFAULT, ef_ver=Config.EFNET_VER_DEFAULT, \
          max_epoch=Config.MAX_EPOCHS_DEFAULT, eval_freq=Config.EVAL_FREQ_DEFAULT, optimizer=Config.OPTIMIZER_DEFAULT, \
          num_workers=Config.WORKERS_DEFAULT, use_gpu=True, folder=Config.DATA_DIR_DEFAULT, DEBUG=True, fd=None, time_tag='WHEN', \
          rs=Config.RESIZE_DEFAULT, dh=Config.DRAW_HAIR_DEFAULT, model = Config.NETWORK_DEFAULT, use_meta=(Config.USE_META_DEFAULT == 1)):
    """
    Performs training and evaluation of the model.
    """
    # Customized GCN-Like CNN model can only accept input size of 128
    if model == 'GCNLikeCNN':
        rs = 128

    # Load Melanoma Datast
    stdLog(sys.stdout, "Loading Melanoma Dataset...\n", DEBUG, fd)
    dataset = MelanomaDataSet(folder, train_transform=Config.get_train_transform(rs, bool(dh)), eval_transform=Config.get_eval_transform(rs), \
                              train=True, valid=True, test=False)
    trainloader = DataLoader(dataset.trainset, batch_size=minibatch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(dataset.validset, batch_size=minibatch_size, shuffle=False, num_workers=num_workers)

    # Initialize the model
    stdLog(sys.stdout, "Initializing the Training Model...\n", DEBUG, fd)
    if model == 'EfficientNet':
        net = EfNet(efnet_version=ef_ver, meta=use_meta)
        stdLog(sys.stdout, "Using EfficientNet {}, images resized to size = {}\n".format(ef_ver, rs), DEBUG, fd)
    elif model == 'ResNeXt':
        net = ResNeXt(meta=use_meta)
        stdLog(sys.stdout, "Using ResNeXt\n".format(ef_ver, rs), DEBUG, fd)
    elif model == 'GCNLikeCNN':
        net = GCNLikeCNN(meta=use_meta)
        stdLog(sys.stdout, "Using GCN-Like CNN\n".format(ef_ver, rs), DEBUG, fd)
    criterion = nn.BCEWithLogitsLoss()

    # Select Optimizer
    if optimizer == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=Config.WEIGHT_DECAY_DEFAULT)    # Adam + L2 Norm
    elif optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(), learning_rate, weight_decay=Config.WEIGHT_DECAY_DEFAULT)
    elif optimizer == 'ADAMW':
        optimizer = torch.optim.AdamW(net.parameters(), learning_rate, weight_decay=Config.WEIGHT_DECAY_DEFAULT)
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
    stdLog(sys.stdout, "Using meta: {}\n".format(use_meta), DEBUG, fd)
    stdLog(sys.stdout, "Start Training!\n", DEBUG, fd)

    stdLog(sys.stdout, "------------------------------------------------------------------------\n", DEBUG, fd)
    
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
                if use_meta:
                    meta_ensemble = metas['ensemble'].to(device)

            # the following line is to deal with exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Config.MAX_NORM_DEFAULT)

            optimizer.zero_grad()
            if use_meta:
                res = net(samples, meta_ensemble)              # [[1], [2], [3]]
            else:
                res = net(samples)
            loss = criterion(res.reshape(-1, 1), labels.type_as(res).reshape(-1, 1))  # BCEWithLogitsLoss does not support Long
            loss.backward()
            optimizer.step()

            preds = torch.round(torch.sigmoid(res)) # set threshold to be 0.5 so that values below 0.5 will be considered 0
            train_correct += (preds.reshape(-1, 1) == labels.type_as(res).reshape(-1, 1)).sum().item()
            train_loss += loss.item()
        train_total = len(dataset.trainset)
        train_acc = train_correct / train_total
        train_loss /= len(trainloader)
        stdLog(sys.stdout, 'Training Round %d: acc = %.2f%%, loss = %.4f\n' % (epoch_i, train_acc * 100, train_loss), DEBUG, fd)
    
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
                        if use_meta:
                            val_meta_ensemble = val_metas['ensemble'].to(device)
                    if use_meta:
                        val_res = net(val_samples, val_meta_ensemble)
                    else:
                        val_res = net(val_samples)
                    val_pred = torch.sigmoid(val_res.reshape(-1, 1))
                    val_pred_list[j * validloader.batch_size : j * validloader.batch_size + len(val_samples)] = val_pred
                val_label_list = dataset.validset.label_list.type_as(val_pred_list).reshape(-1, 1)
                #val_acc = accuracy_score(val_label_list.cpu(), torch.round(val_pred_list.cpu()))    # accuracy on threshold value = 0.5
                val_roc_auc = roc_auc_score(val_label_list.cpu(), val_pred_list.cpu())               # AUC score
                fpr, tpr, thresholds = roc_curve(val_label_list.reshape(-1).cpu(), val_pred_list.reshape(-1).cpu(), pos_label=1)
                tp = tpr * dataset.validset.num_pos
                tn = (1 - fpr) * dataset.validset.num_neg
                val_accs = (tp + tn) / len(dataset.validset)
                val_best_threshold = thresholds[np.argmax(val_accs)]
                val_probs = val_pred_list.reshape(-1).cpu()
                thr = torch.Tensor([val_best_threshold])
                valid_predictions = (val_probs >= thr).float()
                val_acc = accuracy_score(val_label_list.reshape(-1).cpu(), valid_predictions)
                stdLog(sys.stdout, '!!! Validation : acc = %.2f%% under threshold = %.4f, roc_auc = %.2f%% !!!\n' % (val_acc * 100, val_best_threshold, val_roc_auc * 100), DEBUG, fd)

                if train_acc >= 0.9 and val_roc_auc > best_auc:
                    best_auc = val_roc_auc
                    model_name = os.path.join(Config.MODEL_DEFAULT, '{}.pth'.format(time_tag))
                    torch.save(net, model_name)
                    stdLog(sys.stdout, 'Model: {} has been saved.\n'.format(model_name), DEBUG, fd)

def eval(model_name, minibatch_size=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT, use_gpu=True, DEBUG=True, fd=None, \
         rs=Config.RESIZE_DEFAULT, dh=Config.DRAW_HAIR_DEFAULT, folder=Config.DATA_DIR_DEFAULT):
    """
    Evaluate using saved best model (Note that this is a Test API)
    1. Re-evaluate on the validation set
    2. Find the roc_auc score
    3. Plot the ROC curve
    4. Find the optimal threshold for the model/network
    5. List the misclassified images in the validation set
    6. Use the optimal threshold to predict the test set
    """
    net = torch.load(model_name)
    if not hasattr(net, 'use_meta'):    # fix older version issue
        setattr(net, 'use_meta', False)
    use_meta = net.use_meta
    device = torch.device("cuda:0" if (bool(use_gpu) and torch.cuda.is_available()) else "cpu")
    stdLog(sys.stdout, "device: {}\n".format(device), DEBUG, fd)
    if device:
        net.to(device)
        stdLog(sys.stdout, "Best Model sent to CUDA\n", DEBUG, fd)
    # Customized GCN-Like CNN model can only accept input size of 128
    if isinstance(net, GCNLikeCNN):
        rs = 128
    
    # Load Melanoma Datast
    stdLog(sys.stdout, "Loading Melanoma Dataset...\n", DEBUG, fd)
    dataset = MelanomaDataSet(folder, train_transform=Config.get_train_transform(rs, bool(dh)), eval_transform=Config.get_eval_transform(rs), \
                              train=False, valid=True, test=True)
    validloader = DataLoader(dataset.validset, batch_size=minibatch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(dataset.testset, batch_size=minibatch_size, shuffle=False, num_workers=num_workers)
    
    # 1.
    net.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        pred_list = torch.zeros((len(dataset.validset), 1)).to(device)
        
        stdLog(sys.stdout, "Evaluating on Validation Set\n", DEBUG, fd)
        for i, batch in enumerate(validloader):
            samples, metas, labels = batch['image'], batch['meta'], batch['target']
            if device:
                samples, labels = samples.to(device), labels.to(device)
                if use_meta:
                    meta_ensemble = metas['ensemble'].to(device)
            if use_meta:
                res = net(samples, meta_ensemble)
            else:
                res = net(samples)
            pred = torch.sigmoid(res.reshape(-1, 1))
            pred_list[i * validloader.batch_size : i * validloader.batch_size + len(samples)] = pred
        
        # 2.
        label_list = dataset.validset.label_list.type_as(pred_list).reshape(-1, 1)
        pred_list, label_list = pred_list.detach(), label_list.detach()
        roc_auc = roc_auc_score(label_list.cpu(), pred_list.cpu())
        stdLog(sys.stdout, 'Validation Set: roc_auc = %.2f%%\n' % (roc_auc * 100), DEBUG, fd)
        
        # 3.
        true_label = label_list.reshape(-1).cpu().numpy()
        pred_prob = pred_list.reshape(-1).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(true_label, pred_prob, pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (area = %.4f)' % (roc_auc))   # plot the curve
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')                                # plot a diagonal line for reference
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) of Melanoma Model')
        plt.legend(loc="lower right")
        
        eval_base = os.path.basename(model_name)
        eval_filename = os.path.splitext(eval_base)[0]
        eval_path = os.path.join(Config.EVAL_DEFAULT, eval_filename)
        if not os.path.isdir(Config.EVAL_DEFAULT):
            os.mkdir(Config.EVAL_DEFAULT)
        if not os.path.isdir(eval_path):
            os.mkdir(eval_path)
        imgname = os.path.join(eval_path, '{}.png'.format(eval_filename))
        plt.savefig(imgname, bbox_inches='tight')
        stdLog(sys.stdout, 'ROC curve saved to {}\n'.format(imgname), DEBUG, fd)
        #plt.show()
        
        # 4.
        tp = tpr * dataset.validset.num_pos
        tn = (1 - fpr) * dataset.validset.num_neg
        acc = (tp + tn) / len(dataset.validset)
        best_threshold = thresholds[np.argmax(acc)]
        probs = pred_list.cpu().reshape(-1)
        thr = torch.Tensor([best_threshold])
        valid_predictions = (probs >= thr).float()
        val_acc = accuracy_score(label_list.cpu().reshape(-1), valid_predictions)
        stdLog(sys.stdout, 'Optimal Threshold = %.4f, Accuracy under optimal threshold = %.2f%%\n' % (best_threshold, val_acc * 100), DEBUG, fd)
        
        # 5.
        misclassified = []
        for i in range(len(dataset.validset)):
            if int(valid_predictions[i]) != int(dataset.validset[i]['meta']['target']): # misclassified
                misclassified.append(dataset.validset[i]['meta']['image_name'])
        stdLog(sys.stdout, "Misclassified Images: {}\n".format(misclassified), DEBUG, fd)

    # 6.
    torch.cuda.empty_cache()
    with torch.no_grad():
        pred_list = torch.zeros((len(dataset.testset), 1)).to(device)
        stdLog(sys.stdout, "Evaluating on Test Set\n", DEBUG, fd)
        for i, batch in enumerate(testloader):
            samples, metas, labels = batch['image'], batch['meta'], batch['target']
            if device:
                samples, labels = samples.to(device), labels.to(device)
                if use_meta:
                    meta_ensemble = metas['ensemble'].to(device)
            if use_meta:
                res = net(samples, meta_ensemble)
            else:
                res = net(samples)
            pred = torch.sigmoid(res.reshape(-1, 1))
            pred_list[i * testloader.batch_size : i * testloader.batch_size + len(samples)] = pred

        probs = pred_list.detach().reshape(-1)
        thr = torch.Tensor([best_threshold]).to(device)
        test_predictions = (probs >= thr).float()

        resname = os.path.join(eval_path, '{}.csv'.format(eval_filename))
        resname_f = open(resname, 'w')
        for i in range(len(dataset.testset)):
            cur_res = '{},{}\n'.format(dataset.testset[i]['meta']['image_name'], int(test_predictions[i]))
            stdLog(None, cur_res, False, fd)
            resname_f.write(cur_res)
        resname_f.close()
        stdLog(sys.stdout, 'Predictions on test set output to {}\n'.format(resname), DEBUG, fd)

if __name__ == '__main__':
    """
    Usage Example:
        python MelanomaDiscriminator.py -c 10 -m train -net EfficientNet -meta 1
        python MelanomaDiscriminator.py -c 10 -m eval -e model/20200821_08_29_25.pth
    """
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
                        help='Optimizer to be used [ADAM, RMSprop, ADAMW], default = {}'.format(Config.OPTIMIZER_DEFAULT))
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
    parser.add_argument('-m', '--mode', type=str, default=Config.MODE_DEFAULT, \
                        help='Specify which mode the discriminator runs in (train, eval), default = {}'.format(Config.MODE_DEFAULT))
    parser.add_argument('-e', '--eval', type=str, default=Config.EVAL_DEFAULT, \
                        help='Specify the location of saved network to be loaded for evaluation, default = {}'.format(Config.EVAL_DEFAULT))
    parser.add_argument('-meta', '--use_meta', type=int, default=Config.USE_META_DEFAULT, \
                        help='Specify whether to use meta, default = {}'.format(Config.USE_META_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    
    time_tag = datetime.now().strftime('%Y%m%d_%H_%M_%S')

    # Starts a log file in the specified directory
    if FLAGS.log:
        if not os.path.isdir(FLAGS.log):
            os.mkdir(FLAGS.log)
        fd = open(os.path.join(FLAGS.log, '{}.log'.format(time_tag)), 'w')
    else:
        fd = None

    discriminator_mode = FLAGS.mode
    if discriminator_mode == 'train':
        train(learning_rate = FLAGS.learning_rate, minibatch_size = FLAGS.minibatch_size, max_epoch = FLAGS.max_steps, \
              ef_ver=FLAGS.efnet_version, eval_freq = FLAGS.eval_freq, optimizer = FLAGS.optimizer, num_workers=FLAGS.cores, \
              use_gpu = FLAGS.gpu, folder = FLAGS.data_dir, DEBUG = True, fd = fd, time_tag=time_tag, rs = FLAGS.resize, dh = FLAGS.draw_hair, \
              model = FLAGS.network, use_meta = (FLAGS.use_meta == 1))
    elif discriminator_mode == 'eval':
        eval_file = FLAGS.eval
        if (not eval_file) or (not os.path.isfile(eval_file)):
            sys.stderr.write('File for evaluation not found, please check!\n')
            if fd:
                fd.close()
            exit(-1)
        eval(eval_file, minibatch_size = FLAGS.minibatch_size, num_workers = FLAGS.cores, use_gpu = FLAGS.gpu, DEBUG = True, \
             fd = fd, rs = FLAGS.resize, dh = FLAGS.draw_hair, folder = FLAGS.data_dir)
    else:
        sys.stderr.write("Please specify the running mode (train/eval) - 'python MelanomaDiscriminator -m train'\n")
        if fd:
            fd.close()
        exit(-2)