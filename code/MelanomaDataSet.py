################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################
# This file stores the melanoma data as an entity of PyTorch's Dataloader.
# More: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
import sys
import time
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import Config

class MDS_Entity(Dataset):
    """
    Melanoma DataSet Entity (Training / Validation / Test) 
    # image_name, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target
    eg. img_1, female, 35, torso, unknown, benign, 0
    """

    def __init__(self, csv_file, root_dir, transform=None, test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            test (Boolean): whether this is a test set
        """
        self.data_frame = pd.read_csv(csv_file)
        with open(Config.META_MAPPING_DEFAULT) as f:
            self.mappings = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.label_list = []
        self.num_pos = 0
        self.num_neg = 0
        if not self.test:
            for i in range(len(self.data_frame)):
                cur_label = MDS_Entity.tryConvertFloat(self.data_frame.iloc[i, 6])
                self.label_list.append(cur_label)
                if cur_label == 1:
                    self.num_pos += 1
                elif cur_label == 0:
                    self.num_neg += 1
            self.label_list = torch.Tensor(self.label_list)
        else:
            self.label_list = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = '{}.jpg'.format(os.path.join(self.root_dir, \
                                self.data_frame.iloc[idx, 0]))
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        label = self.data_frame.iloc[idx, 6] if not self.test else -1
        meta = { \
            "image_name": str(self.data_frame.iloc[idx, 0]), \
            "sex": str(self.data_frame.iloc[idx, 1]), \
            "age_approx": MDS_Entity.tryConvertInt(self.data_frame.iloc[idx, 2]), \
            "anatom_site_general_challenge": str(self.data_frame.iloc[idx, 3]), \
            "ensemble": self.get_meta_ensemble(self.data_frame.iloc[idx])
            } if self.test else { \
            "image_name": str(self.data_frame.iloc[idx, 0]), \
            "sex": str(self.data_frame.iloc[idx, 1]), \
            "age_approx": MDS_Entity.tryConvertInt(self.data_frame.iloc[idx, 2]), \
            "anatom_site_general_challenge": str(self.data_frame.iloc[idx, 3]), \
            "diagnosis": str(self.data_frame.iloc[idx, 4]), \
            "benign_malignant": str(self.data_frame.iloc[idx, 5]), \
            "target": MDS_Entity.tryConvertFloat(self.data_frame.iloc[idx, 6]), \
            "ensemble": self.get_meta_ensemble(self.data_frame.iloc[idx])
            }

        # return data & label
        sample = {'image': image, 'meta': meta, 'target': label}
        return sample

    def get_meta_ensemble(self, meta_frame):
        """
        Transform metadata into values that can be learned
        Author: idea desgined by Wei Yanbin, code refactored by Peter S
        """
        meta_len, meta_map = self.mappings['meta_len'], self.mappings['map']
        meta_ensemble = [0] * meta_len
        for i in range(len(meta_map)):
            mapped_index = meta_map[i][str(meta_frame[i + 1]) if str(meta_frame[i + 1]) != 'nan' else ""]  # fetch corresponding value
            if mapped_index == -1:
                continue
            meta_ensemble[mapped_index] = 1
        return torch.Tensor(meta_ensemble)

    def tryConvertInt(str):
        """
        Helper function: convert a string into an integer. If fails, return -1 instead
        """
        try:
            return int(str)
        except ValueError:
            return -1

    def tryConvertFloat(str):
        """
        Helper function: convert a string into an float. If fails, return -1.0 instead
        """
        try:
            return float(str)
        except ValueError:
            return -1.0

class MelanomaDataSet:
    """ Melanoma DataSet """

    def __init__(self, path, train_transform=None, eval_transform=None, train=True, valid=True, test=True):
        """
        Inputs:
            train, valid, test (Boolean): whether the training set, validation set, test set
                should be included
            transform (Boolean): whether the sample should be transformed on the fly
        Args:
            path (string): root directory of data set
            rescale (int): rescale dimension for transform
            randCrop (int): random crop dimension for transform
            trainset (MDS_Entity): training set instance
            validset (MDS_Entity): validation set instance
            testset (MDS_Entity): test set instance
            __sets__ (dictionary): helper variable for specifying which sets of data are included
        """
        self.path = path

        self.train_transform = train_transform
        self.eval_transform = eval_transform

        if train:
            sys.stderr.write('Loading training set...\n')
            self.trainset = MDS_Entity(csv_file=os.path.join(self.path, 'training_set.csv'), \
                                       root_dir=os.path.join(self.path, 'Training_set'), \
                                       transform=self.train_transform)

        if valid:
            sys.stderr.write('Loading validation set...\n')
            self.validset = MDS_Entity(csv_file=os.path.join(self.path, 'validation_set.csv'), \
                                       root_dir=os.path.join(self.path, 'Validation_set'), \
                                       transform=self.eval_transform)

        if test:
            sys.stderr.write('Loading test set...\n')
            self.testset = MDS_Entity(csv_file=os.path.join(self.path, 'test_set.csv'), \
                                       root_dir=os.path.join(self.path, 'Test_set'), \
                                       transform=self.eval_transform, test=True)
        
        self.__sets__ = {'train': train, 'validation': valid, 'test': test}
        sys.stderr.write('Melanoma DataSet Ready: {}\n'.format([key for key in self.__sets__ if self.__sets__[key]]))

def testSamplingSpeed(ds, batch_size, shuffle, tag, num_workers=4):
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    time0 = time.time()

    for i, batch in enumerate(dataloader):
        samples, metas, labels = batch['image'], batch['meta'], batch['target']
        sys.stderr.write("\r{} Set - Batch No. {}/{} with time used(s): {}, {}".format(tag, i + 1, len(dataloader), time.time() - time0, samples.size()))
        sys.stderr.flush()
        #if i == 0:
        #    print(samples)

    sys.stderr.write("\n")

if __name__ == '__main__':
    """
    Usage Example:
        py MelanomaDataSet.py -c 6
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, \
                        help='number of workers (cores used), default={}'.format(Config.WORKERS_DEFAULT))
    FLAGS, unparsed = parser.parse_known_args()
    print('Running with {} workers.'.format(FLAGS.cores))

    dataset = MelanomaDataSet(path=Config.DATA_DIR_DEFAULT, train_transform=Config.get_train_transform(), eval_transform=Config.get_eval_transform())
    testSamplingSpeed(dataset.trainset, 32, True, "Training", FLAGS.cores)
    testSamplingSpeed(dataset.validset, 16, False, "Validation", FLAGS.cores)
    testSamplingSpeed(dataset.testset, 16, False, "Test", FLAGS.cores)