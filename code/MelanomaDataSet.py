################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################
# This file stores the melanoma data as an entity of PyTorch's Dataloader.
# More: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, transform
import pickle

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode on

import Config

class MDS_Entity(Dataset):
    """
    Melanoma DataSet Entity (Training / Validation / Test) 
    # image_name, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target
    eg. img_1, female, 35, torso, unknown, benign, 0
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = '{}.jpg'.format(os.path.join(self.root_dir, \
                                self.data_frame.iloc[idx, 0]))
        image = io.imread(img_name)
        meta = {}
        meta['image_name'], meta['sex'], meta['age_approx'], \
        meta['anatom_site_general_challenge'], meta['diagnosis'], \
        meta['benign_malignant'], meta['target'] = self.data_frame.iloc[idx, :]
        sample = {'image': image, 'meta': meta}

        if self.transform:
            sample = self.transform(sample) # TODO: transform on meta needed?

        # reutrn data & label
        return (sample, sample['meta']['target'])

class MelanomaDataSet:
    """ Melanoma DataSet """

    def __init__(self, path, load=True, dump=True):
        self.path = path
        # load
        if load and self.load(self.path):
            pass
        else:
            sys.stderr.write('Loading data from source...\n')
            self.train_meta = self.readCSV_train(self.path)   # Load train metadata csv
            self.valid_meta = self.readCSV_valid(self.path)   # Load validation metadata csv
        # dump
        if dump:
            self.dump(self.path)

    def readCSV_train(self, path):
        """
        Read metadata (.csv) of the training data

        """
        train_metadata_frame = pd.read_csv("{}/training_set.csv".format(path))
        train_metadata = np.asarray(train_metadata_frame.iloc[:, :])
        sys.stderr.write('{} training metadata fetched\n'.format(len(train_metadata)))
        return train_metadata

    def readCSV_valid(self, path):
        """
        Read metadata (.csv) of the validation data
        # image_name, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target
        eg. img_1, female, 35, torso, unknown, benign, 0
        """
        validation_metadata_frame = pd.read_csv("{}/validation_set.csv".format(path))
        validation_metadata = np.asarray(validation_metadata_frame.iloc[:, :])
        sys.stderr.write('{} validation metadata fetched\n'.format(len(validation_metadata)))
        return validation_metadata

    def load(self, path):
        """
        Load the dataset instance from "PATH/MelanomaDataSet.pk"
        """
        file_path = "{}/MelanomaDataSet.pk".format(path)
        if os.path.isfile(file_path):
            dataset = pickle.load(open(file_path, 'rb'))
            sys.stderr.write('Melanoma DataSet loaded from {}\n'.format(file_path))
            self = dataset
            return True
        else:
            sys.stderr.write('Melanoma DataSet not already stored before at {}\n'.format(file_path))
            return False

    def dump(self, path):
        """
        Dump the dataset instance to "PATH/MelanomaDataSet.pk"
        """
        file_path = "{}/MelanomaDataSet.pk".format(path)
        pickle.dump(self, open(file_path, 'wb'))
        sys.stderr.write('Melanoma DataSet dumped as {}\n'.format(file_path))

if __name__ == '__main__':
    #dataset = MelanomaDataSet(Config.data_path, load=False)
    trainset = MDS_Entity(csv_file=os.path.join(Config.data_path, 'training_set.csv'), \
                          root_dir=os.path.join(Config.data_path, 'Training_set'))