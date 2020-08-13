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

class Rescale(object):
    """
    Rescale the image in a sample to a given size
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        #time0 = time.time()
        image = sample

        w, h = image.size
        if isinstance(self.output_size, int):
            if w < h:
                new_w, new_h = self.output_size, self.output_size * h / w
            else:
                new_w, new_h = self.output_size * w / h, self.output_size
        else:
            new_w, new_h = self.output_size

        new_w, new_h = int(new_w), int(new_h)
        #time1 = time.time()
        img = image.resize((new_w, new_h))

        #print("Rescale: {} | {}".format(time1 - time0, time.time() - time1))
        return img

class RandomCrop(object):
    """ 
    Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        #time0 = time.time()
        image = sample

        w, h = image.size
        new_w, new_h = self.output_size
        
        left = np.random.randint(0, w - new_w)
        top = np.random.randint(0, h - new_h)

        image = image.crop((left, top, left + new_w, top + new_h))
        
        #print("RandomCrop: {}".format(time.time() - time0))
        return image

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """

    def __call__(self, sample):
        #time0 = time.time()
        image = sample

        tt = transforms.ToTensor()
        image = tt(image)
        #print("ToTensor: {}".format(time.time() - time0))
        return image

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
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = '{}.jpg'.format(os.path.join(self.root_dir, \
                                self.data_frame.iloc[idx, 0]))
        #time0 = time.time()
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 6] if not self.test else -1
        #time1 = time.time()

        if self.transform:
            image = self.transform(image)
            
        #time2 = time.time()
        #print("load: {}\t\ttransform: {}".format(time1-time0, time2-time1))

        # return data & label
        sample = {'image': image, 'target': label}
        return sample

class MelanomaDataSet:
    """ Melanoma DataSet """

    def __init__(self, path, train=True, valid=True, test=True, transform=True):
        """
        Inputs:
            train, valid, test (Boolean): whether the training set, validation set, test set
                should be included
            transform (Boolean): whether the sample should be transformed on the fly
        Args:
            path (string): root directory of data set
            transform (transforms.Compose): transform function
            trainset (MDS_Entity): training set instance
            validset (MDS_Entity): validation set instance
            testset (MDS_Entity): test set instance
            __sets__ (dictionary): helper variable for specifying which sets of data are included
        """
        self.path = path

        self.transform = transforms.Compose([
            Rescale(512), \
            RandomCrop(500), \
            ToTensor() \
        ])

        if train:
            sys.stderr.write('Loading training set...\n')
            self.trainset = MDS_Entity(csv_file=os.path.join(self.path, 'training_set.csv'), \
                                       root_dir=os.path.join(self.path, 'Training_set'), \
                                       transform=self.transform if transform else None)

        if valid:
            sys.stderr.write('Loading validation set...\n')
            self.validset = MDS_Entity(csv_file=os.path.join(self.path, 'validation_set.csv'), \
                                       root_dir=os.path.join(self.path, 'Validation_set'), \
                                       transform=self.transform if transform else None)

        if test:
            sys.stderr.write('Loading test set...\n')
            self.testset = MDS_Entity(csv_file=os.path.join(self.path, 'test_set.csv'), \
                                       root_dir=os.path.join(self.path, 'Test_set'), \
                                       transform=self.transform if transform else None, \
                                      test=True)
        
        self.__sets__ = {'train': train, 'validation': valid, 'test': test}
        sys.stderr.write('Melanoma DataSet Ready: {}\n'.format([key for key in self.__sets__ if self.__sets__[key]]))

def testSamplingSpeed(ds, batch_size, shuffle, tag):
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    time0 = time.time()

    for i, batch in enumerate(dataloader):
        samples, labels = batch['image'], batch['target']
        if device:
            samples, labels = samples.to(device), labels.to(device)
        sys.stderr.write("\r{} Set - Batch No. {}/{} with time used(s): {}".format(tag, i + 1, len(dataloader), time.time() - time0))
        sys.stderr.flush()
    sys.stderr.write("\n")

if __name__ == '__main__':
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("device: {}".format(device))

    dataset = MelanomaDataSet(Config.DATA_DIR_DEFAULT, transform=True)
    
    testSamplingSpeed(dataset.trainset, 10, True, "Training")
    testSamplingSpeed(dataset.validset, 5, False, "Validation")
    testSamplingSpeed(dataset.testset, 5, False, "Test")