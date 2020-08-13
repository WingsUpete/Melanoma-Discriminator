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
from skimage import io, transform

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
        image, meta = sample['image'], sample['meta']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'meta': meta}

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
        image, meta = sample['image'], sample['meta']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, \
                      left : left + new_w]

        return {'image': image, 'meta': meta}

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """

    def __call__(self, sample):
        image, meta = sample['image'], sample['meta']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), \
                'meta': meta}

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
        #time0 = time.time()
        image = io.imread(img_name)
        meta = {}
        meta['image_name'], meta['sex'], meta['age_approx'], \
        meta['anatom_site_general_challenge'], meta['diagnosis'], \
        meta['benign_malignant'], meta['target'] = self.data_frame.iloc[idx, :]
        sample = {'image': image, 'meta': meta}
        #time1 = time.time()

        if self.transform:
            sample = self.transform(sample)
            
        #time2 = time.time()
        #print("load: {}\t\ttransform: {}".format(time1-time0, time2-time1))

        # return data & label
        return (sample, sample['meta']['target'])

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
            Rescale(256), \
            RandomCrop(224), \
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
                                       transform=self.transform if transform else None)
        
        self.__sets__ = {'train': train, 'validation': valid, 'test': test}
        sys.stderr.write('Melanoma DataSet Ready: {}\n'.format([key for key in self.__sets__ if self.__sets__[key]]))

if __name__ == '__main__':
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("device: {}".format(device))

    dataset = MelanomaDataSet(Config.data_path, transform=True)
    
    trainloader = DataLoader(dataset.trainset, batch_size=32, shuffle=True, num_workers=4)
    time0 = time.time()
    for i, batch in enumerate(trainloader):
        samples, label = batch
        if device:
            samples['image'], label = samples['image'].to(device), label.to(device)
        #print(i, samples['image'].size(), label)
        sys.stderr.write('\rBatch No. {}/{}'.format(i + 1, len(trainloader)))
        sys.stderr.flush()
        #if i == 10:
        #    break
    sys.stderr.write("\n")
    time1 = time.time()
    print("Iteration takes seconds: {}".format(time1 - time0))