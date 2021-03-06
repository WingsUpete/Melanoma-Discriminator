################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 18th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################
# Used for analyzing the data imbalance
# Author: Peter S

import os
import pandas as pd
import argparse
import json

dict_keys = ['sex', 'age_approx', 'anatom_site_general_challenge', \
             'diagnosis', 'benign_malignant', 'target']

def freq_statistics(csv_file):
    """
    Count the frequencies of values appearing in the csv file of a specified set
    Args:
        csv_file (str): path of the csv file
    """
    data_frame = pd.read_csv(csv_file)
    melanoma_map = {}
    for _ in range(len(data_frame)):
        sample_values = data_frame.iloc[_, 1:]
        for i in range(len(sample_values)):
            cur_key = dict_keys[i]
            if cur_key not in melanoma_map:
                melanoma_map[cur_key] = {}
            cur_val = str(sample_values[i]) if str(sample_values[i]) != 'nan' else ''
            if cur_val not in melanoma_map[cur_key]:
                melanoma_map[cur_key][cur_val] = 0
            melanoma_map[cur_key][cur_val] += 1
    print(melanoma_map)
    base = os.path.basename(csv_file)
    fname = os.path.splitext(base)[0]
    full_path = '{}_statistics.json'.format(fname)
    with open(full_path, 'w') as f:
        json.dump(melanoma_map, f)
    print('Statistics saved to {}'.format(full_path))
    return melanoma_map

if __name__ == '__main__':
    """
    Usage Example:
        python DataAnalyzer.py -set train
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-set', '--set', type=str, default='train', \
                        help='Specify which set to analyze frequency on [train/valid/test], default={}'.format('train'))
    FLAGS, unparsed = parser.parse_known_args()

    settype = FLAGS.set
    if settype == 'train':
        freq_statistics('DataSet/training_set.csv')
    elif settype == 'valid':
        freq_statistics('DataSet/validation_set.csv')
    elif settype == 'test':
        freq_statistics('DataSet/test_set.csv')
    else:
        freq_statistics('DataSet/training_set.csv')