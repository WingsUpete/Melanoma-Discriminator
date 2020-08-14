################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################

from torchvision import transforms

# Settings
train_transform = transforms.Compose([ \
    transforms.RandomResizedCrop(size=240, scale=(0.9, 1.0)), \
    transforms.ToTensor(), \
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # standard for EfficientNet \
])
# TODO: test with additional:
#    transforms.RandomHorizontalFlip(), \
#    transforms.RandomVerticalFlip(), \

eval_transform = transforms.Compose([ \
    transforms.RandomResizedCrop(size=240, scale=(0.996, 1.0)), \
    transforms.ToTensor(), \
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # standard for EfficientNet \
])

# Default values
LEARNING_RATE_DEFAULT = 1e-2    # TODO: test with 1e-2
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 666
EVAL_FREQ_DEFAULT = 5
OPTIMIZER_DEFAULT = 'RMSprop'
WEIGHT_DECAY_DEFAULT = 0.01
DATA_DIR_DEFAULT = 'DataSet'
LOG_DEFAULT = 'log'
WORKERS_DEFAULT = 4
USE_GPU_DEFAULT = True
EFNET_VER_DEFAULT = 1
MAX_NORM_DEFAULT = 1.0