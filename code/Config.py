################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################

from torchvision import transforms

# Settings
image_transform = transforms.Compose([ \
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), \
    transforms.RandomHorizontalFlip(), \
    transforms.RandomVerticalFlip(), \
    transforms.ToTensor(), \
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) \
])

# Default values
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_DECAY_DEFAULT = 0.01
DATA_DIR_DEFAULT = 'DataSet'