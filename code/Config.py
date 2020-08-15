################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################

from torchvision import transforms

# Settings
def get_train_transform(img_resize=240):
    return transforms.Compose([ \
        transforms.CenterCrop(512), \
        transforms.Resize(img_resize), \
        transforms.ToTensor(), \
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # standard for EfficientNet \
        ])

def get_eval_transform(img_resize=240):
    return transforms.Compose([ \
        transforms.CenterCrop(512), \
        transforms.Resize(img_resize), \
        transforms.ToTensor(), \
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        ])

# Default values
LEARNING_RATE_DEFAULT = 1e-3
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 100
EVAL_FREQ_DEFAULT = 5
OPTIMIZER_DEFAULT = 'RMSprop'
WEIGHT_DECAY_DEFAULT = 0.01
DATA_DIR_DEFAULT = 'DataSet'
LOG_DEFAULT = 'log'
MODEL_DEFAULT = 'model'
WORKERS_DEFAULT = 4
USE_GPU_DEFAULT = True
EFNET_VER_DEFAULT = 1
MAX_NORM_DEFAULT = 10.0
RESIZE_DEFAULT = 240