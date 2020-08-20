################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################

import random

from torchvision import transforms
from PIL import Image, ImageDraw

# Settings
class DrawHair:
    """
    Draw a random number of pseudo hairs
    https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet/

    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs=4, width=(1, 2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on

        Returns:
            PIL Image: Image with drawn hairs
        """
        if not self.hairs:
            return img

        width, height = img.size
        draw = ImageDraw.Draw(img)

        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)   # color of the hair: Black
            draw.line((origin, end), fill=(0, 0, 0), width=random.randint(self.width[0], self.width[1]))

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'

# https://www.kaggle.com/sayakdasgupta/siim-isic-melanoma-efficientnet-on-pytorch-tpus
# https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/171745
def get_train_transform(img_resize=240, hair=True):
    funcs = [ \
        transforms.RandomResizedCrop(size=img_resize, scale=(0.9, 1.0)), \
        transforms.ToTensor(), \
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        ]
    if hair:
        funcs.insert(0, DrawHair())
    return transforms.Compose(funcs)

def get_eval_transform(img_resize=240):
    return transforms.Compose([ \
        transforms.Resize((img_resize, img_resize)), \
        transforms.ToTensor(), \
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        ])

# Default values
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 75
EVAL_FREQ_DEFAULT = 5
OPTIMIZER_DEFAULT = 'ADAMW'
WEIGHT_DECAY_DEFAULT = 0.01
DATA_DIR_DEFAULT = 'DataSet'
LOG_DEFAULT = 'log'
MODEL_DEFAULT = 'model'
WORKERS_DEFAULT = 4
USE_GPU_DEFAULT = 1
EFNET_VER_DEFAULT = 1
MAX_NORM_DEFAULT = 10.0
RESIZE_DEFAULT = 240
DRAW_HAIR_DEFAULT = 1
NETWORK_DEFAULT = 'ResNeXt'
MODE_DEFAULT = 'train'
EVAL_DEFAULT = None