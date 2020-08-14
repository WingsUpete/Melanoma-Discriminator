# Melanoma-Discriminator
Decide whether an image is a melanoma.

Created by <a href="mailto:petershen815@126.com">Peter S</a> on Aug 11th, 2020

<br>

## Dependencies

-   [`numpy`](https://numpy.org/)
-   [`matplotlib`](https://matplotlib.org/)
-   [`scipy`](https://www.scipy.org/)
-   [`scikit-learn`](https://scikit-learn.org/stable/)
-   [`jupyter`](https://jupyter.org/)
-   [`pytorch` (`torch`, `torchvision`)](https://pytorch.org/)
-   [`PIL [Python Image Library]`](https://python-pillow.org/)
-   [`pandas`](https://pandas.pydata.org/)
-   [`efficientnet_pytorch`](https://github.com/lukemelas/EfficientNet-PyTorch#about-efficientnet)

<br>

## Data

Data Source: https://challenge2020.isic-archive.com/

The data has been simplified and split into training/validation sets.

See `MelanomaDataSet.py` for more details

The training/validation/test sets are packed using PyTorch’s `Dataset` and can be accessed through `DataLoader`. For each set, the image data and metadata are packed together as a sample. The images will be transformed with such procedure by default:

1.  Resize and crop randomly to 256 x 256 with scaling of $(h=0.8, w=1.0)$
3.  Flip horizontally and vertically, each with a $probability = 0.5$
3.  Transform to tensor
4.  Normalize channel values with $mean=(0.485, 0.456, 0.406)$, $std=(0.229, 0.224, 0.225)$

<br>

## Model

[PyTorch’s `EfficientNet`](https://github.com/lukemelas/EfficientNet-PyTorch#about-efficientnet) is used to construct the model network.

The last fully-connected layer is modified so that its number of output features becomes $500$.

A fully connected layer is added to the end to summarize the features to 1 channel. 