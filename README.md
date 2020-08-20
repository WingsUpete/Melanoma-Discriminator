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
-   [`pytorch-lightning`](https://github.com/PyTorchLightning/pytorch-lightning)

<br>

## Data

Data Source: https://challenge2020.isic-archive.com/

The data has been simplified and split into training/validation sets.

See `MelanomaDataSet.py` for more details

The training/validation/test sets are packed using PyTorch’s `Dataset` and can be accessed through `DataLoader`. For each set, the image data and metadata are packed together as a sample.

<br>

## Model

[PyTorch’s `EfficientNet`](https://github.com/lukemelas/EfficientNet-PyTorch#about-efficientnet) is used to construct the model network.

<br>

## Discriminator

-   Train on the training set
-   Evaluate regularly using the validation set (Need to compute ROC & AUC)