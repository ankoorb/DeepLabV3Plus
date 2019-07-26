# Author: Ankoor
## Ref: https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/inference/tta.py
import numpy as np
from torch import nn
from functools import partial

from albumentations import (

    # Non Spatial (image only)
    CLAHE,
    RandomGamma,
    RandomContrast,
    RandomBrightness,
    RandomBrightnessContrast,

    # Blur (image only)
    Blur,

    # Noise (image only)
    GaussNoise,

    # Helper
    Compose,
    Normalize
)

from albumentations.pytorch import ToTensor


def identity(t):
    """
    t: PyTorch Tensor
    """
    return t


def rot90(t):
    """
    t: PyTorch Tensor
    """
    return t.transpose(2, 3).flip(2)


def rot180(t):
    """
    t: PyTorch Tensor
    """
    return t.flip(2).flip(3)


def rot270(t):
    """
    t: PyTorch Tensor
    """
    return t.transpose(2, 3).flip(3)


def transpose(t):
    """
    t: PyTorch Tensor
    """
    return t.transpose(2, 3)


def flip_lr(t):
    """
    t: PyTorch Tensor
    """
    return t.flip(3)


def flip_ud(t):
    """
    t: PyTorch Tensor
    """
    return t.flip(2)


def ArrayToNormTensor(array):
    """
    array: Numpy array of shape [H, W, C]
    """
    convert = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor(sigmoid=False)
    ])

    converted = convert(image=array)

    return converted['image']


def TensorToArray(tensor):
    """
    tensor: PyTorch cuda tensor of shape [1, C, H, W]
    """
    array = tensor.squeeze(0).cpu().data.numpy()  # [C, H, W]

    # [C, H, W] to [H, W, C]
    array = array.transpose(1, 2, 0).astype(np.uint8)

    return array


class UnNormalize(object):
    """
    Ref: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor


def dihedral(model, image):
    """
    TTA for image segmentation that averages predictions
    of all Dihedral group D4 augmentations applied to input 
    image. For segmentation we need to reverse the augmentation 
    after making a prediction on augmented input.

    model: Model to use for making predictions.
    image: PyTorch tensor of size [1, C, H, W]

    return: Arithmetically averaged predictions
    """
    output = model(image)

    for aug, deaug in zip([rot90, rot180, rot270],
                          [rot270, rot180, rot90]):
        temp_output = deaug(model(aug(image)))

        output = output + temp_output

    image = transpose(image)

    for aug, deaug in zip([identity, rot90, rot180, rot270],
                          [identity, rot270, rot180, rot90]):
        temp_output = deaug(model(aug(image)))

        output = output + transpose(temp_output)

    return output * float(1.0 / 8.0)


def non_spatial(model, image):
    """
    model: Model to use for making predictions.
    image: PyTorch tensor (normalized) of size [1, C, H, W]
    """
    output = model(image)

    # Unnormalize tensor
    un_norm = UnNormalize()
    un_tensor = un_norm(image.squeeze(0))  # Values in range [0, 1]
    un_tensor = un_tensor * 255

    # Convert PyTorch Tensor to Numpy Array
    array = TensorToArray(un_tensor)

    augmentations = [CLAHE, RandomBrightness, RandomContrast, RandomBrightnessContrast,
                     RandomGamma, GaussNoise, Blur]

    for aug_type in augmentations:
        aug = aug_type(p=1)

        # Apply augmentation
        augmented = aug(image=array)

        # Convert Numpy Array to Pytorch Tensor
        tensor = ArrayToNormTensor(augmented['image']).unsqueeze(0)  # [1, C, H, W]

        temp_output = model(tensor.cuda())

        output = output + temp_output

    return output * float(1.0 / 8.0)


def dihedral_and_non_spatial(model, image):
    ## Dihedral Group D4
    output = model(image)

    for aug, deaug in zip([rot90, rot180, rot270],
                          [rot270, rot180, rot90]):
        dih_output = deaug(model(aug(image)))

        output = output + dih_output

    dih_image = transpose(image)

    for aug, deaug in zip([identity, rot90, rot180, rot270],
                          [identity, rot270, rot180, rot90]):
        dih_output = deaug(model(aug(dih_image)))

        output = output + transpose(dih_output)

    ## Non Spatial
    augmentations = [CLAHE, RandomBrightness, RandomContrast, RandomBrightnessContrast,
                     RandomGamma, GaussNoise, Blur]

    # Unnormalize tensor
    un_norm = UnNormalize()
    un_tensor = un_norm(image.squeeze(0))  # Values in range [0, 1]
    un_tensor = un_tensor * 255

    # Convert PyTorch Tensor to Numpy Array
    array = TensorToArray(un_tensor)

    for aug_type in augmentations:
        aug = aug_type(p=1)

        # Apply augmentation
        augmented = aug(image=array)

        # Convert Numpy Array to Pytorch Tensor
        tensor = ArrayToNormTensor(augmented['image']).unsqueeze(0)  # [1, C, H, W]

        ns_output = model(tensor.cuda())

        output = output + ns_output

    return output * float(1.0 / 15.0)


def get_tta_function(args):
    if args.tta_fn == 'dihedral':
        return dihedral
    elif args.tta_fn == 'nonspatial':
        return non_spatial
    elif args.tta_fn == 'combined':
        return dihedral_and_non_spatial
    else:
        raise NotImplementedError


class TTA(nn.Module):
    def __init__(self, model, tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, image):
        return self.tta(self.model, image)

