import numpy as np
import cv2
import random
import torch

class Shift(object):
    """
    Shifts input image by random x amount between [-max_shift, max_shift]
      and separate random y amount between [-max_shift, max_shift].
    """

    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape

        x_shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        y_shift = np.random.randint(-self.max_shift, self.max_shift + 1)

        shifted_image = np.zeros((3, H, W))

        for channel in range(3):
            if x_shift > 0:
                shifted_image[channel][:, x_shift:] = image[channel][:, :-x_shift]
            elif x_shift < 0:
                shifted_image[channel][:, :x_shift] = image[channel][:, -x_shift:]

            if y_shift > 0:
                shifted_image[channel][y_shift:, :] = shifted_image[channel][
                    :-y_shift, :
                ]
                shifted_image[channel][:y_shift, :] = 0
            elif y_shift < 0:
                shifted_image[channel][:y_shift, :] = shifted_image[channel][
                    -y_shift:, :
                ]
                shifted_image[channel][y_shift:, :] = 0

        return torch.Tensor(shifted_image)

    def __repr__(self):
        return self.__class__.__name__


class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. 

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        image = image.numpy()

        _, H, W = image.shape
        contrast = random.uniform(self.min_contrast, self.max_contrast)
        for channel in range(3):
            mean = np.mean(image[channel, :, :])
            image[channel] = contrast * image[channel] + (1 - contrast) * mean

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. 

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees

    """

    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        image = image.numpy()
        _, H, W = image.shape

        angle = np.random.uniform(-self.max_angle, self.max_angle)
        rot_matrix = cv2.getRotationMatrix2D((H / 2, W / 2), angle, 1)
        for channel in range(3):
            image[channel] = cv2.warpAffine(image[channel], rot_matrix, (H, W))

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__


class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        image = image.numpy()
        _, H, W = image.shape

        if random.random() <= self.p:
            for channel in range(3):
                image[channel] = np.fliplr(image[channel])

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__