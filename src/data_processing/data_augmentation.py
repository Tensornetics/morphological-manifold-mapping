import numpy as np
import random
import cv2


def random_flip(image, axis=0):
    """
    Randomly flip an image along the given axis (horizontally or vertically).

    Parameters:
        image (np.ndarray): The input image.
        axis (int, optional): The axis along which to flip the image. 0 for vertical and 1 for horizontal.

    Returns:
        np.ndarray: The flipped image.
    """
    if random.random() < 0.5:
        image = np.flip(image, axis)
    return image


def random_rotation(image, angle_range=10):
    """
    Randomly rotate an image by an angle within a given range.

    Parameters:
        image (np.ndarray): The input image.
        angle_range (int, optional): The range of angles within which to randomly rotate the image.

    Returns:
        np.ndarray: The rotated image.
    """
    angle = random.uniform(-angle_range, angle_range)
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return image


def random_noise(image, sigma=0.1):
    """
    Add random Gaussian noise to an image.

    Parameters:
        image (np.ndarray): The input image.
        sigma (float, optional): The standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: The image with added noise.
    """
    noise = np.random.randn(*image.shape) * sigma
    image = image + noise
    return np.clip(image, 0, 1)


def augment_image(image):
    """
    Augment an image by randomly flipping, rotating, and adding noise to it.

    Parameters:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The augmented image.
    """
    image = random_flip(image)
    image = random_rotation(image)
    image = random_noise(image)
    return image
