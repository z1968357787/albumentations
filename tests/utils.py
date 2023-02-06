import random
import typing
from io import StringIO
from typing import Optional, Set

import numpy as np

from albumentations.utils import get_transforms


def convert_2d_to_3d(arrays, num_channels=3):
    # Converts a 2D numpy array with shape (H, W) into a 3D array with shape (H, W, num_channels)
    # by repeating the existing values along the new axis.
    arrays = tuple(np.repeat(array[:, :, np.newaxis], repeats=num_channels, axis=2) for array in arrays)
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def convert_2d_to_target_format(arrays, target):
    if target == "mask":
        return arrays[0] if len(arrays) == 1 else arrays
    if target == "image":
        return convert_2d_to_3d(arrays, num_channels=3)
    if target == "image_4_channels":
        return convert_2d_to_3d(arrays, num_channels=4)

    raise ValueError("Unknown target {}".format(target))


class InMemoryFile(StringIO):
    def __init__(self, value, save_value, file):
        super().__init__(value)
        self.save_value = save_value
        self.file = file

    def close(self):
        self.save_value(self.getvalue(), self.file)
        super().close()


class OpenMock:
    """
    Mocks the `open` built-in function. A call to the instance of OpenMock returns an in-memory file which is
    readable and writable. The actual in-memory file implementation should call the passed `save_value` method
    to save the file content in the cache when the file is being closed to preserve the file content.
    """

    def __init__(self):
        self.values = {}

    def __call__(self, file, *args, **kwargs):
        value = self.values.get(file)
        return InMemoryFile(value, self.save_value, file)

    def save_value(self, value, file):
        self.values[file] = value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def check_all_augs_exists(
    augmentations: typing.List[typing.List],
    except_augmentations: Optional[Set] = None,
) -> typing.List[typing.List]:
    existed_augs = {i[0] for i in augmentations}
    except_augmentations = except_augmentations or set()

    not_existed = []

    for cls, _ in get_transforms(except_augmentations=except_augmentations):
        if cls not in existed_augs:
            not_existed.append(cls.__name__)

    if not_existed:
        raise ValueError(f"These augmentations do not exist in augmentations and except_augmentations: {not_existed}")

    return augmentations
