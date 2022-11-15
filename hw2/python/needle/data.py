import gzip
import struct
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

import numpy as np

from .autograd import Tensor


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        img_pad = np.pad(img, [(self.padding, self.padding), (self.padding, self.padding), (0, 0)], 'constant')
        H, W, _ = img_pad.shape
        return img_pad[self.padding + shift_x: H - self.padding + shift_x, self.padding + shift_y: W - self.padding + shift_y, :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.idx = -1
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        self.idx += 1
        if self.idx >= len(self.ordering):
            raise StopIteration
        samples = [self.dataset[i] for i in self.ordering[self.idx]]
        return [Tensor(np.stack([samples[i][j] for i in range(len(samples))])) for j in range(len(samples[0]))]
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        with gzip.open(image_filename, "rb") as img_file:
            magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
            assert(magic_num == 2051)
            tot_pixels = row * col
            imgs = [np.array(struct.unpack(f"{tot_pixels}B",
                                           img_file.read(tot_pixels)),
                                           dtype=np.float32)
                    for _ in range(img_num)]
            X = np.vstack(imgs)
            X -= np.min(X)
            X /= np.max(X)
            self.X = X

        with gzip.open(label_filename, "rb") as label_file:
            magic_num, label_num = struct.unpack(">2i", label_file.read(8))
            assert(magic_num == 2049)
            self.y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.X[index]
        labels = self.y[index]
        if len(imgs.shape) > 1:
            imgs = np.vstack([self.apply_transforms(img.reshape(28, 28, 1)) for img in imgs])
        else:
            imgs = self.apply_transforms(imgs.reshape(28, 28, 1))
        return (imgs, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
