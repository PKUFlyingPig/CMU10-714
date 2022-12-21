import numpy as np
from .autograd import Tensor
import gzip
import os
import pickle
import struct
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: C x H x W NDArray of an image
        Returns:
            C x H x W ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, :, ::-1]
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: C x H x W NDArray of an image
        Return
            C x H x W NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        img_pad = np.pad(img, [(0, 0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        _, H, W = img_pad.shape
        return img_pad[:, self.padding + shift_x: H - self.padding + shift_x, self.padding + shift_y: W - self.padding + shift_y]
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
        device = None
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

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
        batch_indices = self.ordering[self.idx]
        return tuple([Tensor(x, device = self.device) for x in self.dataset[batch_indices]])
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


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        import pickle
        if train:
            data_batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            data_batch_files = ['test_batch']
        X = []
        Y = []
        for data_batch_file in data_batch_files:
            with open(os.path.join(base_folder, data_batch_file), 'rb') as f:
                data_dict = pickle.load(f, encoding = 'bytes')
                X.append(data_dict[b'data'])
                Y.append(data_dict[b'labels'])
        X = np.concatenate(X, axis = 0)
        # preprocessing X.
        X = X / 255.
        X = X.reshape((-1, 3, 32, 32))
        Y = np.concatenate(Y, axis = None) # Y is just 1-dimensional.
        self.X = X
        self.Y = Y
        self.transforms=transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if self.transforms:
            image = np.array([self.apply_transforms(img) for img in self.X[index]])
        else:
            image = self.X[index]
        label = self.Y[index]
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.Y)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.words = set()

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.words:
            self.words.add(word)
            uid = len(self.idx2word)
            self.word2idx[uid] = word
            self.idx2word.append(word)
        else:
            uid = self.idx2word.index(word)
        return uid
        
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.words)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        eos_id = self.dictionary.add_word("<eos>")

        def tokenize_one_line(line):
            words = line.split()
            for word in words:
                ids.append(self.dictionary.add_word(word))
            ids.append(eos_id)

        with open(path, "r") as f:
            if max_lines:
                for _ in range(max_lines):
                    tokenize_one_line(f.readline())
            else:
                for line in f:
                    tokenize_one_line(line)
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    nbatch = len(data) // batch_size
    data = np.array(data[:nbatch * batch_size]).reshape((nbatch, batch_size))
    return data
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype="float32"):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    tot_seqlen = batches.shape[0]
    assert i < tot_seqlen - 1
    if i + bptt + 1 > tot_seqlen:
        X = batches[i : -1, :]
        y = batches[i+1 : , :].flatten()
    else:
        X = batches[i : i + bptt, :]
        y = batches[i + 1: i + 1 + bptt, :].flatten()
    return Tensor(X, device=device, dtype=dtype), Tensor(y, device=device, dtype=dtype)
    ### END YOUR SOLUTION