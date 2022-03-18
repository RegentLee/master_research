"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import torchvision.transforms as transforms

from data.MyFunction import CryptoSiteDataMD_creator
from data.MyFunction import my_transforms
from util import my_util

import numpy as np
import torch

class TestDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        # self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)
        
        data = CryptoSiteDataMD_creator.CryptoSiteDataMDTestCreator(opt)
        
        transform_A = transforms.Compose([
            # my_transforms.clip(),
            my_transforms.preprocess(),
            transforms.ToTensor(),
            # my_transforms.pad()
            # transforms.Normalize((0.5,), (0.5,)),
            ])

        transform_B = transforms.Compose([
            # my_transforms.clip(),
            my_transforms.preprocessB(),
            transforms.ToTensor(),
            # my_transforms.pad()
            ])

        # self.Flip = my_transforms.rotate()

        data_A = data.data_A
        data_B = data.data_B
        # if not my_util.val:
        #     for i in range(len(data_B)):
        #         if len(data_A[i*1001]) != len(data_B[i]):
        #             print(i, len(data_A[i*1001]), len(data_B[i]))

        if not my_util.val:
            # data_A += [my_transforms.rotate(i) for i in data_A]
            # data_B += [my_transforms.rotate(i) for i in data_B]
            self.data_A = [transform_A(i) for i in data_A]
            self.data_B = [transform_B(i) for i in data_B]
            # self.data_A += [transform_A(i) for i in d_B]
            # self.data_B += [transform_B(i) for i in d_B]
        else:
            self.data_A = [transform_A(i) for i in data_A]
            self.data_B = [transform_B(i) for i in data_B]

        self.rotate = my_transforms.rotate()

        self.x = my_util.x
        self.y = my_util.y
        self.A = []

        if len(self.data_A) == len(self.data_B): # val data
            self.A = self.data_A
        else:
            self.A = [self.data_A[i] for i in range(self.x, len(self.data_A), 10)]
        '''
        self.A = []
        self.B = []
        self.idx = []

        for i in range(len(self.data_A)):
            temp = my_transforms.crop(self.x, self.y, self.data_A[i], self.data_B[i])
            self.A += temp[0]
            self.B += temp[1]
            self.idx += temp[2]
        '''
        '''
        if not my_util.val:
            self.domain = [0 if i < len(self.data_A)//2 else 1 for i in range(len(self.data_A))]
        else:
            self.domain = [0]*len(self.data_A)
        '''

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = 'temp'    # needs to be a string
        # data_A = torch.Tensor(self.data.data_A)    # needs to be a tensor
        # data_B = torch.Tensor(self.data.data_B)    # needs to be a tensor
        
        if my_util.x != self.x or my_util.y != self.y:
            self.x = my_util.x
            self.y = my_util.y
            self.A = []
            # self.B = []

            if len(self.data_A) == len(self.data_B): # val data
                self.A = self.data_A
            else:
                self.A = [self.data_A[i] for i in range(self.x, len(self.data_A), 10)]
            '''
            for i in range(len(self.data_A)):
                if not my_util.val:
                    Ac, Bc = self.choose(self.data_A[i], self.data_B[i])
                else:
                    Ac, Bc = self.data_A[i], self.data_B[i]
                temp = my_transforms.crop(self.x, self.y, Ac, Bc)
                self.A += temp[0]
                self.B += temp[1]
                self.idx += temp[2]
            '''
        
        if len(self.data_A) == len(self.data_B): # val data
            A = self.data_A[index%len(self.data_A)]
            B = self.data_B[index%len(self.data_B)]
        else: # train data
            A = self.A[index%len(self.A)]
            B = self.data_B[index//100]

        if not my_util.val:
            A, B = self.rotate(A, B)
        
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images."""
        return max(len(self.A), len(self.data_B))


def create_test_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = TestDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class TestDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = TestDataset(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data