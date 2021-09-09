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
import torch
import torchvision.transforms as transforms

from data.MyFunction import my_data_creator
from data.MyFunction import my_transforms
from util import my_util

class MasterPix2PixDataset(BaseDataset):
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

        parser.add_argument('--matrix', type=str, default='Cb', help='input matrix')
        parser.add_argument('--LOOid', type=int, default=-1, help='Leave-one-out cross-validation id')
        parser.add_argument('--diff', type=bool, default=False)

        parser.set_defaults(input_nc=1, output_nc=1)  # specify dataset-specific default values
        
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
        
        data = my_data_creator.MyDataCreator(opt)

        matrix_size = [len(i) for i in data.data_A]
        input_n = max(matrix_size)
        
        for i in range(4):
            if input_n%4 == 0:
                break
            input_n += 1
        
        transform_A = transforms.Compose([
            my_transforms.preprocess(input_n),
            transforms.ToTensor()
            ])

        transform_B = transforms.Compose([
            my_transforms.CE(),
            transforms.ToTensor()
            ])

        data_A = data.data_A
        if opt.diff:
            data_B = [data.data_B[i//3] - data_A[i] for i in range(len(data_A))]
        else:
            data_B = [data.data_B[i//3] for i in range(len(data_A))]

        if opt.LOOid < 0:
            val_A = [data_A[i] for i in range(3)]
            val_B = [data_B[i] for i in range(3)]
        else:
            val_A = [data_A[i] for i in range(opt.LOOid*3, opt.LOOid*3 + 3)]
            data_A = data_A[:opt.LOOid*3] + data_A[opt.LOOid*3 + 3:]                
            val_B = data_B[opt.LOOid*3:opt.LOOid*3 + 3]
            data_B = data_B[:opt.LOOid*3] + data_B[opt.LOOid*3 + 3:]

        if not my_util.val:
            self.data_A = [transform(i) for i in data_A]
            self.data_B = [transform(i) for i in data_B]
        else:
            self.data_A = [transform(i) for i in val_A]
            self.data_B = [transform(i) for i in val_B]

        self.x = my_util.x
        self.y = my_util.y
        self.A = []
        self.B = []

        for i in self.data_A:
            self.A += my_transforms.crop(self.x, self.y, i)
        for j in self.data_B:
            self.B += my_transforms.crop(self.x, self.y, j)


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
            self.B = []

            for i in self.data_A:
                self.A += my_transforms.crop(self.x, self.y, i)
            for j in self.data_B:
                self.B += my_transforms.crop(self.x, self.y, j)

        A = self.data_A[index % len(self.A)]

        B = self.data_B[index % len(self.B)]
        
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images."""
        return max(len(self.A), len(self.B))

