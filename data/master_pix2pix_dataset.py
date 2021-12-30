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

import numpy as np

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

        parser.add_argument('--matrix', type=str, default='Ca', help='input matrix')
        parser.add_argument('--LOOid', type=int, default=-1, help='Leave-one-out cross-validation id')
        parser.add_argument('--diff', type=bool, default=False)
        
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
            # my_transforms.clip(),
            my_transforms.preprocess(input_n),
            transforms.ToTensor(),
            # my_transforms.pad()
            # transforms.Normalize((0.5,), (0.5,)),
            ])

        transform_B = transforms.Compose([
            # my_transforms.clip(),
            my_transforms.preprocessB(input_n),
            transforms.ToTensor(),
            # my_transforms.pad()
            ])

        # self.Flip = my_transforms.rotate()

        data_A = data.data_A
        data_B = data.data_B
        d_B = data.data_B
        if opt.diff:
            data_B = [data_B[i//3] - data_A[i] for i in range(len(data_A))]
        else:
            data_B = [data_B[i//3] for i in range(len(data_A))]

        if opt.LOOid < 0:
            val_A = [data_A[i] for i in range(3)]
            val_B = [data_B[i] for i in range(3)]
        else:
            val_A = data_A[opt.LOOid*3:opt.LOOid*3 + 3]
            data_A = data_A[:opt.LOOid*3] + data_A[opt.LOOid*3 + 3:]                
            val_B = data_B[opt.LOOid*3:opt.LOOid*3 + 3]
            data_B = data_B[:opt.LOOid*3] + data_B[opt.LOOid*3 + 3:]
            d_B = d_B[:opt.LOOid] + d_B[opt.LOOid + 1:]

            '''val_A = data_A[1:3] + data_A[4:6] + data_A[7:9] + data_A[10:]
            val_B = data_B[1:3] + data_B[4:6] + data_B[7:9] + data_B[10:]
            data_A = [data_A[0]] + [data_A[3]] + [data_A[6]] + [data_A[9]]
            data_B = [data_B[0]] + [data_B[3]] + [data_B[6]] + [data_B[9]]'''

            
            '''A = data_A
            B = data_B
            val_A = []
            val_B = []
            data_A = []
            data_B = []
            for i in range(0, 12, 3):
                print(i)
                p = np.random.randint(0, 2 + 1)
                if p == 0:
                    val_A += [A[i + 1]] + [A[i + 2]]
                    val_B += [B[i + 1]] + [B[i + 2]]
                elif p == 1:
                    val_A += [A[i + 0]] + [A[i + 2]]
                    val_B += [B[i + 0]] + [B[i + 2]]
                elif p == 2:
                    val_A += [A[i + 0]] + [A[i + 1]]
                    val_B += [B[i + 0]] + [B[i + 1]]
                data_A += [A[i + p]]
                data_B += [B[i + p]]'''

            '''data_A.append(val_A[0])
            data_B.append(val_B[0])
            val_A.pop(0)
            val_B.pop(0)'''

        if not my_util.val:
            # data_A += [my_transforms.rotate(i) for i in data_A]
            # data_B += [my_transforms.rotate(i) for i in data_B]
            self.data_A = [transform_A(i) for i in data_A]
            self.data_B = [transform_B(i) for i in data_B]
            self.data_A += [transform_A(i) for i in d_B]
            self.data_B += [transform_B(i) for i in d_B]
        else:
            self.data_A = [transform_A(i) for i in val_A]
            self.data_B = [transform_B(i) for i in val_B]

        self.rotate = my_transforms.rotate()
        self.choose = my_transforms.mask()

        '''
        self.x = my_util.x
        self.y = my_util.y
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
        '''
        if my_util.x != self.x or my_util.y != self.y:
            self.x = my_util.x
            self.y = my_util.y
            self.A = []
            self.B = []
            self.idx = []

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
        A = self.data_A[index%len(self.data_A)]

        B = self.data_B[index%len(self.data_B)]

        if not my_util.val:
            A, B = self.rotate(A, B)
        A, B, Pos = self.choose(A, B)

        idx = 0# torch.tensor(self.idx[index%len(self.idx)], dtype=torch.long)

        dm = 0# self.domain[index]
        
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path, 'idx': idx, 'domain':dm, 'Pos': Pos}

    def __len__(self):
        """Return the total number of images."""
        return max(len(self.data_A), len(self.data_B))
