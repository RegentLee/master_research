import numpy as np
import torchvision.transforms as transforms
import torch

from util import my_util

import random


class preprocess:
    def __init__(self):
        pass

    def __call__(self, image):
        image = image.astype('float32')
        # image = (image / (np.max(image)/2)) - 1
        m_max = my_util.distance[-1]*2
        image = np.where(image > m_max, m_max, image)
        m_min = 0
        image = (image - m_min)/(m_max - m_min)*2 - 1
        # image = np.pad(image, self.input_n-len(image))
        # image = image[len(image)//2 - self.input_n//2:len(image)//2 + self.input_n//2, len(image)//2 - self.input_n//2:len(image)//2 + self.input_n//2]
        # image = image[np.newaxis, :, :]
        return image

class preprocessB:
    def __init__(self):
        pass
        
    def __call__(self, image):
        image = image.astype('float32')
        # image = (image / (np.max(image)/2)) - 1
        m_max = my_util.distance[-1]
        image = np.where(image > m_max, m_max, image)
        m_min = 0
        image = (image - m_min)/(m_max - m_min)*2 - 1
        # image = np.pad(image, self.input_n-len(image))
        # image = image[len(image)//2 - self.input_n//2:len(image)//2 + self.input_n//2, len(image)//2 - self.input_n//2:len(image)//2 + self.input_n//2]
        # image = image[np.newaxis, :, :]
        return image

class rotate:
    def __init__(self):
        self.Flip = transforms.Compose([
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomHorizontalFlip(p=1)
            ])
        
    def __call__(self, A, B):
        p = np.random.rand()
        if p < 0.5:
            A = self.Flip(A)
            B = self.Flip(B)
        return A, B
'''
def rotate(image):
    img = image[::-1][:, ::-1]
    return img
'''
class CE:
    def __init__(self):
        pass

    def __call__(self, image):
        image = image.astype('float32')
        image = np.floor((image / (np.max(image)))*63)
        return image

'''
def crop(x, y, imgA, imgB):
    assert(len(imgA[0]) == len(imgB[0]))
    if len(imgA[0]) % 64 !=0:
        x = x % (len(imgA[0])%64)
        y = y % (len(imgA[0])%64)
    else:
        x = 0
        y = 0
    # print(x, y)
    img_size = len(imgA[0])
    crop_imgA = []
    crop_imgB = []
    idx = []
    idx_i = 0
    idx_j = 0
    for i in range(y, img_size - 64 + 1, 64):
        for j in range(x, img_size - 64 + 1, 64):
            crop_imgA.append(imgA[:, j:j + 64, i:i + 64])
            crop_imgB.append(imgB[:, j:j + 64, i:i + 64])
            idx.append(idx_i*4 + idx_j)
            idx_j += 1
        idx_i += 1
        idx_j = 0
    idx = [i if i < 8 else 15 - i for i in idx]
    return crop_imgA, crop_imgB, idx
'''

class clip:
    def __init__(self):
        pass

    def __call__(self, image):
        img = np.zeros_like(image, dtype=float)
        for i in range(len(image)):
            for j in range(len(image)):
                if image[i][j] < my_util.distance[0]:
                    pass
                elif image[i][j] >= my_util.distance[-1]:
                    img[i][j] = my_util.distance[-1]
                else:
                    img[i][j] = ((image[i][j] - 2 + 1e-10)//0.32*0.32 + 2)*1000//10/100
        return img


class choose:
    def __init__(self):
        pass

    def __call__(self, A, B):
        p = np.random.rand()
        if p < 0.2:
            # print(len(A[0]) - 256)
            # n_m = min(len(A[0]) - 256, 5)
            n_m = len(A[0]) - 240
            # p = np.random.randint(1, 246 - 1)
            n = np.random.randint(1, n_m + 1)

            # print(A.shape)

            for _ in range(n + 1):
                len_A = len(A[0])
                c = np.random.randint(1, len_A - 1)

                A = torch.cat([A[:, :c], A[:, c + 1:]], dim=1)
                # print("dim 1", A.shape)
                A = torch.cat([A[:, :, :c], A[:, :, c + 1:]], dim=2)
                # print("dim 2", A.shape)

                B = torch.cat([B[:, :c], B[:, c + 1:]], dim=1)
                B = torch.cat([B[:, :, :c], B[:, :, c + 1:]], dim=2)

        return A, B

class mask:
    def __init__(self):
        pass

    def __call__(self, A, B):
        if not my_util.val:    
            # p = np.random.rand()
            p = 1
            if p < 0.5:
                # print(len(A[0]) - 256)
                # n_m = min(len(A[0]) - 256, 5)
                # n_m = len(A[0]) - 240
                n_m = int(len(A[0])*0.2)
                # p = np.random.randint(1, 246 - 1)
                # n = np.random.randint(1, n_m + 1)
                A_mean = torch.mean(A)
                Pos = torch.zeros_like(A)

                n = random.sample(range(len(A[0])), n_m)

                A[:, n] = 0
                A[:, :, n] = 0
                
                Pos[:, n] = 1
                Pos[:, :, n] = 1
            else:
                Pos = torch.zeros_like(A)
        else:
            Pos = torch.zeros_like(A)

        return A, B, Pos

class pad:
    def __init__(self):
        self.padding = transforms.Pad(5, padding_mode='symmetric')

    def __call__(self, img):
        if len(img[0]) < 256:
            img = self.padding(img)
        return img


class crop:
    def __init__(self) -> None:
        pass

    def __call__(self, imgA, imgB, pos):
        assert(len(imgA[0]) == len(imgB[0]))
        if len(imgA[0]) > 256:
            x = np.random.randint(len(imgA[0]) - 256)
            y = np.random.randint(len(imgA[0]) - 256)

            A = imgA[:, x:x + 256, y:y + 256]
            B = imgB[:, x:x + 256, y:y + 256]
            pos = pos[:, x:x + 256, y:y + 256]

            return A, B, pos
        else:
            return imgA, imgB, pos
