import numpy as np
import torchvision.transforms as transforms
import torch

from util import my_util


class preprocess:
    def __init__(self, input_n):
        self.input_n = input_n
        
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
    def __init__(self, input_n):
        self.input_n = input_n
        
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

'''class rotate:
    def __init__(self):
        self.Flip = transforms.Compose([
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomHorizontalFlip(p=1)
            ])
        
    def __call__(self, A, B):
        p = np.random.randint(0, 2)
        if p == 1:
            A = self.Flip(A)
            B = self.Flip(B)
        return A, B'''

def rotate(image):
    img = image[::-1][:, ::-1]
    return img

class CE:
    def __init__(self):
        pass

    def __call__(self, image):
        image = image.astype('float32')
        image = np.floor((image / (np.max(image)))*63)
        return image

def crop(x, y, imgA, imgB):
    assert(len(imgA[0]) == len(imgB[0]))
    x = x % (len(imgA[0])%64)
    y = y % (len(imgA[0])%64)
    # print(x, y)
    img_size = len(imgA[0])
    crop_imgA = []
    crop_imgB = []
    for i in range(y, img_size - 64 + 1, 64):
        for j in range(x, img_size - 64 + 1, 64):
            crop_imgA.append(imgA[:, j:j + 64, i:i + 64])
            crop_imgB.append(imgB[:, j:j + 64, i:i + 64])
    return crop_imgA, crop_imgB

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
        n = np.random.randint(0, 2 + 1)

        # print(A.shape)

        for i in range(n):
            len_A = len(A[0])
            c = np.random.randint(1, len_A - 1)

            A = torch.cat([A[:, :c], A[:, c + 1:]], dim=1)
            # print("dim 1", A.shape)
            A = torch.cat([A[:, :, :c], A[:, :, c + 1:]], dim=2)
            # print("dim 2", A.shape)

            B = torch.cat([B[:, :c], B[:, c + 1:]], dim=1)
            B = torch.cat([B[:, :, :c], B[:, :, c + 1:]], dim=2)

        return A, B


def crop_244(x, y, imgA, imgB):
    assert(len(imgA[0]) == len(imgB[0]))
    x = x % (len(imgA[0])%244)
    y = y % (len(imgA[0])%244)
    # print(x, y)
    A = imgA[:, x:x + 244, y:y + 244]
    B = imgB[:, x:x + 244, y:y + 244]
    return A, B
        
