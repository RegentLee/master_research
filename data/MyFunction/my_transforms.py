import numpy as np


class preprocess:
    def __init__(self, input_n):
        self.input_n = input_n
        
    def __call__(self, image):
        image = image.astype('float32')
        image = (image / (np.max(image)/2)) - 1
        # image = np.pad(image, self.input_n-len(image))
        # image = image[len(image)//2 - self.input_n//2:len(image)//2 + self.input_n//2, len(image)//2 - self.input_n//2:len(image)//2 + self.input_n//2]
        # image = image[np.newaxis, :, :]
        return image

class rotate:
    def __init__(self):
        pass
        
    def __call__(self, image):
        image = image[::-1][:, ::-1]
        return image

class CE:
    def __init__(self):
        pass

    def __call__(self, image):
        image = image.astype('float32')
        image = np.floor((image / (np.max(image)))*63)
        return image

def crop(x, y, img):
    x = x % (len(img)%64)
    y = y % (len(img)%64)
    img_size = len(img)
    crop_img = []
    for i in range(x, img_size, 64):
        for j in range(y, img_size, 64):
            crop_img.append(img[x:x + 64, y:y + 64])
    return crop_img