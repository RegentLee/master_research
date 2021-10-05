import numpy as np
import random

from data.MyFunction import my_transforms

#############################################
#                variable                   #
#############################################
val = False
x = 0
y = 0

category = [i for i in range(64)]
distance = [int(100*(2 + i*0.32))/100 for i in range(63)]

#############################################
#                function                   #
#############################################
'''score function'''
def RMSD(A, B):
    mse = np.sum(np.power(A - B, 2)/B.size)
    return np.sqrt(mse)

def MAE(A, B):
    # A = 59.2/2*(A + 1)
    # B = 59.2/2*(B + 1)
    A = distance[-1]/2*(A + 1)
    B = distance[-1]/2*(B + 1)
    mae = np.sum(np.abs(A - B))/B.size
    return mae

'''def DALI(A, B): not used
    """Citation:
    Holm, Liisa. 
    "DALI and the persistence of protein shape." 
    Protein Science 29.1 (2020): 128-140.
    APPENDIX I: SCORES USED IN DALI
    """
    DALI_score = 0.2*len(B)
    A = 10*((A + 1)*3)
    B = 10*((B + 1)*3)
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            DALI_score += 2*(0.2 - 2*np.abs(A[i][j] - B[i][j])/(A[i][j] + B[i][j]))*np.exp(-((A[i][j] + B[i][j])/(2*20))**2)
    m_L = 7.95 + 0.71*len(B) - 0.000259*len(B)**2 - 0.00000192*len(B)**3
    Z_score = (DALI_score - m_L)/(0.5*m_L)
    return Z_score'''

'''dis2cat & cat2dis'''
def dis2cat(x):
    if x < distance[0]:
        return category[0]
    elif x >= distance[-1]:
        return category[-1]
    else:
        return int((x - 2)//0.32 + 1)

def cat2dis(x):
    if x == 0:
        return 0
    else:
        return distance[int(x) - 1]

def crop(x, y, img):
    x = x % (len(img)%64)
    y = y % (len(img)%64)
    img_size = len(img[0][0])
    crop_img = []
    for i in range(y, img_size - 64 + 1, 64):
        for j in range(x, img_size - 64 + 1, 64):
            crop_img.append(img[:, :, j:j + 64, i:i + 64])
    return crop_img

def train(model, data):
    A = data['A']
    B = data['B']
    A_paths = data['A_paths']
    B_paths = data['B_paths']
    
    data_A = crop(x, y, A)
    data_B = crop(x, y, B)

    idx = [i for i in range(len(data_A))]
    random.shuffle(idx)
    data_A =[data_A[idx[i]] for i in range(len(data_A))]
    data_B = [data_B[idx[i]] for i in range(len(data_A))]

    for i in range(len(data_A)):
        img = {'A': data_A[i], 'B': data_B[i], 'A_paths': A_paths, 'B_paths': B_paths}
        model.set_input(img)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

def test(model, data):
    A = data['A']
    B = data['B']
    A_paths = data['A_paths']
    B_paths = data['B_paths']

    xy = [0, len(A[0][0])%64]
    result = []
    prob = []
    for y in xy:
        for x in xy:
            data_A = crop(x, y, A)
            result_crop = []
            prob_crop = []
            for i in range(len(data_A)):
                img = {'A': data_A[i], 'B': B, 'A_paths': A_paths, 'B_paths': B_paths}
                model.set_input(img)
                answer = model.image.to('cpu').detach().numpy().copy()[0][0]
                prob_c = model.prob.to('cpu').detach().numpy().copy()[0][0]
                result_crop.append(answer)
                prob_crop.append(prob_c)
            result.append(result_crop)
            prob.append(prob_crop)
    result = np.array(result)
    prob = np.array(prob)
    # print(result.shape)

    image = np.zeros([len(result), len(A[0][0]), len(A[0][0])])
    prob_image = np.zeros([len(prob), len(A[0][0]), len(A[0][0])])
    for y in range(len(xy)):
        for x in range(len(xy)):
            for i in range(xy[y], len(A[0][0]) - 64 + 1, 64):
                for j in range(xy[x], len(A[0][0]) - 64 + 1, 64):
                    # print(j, i)
                    # print(y*len(xy) + x, i//64*int(np.sqrt(len(result[y*len(xy) + x]))) + j//64)
                    image[y*len(xy) + x, j:j + 64, i:i + 64] = result[y*len(xy) + x, i//64*int(np.sqrt(len(result[y*len(xy) + x]))) + j//64, :, :]
                    prob_image[y*len(xy) + x, j:j + 64, i:i + 64] = prob[y*len(xy) + x, i//64*int(np.sqrt(len(prob[y*len(xy) + x]))) + j//64, :, :]
            # print(image[y*len(xy) + x])

    img = np.empty([len(A[0][0]), len(A[0][0])])
    for i in range(len(A[0][0])):
        for j in range(len(A[0][0])):
            point = np.argmax(prob_image[:, i, j])
            img[i][j] = image[point, i, j]

    score = MAE
    
    data_B = data['B'].numpy()[0][0]
    data_B = data_B/(np.max(data_B)/2) - 1
    org = score(data['A'].numpy()[0][0], data_B)
    last = score(data['A'].numpy()[0][0], img)
    first = score(img, data_B)

    return np.array([org, last, first])