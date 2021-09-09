import numpy as np

#############################################
#                variable                   #
#############################################
val = False
x = 0
y = 0


#############################################
#                function                   #
#############################################
def RMSD(A, B):
    mse = np.sum(np.power(A - B, 2)/B.size)
    return np.sqrt(mse)

def MAE(A, B):
    A = 59.2/2*(A + 1)
    B = 59.2/2*(B + 1)
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

