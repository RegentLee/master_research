import mdtraj as md

from data.MyFunction.my_matrix import my_get_matrix
from util import my_util

class CryptoSiteDataCreator:
    def __init__(self, opt):
        self.data_A = []
        self.data_B = []

        self.matrix = my_get_matrix(opt)

        self.data_load()
        self.make_matrix()
        self.clean_up()


    def data_load(self):
        # dataload
        data_root = '/gs/hs0/tga-science/lee/master_research/CryptoSiteData/'
        if not my_util.val:
            data_root += 'train/'
            data_len = 80
        else:
            data_root += 'test/'
            data_len = 14
        
        for i in range(data_len):
            apo = md.load(data_root + 'apo_' + str(i + 1) + '.pdb')
            holo = md.load(data_root + 'holo_' + str(i + 1) + '.pdb')
            self.data_A.append(apo)
            self.data_B.append(holo)

    def make_matrix(self):
        self.data_A = [self.matrix(apo) for apo in self.data_A]
        self.data_B = [self.matrix(holo) for holo in self.data_B]

    def clean_up(self):
        data_A = []
        data_B = []
        id = []
        for i in range(len(self.data_A)):
            if len(self.data_A[i]) == len(self.data_B[i]) and len(self.data_A[i]) < 500:
                data_A.append(self.data_A[i])
                data_B.append(self.data_B[i])
                id.append(i + 1)
                print(i + 1, len(self.data_A[i]))
        self.data_A = data_A
        self.data_B = data_B
        print(id)


if __name__ == "__main__":
    data = CryptoSiteDataCreator(None)
    matrix_size = [len(i) for i in data.data_A]
    print(matrix_size)
