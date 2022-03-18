import mdtraj as md
import os
import pickle

from data.MyFunction.my_matrix import my_get_matrix
from util import my_util

class CryptoSiteDataMDCreator:
    """Create Dataset

    Create dataset after md
    """
    def __init__(self, opt):
        """init

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.data_A = []
        self.data_B = []

        self.matrix = my_get_matrix(opt)

        self.data_load()
        self.make_matrix()
        if my_util.val:
            self.clean_up()


    def data_load(self):
        """load data

        load train data.
        load input matrix to self.data_A
        load output data(mdtraj) to self.data_B
        """
        data_root = '/gs/hs0/tga-science/lee/master_research/CryptoSiteData/'
        if not my_util.val:
            apo_root = data_root + 'apo_amber/'
            holo_root = data_root + 'train/'
            data_len = 80

            no = [1, 3, 26, 51, 68]

            for i in range(data_len):
                if os.path.isfile(apo_root + 'apo_' + str(i + 1)) and (i + 1 not in no):
                    with open(apo_root + 'apo_' + str(i + 1), 'rb') as f:
                        # temp = 
                        self.data_A += pickle.load(f)[:1000]
                    holo = md.load(holo_root + 'holo_' + str(i + 1) + '.pdb')
                    self.data_B.append(holo)
        '''
        else:
            apo_root = data_root + 'apo_amber/'
            holo_root = data_root + 'train/'
            data_len = 80

            for i in range(data_len):
                if os.path.isfile(apo_root + 'apo_' + str(i + 1)) and str(i + 1) != '68' and str(i + 1) != '26':
                    apo = md.load(holo_root + 'apo_' + str(i + 1) + '.pdb')
                    holo = md.load(holo_root + 'holo_' + str(i + 1) + '.pdb')
                    self.data_A.append(apo)
                    self.data_B.append(holo)
            
            apo_root = data_root + 'test/'
            holo_root = data_root + 'test/'
            data_len = 14

            for i in range(data_len):
                apo = md.load(apo_root + 'apo_' + str(i + 1) + '.pdb')
                holo = md.load(holo_root + 'holo_' + str(i + 1) + '.pdb')
                self.data_A.append(apo)
                self.data_B.append(holo)
        '''

    def make_matrix(self):
        """make distacne matrix

        if my_util.val is true
            make distance matrix from mdtraj in self.data_A and put it to self.data_A
        else 
            distance matrix is already in self.data_A 
        make distance matrix from mdtraj in self.data_B and put it to self.data_B 
        """
        if my_util.val:
            self.data_A = [self.matrix(apo) for apo in self.data_A]
        self.data_B = [self.matrix(holo) for holo in self.data_B]
        # for i in range(len(self.data_B)):
        #     print(i + 1, len(self.data_A[i*1000]), len(self.data_B[i]), len(self.data_A[i*1000]) - len(self.data_B[i]))

    def clean_up(self):
        """clean up dataset

        make sure len(input matrix) == len(output matrix)
        """
        data_A = []
        data_B = []
        id = []
        for i in range(len(self.data_A)):
            if len(self.data_A[i]) == len(self.data_B[i]) and len(self.data_A[i]) < 500:
                data_A.append(self.data_A[i])
                data_B.append(self.data_B[i])
                id.append(i + 1)
        self.data_A = data_A
        self.data_B = data_B

class CryptoSiteDataMDTestCreator:
    """Create Test Dataset

    Create dataset from structure that download from protein data bank
    """
    def __init__(self, opt):
        """init

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.data_A = []
        self.data_B = []

        self.matrix = my_get_matrix(opt)

        self.data_load()
        self.make_matrix()
        if my_util.val:
            self.clean_up()


    def data_load(self):
        """load data

        load test data.
        load input data(mdtraj) to self.data_A
        load output data(mdtraj) to self.data_B
        """
        data_root = '/gs/hs0/tga-science/lee/master_research/CryptoSiteData/'
        if not my_util.val:
            apo_root = data_root + 'apo_amber/'
            holo_root = data_root + 'train/'
            data_len = 80

            no = [1, 3, 26, 51, 68]

            for i in range(data_len):
                if os.path.isfile(apo_root + 'apo_' + str(i + 1)) and (i + 1 not in no):
                    apo = md.load(holo_root + 'apo_' + str(i + 1) + '.pdb')
                    holo = md.load(holo_root + 'holo_' + str(i + 1) + '.pdb')
                    self.data_A.append(apo)
                    self.data_B.append(holo)
        else:
            apo_root = data_root + 'test/'
            holo_root = data_root + 'test/'
            data_len = 14

            for i in range(data_len):
                apo = md.load(apo_root + 'apo_' + str(i + 1) + '.pdb')
                holo = md.load(holo_root + 'holo_' + str(i + 1) + '.pdb')
                self.data_A.append(apo)
                self.data_B.append(holo)

    def make_matrix(self):
        """make distacne matrix

        make distance matrix from mdtraj in self.data_A and put it to self.data_A
        make distance matrix from mdtraj in self.data_B and put it to self.data_B 
        """
        self.data_A = [self.matrix(apo) for apo in self.data_A]
        self.data_B = [self.matrix(holo) for holo in self.data_B]

    def clean_up(self):
        """clean up dataset

        make sure len(input matrix) == len(output matrix)
        """
        data_A = []
        data_B = []
        id = []
        for i in range(len(self.data_A)):
            if len(self.data_A[i]) == len(self.data_B[i]) and len(self.data_A[i]) < 500:
                data_A.append(self.data_A[i])
                data_B.append(self.data_B[i])
                id.append(i + 1)
        self.data_A = data_A
        self.data_B = data_B

class RMSD:
    def __init__(self, val=False):
        self.val = val
        self.data_A = []
        self.data_B = []

        self.data_load()
        self.rmsd()


    def data_load(self):
        # dataload
        data_root = '/gs/hs0/tga-science/lee/master_research/CryptoSiteData/'
        if not self.val:
            apo_root = data_root + 'apo/'
            holo_root = data_root + 'train/'
            data_len = 80

            for i in range(data_len):
                if os.path.isfile(apo_root + 'apo_' + str(i + 1)) and str(i + 1) != '68' and str(i + 1) != '26':
                    apo = md.load(holo_root + 'apo_' + str(i + 1) + '.pdb')
                    holo = md.load(holo_root + 'holo_' + str(i + 1) + '.pdb')
                    la = apo.topology.n_atoms
                    lh = holo.topology.n_atoms
                    if la > lh:
                        apo = md.load(holo_root + 'apo_' + str(i + 1) + '.pdb', atom_indices=[i for i in range(holo.topology.n_atoms)])
                    elif lh > la:
                        holo = md.load(holo_root + 'holo_' + str(i + 1) + '.pdb', atom_indices=[i for i in range(apo.topology.n_atoms)])
                    self.data_A.append(apo)
                    self.data_B.append(holo)
        else:
            apo_root = data_root + 'test/'
            holo_root = data_root + 'test/'
            data_len = 14
            id = [1, 2, 5, 8, 9, 10]
            id = [0, 1, 4, 7, 8, 9]

            for i in range(data_len):
                if i in id:
                    apo = md.load(apo_root + 'apo_' + str(i + 1) + '.pdb')
                    holo = md.load(holo_root + 'holo_' + str(i + 1) + '.pdb')
                    la = apo.topology.n_atoms
                    lh = holo.topology.n_atoms
                    if la > lh:
                        apo = md.load(holo_root + 'apo_' + str(i + 1) + '.pdb', atom_indices=[i for i in range(holo.topology.n_atoms)])
                    elif lh > la:
                        holo = md.load(holo_root + 'holo_' + str(i + 1) + '.pdb', atom_indices=[i for i in range(apo.topology.n_atoms)])
                    self.data_A.append(apo)
                    self.data_B.append(holo)

    def rmsd(self):
        r = 0
        for i in range(len(self.data_A)):
            top = self.data_A[i].topology
            selectionA = [atom.index for atom in top.atoms if (atom.name == 'CA')]
            top = self.data_B[i].topology
            selectionB = [atom.index for atom in top.atoms if (atom.name == 'CA')]
            r += md.rmsd(self.data_B[i], self.data_A[i])
        print(r/len(self.data_A))

if __name__ == "__main__":
    RMSD()
    RMSD(val = True)