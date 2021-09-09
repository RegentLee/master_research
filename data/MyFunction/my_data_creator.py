import mdtraj as md
import numpy as np

from data.MyFunction.my_matrix import my_get_matrix

class MyDataCreator:
    def __init__(self, opt):
        self.data_A = []
        self.data_B = []
        self.val_A = []
        self.val_B = []
        self.__temp = []

        self.matrix = my_get_matrix(opt)
        self.loo_id = opt.LOOid
        self.diff = opt.diff
        self.model = opt.model

        self.MD_result_load()
        self.make_matrix()


    def MD_result_load(self):
        # dataload
        data_root = '/gs/hs0/tga-science/lee/master_research/frames/'
        # 1byg
        traj_1byg_0 = md.load(data_root + '1byg_0.pdb')
        traj_1byg_1 = md.load(data_root + '1byg_1.pdb')
        traj_1byg_2 = md.load(data_root + '1byg_2.pdb')
        traj_1byg_3 = md.load(data_root + '1byg_3.pdb')
        self.__temp.append([traj_1byg_0, traj_1byg_1, traj_1byg_2, traj_1byg_3])

        # 1qpj
        traj_1qpj_0 = md.load(data_root + '1qpj_0.pdb')
        traj_1qpj_1 = md.load(data_root + '1qpj_1.pdb')
        traj_1qpj_2 = md.load(data_root + '1qpj_2.pdb')
        traj_1qpj_3 = md.load(data_root + '1qpj_3.pdb')
        self.__temp.append([traj_1qpj_0, traj_1qpj_1, traj_1qpj_2, traj_1qpj_3])

        # 2dq7
        traj_2dq7_0 = md.load(data_root + '2dq7_0.pdb')
        traj_2dq7_1 = md.load(data_root + '2dq7_1.pdb')
        traj_2dq7_2 = md.load(data_root + '2dq7_2.pdb')
        traj_2dq7_3 = md.load(data_root + '2dq7_3.pdb')
        self.__temp.append([traj_2dq7_0, traj_2dq7_1, traj_2dq7_2, traj_2dq7_3])

        # 3a4o
        traj_3a4o_0 = md.load(data_root + '3a4o_0.pdb')
        traj_3a4o_1 = md.load(data_root + '3a4o_1.pdb')
        traj_3a4o_2 = md.load(data_root + '3a4o_2.pdb')
        traj_3a4o_3 = md.load(data_root + '3a4o_3.pdb')
        self.__temp.append([traj_3a4o_0, traj_3a4o_1, traj_3a4o_2, traj_3a4o_3])


    def make_matrix(self):
        for i in range(len(self.__temp)):
            first = self.matrix(self.__temp[i][0])
            self.data_B.append(first)
            for j in range(1, len(self.__temp[i])):
                last = self.matrix(self.__temp[i][j])
                self.data_A.append(last)

