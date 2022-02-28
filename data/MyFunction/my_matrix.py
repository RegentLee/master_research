import numpy as np
import mdtraj as md

def my_get_matrix(opt):
    """Return which distance matrix to use

    Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

    Returns which distance matrix to use
        Ca/Cb/... -- distance matrix function
    """
    if opt.matrix == 'Ca':
        return myCa
    elif opt.matrix == 'Cb':
        return Cb

def Ca(traj):
    """Calculate the distance matrix with alpha Carbon by mdtraj, not good

    Parameters:
        traj -- mdtraj.Trajectory

    Returns distance matrix 
        10*d_matrix[0] (2d numpy.array) -- distance matrix
    """

    temp = md.compute_contacts(traj, scheme='ca')
    d_matrix = md.geometry.squareform(temp[0],temp[1])
    return 10*d_matrix[0]

def myCa(traj):
    """Calculate the distance matrix with alpha Carbon by coordinate

    Parameters:
        traj -- mdtraj.Trajectory

    Returns distance matrix 
        10*temp (2d numpy.array) -- distance matrix
    """

    ca = [atom.index for atom in traj.topology.atoms if (atom.name == 'CA')]
    temp = np.zeros((len(ca), len(ca)), dtype=float)
    for i in range(0, len(ca)):
        for j in range(i + 1, len(ca)):
            temp[i][j] = np.sqrt(np.sum((traj.xyz[:, ca[i], :] - traj.xyz[:, ca[j], :])**2, axis=1))
            temp[j][i] = temp[i][j]
    return 10*temp

def Cb(traj):
    """Calculate the distance matrix with beta Carbon while GLY use alpha Carbon

    Parameters:
        traj -- mdtraj.Trajectory

    Returns distance matrix 
        10*first (2d numpy.array) -- distance matrix
    """

    topology = traj.topology
    temp_name = [atom.name for atom in topology.atoms if atom.element.symbol is 'C' and atom.is_sidechain]
    temp_idx = [atom.index for atom in topology.atoms if atom.element.symbol is 'C' and atom.is_sidechain]
    CB = [temp_idx[i] for i in range(len(temp_name)) if temp_name[i] == 'CB']

    GLY_name = [atom.name for atom in topology.atoms if atom.element.symbol is 'C' and (atom.residue.name == 'GLY')]
    GLY_idx = [atom.index for atom in topology.atoms if atom.element.symbol is 'C' and (atom.residue.name == 'GLY')]
    GLY_CA = [GLY_idx[i] for i in range(len(GLY_name)) if GLY_name[i] == 'CA']

    CB = CB + GLY_CA
    CB.sort()

    first = np.zeros([len(CB), len(CB)])
    # last = np.zeros([len(CB), len(CB)])

    for i in range(0, len(CB)):
        for j in range(i+1, len(CB)):
            x_i = traj.xyz[0,CB[i],0]; y_i = traj.xyz[0,CB[i],1]; z_i = traj.xyz[0,CB[i],2]
            x_j = traj.xyz[0,CB[j],0]; y_j = traj.xyz[0,CB[j],1]; z_j = traj.xyz[0,CB[j],2]
            first[i][j] = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2 + (z_i - z_j)**2)
            first[j][i] = first[i][j]
    '''
    for i in range(0, len(CB)):
        for j in range(i+1, len(CB)):
            x_i = traj.xyz[-1,CB[i],0]; y_i = traj.xyz[-1,CB[i],1]; z_i = traj.xyz[-1,CB[i],2]
            x_j = traj.xyz[-1,CB[j],0]; y_j = traj.xyz[-1,CB[j],1]; z_j = traj.xyz[-1,CB[j],2]
            last[i][j] = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2 + (z_i - z_j)**2)
            last[j][i] = last[i][j]
    '''
    return 10*first

