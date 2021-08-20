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
        return Ca
    elif opt.matrix == 'Cb':
        return Cb

def Ca(traj):
    """Calculate the distance matrix with alpha Carbon

    Parameters:
        traj -- mdtraj.Trajectory

    Returns distance matrix 
        d_matrix[0] (2d numpy.array) -- distance matrix for first frame
        d_matrix[-1] (2d numpy.array) -- distance matrix for last frame
    """

    temp = md.compute_contacts(traj, scheme='ca')
    d_matrix = md.geometry.squareform(temp[0],temp[1])
    return d_matrix[0]

def Cb(traj):
    """Calculate the distance matrix with beta Carbon while GLY use alpha Carbon

    Parameters:
        traj -- mdtraj.Trajectory

    Returns distance matrix 
        d_matrix[0] (2d numpy.array) -- distance matrix for first frame
        d_matrix[-1] (2d numpy.array) -- distance matrix for last frame
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
    return first

def Ca_xyz(traj):
    pass
