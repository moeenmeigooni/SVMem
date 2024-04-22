from numba_utils.SVMem_numba_old import SVMem
import numpy as np
import mdtraj as md

# load structure into mdtraj trajectory object
trajectory = md.load('membrane-cdl-1d.pdb')

# remove water, ions
lipid = trajectory.atom_slice(trajectory.top.select('not name W WF NA CL'))

# define selection for training set
head_selection_text = 'name GL0 PO4'
head_selection = lipid.top.select(head_selection_text)

# define periodicity of system in x,y,z directions
periodic = np.array([True, True, False])

# get indices of each lipid, required for COM calculation
atom_ids_per_lipid = [np.array([atom.index for atom in residue.atoms]) for residue in lipid.top.residues]

# define gamma, hyperparameter used for RBF kernel
gamma = 0.1

svmem = SVMem(lipid.xyz, # atomic xyz coordinates of all lipids; shape = (n_frames, n_atoms)
              head_selection, # indices of training points; shape = (n_lipids)
              atom_ids_per_lipid, # list of atom ids for each lipid; shape = (n_lipids,
              lipid.unitcell_lengths, # unitcell dimensions; shape = (n_frames, 3)
              periodic,
              gamma)

svmem.calculate_curvature(frames='all')

# curvature and normal vectors are stored in the svmem object
print(svmem.mean_curvature, svmem.gaussian_curvature, svmem.normal_vectors)
