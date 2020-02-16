# SVMem  

### A self-supervised learning approach for membrane curvature calculation  
Version 0.0.2  

## Requirements:  
SVMem.py uses the following packages:  
* numpy >= 1.16.4
* numba >= 0.45.1
* scikit-learn >= 0.21.2  
* mdtraj >= 1.9.3

## Basic Usage:
```python
from SVMem import SVMem
import numpy as np
import mdtraj as md

# load structure into mdtraj trajectory object
trajectory = md.load('membrane.pdb') 

# remove water, ions
lipid = trajectory.atom_slice(trajectory.top.select('not name W WF NA CL'))

# define selection for training set
head_selection_text = 'name PO4' 
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
svmem.mean_curvature
svmem.gaussian_curvature
svmem.normal_vectors

```

## Troublshooting:
Having problems running SVMem? Documentation will be limited until the release of version 0.1.  
Until then, email me at meigoon2@illinois.edu with questions/comments/concerns, and I'll get back to you as soon as I can.
