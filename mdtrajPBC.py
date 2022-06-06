import torch
import numpy as np
import functools
import matplotlib.pyplot as plt
import gpytorch as gtorch 
import sklearn.svm 
import sklearn.cluster
import warnings
import attrs
import warnings
import deprecation
from typing import *
import sklearn.base
import functorch
warnings.simplefilter("ignore")
import mdtraj
import nglview
import collections
import copy
import tqdm
import itertools
import math
from curtsies import fmtfuncs as cf
from torch.nn.modules import padding
import pandas as pd

@attrs.define
class MDtrajTorch():
    traj: mdtraj.Trajectory = None
    box_vectors: torch.Tensor = None
    periodic: torch.BoolTensor = None

    def compute_distances_core(self, 
            positions,
            atom_pairs,
            unitcell_vectors = None,
            periodic=True,
            opt = True,
    ):

        """Compute the distances between pairs of atoms in each frame.
        Parameters
        ----------
        positions : np.ndarray of shape=(n_frames, n_atoms, 3), dtype=float
            The positions of all atoms for a given trajectory.
        atom_pairs : np.ndarray of shape=(num_pairs, 2), dtype=int
            Each row gives the indices of two atoms involved in the interaction.
        unitcell_vectors : None or np.ndarray of shape(n_frames, 3 x 3), default=None
            A numpy array that specifies the box vectors for all frames for a trajectory.
        periodic : bool, default=True
            If `periodic` is True and the trajectory contains unitcell
            information, we will compute distances under the minimum image
            convention.
        opt : bool, default=True
            Use an optimized native library to calculate distances. Our optimized
            SSE minimum image convention calculation implementation is over 1000x
            faster than the naive numpy implementation.
        Returns
        -------
        distances : np.ndarray, shape=(n_frames, num_pairs), dtype=float
            The distance, in each frame, between each pair of atoms.
        """
        xyz = torch.from_numpy(positions).requires_grad_()
        pairs = torch.from_numpy(atom_pairs)

        if not torch.all(torch.logical_and(pairs < positions.shape[1], pairs >= 0)):
            raise ValueError('atom_pairs must be between 0 and %d' % traj.n_atoms)

        if len(pairs) == 0:
            return torch.zeros((len(xyz), 0), dtype=torch.float32)

        # if periodic.any() and (unitcell_vectors is not None):
        if (unitcell_vectors is not None):
            unitcell_vectors = torch.from_numpy(unitcell_vectors).requires_grad_()
            box = unitcell_vectors.clone()
            # print(unitcell_vectors)
            # convert to angles
            unitcell_angles = []
            for fr_unitcell_vectors in unitcell_vectors:
                # print(fr_unitcell_vectors[2])
                _, _, _, alpha, beta, gamma = self.box_vectors_to_lengths_and_angles(
                    fr_unitcell_vectors[0],
                    fr_unitcell_vectors[1],
                    fr_unitcell_vectors[2],
                )
                unitcell_angles.append(torch.tensor([alpha, beta, gamma]))
            cat_units = torch.cat(unitcell_angles, dim=0)
            orthogonal = torch.allclose(cat_units, cat_units.new_full(cat_units.size(), fill_value=90.))

            if opt:
                # out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
                # _geometry._dist_mic(xyz, pairs, box.transpose(0, 2, 1).copy(), out, orthogonal)
                # return out
                pass
            else:
                print("Using MIC")
                return self._distance_mic2(xyz, pairs, box.permute(0, 2, 1), orthogonal, periodic)

        # either there are no unitcell vectors or they dont want to use them
        if opt:
            # out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
            # _geometry._dist(xyz, pairs, out)
            # return out
            pass
        else:
            print("Using non-MIC")
            return self._distance_mic2(xyz, pairs)


    def compute_distances(self, traj, atom_pairs, periodic=True, opt=False):
        """Compute the distances between pairs of atoms in each frame.
        Parameters
        ----------
        traj : Trajectory
            An mtraj trajectory.
        atom_pairs : np.ndarray, shape=(num_pairs, 2), dtype=int
            Each row gives the indices of two atoms involved in the interaction.
        periodic : bool, default=True
            If `periodic` is True and the trajectory contains unitcell
            information, we will compute distances under the minimum image
            convention.
        opt : bool, default=True
            Use an optimized native library to calculate distances. Our optimized
            SSE minimum image convention calculation implementation is over 1000x
            faster than the naive numpy implementation.
        Returns
        -------
        distances : np.ndarray, shape=(n_frames, num_pairs), dtype=float
            The distance, in each frame, between each pair of atoms.
        """

        return self.compute_distances_core(
            traj.xyz,
            atom_pairs,
            unitcell_vectors=traj.unitcell_vectors,
            periodic=periodic,
            opt=opt,
        )

    def box_vectors_to_lengths_and_angles(self, a, b, c):
        """Convert box vectors into the lengths and angles defining the box.
        Parameters
        ----------
        a : np.ndarray
            the vector defining the first edge of the periodic box (length 3), or
            an array of this vector in multiple frames, where a[i,:] gives the
            length 3 array of vector a in each frame of a simulation
        b : np.ndarray
            the vector defining the second edge of the periodic box (length 3), or
            an array of this vector in multiple frames, where b[i,:] gives the
            length 3 array of vector a in each frame of a simulation
        c : np.ndarray
            the vector defining the third edge of the periodic box (length 3), or
            an array of this vector in multiple frames, where c[i,:] gives the
            length 3 array of vector a in each frame of a simulation
        Examples
        --------
        >>> a = np.array([2,0,0], dtype=float)
        >>> b = np.array([0,1,0], dtype=float)
        >>> c = np.array([0,1,1], dtype=float)
        >>> l1, l2, l3, alpha, beta, gamma = box_vectors_to_lengths_and_angles(a, b, c)
        >>> (l1 == 2.0) and (l2 == 1.0) and (l3 == np.sqrt(2))
        True
        >>> np.abs(alpha - 45) < 1e-6
        True
        >>> np.abs(beta - 90.0) < 1e-6
        True
        >>> np.abs(gamma - 90.0) < 1e-6
        True
        Returns
        -------
        a_length : scalar or np.ndarray
            length of Bravais unit vector **a**
        b_length : scalar or np.ndarray
            length of Bravais unit vector **b**
        c_length : scalar or np.ndarray
            length of Bravais unit vector **c**
        alpha : scalar or np.ndarray
            angle between vectors **b** and **c**, in degrees.
        beta : scalar or np.ndarray
            angle between vectors **c** and **a**, in degrees.
        gamma : scalar or np.ndarray
            angle between vectors **a** and **b**, in degrees.
        """
        if not a.shape == b.shape == c.shape:
            raise TypeError('Shape is messed up.')
        if not a.shape[-1] == 3:
            raise TypeError('The last dimension must be length 3')
        if not (a.ndim in [1,2]):
            raise ValueError('vectors must be 1d or 2d (for a vectorized '
                            'operation on multiple frames)')
        last_dim = a.ndim-1

        a_length = torch.sqrt(torch.sum(a*a, dim=last_dim))
        b_length = torch.sqrt(torch.sum(b*b, dim=last_dim))
        c_length = torch.sqrt(torch.sum(c*c, dim=last_dim))

        # we allow 2d input, where the first dimension is the frame index
        # so we want to do the dot product only over the last dimension
        alpha = torch.arccos(torch.einsum('...i, ...i', b, c) / (b_length * c_length))
        beta = torch.arccos(torch.einsum('...i, ...i', c, a) / (c_length * a_length))
        gamma = torch.arccos(torch.einsum('...i, ...i', a, b) / (a_length * b_length))

        # convert to degrees
        alpha = alpha * 180.0 / np.pi
        beta = beta * 180.0 / np.pi
        gamma = gamma * 180.0 / np.pi

        return a_length, b_length, c_length, alpha, beta, gamma

    def _distance(self, xyz, pairs):
        "Distance between pairs of points in each frame"
        # delta = torch.diff(xyz[:, pairs], dim=2)[:, :, 0]
        # return (delta ** 2.).sum(-1) ** 0.5
        # r12_tmp = (xyz:, :, None] - xyz[:, :, :]) #(batch, natoms, 1, 3) - (batch, 1, natoms, 3)
        # r12 = r12_tmp #(batch, natoms, 1, 3) - (batch, 1, natoms, 3) --> (batch, natoms, natoms)
        return torch.cdist(xyz, xyz)

    def _distance_mic(self, xyz, pairs, box_vectors, orthogonal, periodic):
        """Distance between pairs of points in each frame under the minimum image
        convention for periodic boundary conditions.
        The computation follows scheme B.9 in Tukerman, M. "Statistical
        Mechanics: Theory and Molecular Simulation", 2010.
        This is a slow pure python implementation, mostly for testing.
        """
        # out = torch.empty((xyz.shape[0], pairs.shape[0]), dtype=torch.float32) #nframes, pairs
        out = [ []*len(xyz) ]
        for i in range(len(xyz)):
            bv1, bv2, bv3 = self._reduce_box_vectors(box_vectors[i].T)

            for j, (a,b) in enumerate(pairs):
                r12 = xyz[i,b,:] - xyz[i,a,:]
                r12.data -= (bv3 * torch.round(r12[2]/bv3[2])).data if periodic[2] else bv3.data; 
                r12.data -= (bv2 * torch.round(r12[1]/bv2[1])).data if periodic[1] else bv2.data;
                r12.data -= (bv1 * torch.round(r12[0]/bv1[0])).data if periodic[0] else bv1.data;
                dist = torch.linalg.norm(r12)
                if not orthogonal:
                    for ii in range(-1, 2):
                        v1 = bv1*ii
                        for jj in range(-1, 2):
                            v12 = bv2*jj + v1
                            for kk in range(-1, 2):
                                new_r12 = r12 + v12 + bv3*kk
                                dist = min(dist, torch.linalg.norm(new_r12))
                out[i].append(dist)

        out = torch.stack(out, dim=0)
        return out

    def _distance_mic2(self, xyz, pairs, box_vectors, orthogonal, periodic):
        """Distance between pairs of points in each frame under the minimum image
        convention for periodic boundary conditions.
        The computation follows scheme B.9 in Tukerman, M. "Statistical
        Mechanics: Theory and Molecular Simulation", 2010.
        This is a slow pure python implementation, mostly for testing.
        """
        # out = torch.empty((xyz.shape[0], pairs.shape[0]), dtype=torch.float32) #nframes, pairs
        dists = []
        tril_ind = torch.tril_indices(len(xyz[0]), len(xyz[0]), offset=-1)

        for i in range(len(xyz)):
            bv1, bv2, bv3 = self._reduce_box_vectors(box_vectors[i].T) #each (3,)
            # zipped_pairs = pairs.t().unbind() #(2,pairs)
            # r12 = xyz[i, zipped_pairs[0], :] - xyz[i, zipped_pairs[1], :] #(pairs, 3)
            r12_tmp = (xyz[i, :, None] - xyz[i, :, :]) #(natoms, 1, 3) - (1, natoms, 3)
            r12 = r12_tmp.view(-1,3) #(natoms, 1, 3) - (1, natoms, 3) --> (pairs, 3)
            r12.data -= (bv3.view(1,-1) * torch.round(r12[:,2:] / bv3.view(1,-1)[:,2:])).data * periodic[1].data;
            r12.data -= (bv2.view(1,-1) * torch.round(r12[:,1:2] / bv2.view(1,-1)[:,1:2])).data * periodic[0].data;
            r12.data -= (bv1.view(1,-1) * torch.round(r12[:,0:1] / bv1.view(1,-1)[:,0:1])).data * periodic[2].data;
            dist = r12.norm(dim=-1) #(pairs,  ); without diagonals
            dists.append(dist)
        outs = torch.stack(dists, dim=0).view(-1, len(xyz[0]), len(xyz[0])) #(batch, natoms, natoms)
        # print(outs.shape)
        out = torch.atleast_3d(outs)        
        return out

    def _reduce_box_vectors(self, vectors):
        """Make sure box vectors are in reduced form."""
        (bv1, bv2, bv3) = vectors
        bv3.data -= (bv2*torch.round(bv3[1]/bv2[1])).data;
        bv3.data -= (bv1*torch.round(bv3[0]/bv1[0])).data;
        bv2.data -= (bv1*torch.round(bv2[0]/bv1[0])).data;
        return (bv1, bv2, bv3)

    def __call__(self, x1, x2):
        return self.covar_dist_custom(x1, x2)

    def covar_dist_custom(self, x1, x2):
        dists = []
        assert len(x1) == len(x2), "Must have same number of frames..."

        for i in range(len(x1)):
            bv1, bv2, bv3 = self._reduce_box_vectors(self.box_vectors[i]) #each (3,)
            # zipped_pairs = pairs.t().unbind() #(2,pairs)
            # r12 = xyz[i, zipped_pairs[0], :] - xyz[i, zipped_pairs[1], :] #(pairs, 3)
            r12_tmp = (x1[i, :, None] - x2[i, :, :]) #(natoms_, 1, 3) - (1, natoms, 3)
            r12 = r12_tmp.view(-1,3) #(natoms_, 1, 3) - (1, natoms, 3) --> (pairs_, 3)
            r12.data -= (bv3.view(1,-1) * torch.round(r12[:,2:] / bv3.view(1,-1)[:,2:])).data * self.periodic[1].data;
            r12.data -= (bv2.view(1,-1) * torch.round(r12[:,1:2] / bv2.view(1,-1)[:,1:2])).data * self.periodic[0].data;
            r12.data -= (bv1.view(1,-1) * torch.round(r12[:,0:1] / bv1.view(1,-1)[:,0:1])).data * self.periodic[2].data;
            dist = r12.norm(dim=-1) #(pairs_); without diagonals
            dists.append(dist)
        outs = torch.stack(dists, dim=0).view(-1, len(x1[0]), len(x2[0])) #(batch, natoms_, natoms)
        out = torch.atleast_3d(outs)   
        return out

"""
trajectory = mdtraj.load('membrane-cdl-1d.pdb') 
head_selection_text = 'name PO4' 
head_selection = trajectory.top.select(head_selection_text)
lipid_head = trajectory.atom_slice(head_selection)
torch.from_numpy(lipid_head.unitcell_vectors).pow(2).sum(dim=-1).sqrt()
lipid_head.xyz
lipid_head.unitcell_lengths

pbc = torch.BoolTensor([True, True, True]) 
cell = torch.from_numpy(lipid_head.unitcell_vectors)
# mdt = MDtrajTorch()
coords = torch.from_numpy(lipid_head.xyz).requires_grad_()

# lipid_indices = lipid_head.topology.select_atom_indices("all")
# atom_pairs = np.array(
#     [(i,j) for (i,j) in itertools.combinations(lipid_indices, 2)])
# mddistmap = mdt.compute_distances(lipid_head, atom_pairs=atom_pairs, periodic=pbc)

mdt = MDtrajTorch(traj = lipid_head, box_vectors = cell, periodic = pbc)
mddistmap = mdt.covar_dist_custom(coords, coords)
cdistmap = torch.cdist(coords, coords)

# plt.hist([asedistmap.detach().triu(1).view(-1,).numpy(), mddistmap.detach().triu(1).view(-1,).numpy()])
plt.hist([cdistmap.detach().triu(1).view(-1,).numpy(), mddistmap.detach().triu(1).view(-1,).numpy()])

# k = gtorch.kernels.MaternKernel()
# mat_dist = k.covar_dist(coords, coords)
# k.covar_dist = mdt.compute_distances(lipid_head, atom_pairs=atom_pairs, periodic=pbc)
# # plt.hist([cdistmap.detach().triu(1).view(-1,).numpy(), mat_dist.detach().triu(1).view(-1,).numpy()])
# help(k.covar_dist)


"""
