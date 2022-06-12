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
from mdtrajPBC import *

# traj = lipid_head
# box_vectors = cell
# periodic = pbc

def default_postprocess_script(x):
    return x

class DistanceCustom(gtorch.kernels.kernel.Distance):
    def __init__(self, traj = None, box_vectors = None, periodic = None, **kwargs):
        super().__init__(**kwargs)
        self.periodic_distance = MDtrajTorch(traj = traj, box_vectors = box_vectors, periodic = periodic) #Must define these somehow! WIP!!!
        # print(self.periodic_distance.box_vectors)

    def _sq_dist(self, x1, x2, postprocess, x1_eq_x2=False):
        res = self.periodic_distance.covar_dist_custom(x1, x2)
        res.clamp_min_(1e-30).pow_(2)
        return self._postprocess(res) if postprocess else res

    def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
        res = self.periodic_distance.covar_dist_custom(x1, x2)
        res = res.clamp_min_(1e-30)
        return self._postprocess(res) if postprocess else res

class KernelCustom(gtorch.kernels.kernel.Kernel):
    def __init__(self, *args, traj = None, box_vectors = None, periodic = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.traj = traj
        self.box_vectors = box_vectors
        self.periodic = periodic

    def covar_dist(
        self,
        x1,
        x2,
        diag=False,
        last_dim_is_batch=False,
        square_dist=False,
        dist_postprocess_func=default_postprocess_script,
        postprocess=True,
        **params,
    ):
        r"""
        This is a helper method for computing the Euclidean distance between
        all pairs of points in x1 and x2.

        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`diag` (bool):
                Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?
            :attr:`square_dist` (bool):
                Should we square the distance matrix before returning?

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        # torch scripts expect tensors
        postprocess = torch.tensor(postprocess)

        res = None

        # Cache the Distance object or else JIT will recompile every time
        if not self.distance_module or self.distance_module._postprocess != dist_postprocess_func:
            self.distance_module = DistanceCustom(traj = self.traj, box_vectors = self.box_vectors, periodic = self.periodic, postprocess_script=dist_postprocess_func)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                res = torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
                if postprocess:
                    res = dist_postprocess_func(res)
                return res
            else:
                res = torch.norm(x1 - x2, p=2, dim=-1)
                if square_dist:
                    res = res.pow(2)
            if postprocess:
                res = dist_postprocess_func(res)
            return res

        elif square_dist:
            res = self.distance_module._sq_dist(x1, x2, postprocess, x1_eq_x2)
        else:
            res = self.distance_module._dist(x1, x2, postprocess, x1_eq_x2)

        return res

    def __add__(self, other):
        kernels = []
        kernels += self.kernels if isinstance(self, AdditiveKernelCustom) else [self]
        kernels += other.kernels if isinstance(other, AdditiveKernelCustom) else [other]
        return AdditiveKernelCustom(*kernels)

class AdditiveKernelCustom(KernelCustom):
    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if all components are stationary.
        """
        return all(k.is_stationary for k in self.kernels)

    def __init__(self, *kernels):
        super(AdditiveKernelCustom, self).__init__()
        self.kernels = torch.nn.ModuleList(kernels)

    def forward(self, x1, x2, diag=False, **params):
        res = gtorch.lazy.ZeroLazyTensor() if not diag else 0
        for kern in self.kernels:
            next_term = kern(x1, x2, diag=diag, **params)
            if not diag:
                res = res + gtorch.lazy.lazify(next_term)
            else:
                res = res + next_term
        return res

    def num_outputs_per_input(self, x1, x2):
        return self.kernels[0].num_outputs_per_input(x1, x2)

    def __getitem__(self, index):
        new_kernel = copy.deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = self.kernels[i].__getitem__(index)
        return new_kernel

def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()

class RBFCustom(KernelCustom):
    has_lengthscale = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
        ):
            print(x1.device, x2.device, self.lengthscale.device)
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return self.covar_dist(
                x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )

class MaternCustom(KernelCustom):
    has_lengthscale = True

    def __init__(self, nu: Optional[float] = 2.5, *args, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternCustom, self).__init__(*args, **kwargs)
        self.nu = nu

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
        ):
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component





# rbf_custom = RBFCustom(ard_num_dims=3, traj = lipid_head, box_vectors = cell, periodic = pbc,) + RBFCustom(ard_num_dims=3, traj = lipid_head, box_vectors = cell, periodic = pbc,)
# rbf_custom(coords, coords).evaluate()
# rbf_custom.kernels[0].covar_dist(coords, coords)
# rbf_custom.kernels[0].box_vectors
# # mddistmap2 = rbf_custom.covar_dist(coords, coords)
# rbf_custom.periodic
# plt.hist([cdistmap.detach().triu(1).view(-1,).numpy(), mddistmap2.detach().triu(1).view(-1,).numpy()])
