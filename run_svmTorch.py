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
from gpytorchKernel import *
from mdtrajPBC import *
from svmemTorch import *

"""Use is for PER-FRAME..."""

svt = svmemTorch()
svt.prepare("membrane-cdl-1d.pdb")
svt.fit()

df = svt.create_svm_decision_boundary(N=4) #INFERENCE
grads = svt.differentiate_surface(svt._to_tensor(svt.references, differentiable=True), return_hessian=True) #INFERENCE
jac = grads.jac.data.clone()
hess = grads.hess.data.clone()
hess.unique()

# svt.get_jac_and_hes.hess.shape
# curves = svt.get_curvature
# points = final_points
final_points = svt(points=None, iters=50, coeffs=torch.tensor([0.,0.,0.])) #__call__
#svt.draw_3d_plots(data=final_points, labels=torch.full((final_points.size(1),), fill_value=2), ref_jac=jac, draw_jac=True)
# svt.draw_3d_plots(ref_jac=grads, draw_jac=True)
# svt.get_jac_and_hes.jac
curves = svt.get_curvature(final_points)
svt.get_histogram_analysis(curves.mean.detach())

# plt.hist(curves.mean.detach().numpy())
# import pandas as pd
# pd.DataFrame(curves.mean.detach().numpy()).describe()
# curves.mean.detach().isnan().count_nonzero()
