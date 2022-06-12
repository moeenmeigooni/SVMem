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

class NotFittedException(Exception):
    def __init__(self, ):
        msg = f"svmemTorch has not be fitted with training set..."
        return super().__init__(msg)

@attrs.define
class svmemTorch(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin, sklearn.base.ClassifierMixin):
    samples: torch.Tensor = None #auto initialization with attrs; call with self of shape (batch, natoms_, 3); natoms_ and natoms may not be the same
    references: torch.Tensor = None #auto initialization with attrs; call with self of shape (batch, natoms, 3)
    labels: torch.Tensor = None #labels for reference (batch, natoms); -1 or 1 ONLY!
    train_kernel: torch.Tensor = None #auto initialization with attrs; call with self of shape (batch, natoms, natoms)
    test_kernel: torch.Tensor = None #auto initialization with attrs; call with self of shape (batch, natoms_, natoms); natoms_ and natoms may not be the same
    train_boundary_function: torch.Tensor = None #auto initialization with attrs; call with self of shape (batch, _natoms_, 3); decision function value computed at higher dim (F(x,y,z)=k) 

    kernelized: bool = False #is implicitDecisionBoundary done first?
    fitted: bool = False #if fit and _fit method done first?
    kernel_weights: torch.Tensor = None
    kernel_biases: torch.Tensor = None
    svc_list: List[np.ndarray] = None #fit and _fit methods
    got_dual_coefs: bool = False #after fit and get_svm_weights methods

    jac_and_hes: bool = False #caluate jac/hes
    periodic: torch.Tensor = torch.BoolTensor([True, True, True]) 
    unitcell_lengths: torch.Tensor = None
    unitcell_vectors: torch.Tensor = None
    curvature: bool = None

    def prepare(self, topology: str):
        # load structure into mdtraj trajectory object
        topology = topology.split(".")[0] + ".pdb"
        trajectory = mdtraj.load(f'{topology}') 
        # remove water, ions
        lipid = trajectory.atom_slice(trajectory.top.select('not name W WF NA CL'))
        # define selection for training set
        head_selection_text = 'name PO4' 
        head_selection = lipid.top.select(head_selection_text)
        self.traj = lipid_head_traj = trajectory.atom_slice(trajectory.top.select(head_selection_text))
        self.references = self._to_tensor(lipid_head_traj.xyz) #(batch, natoms, 3)
        self.unitcell_lengths = self._to_tensor(lipid_head_traj.unitcell_lengths)
        self.unitcell_vectors = self._to_tensor(lipid_head_traj.unitcell_vectors)
        self.labels = self.get_labels #(batch, natoms)

    @staticmethod
    def _to_tensor(samples: Union[torch.Tensor, np.ndarray, List], differentiable=False):   
        samples = torch.from_numpy(np.array(samples)).requires_grad_(differentiable) if isinstance(samples, (np.ndarray, list)) else samples.requires_grad_(differentiable) 
        return samples

    @staticmethod
    def _to_numpy(samples: Union[torch.Tensor, np.ndarray]):   
        samples = samples.detach().numpy() if isinstance(samples, torch.Tensor) else samples
        return samples

    def implicitDecisionBoundary(self, samples: Union[np.ndarray, torch.Tensor], use_kernel: bool=False, get_surface: bool=False, weight_type: str="uniform"):
        #samples #(batch, natoms_, 3)
        #references #(batch, natoms, 3)
        #Necessary for svm_and_dual method!
        #For TRAINING and INFERENCE
        #ASSUME all data fit into one GPU!

        # covar = gtorch.kernels.ScaleKernel(
        #     MaternCustom(nu=0.5, ard_num_dims=3,             
        #     traj = self.traj, 
        #     box_vectors = self.unitcell_vectors, 
        #     periodic = self.periodic), 
        #     outputscale_constraint=gtorch.constraints.Positive(),
        #     lengthscale_prior=gtorch.priors.GammaPrior(3,1),
        #     ) + gtorch.kernels.ScaleKernel(
        #         RBFCustom(ard_num_dims=3, 
        #                   lengthscale_constraint=gtorch.constraints.Positive(), 
        #                   lengthscale_prior=gtorch.priors.GammaPrior(3,1),                          
        #                   traj = self.traj, 
        #                   box_vectors = self.unitcell_vectors, 
        #                   periodic = self.periodic), 
        #                   outputscale_constraint=gtorch.constraints.Positive(),
        #         )           #WIP!
        covar = gtorch.kernels.ScaleKernel(
                RBFCustom(ard_num_dims=3, 
                          lengthscale_constraint=gtorch.constraints.Positive(), 
                          lengthscale_prior=gtorch.priors.GammaPrior(3,1),                          
                          traj = self.traj, 
                          box_vectors = self.unitcell_vectors, 
                          periodic = self.periodic), 
                          outputscale_constraint=gtorch.constraints.Positive(),
                )           #WIP!
        references = self._to_tensor(self.references, differentiable=True) #(batch, natoms, 3)

        if not use_kernel:
            assert samples.size(-1) == 3, "Must be Cartesian coordinate..."
            #Pass a precomputed kernel or not?
            samples = self._to_tensor(samples, differentiable=True) #(batch, natoms_, 3)
            references = references.to(samples).detach()
            kernel = covar(samples, references) #(batch, natoms_, natoms) instance of LazyTensor
            # print(covar.periodic, covar.periodic)
            kernel = kernel.evaluate() #(batch, natoms_, natoms)
        else:
            assert samples.requires_grad and samples.is_leaf, "To use kernel directly, samples MUST be both LEAF and DIFFERENTIABLE..."
            kernel = samples #(batch, natoms_, natoms);;; WIP;; currently not used for def predict!!
       
        assert weight_type in ["uniform", "custom"], "wrong keyword is chosen..."
        if weight_type == "uniform":
            #Unspecified weights for ea data point before marginalization
            weights = kernel.new_ones(kernel.size()).to(references) #(batch, natoms_, natoms)
            biases = weights.new_zeros(weights.size(0)) #(batch, )
        else:
            weights, biases = self.fitted_dual_coef_weights
            assert weights.ndim == 2, "weight dimension is wrong or not computed... from list of svm.dual_coef_ modifed to torch.Tensor..."
            weights = self._to_tensor(weights).to(references) #(batch, natoms)
            biases = self._to_tensor(biases).to(references) ##(batch, 1)
            # ndims = kernel.ndim
            # dim_expansion = [None] * (ndims - 1)
            weights = weights[:, None, :].expand_as(kernel) #Hardcode for 3D membrane only! (WIP) #(batch, natoms_, natoms)
        if not get_surface:
            return kernel * weights #(batch, natoms_, natoms);  differentiable
        else:
            return (kernel * weights).sum(dim=-1, keepdim=True) + biases[:, None] #(batch, natoms_, 1) + (batch, 1, 1); decisision function value; differentiable

    def __add__(self, other):
        """for DATALOADER class in case..."""
        pass 

    def _fit_for_train(self, ):
        #THIS is for PER-FOR-LOOP method for SVM instances! 
        #Use AFTER implicitDecisionBoundary and BEFORE get_svm_weights
        #BE careful!
        #TRAINING USE-ONLY

        kernel = self.implicitDecisionBoundary(self.references, get_surface=False) # then get kernel #(batch, natoms, natoms) for TRAINING weights
        svc_ = [sklearn.svm.SVC(kernel="precomputed") for _ in range(kernel.size(0))] #Multiple instances of SVM
        X = self._to_numpy(kernel) #(batch, natoms, natoms); sklearn convention
        Y = self._to_numpy(self.labels) #(batch, natoms); sklearn convention
        #Should SVC be declared every time????? (WIP)
        self.svc_list = [svc.fit(x, y) for svc, x, y in zip(svc_, X, Y)] #List of length batch; each batch is an svm instance... ASSUME all data fit into one GPU
        return self

    def fit(self, ):
        self._fit_for_train()
        self.fitted = True
        self.get_svm_weights
        return self

    def predict(self, samples: torch.Tensor):
        # svc = sklearn.svm.SVC(kernel="precomputed")
        # X = self._to_numpy(test_kernel) #(batch, natoms, natoms); sklearn convention
        svc_list = self.fitted_svm_list #List of fitted svms (batch)
        assert samples.shape[0] == len(svc_list), "test samples and fitted svm list MUST have the same batch size..." #(batch, _natoms/_natoms_, 3)
        # if use_sklearn: 
        #     svc_preds_list = torch.stack([svc.decision_function(x).reshape(-1,1) for svc, x in zip(svc_list, samples)], dim=0).squeeze() #List of length batch; each batch is an svm decision function... ASSUME all data fit into one GPU; (batch, _natoms/_natoms_)
        # else:
        svc_decision_funcs = self.implicitDecisionBoundary(samples, get_surface=True, weight_type="custom") #(batch, _natoms/_natoms_)
        return svc_decision_funcs #List of decision function results (batch, _natoms/_natoms_)

    @property
    def fitted_svm_list(self, ):
        assert self.fitted, "Training data must be fit first..."
        return self.svc_list #List[svm] #size of batch (aka nframes)
 
    @property
    def fitted_dual_coef_weights(self, ):
        assert self.fitted and self.got_dual_coefs, "Training data must be fit first AND obtain dual coefs..."
        return self.kernel_weights, self.kernel_biases #(batch, natoms) and #(batch, 1)
 
    @property
    def get_jac_and_hes(self, ):
        assert self.jac_and_hes, "Jacobian and Hessian must be calculated first..."
        return self.gradient_results #collections.namedtuple

    # def fit_predict(self, X, y):
    #     return super().fit_predict(X, y=y) 

    # def score(self, X, y, sample_weight):
    #     return super().score(X, y, sample_weight=sample_weight)

    @property
    def get_svm_weights(self, ):
        #THIS is for PER-FOR-LOOP method for SVM instances! 
        #Use AFTER implicitDecisionBoundary and _fit!
        #BE careful!

        if self.fitted:
            sizes = self.references.size()
            weights =  self.references.new_zeros(size=torch.Size([sizes[0], sizes[1]+1])) #(batch, natoms+1)s
            svc_list = self.fitted_svm_list #fitted list of svc instances
            indices_list = torch.nn.utils.rnn.pad_sequence( list(map(lambda inp: self._to_tensor(getattr(inp, "support_")), svc_list)), batch_first=True, padding_value=sizes[1] ).long() #NOW: (batch, Longest_svNum);; OBSOLETE: (batch, svNum) list of size batch and each batch example will have different lengths of support indices...
            dual_coef_list = torch.nn.utils.rnn.pad_sequence( list(map(lambda inp: self._to_tensor(getattr(inp, "dual_coef_")), svc_list)), batch_first=True, padding_value=-100. ).to(weights).squeeze(dim=1) #NOW: tensor (batch, Longest_svNum); OBSOLETE: list of size batch and each batch example will have different lengths of support indices...
            intercept_list = self._to_tensor(list(map(lambda inp: getattr(inp, "intercept_"), svc_list)) ).to(weights) #NOW: tensor (batch, 1); OBSOLETE: list of size batch and each batch example will have different lengths of support indices...
            # for b, (indices, dual_coef) in enumerate(zip(indices_list, dual_coef_list)):
            #     weights.data[b, indices] = self._to_tensor(dual_coef.reshape(-1,)).float() #(svNum,) for dual_coef_ corresponding to indices ##svNum is PER-LOOP's SV points; WIP!!
            # biases.data = self._to_tensor(intercept.reshape(-1,)).float()
            # print(indices_list.shape, dual_coef_list.shape, sizes)
            weights.scatter_(dim=1, index=indices_list, src=dual_coef_list) #(batch, natoms+1);; carry corresponding dual_coeff to index of weights...
            weights = weights.narrow(dim=1, start=0, length=sizes[1]) #(batch, natoms) cut fit
            biases = intercept_list #(batch, 1)

            self.kernel_weights = weights #(batch, natoms)
            self.kernel_biases = biases #(batch,1 )
            self.got_dual_coefs = True
            return weights, biases
        else:
            raise NotFittedException()
    
    def differentiate_surface(self, samples: Union[np.ndarray, torch.Tensor], jax_like=False, return_hessian=False, device=None):
        """NEITHER functorch/autograd auto-creates a graph...
        Use this during INFERENCE!"""

        if self.fitted:
            samples = self._to_tensor(samples, differentiable=True).to(device) #(batch, natoms_, 3)
            assert samples.device.type[:4] == "cuda", "must be on GPU" #WIP 
            batch_size = samples.size(0)
            natoms = samples.size(1)

            print(samples.requires_grad, samples.is_leaf)
            assert samples.requires_grad and samples.is_leaf, "This cloned leaf Tensor does is not differentiable..."
            # surface_value = self.implicitDecisionBoundary(samples,  get_surface=True, weight_type="custom") # FOR AUTOGRAD; then get kernel from x (batch, natoms_, 3) -> weighted distance (batch*natoms_, 1);  differentiable
            surface_value = self.predict(samples) # FOR AUTOGRAD; then get kernel from x (batch, natoms_, 3) -> weighted distance (batch*natoms_, 1);  differentiable
            gradient_results = collections.namedtuple("gradient_results", ["jac", "hess", "surface_value"])
            ind = np.diag_indices(natoms, ndim=2)
            print(f"samples {samples}")
            print(cf.on_yellow(f"surf"), f"{surface_value}")
            if not return_hessian:
                if jax_like:
                    jac = functorch.vmap(functorch.jacrev(functools.partial(self.implicitDecisionBoundary, 
                                                                            get_surface=True, 
                                                                            weight_type="custom")))(samples).squeeze() #samples:(batch, natoms_, 3) -> jac:(batch, natoms_, natoms_, 3) 
                    grad = jac[:, ind[0], ind[1], :] #-> finally (batch, natoms_, 3)
                else:
                    grad = torch.autograd.grad(surface_value, samples, grad_outputs=torch.ones_like(surface_value), create_graph=True)[0].view(samples.size()) #Obtain grad:(batch, natoms_, 3) gradient w/o functorch
                self.jac_and_hes = True
                self.gradient_results = gradient_results(grad, None, surface_value)

                return self.gradient_results #Namedtuple: gradient is DelF(x,y,z) of (batch, natoms_, 3) and surface_value is F(x,y,z)=k value! of (batch, natoms_) computed by weighted sum...
            else:
                if jax_like:
                    jac = functorch.vmap(functorch.jacrev(functools.partial(self.implicitDecisionBoundary, 
                                                          get_surface=True, 
                                                          weight_type="custom")))(samples).squeeze() #samples:(batch, natoms_, 3) -> jac:(batch, natoms_, 3)
                    hess = functorch.vmap(functorch.hessian(functools.partial(self.implicitDecisionBoundary, 
                                                                            get_surface=True, 
                                                                            weight_type="custom")))(samples).squeeze() #samples:(batch, natoms_, 3) -> hes:(batch, natoms_, natoms_, 3, natoms_, 3)
                    grad = jac[:, ind[0], ind[1], :] #-> finally (batch, natoms_, 3)
                    hess = hess[:, ind[0], ind[1], :].permute(0,1,3,2,4)[:, ind2[0], ind[1], :, :] #-> finally (batch, natoms_, 3, 3)
                else:
                    # assert jax_like, "Hessian should be calculated with JAX-like framework..."
                    #https://github.com/noegroup/bgflow/blob/main/bgflow/utils/autograd.py
                    grad = torch.autograd.grad(surface_value, samples, grad_outputs=torch.ones_like(surface_value), create_graph=True, retain_graph=True)[0] #.view(samples.size()) #Obtain grad:(batch, natoms_, 3) gradient w/o functorch
                    hess = torch.stack([torch.autograd.grad(grad[...,i], samples, grad_outputs=torch.ones_like(grad[...,i]), create_graph=True)[0] for i in range(grad.size(-1))], dim=-1) #.view(samples.size()) #Obtain grad:(batch, natoms_, 3) gradient w/o functorch
                    # https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7  
                    # def jacobian(y, x, create_graph=False):
                    #     # xx, yy = x.detach().numpy(), y.detach().numpy()
                    #     jac = []
                    #     flat_y = y.reshape(-1)
                    #     grad_y = torch.zeros_like(flat_y)
                    #     for i in range(len(flat_y)):
                    #         grad_y[i] = 1.
                    #         grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=True)
                    #         jac.append(grad_x.reshape(x.shape))
                    #         grad_y[i] = 0.
                    #     return torch.stack(jac).reshape(y.shape + x.shape)
                    # def hessian(y, x):
                    #     return jacobian(jacobian(y, x, create_graph=True), x)
                    # grad = jacobian(surface_value, samples)
                    # hess = hessian(surface_value, samples)
                self.jac_and_hes = True
                self.gradient_results = gradient_results(grad, hess, surface_value)

                return self.gradient_results #Namedtuple: gradient is DelF(x,y,z) of (batch, natoms_, 3) and surface_value is F(x,y,z)=k value! of (batch, natoms_) computed by weighted sum...
        else:
            raise NotFittedException()

    def draw_3d_plots(self, data: torch.Tensor=None, labels: torch.Tensor=None, ref_jac=None, frame=0, draw_jac: bool=False):
        #For training set viz
        from mpl_toolkits import mplot3d
        import plotly.graph_objects as go

        ref_labels = self.labels[frame].reshape(-1,) #if labels != None else labels[frame] #(natoms, )
        references = self.references[frame] #if data != None else data[frame]#(natoms, 3)
        ref_jac = torch.nn.functional.normalize(ref_jac[frame], dim=-1) #(natoms, 3)
        
        jac = self.get_jac_and_hes.jac #Optimized ones
        surface_values = self.predict(data)
        jac = - torch.nn.functional.normalize(jac[frame], dim=-1) * surface_values.sign().data[frame] #(natoms, 3); sign is considered..
        data = data[frame]
        labels = labels[frame].reshape(-1, )
        references, ref_jac, ref_labels, data, labels, jac = list(map(lambda inp: self._to_numpy(inp), (references, ref_jac, ref_labels, data, labels, jac) ))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
              x=references[:,0],
              y=references[:,1],
              z=references[:,2],
              mode='markers',
              marker=dict(
                  size=6,
                  color=ref_labels,                # set color to an array/list of desired values
                  colorscale='Viridis',   # choose a colorscale
                  opacity=1.
                    )
              )
        )
        
        # fig.add_trace(go.Cone(
        #       x=references[:,0],
        #       y=references[:,1],
        #       z=references[:,2],
        #       u=ref_jac[:,0],
        #       v=ref_jac[:,1],
        #       w=ref_jac[:,2],
        #     colorscale='Blues',
        #     sizemode="absolute",
        #     sizeref=1)
        # )

        fig.add_trace(
            go.Scatter3d(
              x=data[:,0],
              y=data[:,1],
              z=data[:,2],
              mode='markers',
              marker=dict(
                  size=6,
                  color=labels,                # set color to an array/list of desired values
                  colorscale='Reds',   # choose a colorscale
                  opacity=1.
                    )
              )
        )

        fig.add_trace(go.Cone(
              x=data[:,0],
              y=data[:,1],
              z=data[:,2],
              u=jac[:,0],
              v=jac[:,1],
              w=jac[:,2],
            colorscale='Greens',
            sizemode="absolute",
            sizeref=1)
        )

        fig.show()

    def get_histogram_analysis(self, values: Union[np.ndarray, torch.Tensor]):
        plt.hist(self._to_numpy(values), bins=30)
        plt.show()
        df = pd.DataFrame(values)
        print(df.describe())

    def create_svm_decision_boundary(self, N: int=5, samples: torch.Tensor=None):
        #samples (Tensor) is test atoms for nframes!
        if samples == None:
            sample_test_mesh = [torch.linspace(-10, 10, N)] * 3 #for meshgrid with N fragmentations; use BROADCASTING for batch (aka nframes)!!
            X, Y, Z = torch.meshgrid(*sample_test_mesh) #torch.Tensor
            samples = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1) #_natoms_, 3; Tensor; use broadcasting soon!
            samples = self._to_tensor(samples.expand(self.references.size(0), -1, 3), differentiable=True) #torch (batch, _natoms_, 3); use broadcasting soon!
        else:
            assert self._to_tensor(samples).ndim == 3, "Must be 3 dim!"
            samples = self._to_tensor(samples, differentiable=True) #numpy (batch, _natoms, 3)

        assert samples.requires_grad and samples.is_leaf, "To determine decision function, samples MUST be both LEAF and DIFFERENTIABLE..."
        svc_decision_funcs = self.predict(samples) #(batch, _natoms/_natoms_); torch.Tensor
        return svc_decision_funcs

    @property
    def get_labels(self, ):
        #Periodicy should be considered... WIP
        assert self.references != None, "Read reference PDBs first..."
        hierarchical_clusters_ = [sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='single') for _ in range(self.references.size(0))] #Multiple instances of Agglo (batch, )
        # kernel_for_dist = gtorch.kernels.RBFKernel(ard_num_dims=3)
        # kernel_for_dist.lengthscale = torch.ones([3])
        # cov = kernel_for_dist(self.references).evaluate() #(batch, natoms, natoms)
        # distmap = cov.add(torch.finfo().eps).log().mul(-2.) #Distance mapping
        dist_module = MDtrajTorch(traj = self.traj, box_vectors = self.unitcell_vectors, periodic = self.periodic)
        distmap = dist_module(self.references, self.references)
        # distmap = torch.cdist(self.references, self.references) #BETTER than kernelized
        distmap = self._to_numpy(distmap)
        labels = np.stack([cluster.fit_predict(distance_matrix) for cluster, distance_matrix in zip(hierarchical_clusters_, distmap)], axis=0) #(batch, natoms)
        labels = self._to_tensor(labels)
        return labels

    @staticmethod
    def vec_mag(vec):
        # n = len(vec)
        # l = 0.
        # for i in range(n):
        #     l += (vec[i])**2.
        # return np.sqrt(l)
        return vec.norm(dim = -1) #natoms, 

    def gaussian_curvature(self, grad, hess, frame=0):
        grad = grad.detach()[frame] #(natoms, 3)
        hess = hess.detach()[frame] #(natoms, 3, 3)
        n = grad.shape[1] #3
        X = torch.empty((grad.shape[0], n+1, n+1)) #(natoms, 4, 4)
        X[n,n] = 0.
        # for i in range(n):
        #     for j in range(n):
        #         X[i,j] = hess[i,j]
        X[:, :-1,:-1] = hess #(natoms, 3, 3); SHAPE must MATCH
        # for i in range(n):
        #     X[n,i] = grad[i]
        #     X[i,n] = grad[i]
        X[:, [-1],:-1] = grad[:,None,:] #(natoms, 1, 4); SHAPE must MATCH
        X[:, :-1, [-1]] = grad[...,None]  #(natoms, 4, 1); SHAPE must MATCH
        div = -(self.vec_mag(grad)**4.) #natoms, 
        return torch.linalg.det(X) / div   #determinant: (natoms,) / div: (natoms) -> (natoms, )

    def mean_curvature(self, grad, hess, frame=0):
        grad = grad.detach()[frame] #(natoms, 3)
        hess = hess.detach()[frame] #(natoms, 3, 3)
        grad_mag = self.vec_mag(grad) #natoms
        div = 2. * (grad_mag**3.) #natoms
        traces = hess.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1) #natoms, 
        return (torch.einsum("ni,nik,nk->n", grad, hess, grad) + (-(grad_mag**2.) * traces)) / div #(natoms)

    def get_curvature(self, samples: torch.Tensor, frame=0):
        self.differentiate_surface(samples=samples, jax_like=False, return_hessian=True) #To compute jac/hess
        curvature = collections.namedtuple("curvature", ["mean", "gauss"])
        grads = self.get_jac_and_hes
        jac = grads.jac
        hess = grads.hess

        mean = self.mean_curvature(jac, hess, frame)
        gauss = self.gaussian_curvature(jac, hess, frame)
        curvature.mean = mean
        curvature.gauss = gauss
        self.curvature = curvature
        return self.curvature #namedtuple

    @property
    def _initialize_decision_points(self, ):
        refs = self.references.data.clone()
        initial_points = self._to_tensor(refs, differentiable=True) #(batch, natoms, 3)
        assert initial_points.is_leaf and initial_points.requires_grad, "decision boundary points must be both leaf and differentiable..."
        return initial_points

    @_initialize_decision_points.setter
    def _initialize_decision_points(self, values: torch.Tensor):
        assert values.ndim == 3, "values to set must be (batch, natoms, 3) size..."
        self.references.data = self._to_tensor(values).data

    def determined_system(self, points: torch.Tensor, coeffs: torch.Tensor=torch.FloatTensor([1.,1.,1.])):
        assert points.is_leaf and points.requires_grad, "decision boundary points must be both leaf and differentiable..."
        #poinst (batch, natoms, 3)
        def must_be_zero_surface(points):
            surface_values = self.predict(points) #(batch, natoms)
            mean_surface_loss = surface_values.pow(2).sum(-1).mean() #(batch, ) -> (1,)
            std_surface_loss = surface_values.pow(2).sum(-1).std() #(batch, ) -> (1,)
            return mean_surface_loss, std_surface_loss

        def must_be_dot_one(points):
            grad = self.differentiate_surface(points).jac #points: (batch, natoms, 3) -> grad: (batch, natoms, 3); differentiable
            grad_normal = torch.nn.functional.normalize(grad, dim=-1) #(batch, natoms, 3); differentiable
            diff_vec_normal = torch.nn.functional.normalize(points.sub(self.references), dim=-1) #(batch, natoms, 3); differentiable
            dot_normal_vecs = grad_normal.mul(diff_vec_normal).sum(dim=-1) #(batch, natoms); dot product
            dot_normal_zeros_loss = (dot_normal_vecs.pow(2) - 1.).pow(2).sum(-1).mean() #(batch, natoms) -> (1,)
            return dot_normal_zeros_loss

        def must_be_distance_const(points):
            diff_dist_loss = ((points - self.references).pow(2).sum(dim=-1) - 15.).pow(2).mean() #(batch, natoms) -> (1, )
            return diff_dist_loss

        loss_mean, _ = must_be_zero_surface(points)
        loss_dot = must_be_dot_one(points)
        loss_dist = must_be_distance_const(points)
        
        return functools.reduce(lambda x, y: x+y, [l*c for l, c in zip([loss_mean, loss_dot, loss_dist], coeffs)] ) #total loss

    @staticmethod
    def map2central(cell: torch.Tensor, coordinates: torch.Tensor, pbc: torch.Tensor) -> torch.Tensor:
        """https://aiqm.github.io/nnp-test-docs/_modules/nnp/pbc.html#map2central"""
        # Step 1: convert coordinates from standard cartesian coordinate to unit
        # cell coordinates
        inv_cell = torch.inverse(cell.data)
        coordinates_cell = coordinates.data @ inv_cell
        # Step 2: wrap cell coordinates into [0, 1)
        coordinates_cell.data -= coordinates_cell.data.floor() * pbc.to(coordinates_cell.data.dtype)
        # Step 3: convert from cell coordinates back to standard cartesian
        # coordinate
        return coordinates_cell.data @ cell

    def optimize_decision_boundary(self, points: torch.Tensor=None, iters=50, learning_rate=0.08, coeffs=torch.tensor([1.,1.,1.])):
        print(f"points value is {points}...")
        device = torch.cuda.current_device()
        points = self._initialize_decision_points if points == None else points #differentiable
        points = points.to(device)
        print(device)
        # optimizer = torch.optim.Adam([points], lr=0.001)
        # for _ in range(iters):
        #     optimizer.zero_grad()
        #     loss = self.determined_system(points)
        #     loss.backward()
        #     optimizer.step()
        tqloading = tqdm.tqdm(range(iters))
        for _ in tqloading:
            loss = self.determined_system(points, coeffs=coeffs)
            loss_grads = torch.nn.functional.normalize(torch.autograd.grad(loss, points)[0], dim=-1)
            surface_values = self.predict(points) #(batch, natoms, 1)
            grad = self.differentiate_surface(points, device=device).jac
            grad_norm = torch.nn.functional.normalize(grad, dim=-1) #(batch, natoms, 3)
            points.data -= learning_rate * (surface_values.sign().data * (grad_norm.data + loss_grads.data)) #* self.labels.data.float()
            points.data = self.map2central(cell=self.unitcell_vectors, coordinates=points.data, pbc=self.periodic) #Wrapped
        return points

    def __call__(self, points=None, iters=50, learning_rate=0.08, coeffs=torch.tensor([1.,1.,1.])):
        points = self.optimize_decision_boundary(points=points, iters=iters, learning_rate=learning_rate, coeffs=coeffs)
        return points 

    # def unravel_upper_triangle_index(self, n):
    #     n_unique = (n * (n-1)) // 2
    #     a, b = np.empty((n_unique),dtype=np.int64), np.empty((n_unique),dtype=np.int64)
    #     k = 0
    #     for i in range(n):
    #         for j in range(n):
    #             if i < j:
    #                 a[k], b[k] = i, j
    #                 k += 1
    #     return a, b

    # def sym_dist_mat_(self, xyzs, box_dims, periodic):
    #     n = xyzs.shape[0]
    #     n_unique = (n * (n-1)) // 2
    #     ndim = xyzs.shape[1]
    #     i, j = self.unravel_upper_triangle_index(n)
    #     dist_mat = np.zeros((n_unique))
    #     for k in range(n_unique):
    #         for ri in range(ndim):
    #             dr = np.abs(xyzs[i[k],ri] - xyzs[j[k],ri])
    #             if periodic[ri] == True:
    #                 while (dr >  (box_dims[ri]*0.5)):
    #                     dr -= box_dims[ri]
    #             dist_mat[k] += np.square(dr)
    #     return np.sqrt(dist_mat)

    # def sym_dist_mat(self, xyzs, box_dims, periodic):
    #     n = xyzs.shape[0]
    #     dist_mat_flat = self.sym_dist_mat_(xyzs, box_dims, periodic)
    #     dist_mat = np.zeros((n,n))
    #     k = 0
    #     for i in range(n):
    #         for j in range(n):
    #             if i < j:
    #                 dist_mat[i,j] = dist_mat_flat[k]
    #                 dist_mat[j,i] = dist_mat_flat[k]
    #                 k += 1
    #     return dist_mat

class CustomPeriodicDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, boxlength: torch.Tensor):
        pass
    
    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        pass

"""
np.random.seed(42)
print(np.random.random((3,)) @ np.random.random((3,3)) @ np.random.random((3,)).T)
np.random.seed(42)
print(np.einsum("i,ik,k->", np.random.random((3,)) , np.random.random((3,3)) , np.random.random((3,)).T))
"""
