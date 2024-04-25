import warnings
import numpy as np
import mdtraj as md
from numba import njit, prange
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, lax, random
from jax.typing import Tuple, Any, Callable, Array
import os
import sys
import argparse
import pathlib
from functools import partial
import time

# JAX utils
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots)
from jax_utils.main import get_args
from numba_utils.SVMem_numba_old import SVMem
# https://antixk.netlify.app/blog/linearization_ad/ #JAX jvp etc.

warnings.simplefilter('ignore')

def ndot(a: Array, b: Array) -> Array:
    return jnp.dot(a, b)

def nsign(x: Array) -> Array:
    return jnp.where(x > 0, 1.0, 0.0)

def nsign_int(x: Array) -> Array:
    return jnp.where(x > 0, 1, 0)

def vec_mag(vec: Array) -> Array:
    return jnp.linalg.norm(vec)

def vec_mags(vecs: Array) -> Array:
    return jnp.linalg.norm(vecs, axis=1)

def vec_norm(vec: Array) -> Array:
    return vec / vec_mag(vec)

def vec_norms(vecs: Array) -> Array:
    return vecs / vec_mags(vecs).reshape(-1, 1)

def vec_sum(vecs: Array) -> Array:
    return jnp.sum(vecs, axis=0)

def unravel_index(n1: int, n2: int) -> Tuple[Array, Array]:
    a = jnp.arange(n1)
    b = jnp.arange(n2)
    a_grid, b_grid = jnp.meshgrid(a, b, indexing='ij')
    return a_grid.ravel(), b_grid.ravel()

def unravel_upper_triangle_index(n: int) -> Tuple[Array, Array]:
    idx = jnp.tril_indices(n, k=-1)
    return idx[0], idx[1]

def sym_dist_mat_(xyzs: Array, box_dims: Array, periodic: bool) -> Array:
    n = xyzs.shape[0]
    i, j = unravel_upper_triangle_index(n)

    def compute_dist(k: int) -> Array:
        i_k = i[k]
        j_k = j[k]
        dr = jnp.abs(xyzs[i_k] - xyzs[j_k])
        def apply_periodic_correction(args: Tuple[Array, Array]) -> Array:
            dr, box_dims = args
            return lax.cond(dr > box_dims * 0.5, (dr, box_dims), lambda args: args[0] - args[1], dr, operand=None)
        dr = lax.cond(periodic, (dr, box_dims), apply_periodic_correction, dr, operand=None)
        return jnp.sum(dr ** 2)

    dist_mat = lax.map(compute_dist, jnp.arange(n * (n - 1) // 2))
    return jnp.sqrt(dist_mat)

def sym_dist_mat(xyzs: Array, box_dims: Array, periodic: bool) -> Array:
    n = xyzs.shape[0]
    dist_mat_flat = sym_dist_mat_(xyzs, box_dims, periodic)
    i, j = unravel_upper_triangle_index(n)

    def fill_dist_mat(k: int, dist_mat: Array) -> Array:
        i_k = i[k]
        j_k = j[k]
        return lax.cond(i_k < j_k,
                        lambda dist_mat: dist_mat.at[i_k, j_k].set(dist_mat_flat[k]),
                        lambda dist_mat: dist_mat.at[j_k, i_k].set(dist_mat_flat[k]),
                        operand=dist_mat)

    dist_mat = lax.fori_loop(0, n * (n - 1) // 2, fill_dist_mat, jnp.zeros((n, n)))
    return dist_mat

def dist_mat_(xyz1: Array, xyz2: Array, box_dims: Array, periodic: bool) -> Array:
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    i, j = unravel_index(n1, n2)

    def compute_dist(k: int) -> Array:
        i_k = i[k]
        j_k = j[k]
        dr = jnp.abs(xyz1[i_k] - xyz2[j_k])
        def apply_periodic_correction(args: Tuple[Array, Array]) -> Array:
            dr, box_dims = args
            return lax.cond(dr > box_dims * 0.5, (dr, box_dims), lambda args: args[0] - args[1], dr, operand=None)
        dr = lax.cond(periodic, (dr, box_dims), apply_periodic_correction, dr, operand=None)
        return jnp.sum(dr ** 2)

    dist_mat = lax.map(compute_dist, jnp.arange(n1 * n2))
    return jnp.sqrt(dist_mat)

def dist_mat(xyz1: Array, xyz2: Array, box_dims: Array, periodic: bool) -> Array:
    dist_flat = dist_mat_(xyz1, xyz2, box_dims, periodic)
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    return dist_flat.reshape(n1, n2)

def dist_mat_parallel_(xyz1: Array, xyz2: Array, box_dims: Array, periodic: bool) -> Array:
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    ndim = xyz1.shape[1]
    i, j = unravel_index(n1, n2)

    def compute_dist(k: int) -> Array:
        i_k = i[k]
        j_k = j[k]
        dr = jnp.abs(xyz1[i_k] - xyz2[j_k])
        for ri in range(ndim):
            if periodic[ri]:
                dr = lax.cond(dr[ri] > box_dims[ri] * 0.5, (dr, box_dims[ri]), lambda args: args[0] - args[1], dr, operand=None)
        return jnp.sum(dr ** 2)

    dist_mat = lax.map(compute_dist, jnp.arange(n1 * n2))
    return jnp.sqrt(dist_mat)

def dist_mat_parallel(xyz1: Array, xyz2: Array, box_dims: Array, periodic: bool) -> Array:
    dist_flat = dist_mat_parallel_(xyz1, xyz2, box_dims, periodic)
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    return dist_flat.reshape(n1, n2)

def dist_vec(xyz: Array, xyzs: Array, box_dims: Array, periodic: bool) -> Array:
    ndim = xyz.shape[0]
    n = xyzs.shape[0]

    def compute_dist(i: int) -> Array:
        dr = jnp.abs(xyzs[i] - xyz)
        for ri in range(ndim):
            if periodic[ri]:
                dr = lax.cond(dr[ri] > box_dims[ri] * 0.5, dr, lambda dr: dr - box_dims[ri], dr, operand=None)
        return jnp.sum(dr ** 2)

    dist_vec = vmap(compute_dist)(jnp.arange(n))
    return jnp.sqrt(dist_vec)

def disp(xyz1: Array, xyz2: Array, box_dims: Array, periodic: bool) -> Array:
    def apply_periodic_correction(args: Tuple[Array, Array]) -> Array:
        dr, dim = args
        return lax.cond(dr > box_dims[dim] * 0.5, (dr, box_dims[dim]), lambda args: args[0] - args[1], dr, operand=None)

    dr = xyz1 - xyz2
    dr = lax.cond(periodic, (dr, jnp.arange(3)), apply_periodic_correction, dr, operand=None)
    return dr

def disp_vec(xyz: Array, xyzs: Array, box_dims: Array, periodic: bool) -> Array:
    def disp_single(xyz1: Array) -> Array:
        return disp(xyz, xyz1, box_dims, periodic)
    return vmap(disp_single)(xyzs)

def gaussian_transform_vec(array: Array, gamma: float) -> Array:
    return jnp.exp(-gamma * jnp.square(array))

def gaussian_transform_vec_parallel(array: Array, gamma: float) -> Array:
    return vmap(lambda x: jnp.exp(-gamma * jnp.square(x)))(array)

def gaussian_transform_mat_(array: Array, gamma: float) -> Array:
    return jnp.exp(-gamma * jnp.square(array))

def gaussian_transform_mat(mat: Array, gamma: float) -> Array:
    return gaussian_transform_mat_(mat.ravel(), gamma).reshape(mat.shape)

def decision_function(vec: Array, weights: Array, intercept: float) -> Array:
    return jnp.dot(weights, vec) + intercept

def decision_function_mat(mat: Array, weights: Array, intercept: float) -> Array:
    return jnp.dot(mat, weights) + intercept

def predict(vec: Array, weights: Array, intercept: float) -> Array:
    return jnp.sign(decision_function(vec, weights, intercept))

def predict_mat(mat: Array, weights: Array, intercept: float) -> Array:
    return jnp.sign(decision_function_mat(mat, weights, intercept))

def pbc_center(xyzs: Array, box_dims: Array) -> Array:
    rmax = jnp.max(xyzs)
    thetai = 2. * jnp.pi * xyzs / rmax
    xi = jnp.mean(jnp.cos(thetai))
    zeta = jnp.mean(jnp.sin(thetai))
    theta = jnp.arctan2(zeta, xi) + jnp.pi
    center = theta * rmax / (2. * jnp.pi)
    return center

def calculate_lipid_coms(lipids_xyz: Array, atom_ids_per_lipid: Array, box_dims: Array) -> Array:
    coms = vmap(lambda atom_ids: pbc_center(lipids_xyz[atom_ids], box_dims))(atom_ids_per_lipid)
    return coms

def update_disps(disps: Array, step: Array, box_dims: Array, periodic: bool) -> Array:
    def apply_periodic_correction(args: Tuple[Array, int]) -> Array:
        disps_j, dim = args
        return lax.cond(disps_j > box_dims[dim] * 0.5, disps_j - box_dims[dim], disps_j, disps_j, lambda x: x)
    
    disps = disps + step
    disps = lax.cond(periodic, (disps, jnp.arange(3)), apply_periodic_correction, disps, operand=None)
    return disps

def gradient(disps: Array, gxdists: Array, gamma: float, weights: Array) -> Array:
    factor = -2. * gamma
    del_F = jnp.sum(factor * weights * disps * gxdists, axis=0)
    return del_F

def gradient_descent(point_: Array, support_points: Array, box_dims: Array, periodic: bool, weights: Array, intercept: float, gamma: float, learning_rate: float, max_iter: int) -> Tuple[Array, Array, Array]:
    # Body function for while loop
    def body_fn(vals: Tuple[Array, Array, Array, float, Array, int]) -> Tuple[Array, Array, Array, float, Array, int]:
        point, disps, gxdists, sign, step, max_iter = vals
        step = -learning_rate * jnp.sign(d) * vec_norm(gradient(disps, gxdists, gamma, weights))
        point += step
        disps = update_disps(disps, step, box_dims, periodic)
        gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
        d = decision_function(gxdists, weights, intercept)
        return point, disps, gxdists, jnp.sign(d), step, max_iter - 1

    # Initialization
    point = jnp.array(point_, copy=True)
    disps = disp_vec(point, support_points, box_dims, periodic)
    gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
    d = decision_function(gxdists, weights, intercept)
    sign = jnp.sign(d)
    step = -learning_rate * jnp.sign(d) * vec_norm(gradient(disps, gxdists, gamma, weights))

    # Iterative optimization
    _, disps, _, _, _, _ = lax.while_loop(cond_fn, body_fn, (point, disps, gxdists, sign, step, max_iter))
    return point, vec_norm(step), disps

def coordinate_descent(point_: Array, step: float, disps: Array, box_dims: Array, periodic: bool, weights: Array, intercept: float, gamma: float, step_init: float, max_iter: int, tol: float) -> Array:
    # Body function for while loop
    def body_fn(vals: Tuple[Array, float, int]) -> Tuple[Array, float, int]:
        point, s, max_iter = vals
        point += step
        disps = update_disps(disps, step, box_dims, periodic)
        gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
        d = decision_function(gxdists, weights, intercept)
        news = jnp.sign(d)
        step = step * (news != s) * -0.5 + step * (news == s)
        return point, news, max_iter - 1

    # Initialization
    point = jnp.array(point_, copy=True)
    step = step_init * step
    gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
    d = decision_function(gxdists, weights, intercept)
    s = jnp.sign(d)

    # Iterative optimization
    point, _, _ = lax.while_loop(cond_fn, body_fn, (point, s, max_iter))
    return point

def descend_to_boundary(points: Array, support_points: Array, box_dims: Array, periodic: bool, weights: Array, intercept: float, gamma: float, learning_rate: float, max_iter: int, tol: float) -> Tuple[Array, Array]:
    # Body function for loop
    def body_fn(i: int, bounds_normal: Array) -> Array:
        approx_bound, _, disps = gradient_descent(
            points[i], support_points, 
            box_dims, periodic, weights, intercept, gamma, 
            learning_rate, max_iter)
        final_bound = coordinate_descent(
            approx_bound, normal_vectors[i], disps, 
            box_dims, periodic, weights, intercept, gamma, 
            learning_rate, max_iter, tol)
        return jax.at[i, :].set(bounds_normal, final_bound)

    # Initialization
    bounds = jnp.zeros((points.shape[0], points.shape[1]))
    normal_vectors = jnp.zeros_like(bounds)

    # Iterative optimization
    bounds = lax.fori_loop(0, points.shape[0], body_fn, bounds)
    return bounds, -1.*normal_vectors

def analytical_derivative(point: Array, support_points: Array, box_dims: Array, periodic: bool, gamma: float, weights: Array) -> Tuple[Array, Array]:
    disps = disp_vec(point, support_points, box_dims, periodic)
    gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
    grad = -2. * gamma * disps * gxdists.reshape(-1,1) * weights.reshape(-1,1)
    hess = jnp.zeros((point.shape[0], point.shape[0]))

    # Fill the Hessian matrix
    def fill_hess(i: int, j: int, hess: Array) -> Array:
        return jax.at[hess, (i, j)].set(-2. * gamma * jnp.sum(disps[:,j] * grad[:,i]))

    hess = lax.fori_loop(0, point.shape[0], lambda i, hess: lax.fori_loop(0, point.shape[0], lambda j, hess: fill_hess(i, j, hess), hess), hess)
    
    for i in range(point.shape[0]):
        hess = jax.at[hess, (i, i)].set(-2. * gamma * jnp.sum((1. - 2. * gamma * jnp.square(disps[:,i])) * gxdists * weights))
    
    return jnp.sum(grad,axis=0), hess

def gaussian_curvature(grad: Array, hess: Array) -> Array:
    X = jnp.pad(hess, ((0,1),(0,1)))
    div = -(vec_mag(grad)**4.)
    return jnp.linalg.det(X) / div

def mean_curvature(grad: Array, hess: Array) -> Array:
    grad_mag = vec_mag(grad)
    div = 2. * (grad_mag**3.)
    return ((grad @ hess @ grad.T) + (-(grad_mag**2.) * jnp.trace(hess))) / div

def curvatures(points: Array, support_points: Array, box_dims: Array, periodic: bool, gamma: float, weights: Array) -> Array:
    # Body function for loop
    def body_fn(i: int, curvatures: Array) -> Array:
        grad, hess = analytical_derivative(points[i], support_points, box_dims, periodic, gamma, weights)
        gaussian_curvatures = gaussian_curvature(grad, hess)
        mean_curvatures = mean_curvature(grad, hess)
        return jax.at[curvatures, i].set(jnp.array([gaussian_curvatures, mean_curvatures]))

    # Initialization
    initial_curvatures = jnp.zeros((points.shape[0], 2))

    # Iterative optimization
    curvatures = lax.fori_loop(0, points.shape[0], body_fn, initial_curvatures)
    return curvatures

class Backend(object):
    def __init__(self, xyz, train_indices, atom_ids_per_lipid, box_dims, periodic, gamma, 
                 train_labels='auto', learning_rate=None, max_iter=None, tol=None):
        if xyz.shape[0] != box_dims.shape[0]:
            raise ValueError('Lengths of inputs (xyz, box_dims) must match (%i != %i)'%(xyz.shape[0], box_dims.shape[0]))
        elif xyz.shape[-1] != box_dims.shape[-1] or box_dims.shape[-1] != periodic.shape[0]:
            raise ValueError('Dimensions of inputs (xyz, box_dims, periodic) must match')
        elif len(train_indices.shape) > 1:
            raise ValueError('training indices (train_indices) must be a one-dimensional array of integers (%i-d supplied)'%len(train_indices.shape))
        else:
            self.xyz = xyz.astype(np.float64)
            self.box_dims = box_dims.astype(np.float64)
            self.periodic = periodic
            self.gamma = gamma
            self.train_indices = train_indices.astype(np.int64)
            self.train_points = xyz[:,train_indices].copy()
            self.n_train_points = len(train_indices)
            self.train_labels = None
            self.weights_list = None
            self.intercept_list = None
            self.support_indices_list = None
        if np.sum([len(atom_id_per_lipid) for atom_id_per_lipid in atom_ids_per_lipid]) != self.xyz.shape[1]:
            raise ValueError('list of lipid atom ids (atom_ids_per_lipid) must sum to the total number of atoms')
        else:
            self.atom_ids_per_lipid = atom_ids_per_lipid
        if learning_rate is not None:
            if learning_rate > 1. or learning_rate < 0.:
                raise ValueError('learning rate (learning_rate) must be decimal between 0 and 1')
            else:
                self.learning_rate = learning_rate
        else:
            self.learning_rate = 0.01 # default learning_rate
        if max_iter is not None:
            self.max_iter = max_iter
        else:
            self.max_iter = 500 # default max_iter
        if tol is not None:
            if tol > 1. or tol < 0.:
                raise ValueError('tolerance (tol) must be decimal between 0 and 1')
            else:
                self.tol = tol
        else:
            self.tol = 0.0001 # default tol
        if train_labels != 'auto':
            if len(np.where(train_labels == 1.)[0]) + len(np.where(train_labels == -1.)[0]) != self.n_train_points:
                raise ValueError('supplied training labels (train_labels) must be either -1.0 (for bottom leaflet) or 1.0 (for top leaflet)')
            self.train_labels = train_labels
            self.autogenerate_labels = False
        else:
            self.autogenerate_labels = True
    
    def _calculate_train_labels(self, frame, distance_matrix):
        cluster = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='single')
        self.train_labels = cluster.fit_predict(distance_matrix)
        zmean0 = np.mean(self.train_points[frame,:,2][np.where(self.train_labels==0)[0]])
        zmean1 = np.mean(self.train_points[frame,:,2][np.where(self.train_labels==1)[0]])
        if zmean0 > zmean1:
            self.train_labels = np.where(self.train_labels==0, 1., -1.)
        else:
            self.train_labels = np.where(self.train_labels==1, 1., -1.)
    
    def _train_svm(self, kernel_matrix):
        model = SVC(C=1.,kernel='precomputed', verbose=0, cache_size=1000)
        model.fit(X=kernel_matrix, y=self.train_labels)
        return model.dual_coef_.astype(np.float64).flatten(), model.intercept_.astype(np.float64)[0], model.support_.astype(np.int64).flatten()
    
    def _calculate_curvature(self, frame):
        train_mat = sym_dist_mat(self.train_points[frame], self.box_dims[frame], self.periodic)
        if self.autogenerate_labels:
            self._calculate_train_labels(frame, train_mat)
        train_kernel = gaussian_transform_mat(train_mat, self.gamma) #04.25.2024 change this for JAX??
        weights, intercept, support_indices = self._train_svm(train_kernel)
        coms = calculate_lipid_coms(self.xyz[frame], self.atom_ids_per_lipid, self.box_dims[frame]) #04.25.2024 change below for JAX??
        bounds, normal_vectors = descend_to_boundary(coms, self.train_points[frame,support_indices], 
                    self.box_dims[frame], self.periodic, 
                    weights, intercept, self.gamma, self.learning_rate, self.max_iter, self.tol)
        gauss, mean = curvatures(bounds, self.train_points[frame,support_indices], self.box_dims[frame], self.periodic, self.gamma, weights)
        mean *= -1.
        mean *= self.train_labels
        return mean, gauss, normal_vectors, weights, intercept, support_indices
    
    def calculate_curvature(self, frames=None, stride=None):
        if (frames is None and stride is None) or (frames is not None and stride is not None):
            raise ValueError('Please supply only one of the parameters frames or stride')
        elif frames is None and stride is not None:
            frames = np.arange(0, len(self.xyz), stride)
        elif frames is not None and stride is None:
            if type(frames) == str:
                if frames == 'all':
                    frames = np.arange(0, len(self.xyz))
            else:
                try:
                    assert len(frames.shape) == 1
                except AssertionError:
                    raise ValueError('Parameter frames must be a 1-dimensional np array')            
        n_frames = len(frames)
        self.weights_list = []
        self.intercept_list = []
        self.support_indices_list = []
        self.normal_vectors = np.empty((n_frames, self.n_train_points, 3))
        self.mean_curvature = np.empty((n_frames, self.n_train_points))
        self.gaussian_curvature = np.empty((n_frames, self.n_train_points))
        for i,frame in enumerate(frames):
            self.mean_curvature[i], self.gaussian_curvature[i], self.normal_vectors[i], weights_i, intercept_i, support_indices_i = self._calculate_curvature(frame)
            self.weights_list.append(weights_i)
            self.intercept_list.append(intercept_i)
            self.support_indices_list.append(support_indices_i)

if __name__ == "__main__":
    args = get_args()
    
    # load structure into mdtraj trajectory object
    trajectory = md.load(os.path.join(args.data_dir, args.pdb)) 

    # remove water, ions
    lipid = trajectory.atom_slice(trajectory.top.select(args.atom_selection)) #'not name W WF NA CL'

    # define selection for training set
    head_selection_text = args.head_selection #'name PO4' 
    head_selection = lipid.top.select(head_selection_text)

    # define periodicity of system in x,y,z directions
    periodic = np.array([True, True, False]) 

    # get indices of each lipid, required for COM calculation
    atom_ids_per_lipid = [np.array([atom.index for atom in residue.atoms]) for residue in lipid.top.residues] 

    # define gamma, hyperparameter used for RBF kernel 
    gamma = 0.1 

    
    ########TEST########
    xyzs = jnp.array(np.random.normal(size=(20,3)))

    r = unravel_upper_triangle_index(10)
    print(r)
    
    r = vec_sum(xyzs)
    print(r)
    
    start = time.perf_counter()
    periodic = jnp.array(periodic)
    box_dims=jnp.array([4,3,5])
    r = sym_dist_mat(xyzs, box_dims, periodic) #.block_until_ready()       
    print(r)
    end = time.perf_counter()
    print(end-start)
    
    
    start = time.perf_counter()
    for _ in range(10000):
        periodic = jnp.array(periodic)
        box_dims=jnp.array([4,3,5])
        r = sym_dist_mat(xyzs, box_dims, periodic) #.block_until_ready()       
#         print(r)
    end = time.perf_counter()
    print(end-start)
    
    
    
    
    
    
    
##########FINAL TEST############
#     svmem = SVMem(lipid.xyz, # atomic xyz coordinates of all lipids; shape = (n_frames, n_atoms)
#                   head_selection, # indices of training points; shape = (n_lipids)
#                   atom_ids_per_lipid, # list of atom ids for each lipid; shape = (n_lipids, 
#                   lipid.unitcell_lengths, # unitcell dimensions; shape = (n_frames, 3)
#                   periodic, 
#                   gamma) 

#     svmem.calculate_curvature(frames='all')

#     # curvature and normal vectors are stored in the svmem object
#     print(svmem.mean_curvature, svmem.gaussian_curvature, svmem.normal_vectors)
