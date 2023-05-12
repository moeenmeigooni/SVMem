import warnings
import numpy as np
import mdtraj as md
from numba import njit, prange
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax import random
import os, sys, argparse, pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots)
from jax_utils.main import get_args
from SVMem import SVMem
from functools import partial

warnings.simplefilter('ignore')

@jit
def ndot(a, b):
    n = len(a)
    s = 0.
    for i in range(n):
        s += a[i] * b[i]
    return s.item()

@jit
def nsign(x):
#     if x > 0.:
#         s = 1.
#     else:
#         s = -1.
    s = jax.lax.cond(pred=(x>0.), true_fun=lambda s: 1., false_fun=lambda s: -1., operand=x).item() #.item() for scalar;; necessary for differrentiation!
    #https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#:~:text=Summary,unrolls%20the%20loop
    return s

@jit
def nsign_int(x):
#     if x > 0.:
#         s = 1
#     else:
#         s = -1
    s = jax.lax.cond(pred=(x>0.), true_fun=lambda s: 1, false_fun=lambda s: -1, operand=x).item() #.item() for scalar;; necessary for differrentiation!
    return s

####EQUIVALENT TO ABOVE!
# @partial(jit, static_argnums=(0,))
# def nsign_int(x):
#     if x > 0.:
#         s = 1
#     else:
#         s = -1
#     return s

@jit
def vec_mag(vec):
    n = len(vec)
    l = 0.
    for i in range(n):
        l += (vec[i])**2.
    return jnp.sqrt(l).item()

@jit
def vec_mags(vecs):
    n = vecs.shape[0]
    d = vecs.shape[1]
    mags = jnp.empty((n))
    for i in range(n):
        mags = mags.at[i].set(vec_mag(vecs[i]))
    return mags

@jit
def vec_norm(vec):
    return vec / vec_mag(vec)

@jit
def vec_norms(vecs):
    n = len(vecs)
    norm_vecs = jnp.empty_like(vecs)
    for i in range(n):
        norm_vecs = norm_vecs.at[i].set(vec_norm(vecs[i]))
    return norm_vecs

@jit
def vec_sum(vecs):
    n = vecs.shape[0]
    d = vecs.shape[1]
    vecsum = jnp.zeros((d))
    for i in range(n):
        for j in range(d):
            vecsum = vecsum.at[j].add(vecs[i,j])
    return vecsum

@partial(jit, static_argnames=["n1", "n2"])
def unravel_index(n1, n2):
    a, b = jnp.empty((n1, n2), dtype=jnp.int64), jnp.empty((n1, n2), dtype=jnp.int64)
    for i in range(n1):
        for j in range(n2):
            a = a.at[i,j].set(i)
            b = b.at[i,j].set(j)
    return a.ravel(), b.ravel()

@partial(jit, static_argnames=["n"])
def unravel_upper_triangle_index(n):
    n_unique = (n * (n-1)) // 2
    a, b = jnp.empty((n_unique),dtype=jnp.int64), jnp.empty((n_unique),dtype=jnp.int64)
    k = 0
    for i in range(n):
        for j in range(n):
#             if i < j: #not subject to jax(static_argnums) abstraction
#                 a = a.at[k].set(i)
#                 b = b.at[k].set(j)
#                 k += 1
            (a, b, k) = jax.lax.cond(pred=(i < j), true_fun=lambda a, b, k, i, j: (a.at[k].set(i), b.at[k].set(j), k+1), false_fun=lambda a, b, k, i, j: (a, b, k), operand=(a, b, k, i, j) ).item() #.item() for scalar;; necessary for differrentiation!
    return a, b

# @njit(parallel=True)
@jit
def sym_dist_mat_(xyzs, box_dims, periodic):
    n = xyzs.shape[0]
    n_unique = (n * (n-1)) // 2
    ndim = xyzs.shape[1]
    i, j = unravel_upper_triangle_index(n)
    dist_mat = jnp.zeros((n_unique))
    for k in prange(n_unique): #n_unique: upper triangle of pairwise dist
        for ri in prange(ndim): #ndim: 3 (xyz)
            dr = jnp.abs(xyzs[i[k],ri] - xyzs[j[k],ri])
#             if periodic[ri] == True:
#                 while (dr >  (box_dims[ri]*0.5)):
#                     dr -= box_dims[ri]
            def while_func(dr):
                cond_fun = lambda dr : dr >  (box_dims[ri]*0.5)
                body_fun = lambda dr : dr - box_dims[ri]
                dr = jax.lax.while_loop(cond_fun=cond_fun, body_fun=body_fun, init_val=dr)
                return dr
            dr = jax.lax.cond(pred=periodic[ri], true_fun=while_func, false_fun=lambda dr: dr, operand=dr)
            dist_mat.at[k].add(jnp.square(dr))
    return jnp.sqrt(dist_mat)

@jit
def sym_dist_mat(xyzs, box_dims, periodic):
    n = xyzs.shape[0]
    dist_mat_flat = sym_dist_mat_(xyzs, box_dims, periodic)
    dist_mat = jnp.zeros((n,n))
    k = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                dist_mat = dist_mat.at[i,j].set(dist_mat_flat[k])
                dist_mat = dist_mat.at[j,i].set(dist_mat_flat[k])
                k += 1
    return dist_mat

@jit#(parallel=True)
def dist_mat_(xyz1, xyz2, box_dims, periodic):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    ndim = xyz1.shape[1]
    dist_mat = jnp.zeros((n1 * n2))
    i, j = unravel_index(n1, n2)
    for k in prange(n1 * n2):
        dr = jnp.abs(xyz1[i[k]] - xyz2[j[k]])
        for ri in range(ndim):
            if periodic[ri] == True:
                while (dr[ri] >  (box_dims[ri]*0.5)):
                    dr[ri] -= box_dims[ri]
            dist_mat[k] += jnp.square(dr[ri])
    return jnp.sqrt(dist_mat)

@jit
def dist_mat(xyz1, xyz2, box_dims, periodic):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    return dist_mat_(xyz1, xyz2, box_dims, periodic).reshape(n1, n2)

@njit(parallel=True)
def dist_mat_parallel_(xyz1, xyz2, box_dims, periodic):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    ndim = xyz1.shape[1]
    dist_mat = jnp.zeros((n1 * n2))
    i, j = unravel_index(n1, n2)
    for k in prange(n1 * n2):
        for ri in prange(ndim):
            dr = jnp.abs(xyz1[i[k],ri] - xyz2[j[k],ri])
            if periodic[ri] == True:
                if (dr >  (box_dims[ri]*0.5)):
                    dr -= box_dims[ri]
            dist_mat[k] += jnp.square(dr)
    return jnp.sqrt(dist_mat)

@jit
def dist_mat_parallel(xyz1, xyz2, box_dims, periodic):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    return dist_mat_parallel_(xyz1, xyz2, box_dims, periodic).reshape(n1, n2)

@njit(parallel=True)
def dist_vec(xyz, xyzs, box_dims, periodic):
    n = len(xyzs)
    ndim = len(xyz)
    dist_vec = jnp.zeros((n))
    for i in prange(n):
        for ri in prange(ndim):
            dr = jnp.abs(xyzs[i,ri] - xyz[ri])
            if periodic[ri] == True:
                while (dr >  (box_dims[ri]*0.5)):
                    dr -= box_dims[ri]
            dist_vec[i] += jnp.square(dr)
    return jnp.sqrt(dist_vec)

@jit
def disp(xyz1, xyz2, box_dims, periodic):
    d = len(xyz1)
    disp = jnp.zeros((d))
    for ri in range(d):
        dr = xyz1[ri] - xyz2[ri]
        if periodic[ri] == True:
            while (dr >  ( box_dims[ri]*0.5)):
                dr -= box_dims[ri]
            while (dr <= (-box_dims[ri]*0.5)):
                dr += box_dims[ri]
        disp[ri] = dr
    return disp

@jit
def disp_vec(xyz, xyzs, box_dims, periodic):
    n = xyzs.shape[0]
    d = xyzs.shape[1]
    disps = jnp.empty((n,d))
    for i in range(n):
        disps[i] = disp(xyz, xyzs[i], box_dims, periodic)
    return disps        

@jit
def gaussian_transform_vec(array, gamma):
    g_array = jnp.empty_like(array)
    n = array.shape[0]
    for i in range(n):
        g_array[i] = jnp.exp(-gamma * jnp.square(array[i]))
    return g_array

@njit(parallel=True)
def gaussian_transform_vec_parallel(array, gamma):
    g_array = jnp.empty_like(array)
    n = array.shape[0]
    for i in prange(n):
        g_array[i] = jnp.exp(-gamma * jnp.square(array[i]))
    return g_array

@njit(parallel=True)
def gaussian_transform_mat_(array, gamma):
    g_array = jnp.empty_like(array)
    n = array.shape[0]
    for i in prange(n):
        g_array[i] = jnp.exp(-gamma * jnp.square(array[i]))
    return g_array

@jit
def gaussian_transform_mat(mat, gamma):
    return gaussian_transform_mat_(mat.ravel(), gamma).reshape(mat.shape)

@jit
def decision_function(vec, weights, intercept):
    n = vec.shape[0]
    decision = 0.
    for i in range(n):
        decision += weights[i] * vec[i]
    return decision + intercept

@jit
def decision_function_mat(mat, weights, intercept):
    n = mat.shape[0]
    decisions = jnp.zeros((n))
    for i in range(n):
        decisions[i] = decision_function(mat[i], weights, intercept)
    return decisions

@njit
def predict(vec, weights, intercept):
    return jnp.sign(decision_function(vec, weights, intercept))

@jit
def predict_mat(vec, weights, intercept):
    return jnp.sign(decision_function_mat(vec, weights, intercept))

@njit(parallel=True)
def pbc_center(xyzs, box_dims):
    n = xyzs.shape[0]
    d = xyzs.shape[1]
    center = jnp.empty((d))
    for ri in prange(d):
        rmax = jnp.max(xyzs[:,ri])
        xi = 0.
        zeta = 0.
        for j in prange(n):
            thetai = 2.*jnp.pi*xyzs[j,ri]/rmax
            xi += jnp.cos(thetai)
            zeta += jnp.sin(thetai)
        xi /= -n
        zeta /= -n
        theta = jnp.arctan2(zeta,xi) + jnp.pi
        center[ri] = theta * rmax / (2.*jnp.pi)
    return center

@jit
def calculate_lipid_coms(lipids_xyz, atom_ids_per_lipid, box_dims):
    n_lipids = len(atom_ids_per_lipid)
    coms = jnp.empty((n_lipids, 3))
    for i in range(n_lipids):
        coms[i] = pbc_center(lipids_xyz[atom_ids_per_lipid[i]], box_dims)
    return coms

@jit
def update_disps(disps, step, box_dims, periodic):
    n = disps.shape[0]
    d = 3
    for i in range(n):
        for j in range(d):
            disps[i,j] += step[j]
            if periodic[j] == True:
                if (disps[i,j] >  ( box_dims[j]*0.5)):
                    disps[i,j] -= box_dims[j]
                if (disps[i,j] <= (-box_dims[j]*0.5)):
                    disps[i,j] += box_dims[j]
    return disps

@jit
def gradient(disps, gxdists, gamma, weights):
    n = disps.shape[0]
    del_F = jnp.zeros((3))
    factor = -2.*gamma
    for i in range(n):
        for j in range(3):
            del_F[j] += factor * weights[i] * disps[i,j] * gxdists[i]
    return del_F

@jit
def gradient_descent(point_, support_points, box_dims, periodic, weights, intercept, gamma, learning_rate, max_iter):
    point = point_.copy()
    disps = disp_vec(point, support_points, box_dims, periodic)
    gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
    d = decision_function(gxdists, weights, intercept)
    sign = nsign_int(d)
    step = -learning_rate * nsign(d) * vec_norm(gradient(disps, gxdists, gamma, weights))
    for i in range(max_iter):
        point += step
        disps = update_disps(disps, step, box_dims, periodic)
        gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
        d = decision_function(gxdists, weights, intercept)
        newsign = nsign_int(d)
        if newsign != sign:
            step *= -1.
            break
        step = -learning_rate * nsign(d) * vec_norm(gradient(disps, gxdists, gamma, weights))
    return point, vec_norm(step), disps

@jit
def coordinate_descent(point_, step, disps, box_dims, periodic, weights, intercept, gamma, step_init, max_iter, tol):
    point = point_.copy()
    step = step_init * step
    gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
    d = decision_function(gxdists, weights, intercept)
    s = nsign_int(d)
    for i in range(max_iter):
        point += step
        disps = update_disps(disps, step, box_dims, periodic)
        gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
        d = decision_function(gxdists, weights, intercept)
        news = nsign_int(d)
        if news != s:
            step *= -0.5
        if jnp.abs(d) < tol:
            break
        s = news
    return point
    
@njit(parallel=True)
def descend_to_boundary(points, support_points, box_dims, periodic, weights, intercept, gamma, learning_rate, max_iter, tol):
    n = points.shape[0]
    d = points.shape[1]
    bounds = jnp.empty((n, d))
    normal_vectors = jnp.empty((n, d))
    for i in prange(n):
        approx_bound, normal_vectors[i], disps  = gradient_descent(
            points[i], support_points, 
            box_dims, periodic, weights, intercept, gamma, 
            learning_rate, max_iter)
        bounds[i] = coordinate_descent(
            approx_bound, normal_vectors[i], disps, 
            box_dims, periodic, weights, intercept, gamma, 
            learning_rate, max_iter, tol)
    return bounds, -1.*normal_vectors

@jit
def analytical_derivative(point, support_points, box_dims, periodic, gamma, weights):
    d = point.shape[0]
    n = support_points.shape[0]
    disps = disp_vec(point, support_points, box_dims, periodic)
    gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
    grad = -2. * gamma * disps * gxdists.reshape(-1,1) * weights.reshape(-1,1)
    hess = jnp.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if i < j:
                hess[i,j] = jnp.sum(-2. * gamma * disps[:,j] * grad[:,i])
    for i in range(d):
        hess[i,i] = jnp.sum(-2. * gamma * (1. - 2. * gamma * jnp.square(disps[:,i])) * gxdists * weights)
    for i in range(d):
        for j in range(i+1,d):
            hess[j,i] = hess[i,j]
    return jnp.sum(grad,axis=0), hess

@jit
def gaussian_curvature(grad, hess):
    n = len(grad)
    X = jnp.empty((n+1, n+1))
    X[n,n] = 0.
    for i in range(n):
        for j in range(n):
            X[i,j] = hess[i,j]
    for i in range(n):
        X[n,i] = grad[i]
        X[i,n] = grad[i]
    div = -(vec_mag(grad)**4.)
    return jnp.linalg.det(X) / div   

@jit
def mean_curvature(grad, hess):
    grad_mag = vec_mag(grad)
    div = 2. * (grad_mag**3.)
    return ((grad @ hess @ grad.T) + (-(grad_mag**2.) * jnp.trace(hess))) / div

@njit(parallel=True)
def curvatures(points, support_points, box_dims, periodic, gamma, weights):
    n = points.shape[0]
    mean_curvatures = jnp.zeros((n))
    gaussian_curvatures = jnp.zeros((n))
    for i in prange(n):
        grad, hess = analytical_derivative(points[i], support_points, box_dims, periodic, gamma, weights)
        gaussian_curvatures[i] = gaussian_curvature(grad, hess)
        mean_curvatures[i] = mean_curvature(grad, hess)
    return gaussian_curvatures, mean_curvatures

class SVMem_jax(object):
    def __init__(self, xyz, train_indices, atom_ids_per_lipid, box_dims, periodic, gamma, 
                 train_labels='auto', learning_rate=None, max_iter=None, tol=None):
        if xyz.shape[0] != box_dims.shape[0]:
            raise ValueError('Lengths of inputs (xyz, box_dims) must match (%i != %i)'%(xyz.shape[0], box_dims.shape[0]))
        elif xyz.shape[-1] != box_dims.shape[-1] or box_dims.shape[-1] != periodic.shape[0]:
            raise ValueError('Dimensions of inputs (xyz, box_dims, periodic) must match')
        elif len(train_indices.shape) > 1:
            raise ValueError('training indices (train_indices) must be a one-dimensional array of integers (%i-d supplied)'%len(train_indices.shape))
        else:
            self.xyz = xyz.astype(jnp.float64)
            self.box_dims = box_dims.astype(jnp.float64)
            self.periodic = periodic
            self.gamma = gamma
            self.train_indices = train_indices.astype(jnp.int64)
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
            if len(jnp.where(train_labels == 1.)[0]) + len(jnp.where(train_labels == -1.)[0]) != self.n_train_points:
                raise ValueError('supplied training labels (train_labels) must be either -1.0 (for bottom leaflet) or 1.0 (for top leaflet)')
            self.train_labels = train_labels
            self.autogenerate_labels = False
        else:
            self.autogenerate_labels = True
    
    def _calculate_train_labels(self, frame, distance_matrix):
        cluster = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='single')
        self.train_labels = cluster.fit_predict(distance_matrix)
        zmean0 = jnp.mean(self.train_points[frame,:,2][jnp.where(self.train_labels==0)[0]])
        zmean1 = jnp.mean(self.train_points[frame,:,2][jnp.where(self.train_labels==1)[0]])
        if zmean0 > zmean1:
            self.train_labels = jnp.where(self.train_labels==0, 1., -1.)
        else:
            self.train_labels = jnp.where(self.train_labels==1, 1., -1.)
    
    def _train_svm(self, kernel_matrix):
        model = SVC(C=1.,kernel='precomputed', verbose=0, cache_size=1000)
        model.fit(X=kernel_matrix, y=self.train_labels)
        return model.dual_coef_.astype(jnp.float64).flatten(), model.intercept_.astype(jnp.float64)[0], model.support_.astype(jnp.int64).flatten()
    
    def _calculate_curvature(self, frame):
        train_mat = sym_dist_mat(self.train_points[frame], self.box_dims[frame], self.periodic)
        if self.autogenerate_labels:
            self._calculate_train_labels(frame, train_mat)
        train_kernel = gaussian_transform_mat(train_mat, self.gamma)
        weights, intercept, support_indices = self._train_svm(train_kernel)
        coms = calculate_lipid_coms(self.xyz[frame], self.atom_ids_per_lipid, self.box_dims[frame])
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
            frames = jnp.arange(0, len(self.xyz), stride)
        elif frames is not None and stride is None:
            if type(frames) == str:
                if frames == 'all':
                    frames = jnp.arange(0, len(self.xyz))
            else:
                try:
                    assert len(frames.shape) == 1
                except AssertionError:
                    raise ValueError('Parameter frames must be a 1-dimensional np array')            
        n_frames = len(frames)
        self.weights_list = []
        self.intercept_list = []
        self.support_indices_list = []
        self.normal_vectors = jnp.empty((n_frames, self.n_train_points, 3))
        self.mean_curvature = jnp.empty((n_frames, self.n_train_points))
        self.gaussian_curvature = jnp.empty((n_frames, self.n_train_points))
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
    periodic = jnp.array(periodic)
    box_dims=jnp.array([4,3,5])
    xyzs = jnp.array(np.random.normal(size=(20,3)))
    r = sym_dist_mat_(xyzs, box_dims, periodic)       
    print(r)
    
    
    
    
    
    
    
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
