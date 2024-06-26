import numpy as np
from numba import njit, prange
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering

@njit
def ndot(a, b):
    n = len(a)
    s = 0.
    for i in range(n):
        s += a[i] * b[i]
    return s

@njit
def nsign(x):
    if x > 0.:
        s = 1.
    else:
        s = -1.
    return s

@njit
def nsign_int(x):
    if x > 0.:
        s = 1
    else:
        s = -1
    return s

@njit
def vec_mag(vec):
    n = len(vec)
    l = 0.
    for i in range(n):
        l += (vec[i])**2.
    return np.sqrt(l)

@njit
def vec_mags(vecs):
    n = vecs.shape[0]
    d = vecs.shape[1]
    mags = np.empty((n))
    for i in range(n):
        mags[i] = vec_mag(vecs[i])
    return mags

@njit
def vec_norm(vec):
    return vec / vec_mag(vec)

@njit
def vec_norms(vecs):
    n = len(vecs)
    norm_vecs = np.empty_like(vecs)
    for i in range(n):
        norm_vecs[i] = vec_norm(vecs[i])
    return norm_vecs

@njit
def vec_sum(vecs):
    n = vecs.shape[0]
    d = vecs.shape[1]
    vecsum = np.zeros((d))
    for i in range(n):
        for j in range(d):
            vecsum[j] += vecs[i,j]
    return vecsum

@njit
def unravel_index(n1, n2):
    a, b = np.empty((n1, n2), dtype=np.int64), np.empty((n1, n2), dtype=np.int64)
    for i in range(n1):
        for j in range(n2):
            a[i,j], b[i,j] = i, j
    return a.ravel(),b.ravel()

@njit
def unravel_upper_triangle_index(n):
    n_unique = (n * (n-1)) // 2
    a, b = np.empty((n_unique),dtype=np.int64), np.empty((n_unique),dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                a[k], b[k] = i, j
                k += 1
    return a, b

@njit(parallel=True)
def sym_dist_mat_(xyzs, box_dims, periodic):
    n = xyzs.shape[0]
    n_unique = (n * (n-1)) // 2
    ndim = xyzs.shape[1]
    i, j = unravel_upper_triangle_index(n)
    dist_mat = np.zeros((n_unique))
    for k in prange(n_unique):
        for ri in prange(ndim):
            dr = np.abs(xyzs[i[k],ri] - xyzs[j[k],ri])
            if periodic[ri] == True:
                while (dr >  (box_dims[ri]*0.5)):
                    dr -= box_dims[ri]
            dist_mat[k] += np.square(dr)
    return np.sqrt(dist_mat)

@njit
def sym_dist_mat(xyzs, box_dims, periodic):
    n = xyzs.shape[0]
    dist_mat_flat = sym_dist_mat_(xyzs, box_dims, periodic)
    dist_mat = np.zeros((n,n))
    k = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                dist_mat[i,j] = dist_mat_flat[k]
                dist_mat[j,i] = dist_mat_flat[k]
                k += 1
    return dist_mat

@njit#(parallel=True)
def dist_mat_(xyz1, xyz2, box_dims, periodic):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    ndim = xyz1.shape[1]
    dist_mat = np.zeros((n1 * n2))
    i, j = unravel_index(n1, n2)
    for k in prange(n1 * n2):
        dr = np.abs(xyz1[i[k]] - xyz2[j[k]])
        for ri in range(ndim):
            if periodic[ri] == True:
                while (dr[ri] >  (box_dims[ri]*0.5)):
                    dr[ri] -= box_dims[ri]
            dist_mat[k] += np.square(dr[ri])
    return np.sqrt(dist_mat)

@njit
def dist_mat(xyz1, xyz2, box_dims, periodic):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    return dist_mat_(xyz1, xyz2, box_dims, periodic).reshape(n1, n2)

@njit(parallel=True)
def dist_mat_parallel_(xyz1, xyz2, box_dims, periodic):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    ndim = xyz1.shape[1]
    dist_mat = np.zeros((n1 * n2))
    i, j = unravel_index(n1, n2)
    for k in prange(n1 * n2):
        for ri in prange(ndim):
            dr = np.abs(xyz1[i[k],ri] - xyz2[j[k],ri])
            if periodic[ri] == True:
                if (dr >  (box_dims[ri]*0.5)):
                    dr -= box_dims[ri]
            dist_mat[k] += np.square(dr)
    return np.sqrt(dist_mat)

@njit
def dist_mat_parallel(xyz1, xyz2, box_dims, periodic):
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    return dist_mat_parallel_(xyz1, xyz2, box_dims, periodic).reshape(n1, n2)

@njit
def dist_vec(xyz, xyzs, box_dims, periodic):
    n = len(xyzs)
    ndim = len(xyz)
    dist_vec = np.zeros((n))
    for i in prange(n):
        for ri in prange(ndim):
            dr = np.abs(xyzs[i,ri] - xyz[ri])
            if periodic[ri] == True:
                while (dr >  (box_dims[ri]*0.5)):
                    dr -= box_dims[ri]
            dist_vec[i] += np.square(dr)
    return np.sqrt(dist_vec)

@njit
def disp(xyz1, xyz2, box_dims, periodic):
    d = len(xyz1)
    disp = np.zeros((d))
    for ri in range(d):
        dr = xyz1[ri] - xyz2[ri]
        if periodic[ri] == True:
            while (dr >  ( box_dims[ri]*0.5)):
                dr -= box_dims[ri]
            while (dr <= (-box_dims[ri]*0.5)):
                dr += box_dims[ri]
        disp[ri] = dr
    return disp

@njit
def disp_vec(xyz, xyzs, box_dims, periodic):
    n = xyzs.shape[0]
    d = xyzs.shape[1]
    disps = np.empty((n,d))
    for i in range(n):
        disps[i] = disp(xyz, xyzs[i], box_dims, periodic)
    return disps        

@njit
def gaussian_transform_vec(array, gamma):
    g_array = np.empty_like(array)
    n = array.shape[0]
    for i in range(n):
        g_array[i] = np.exp(-gamma * np.square(array[i]))
    return g_array

@njit(parallel=True)
def gaussian_transform_vec_parallel(array, gamma):
    g_array = np.empty_like(array)
    n = array.shape[0]
    for i in prange(n):
        g_array[i] = np.exp(-gamma * np.square(array[i]))
    return g_array

@njit(parallel=True)
def gaussian_transform_mat_(array, gamma):
    g_array = np.empty_like(array)
    n = array.shape[0]
    for i in prange(n):
        g_array[i] = np.exp(-gamma * np.square(array[i]))
    return g_array

@njit
def gaussian_transform_mat(mat, gamma):
    return gaussian_transform_mat_(mat.ravel(), gamma).reshape(mat.shape)

@njit
def decision_function(vec, weights, intercept):
    n = vec.shape[0]
    decision = 0.
    for i in range(n):
        decision += weights[i] * vec[i]
    return decision + intercept

@njit
def decision_function_mat(mat, weights, intercept):
    n = mat.shape[0]
    decisions = np.zeros((n))
    for i in range(n):
        decisions[i] = decision_function(mat[i], weights, intercept)
    return decisions

@njit
def predict(vec, weights, intercept):
    return np.sign(decision_function(vec, weights, intercept))

@njit
def predict_mat(vec, weights, intercept):
    return np.sign(decision_function_mat(vec, weights, intercept))

@njit
def pbc_center(xyzs, box_dims):
    n = xyzs.shape[0]
    d = xyzs.shape[1]
    center = np.empty((d))
    for ri in prange(d):
        rmax = np.max(xyzs[:,ri])
        xi = 0.
        zeta = 0.
        for j in prange(n):
            thetai = 2.*np.pi*xyzs[j,ri]/rmax
            xi += np.cos(thetai)
            zeta += np.sin(thetai)
        xi /= -n
        zeta /= -n
        theta = np.arctan2(zeta,xi) + np.pi
        center[ri] = theta * rmax / (2.*np.pi)
    return center

@njit
def calculate_lipid_coms(lipids_xyz, atom_ids_per_lipid, box_dims):
    n_lipids = len(atom_ids_per_lipid)
    coms = np.empty((n_lipids, 3))
    for i in range(n_lipids):
        coms[i] = pbc_center(lipids_xyz[atom_ids_per_lipid[i]], box_dims)
    return coms

@njit
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

@njit
def gradient(disps, gxdists, gamma, weights):
    n = disps.shape[0]
    del_F = np.zeros((3))
    factor = -2.*gamma
    for i in range(n):
        for j in range(3):
            del_F[j] += factor * weights[i] * disps[i,j] * gxdists[i]
    return del_F

@njit
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

@njit
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
        if np.abs(d) < tol:
            break
        s = news
    return point
    
@njit(parallel=True)
def descend_to_boundary(points, support_points, box_dims, periodic, weights, intercept, gamma, learning_rate, max_iter, tol):
    n = points.shape[0]
    d = points.shape[1]
    bounds = np.empty((n, d))
    normal_vectors = np.empty((n, d))
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

@njit
def analytical_derivative(point, support_points, box_dims, periodic, gamma, weights):
    d = point.shape[0]
    n = support_points.shape[0]
    disps = disp_vec(point, support_points, box_dims, periodic)
    gxdists = gaussian_transform_vec(vec_mags(disps), gamma)
    grad = -2. * gamma * disps * gxdists.reshape(-1,1) * weights.reshape(-1,1)
    hess = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if i < j:
                hess[i,j] = np.sum(-2. * gamma * disps[:,j] * grad[:,i])
    for i in range(d):
        hess[i,i] = np.sum(-2. * gamma * (1. - 2. * gamma * np.square(disps[:,i])) * gxdists * weights)
    for i in range(d):
        for j in range(i+1,d):
            hess[j,i] = hess[i,j]
    return np.sum(grad,axis=0), hess

@njit
def gaussian_curvature(grad, hess):
    n = len(grad)
    X = np.empty((n+1, n+1))
    X[n,n] = 0.
    for i in range(n):
        for j in range(n):
            X[i,j] = hess[i,j]
    for i in range(n):
        X[n,i] = grad[i]
        X[i,n] = grad[i]
    div = -(vec_mag(grad)**4.)
    return np.linalg.det(X) / div   

@njit
def mean_curvature(grad, hess):
    grad_mag = vec_mag(grad)
    div = 2. * (grad_mag**3.)
    return ((grad @ hess @ grad.T) + (-(grad_mag**2.) * np.trace(hess))) / div

@njit
def curvatures(points, support_points, box_dims, periodic, gamma, weights):
    n = points.shape[0]
    mean_curvatures = np.zeros((n))
    gaussian_curvatures = np.zeros((n))
    for i in prange(n):
        grad, hess = analytical_derivative(points[i], support_points, box_dims, periodic, gamma, weights)
        gaussian_curvatures[i] = gaussian_curvature(grad, hess)
        mean_curvatures[i] = mean_curvature(grad, hess)
    return gaussian_curvatures, mean_curvatures

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
        cluster = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='single')
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
    import time

    ########TEST######
    xyzs = np.array(np.random.normal(size=(20,3)))
    
    r = unravel_upper_triangle_index(10)
    print(r)
    
    r = vec_sum(xyzs)
    print(r)
    
    start = time.perf_counter()
    periodic = np.array([True, True, False])
    box_dims=np.array([4,3,5])
    r = sym_dist_mat(xyzs, box_dims, periodic)       
    print(r)
    end = time.perf_counter()
    print(end-start)
    
    start = time.perf_counter()
    for _ in range(10000):
        periodic = np.array([True, True, False])
        box_dims=np.array([4,3,5])
        r = sym_dist_mat(xyzs, box_dims, periodic)      
    end = time.perf_counter()
    print(end-start)
    
