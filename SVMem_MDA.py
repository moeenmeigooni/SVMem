import argparse 
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
import numpy as np
from typing import List

parser = argparse.ArgumentParser(description='')

parser.add_argument('topology', help='Topology file for trajectory (e.g. psf, parm7, pdb)')
parser.add_argument('trajectory', help='Trajectory file or list of trajectory files (e.g. dcd, xtc)')
parser.add_argument('-b', '--backend', default='numba', choices=['jax', 'numba', 'numpy'],
                    help='Backend for curvature calculations.')
parser.add_argument('-m', '--membrane', default='segname MEMB', help='Atom selection text\
    that adheres to the MDAnalysis conventions for selecting your whole membrane.')
parser.add_argument('-ff', '--forcefield', choices=['martini', 'charmm', 'amber'], 
                    help='Forcefield used for simulation')
parser.add_argument('-p', '--periodic', default=[True, True, False],
                    help='Boolean list of shape (3,1) which describes the dimensions \
                        to consider periodicity in. Should be [True, True, False]')
parser.add_argument('-g', '--gamma', default=0.1, help='Value for gamma hyperparameter.')
parser.add_argument('-lr', '--learning_rate', default=0.01, help='Learning rate for SVM training.')
parser.add_argument('-mi', '--max_iter', default=500, help='Maximum number of iterations of training.')
parser.add_argument('-t', '--tolerance', default=0.001, help='Tolerance.')
parser.add_argument('-l', '--labels', default=None, help='Precomputed training labels.')

class MembraneCurvature(AnalysisBase):
    """
    Analysis class that computes membrane curvature for a given simulation. 
    Must provide an MDAnalysis universe object, membrane atom selection and
    a methodology for computing the curvature. Options include: JAX, numba or
    simply numpy.
    ------
    Inputs
    ------
    
    """
    def __init__(self, memb: mda.AtomGroup,
                 method: str='numba', forcefield: str='martini',
                 periodic: List[bool]=[True, True, False], gamma: float=.1,
                 learning_rate: float=0.01, max_iter: int=500,
                 tolerance: float=0.0001, train_labels: None):
        super().__init__(memb.universe.trajectory)
        self.u = memb.universe
        self.memb = memb
        
        # Defined headgroup atom selections by forcefield
        if forcefield == 'martini':
            head_sel = 'name GL0 PO4'
        else:
            head_sel = 'name ? P'
            
        self.train_points = u.select_atoms(head_sel)
        self.n_train_points = self.train_points.n_atoms
        self.n_frames = len(u.trajectory)
        
        # Switch for underlying methodology
        if method.lower() == 'jax':
            from jax_utils import SVMem_jax as SVMem
        elif method.lower() == 'numpy':
            from numpy_utils import SVMem_numpy as SVMem
        else:
            from numba_utils import SVMem_numba as SVMem
            
        atom_ids_per_lipid = [residue.atoms.indices for residue in memb.residues]
        self.backend = SVMem.Backend(periodic, train_labels, gamma, learning_rate, 
                               max_iter, tolerance, atom_ids_per_lipid)
        
    def _prepare(self):
        """
        Preprocessing of data structures and universe object for per-frame calculations.
        """
        self.weights_list = []
        self.intercept_list = []
        self.support_indices_list = []
        self.mean_curvature = np.empty((self.n_frames, self.n_train_points))
        self.gaussian_curvature = np.empty_like(self.mean_curvature)
        self.normal_vectors = np.empty((self.n_frames, self.n_train_points, 3))

    def _single_frame(self):
        """
        Calling the specified backend, compute the curvature for each frame of simulation.
        """
        fr = self.u.trajectory.frame
        mean, gaussian, normals, weights, intercept, support_indices = self.backend.calculate_curvature(self.train_points.positions,
                                                                                                        self.u.dimensions[:3],
                                                                                                        self.memb)
        self.mean_curvature[fr] = mean
        self.gaussian_curvature[fr] = gaussian
        self.normal_vectors[fr] = normals
        self.weights_list.append(weights)
        self.intercept_list.append(intercept)
        self.support_indices_list.append(support_indices)
        
    def _conclude(self):
        pass
    

if __name__ == '__main__':
    u = mda.Universe('membrane-cdl-1d.pdb')
    sel = u.select_atoms('not name W WF NA CL')
    test_class = MembraneCurvature(sel, method='numba')
    test_class.run()
    print(test_class.mean_curvature)

else:
    args = parser.parse_args()
    top = args.topology
    traj = args.trajectory
    backend = args.backend
    memb = args.membrane_selection
    forcefield = args.forcefield
    periodic = args.periodic
    gamma = args.gamma
    learning_rate = args.learning_rate
    max_iter = args.max_iter
    tolerance = args.tolerance
    labels = args.labels

    u = mda.Universe(top, traj)
    sel = u.select_atoms(memb)
    analysis = MembraneCurvature(sel, method=backend, forcefield=forcefield,
                                 periodic=periodic, gamma=gamma, learning_rate=learning_rate,
                                 max_iter=max_iter, tolerance=tolerance, train_labels=labels)