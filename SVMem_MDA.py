import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
import numpy as np

class SVMem(AnalysisBase):
    """
    Analysis class that computes membrane curvature for a given simulation. 
    Must provide an MDAnalysis universe object, membrane atom selection and
    a methodology for computing the curvature. Options include: JAX or
    simply numba.
    ------
    Inputs
    ------
    
    """
    def __init__(self, u: mda.Universe, memb: mda.AtomGroup,
                 method: str='numba', forcefield: str='martini',
                 periodic: list(bool)=[True, True, False], gamma: float=.1,
                 learning_rate: float=0.01, max_iter: int=500,
                 tolerance: float=0.0001, train_labels: str='auto'):
        self.u = u
        self.memb = memb
        self.periodic = periodic
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        if train_labels != 'auto':
            self.train_labels = train_labels
            self.autogenerate_labels = False
        else:
            self.train_labels = None
            self.autogenerate_labels = True
        
        # Defined headgroup atom selections by forcefield
        if forcefield == 'martini':
            head_sel = 'name GL0 PO4'
        else:
            head_sel = 'name ? P'
            
        self.train_points = u.select_atoms(head_sel)
        self.n_train_points = self.train_points.n_atoms
        self.train_labels = None
        self.weights_list = None
        self.intercept_list = None
        self.support_indices_list = None
        
        # Switch for underlying methodology
        if method.lower() == 'jax':
            from jax_utils.SVMem_jax import Backend as backend
        elif method.lower() == 'numpy':
            from numpy_utils.SVMem_numpy import Backend as backend
        else:
            from numba_utils.SVMem_numba import Backend as backend
            
        self.backend = backend()
        
    def _prepare(self):
        pass

    def _single_frame(self):
        self.backend.compute_curvature()
    
    def _conclude(self):
        pass
    
if __name__ == '__main__':
    # testing happens here
    print('Hello')
    test_class = SVMem(u, sel, method='numba')