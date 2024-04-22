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
                 periodic: list(bool)=[True, True, False], gamma: float=.1):
        self.u = u
        self.memb = memb
        
        if forcefield == 'martini':
            head_sel = 'name GL0 PO4'
        else:
            head_sel = 'name P ??'
            
        self.head_selection = u.select_atoms(head_sel)
        self.periodic = periodic
        self.gamma = gamma
        
        # Switch for underlying methodology
        if method.lower() == 'jax':
            import jax_utils.SVMem_jax.SVMem as backend
        elif method.lower() == 'numpy':
            import numpy_utils.SVMem as backend
        else:
            import SVMem.SVMem as backend
            
        self.curve_computer = backend()
        
    def _prepare(self):
        pass

    def _single_frame(self):
        SVMem.compute()
        pass
    
    def _conclude(self):
        pass
    
if __name__ == '__main__':
    # testing happens here
    print('Hello')
    test_class = SVMem(u, sel, method='numba')