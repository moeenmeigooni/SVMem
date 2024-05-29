import argparse
import MDAnalysis as mda
from SVMem_MDA import MembraneCurvature

parser = argparse.ArgumentParser(description='')

parser.add_argument('topology', help='Topology file for trajectory (e.g. psf, parm7, pdb)')
parser.add_argument('trajectory', default=None, 
                    help='Trajectory file or list of trajectory files (e.g. dcd, xtc)')
parser.add_argument('-b', '--backend', default='numba', choices=['jax', 'numba'],
                    help='Backend for curvature calculations.')
parser.add_argument('-m', '--membrane', default='segid MEMB', help='Atom selection text\
    that adheres to the MDAnalysis conventions for selecting your whole membrane.')
parser.add_argument('-ff', '--forcefield', choices=['martini', 'charmm', 'amber'], 
                    help='Forcefield used for simulation')
parser.add_argument('-p', '--periodic', default=[True, True, False],
                    help='Boolean list of shape (3,1) which describes the dimensions \
                        to consider periodicity in. Should be [True, True, False]')
parser.add_argument('-g', '--gamma', default=0.1, help='Value for gamma hyperparameter.')
parser.add_argument('-lr', '--learning_rate', default=0.01, 
                    help='Learning rate for SVM training.')
parser.add_argument('-mi', '--max_iter', default=500, 
                    help='Maximum number of iterations of training.')
parser.add_argument('-t', '--tolerance', default=0.001, help='Tolerance.')
parser.add_argument('-l', '--labels', default=None, help='Precomputed training labels.')

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

if isinstance(traj, None):
    u = mda.Universe(top)
else:
    u = mda.Universe(top, traj)
    
sel = u.select_atoms(memb)
analysis = MembraneCurvature(sel, method=backend, forcefield=forcefield,
                             periodic=periodic, gamma=gamma, learning_rate=learning_rate,
                             max_iter=max_iter, tolerance=tolerance, train_labels=labels)