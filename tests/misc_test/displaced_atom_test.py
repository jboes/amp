"""
An atom in an Atoms object is displaced a little bit and the potential energy
both before and after displacement is calculated. The result should be
different.

"""

###############################################################################

import numpy as np
from ase import Atoms
from collections import OrderedDict
from amp import Amp
from amp.descriptor import Behler
from amp.regression import NeuralNetwork

###############################################################################


def test():

    ###########################################################################
    # Parameters

    atoms = Atoms(symbols='PdOPd2',
                  pbc=np.array([False, False, False], dtype=bool),
                  cell=np.array(
                      [[1.,  0.,  0.],
                       [0.,  1.,  0.],
                          [0.,  0.,  1.]]),
                  positions=np.array(
                      [[0.,  1.,  0.],
                       [1.,  2.,  1.],
                          [-1.,  1.,  2.],
                          [1.,  3.,  2.]]))

    ###########################################################################
    # Parameters

    Gs = {'O': [{'type': 'G2', 'element': 'Pd', 'eta': 0.8},
                {'type': 'G4', 'elements': [
                    'Pd', 'Pd'], 'eta':0.2, 'gamma':0.3, 'zeta':1},
                {'type': 'G4', 'elements': ['O', 'Pd'], 'eta':0.3, 'gamma':0.6,
                 'zeta':0.5}],
          'Pd': [{'type': 'G2', 'element': 'Pd', 'eta': 0.2},
                 {'type': 'G4', 'elements': ['Pd', 'Pd'],
                  'eta':0.9, 'gamma':0.75, 'zeta':1.5},
                 {'type': 'G4', 'elements': ['O', 'Pd'], 'eta':0.4,
                  'gamma':0.3, 'zeta':4}]}

    hiddenlayers = {'O': (2), 'Pd': (2)}

    weights = OrderedDict([('O', OrderedDict([(1, np.matrix([[-2.0, 6.0],
                                                             [3.0, -3.0],
                                                             [1.5, -0.9],
                                                             [-2.5, -1.5]])),
                                              (2, np.matrix([[5.5],
                                                             [3.6],
                                                             [1.4]]))])),
                           ('Pd', OrderedDict([(1, np.matrix([[-1.0, 3.0],
                                                              [2.0, 4.2],
                                                              [1.0, -0.7],
                                                              [-3.0, 2.0]])),
                                               (2, np.matrix([[4.0],
                                                              [0.5],
                                                              [3.0]]))]))])

    scalings = OrderedDict([('O', OrderedDict([('intercept', -2.3),
                                               ('slope', 4.5)])),
                            ('Pd', OrderedDict([('intercept', 1.6),
                                                ('slope', 2.5)]))])

    fingerprints_range = {"O": [[0.21396177208585404, 2.258090276328769],
                                [0.0, 2.1579067008202975],
                                [0.0, 0.0]],
                          "Pd": [[0.0, 1.4751761770313006],
                                 [0.0, 0.697686078889583],
                                 [0.0, 0.37848964715610417]]}

    ###########################################################################

    calc = Amp(descriptor=Behler(cutoff=6.5, Gs=Gs,),
               regression=NeuralNetwork(hiddenlayers=hiddenlayers,
                                        weights=weights,
                                        scalings=scalings,),
               fingerprints_range=fingerprints_range,)

    atoms.set_calculator(calc)

    e1 = atoms.get_potential_energy(apply_constraint=False)
    e2 = calc.get_potential_energy(atoms)
    f1 = atoms.get_forces(apply_constraint=False)

    atoms[0].x += 0.5

    boolean = atoms.calc.calculation_required(atoms, properties=['energy'])

    e3 = atoms.get_potential_energy(apply_constraint=False)
    e4 = calc.get_potential_energy(atoms)
    f2 = atoms.get_forces(apply_constraint=False)

    assert (e1 == e2 and
            e3 == e4 and
            abs(e1 - e3) > 10. ** (-3.) and
            (boolean is True) and
            (not (f1 == f2).all())), 'Displaced-atom test broken!'

###############################################################################

if __name__ == '__main__':
    test()
