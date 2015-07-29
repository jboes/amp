"""
Exact behler-neural scheme forces and energies of five different non-periodic
configurations and three different periodic configurations have been calculated
in Mathematica, and are given below.  This script checks the values calculated
by the code with and without fortran modules.

"""

###############################################################################

import numpy as np
from ase import Atoms
from collections import OrderedDict
from amp import AMP
from amp.fingerprint import Behler
from amp.regression import NeuralNetwork

###############################################################################
# The test function for non-periodic systems


def non_periodic_test():

    ###########################################################################
    # Making the list of non-periodic images

    images = [Atoms(symbols='PdOPd2',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                            [0.,  0.,  1.]]),
                    positions=np.array(
                        [[0.,  0.,  0.],
                         [0.,  2.,  0.],
                            [0.,  0.,  3.],
                            [1.,  0.,  0.]])),
              Atoms(symbols='PdOPd2',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                            [0.,  0.,  1.]]),
                    positions=np.array(
                        [[0.,  1.,  0.],
                         [1.,  2.,  1.],
                            [-1.,  1.,  2.],
                            [1.,  3.,  2.]])),
              Atoms(symbols='PdO',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                         [0.,  0.,  1.]]),
                    positions=np.array(
                        [[2.,  1., -1.],
                         [1.,  2.,  1.]])),
              Atoms(symbols='Pd2O',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                         [0.,  0.,  1.]]),
                    positions=np.array(
                        [[-2., -1., -1.],
                         [1.,  2.,  1.],
                         [3.,  4.,  4.]])),
              Atoms(symbols='Cu',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                         [0.,  0.,  1.]]),
                    positions=np.array(
                        [[0.,  0.,  0.]]))]

    ###########################################################################
    # Correct energies and forces

    correct_predicted_energies = [14.268069305989567, 14.329747813650384,
                                  5.5742510565528285, 9.414582798669237,
                                  -0.5019297954597407]

    correct_predicted_forces = \
        [[[-0.028809124082570036, -0.043821143919242196, -0.05849702654749431],
          [-0.014500156234202243, 0.08790999055601932, -0.022632801252559023],
            [-0.010798945603058573, -0.015088534168372685,
                0.11352666460922906],
            [0.054108225919830846, -0.029000312468404486,
             -0.03239683680917572]],
            [[-0.012198225632368928, -0.029005619419009913,
              -0.048689506663307056],
             [0.038210519843916824,
                 0.009291267196124579, -0.013844533557767153],
             [-0.04814879678944527, -0.026738952790350826,
                 0.03125178762124301],
             [0.022136502577897378, 0.04645330501323616,
              0.031282252599831195]],
            [[0.10151747265164628, -0.10151747265164628, -0.20303494530329255],
             [-0.10151747265164628, 0.10151747265164628, 0.20303494530329255]],
            [[-0.0003106152026409805, -0.0003106152026409805,
              -0.00020974040523535917],
             [0.004815446319476237, 0.004815446319476237,
                 0.006970982485700309],
             [-0.004504831116835257, -0.004504831116835257,
              -0.00676124208046495]],
            [[0.0, 0.0, 0.0]]]

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
                  'gamma':0.3, 'zeta':4}],
          'Cu': [{'type': 'G2', 'element': 'Cu', 'eta': 0.8},
                 {'type': 'G4', 'elements': ['Cu', 'O'],
                  'eta':0.2, 'gamma':0.3, 'zeta':1},
                 {'type': 'G4', 'elements': ['Cu', 'Cu'], 'eta':0.3,
                  'gamma':0.6, 'zeta':0.5}]}

    hiddenlayers = {'O': (2), 'Pd': (2), 'Cu': (2)}

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
                                                              [3.0]]))])),
                           ('Cu', OrderedDict([(1, np.matrix([[0.0, 1.0],
                                                              [-1.0, -2.0],
                                                              [2.5, -1.9],
                                                              [-3.5, 0.5]])),
                                               (2, np.matrix([[0.5],
                                                              [1.6],
                                                              [-1.4]]))]))])

    scalings = OrderedDict([('O', OrderedDict([('intercept', -2.3),
                                               ('slope', 4.5)])),
                            ('Pd', OrderedDict([('intercept', 1.6),
                                                ('slope', 2.5)])),
                            ('Cu', OrderedDict([('intercept', -0.3),
                                                ('slope', -0.5)]))])

    fingerprints_range = {"Cu": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                          "O": [[0.21396177208585404, 2.258090276328769],
                                [0.0, 2.1579067008202975],
                                [0.0, 0.0]],
                          "Pd": [[0.0, 1.4751761770313006],
                                 [0.0, 0.697686078889583],
                                 [0.0, 0.37848964715610417]]}

    ###########################################################################
    # Testing pure-python and fortran versions of behler-neural force call

    for fortran in [False, True]:

        calc = AMP(fingerprint=Behler(cutoff=6.5, Gs=Gs,),
                   regression=NeuralNetwork(hiddenlayers=hiddenlayers,
                                            weights=weights,
                                            scalings=scalings,
                                            activation='sigmoid',),
                   fingerprints_range=fingerprints_range,
                   fortran=fortran)

        predicted_energies = [calc.get_potential_energy(image) for image in
                              images]

        for image_no in range(len(predicted_energies)):
            assert (abs(predicted_energies[image_no] -
                        correct_predicted_energies[image_no]) < 10.**(-10.)), \
                'The predicted energy of image %i is wrong!' % (image_no + 1)

        predicted_forces = [calc.get_forces(image) for image in images]

        for image_no in range(len(predicted_forces)):
            for index in range(np.shape(predicted_forces[image_no])[0]):
                for direction in range(
                        np.shape(predicted_forces[image_no])[1]):
                    assert (abs(predicted_forces[image_no][index][direction] -
                                correct_predicted_forces[image_no][index]
                                [direction]) < 10.**(-10.)), \
                        'The predicted %i force of atom %i of image %i is' \
                        'wrong!' % (direction, index, image_no + 1)


###############################################################################
###############################################################################
# The test function for periodic systems


def periodic_test():

    ###########################################################################
    # Making the list of periodic images

    images = [Atoms(symbols='PdOPd',
                    pbc=np.array([True, False, False], dtype=bool),
                    cell=np.array(
                        [[2.,  0.,  0.],
                         [0.,  2.,  0.],
                         [0.,  0.,  2.]]),
                    positions=np.array(
                        [[0.5,  1., 0.5],
                         [1.,  0.5,  1.],
                         [1.5,  1.5,  1.5]])),
              Atoms(symbols='PdO',
                    pbc=np.array([True, True, False], dtype=bool),
                    cell=np.array(
                        [[2.,  0.,  0.],
                         [0.,  2.,  0.],
                            [0.,  0.,  2.]]),
                    positions=np.array(
                        [[0.5,  1., 0.5],
                         [1.,  0.5,  1.]])),
              Atoms(symbols='Cu',
                    pbc=np.array([True, True, False], dtype=bool),
                    cell=np.array(
                        [[1.8,  0.,  0.],
                         [0.,  1.8,  0.],
                            [0.,  0.,  1.8]]),
                    positions=np.array(
                        [[0.,  0., 0.]]))]

    ###########################################################################
    # Correct energies and forces

    correct_predicted_energies = [3.856095432699593, 1.6120748520627295,
                                  0.1990889488055934]

    correct_predicted_forces = \
        [[[0.14422386766021048, -3.3861828000174965, 3.385677307965187],
          [0.03842093539765022, 9.256953780593706, -0.45054456370323664],
            [-0.18264480305786068, -5.87077098057621, -2.93513274426195]],
            [[6.539292540693543 * (10**(-6)), -6.539292540693531 * (10**(-6)),
              0.00011301227788167808],
             [-6.539292540693543 * (10**(-6)), 6.539292540693531 * (10**(-6)),
              -0.00011301227788167808]],
            [[0.0, 0.0, 0.0]]]

    ###########################################################################
    # Parameters

    Gs = {'O': [{'type': 'G2', 'element': 'Pd', 'eta': 0.8},
                {'type': 'G4', 'elements': ['O', 'Pd'], 'eta':0.3, 'gamma':0.6,
                 'zeta':0.5}],
          'Pd': [{'type': 'G2', 'element': 'Pd', 'eta': 0.2},
                 {'type': 'G4', 'elements': ['Pd', 'Pd'],
                  'eta':0.9, 'gamma':0.75, 'zeta':1.5}],
          'Cu': [{'type': 'G2', 'element': 'Cu', 'eta': 0.8},
                 {'type': 'G4', 'elements': ['Cu', 'Cu'], 'eta':0.3,
                          'gamma':0.6, 'zeta':0.5}]}

    hiddenlayers = {'O': (2), 'Pd': (2), 'Cu': (2)}

    weights = OrderedDict([('O', OrderedDict([(1, np.matrix([[-2.0, 6.0],
                                                             [3.0, -3.0],
                                                             [1.5, -0.9]])),
                                              (2, np.matrix([[5.5],
                                                             [3.6],
                                                             [1.4]]))])),
                           ('Pd', OrderedDict([(1, np.matrix([[-1.0, 3.0],
                                                              [2.0, 4.2],
                                                              [1.0, -0.7]])),
                                               (2, np.matrix([[4.0],
                                                              [0.5],
                                                              [3.0]]))])),
                           ('Cu', OrderedDict([(1, np.matrix([[0.0, 1.0],
                                                              [-1.0, -2.0],
                                                              [2.5, -1.9]])),
                                               (2, np.matrix([[0.5],
                                                              [1.6],
                                                              [-1.4]]))]))])

    scalings = OrderedDict([('O', OrderedDict([('intercept', -2.3),
                                               ('slope', 4.5)])),
                            ('Pd', OrderedDict([('intercept', 1.6),
                                                ('slope', 2.5)])),
                            ('Cu', OrderedDict([('intercept', -0.3),
                                                ('slope', -0.5)]))])

    fingerprints_range = {"Cu": [[2.8636310860653253, 2.8636310860653253],
                                 [15.723158310819452, 15.723158310819452]],
                          "O": [[2.9409056366723028,
                                 2.9724949026043914],
                                [7.96834050689743,
                                 29.772479983967003]],
                          "Pd": [[2.4629488092411096,
                                  2.616013877408712],
                                 [2.3445785843644544,
                                  2.5119093855123076]]}

    ###########################################################################
    # Testing pure-python and fortran versions of behler-neural force call

    for fortran in [False, True]:

        calc = AMP(fingerprint=Behler(cutoff=4., Gs=Gs,),
                   regression=NeuralNetwork(hiddenlayers=hiddenlayers,
                                            weights=weights,
                                            scalings=scalings,
                                            activation='tanh',),
                   fingerprints_range=fingerprints_range,
                   fortran=fortran)

        predicted_energies = [calc.get_potential_energy(image) for image in
                              images]

        for image_no in range(len(predicted_energies)):
            assert (abs(predicted_energies[image_no] -
                        correct_predicted_energies[image_no]) < 10.**(-10.)), \
                'The predicted energy of image %i is wrong!' % (image_no + 1)

        predicted_forces = [calc.get_forces(image) for image in images]

        for image_no in range(len(predicted_forces)):
            for index in range(np.shape(predicted_forces[image_no])[0]):
                for direction in range(
                        np.shape(predicted_forces[image_no])[1]):
                    assert (abs(predicted_forces[image_no][index][direction] -
                                correct_predicted_forces[image_no][index]
                                [direction]) < 10.**(-10.)), \
                        'The predicted %i force of atom %i of image %i is' \
                        'wrong!' % (direction, index, image_no + 1)

    ###########################################################################

if __name__ == '__main__':
    non_periodic_test()
    periodic_test()
