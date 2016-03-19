"""
Exact Gaussian-neural scheme forces and energies of five different non-periodic
configurations and three different periodic configurations have been calculated
in Mathematica, and are given below.  This script checks the values calculated
by the code with and without fortran modules.

"""

###############################################################################

import numpy as np
from ase import Atoms
from collections import OrderedDict
from amp import Amp
from amp.descriptor import Gaussian
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

    correct_predicted_energies = [14.231186811226152, 14.327219917287948,
                                  5.5742510565528285, 9.41456771216968,
                                  -0.5019297954597407]

    correct_predicted_forces = \
        [[[-0.05095024246182649, -0.10709193432146558, -0.09734321482638622],
          [-0.044550772904033635, 0.2469763195486647, -0.07617425912869778],
            [-0.02352490951707703, -0.050782839419131864, 0.24409220250631508],
            [0.11902592488293715, -0.08910154580806727, -0.07057472855123109]],
            [[-0.024868720575099375, -0.07417891957113862,
              -0.12121240797223251],
             [0.060156158438252574, 0.017517013378773042,
              -0.020047135079325505],
             [-0.10901144291312388, -0.06671262448352767, 0.06581556263014315],
             [0.07372400504997068, 0.12337453067589325, 0.07544398042141486]],
            [[0.10151747265164626, -0.10151747265164626, -0.20303494530329252],
             [-0.10151747265164626, 0.10151747265164626, 0.20303494530329252]],
            [[-0.00031177673224312745, -0.00031177673224312745,
              -0.0002078511548287517],
             [0.004823209772264884, 0.004823209772264884,
              0.006975000714861393],
             [-0.004511433040021756, -0.004511433040021756,
              -0.006767149560032641]],
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
                          "O": [[0.2139617720858539, 2.258090276328769],
                                [0.0, 1.085656080548734],
                                [0.0, 0.0]],
                          "Pd": [[0.0, 1.4751761770313006],
                                 [0.0, 0.28464992134267897],
                                 [0.0, 0.20167521020630502]]}

    ###########################################################################
    # Testing pure-python and fortran versions of Gaussian-neural force call

    for fortran in [False, True]:

        calc = Amp(descriptor=Gaussian(cutoff=6.5, Gs=Gs),
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

    correct_predicted_energies = [3.8560954326995978, 1.6120748520627273,
                                  0.19433107801410093]

    correct_predicted_forces = \
        [[[0.14747720528015523, -3.3010645563584973, 3.3008168318984463],
          [0.03333579762326405, 9.050780376599887, -0.42608278400777605],
            [-0.1808130029034193, -5.7497158202413905, -2.8747340478906698]],
            [[6.5035267996045045 * (10.**(-6.)), -6.503526799604495 * (10.**(-6.)),
              0.00010834689201069249],
             [-6.5035267996045045 * (10.**(-6.)), 6.503526799604495 * (10.**(-6.)),
              -0.00010834689201069249]],
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
                                 [1.5435994865298275, 1.5435994865298275]],
                          "O": [[2.9409056366723028, 2.972494902604392],
                                [1.9522542722823606, 4.0720361595017245]],
                          "Pd": [[2.4629488092411096, 2.6160138774087125],
                                 [0.27127576524253594, 0.5898312261433813]]}

    ###########################################################################
    # Testing pure-python and fortran versions of Gaussian-neural force call

    for fortran in [False, True]:

        calc = Amp(descriptor=Gaussian(cutoff=4., Gs=Gs),
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
