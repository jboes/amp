"""
Exact behler-neural scheme cost function, energy per atom RMSE and force RMSE
for five different non-periodic configurations and three three different
periodic configurations have been calculated in Mathematica. This script
checks the values calculated by the code during training with and without
fortran modules and also on different number of cores.

"""

###############################################################################

import numpy as np
from collections import OrderedDict
from ase import Atoms
from ase.calculators.emt import EMT
from amp import AMP
from amp.fingerprint import Behler
from amp.regression import NeuralNetwork

###############################################################################
# The test function for non-periodic systems


def non_periodic_test():

    ###########################################################################
    # Making the list of periodic image

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

    for image in images:
        image.set_calculator(EMT())
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)

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

    ###########################################################################
    # Derivative of the cost function values

    correct_der_cost_fxn = [0, 0, 0, 0, 0, 0, 0.01374139170953901,
                            0.36318423812749656, 0.028312691567496464,
                            0.6012336354445753, 0.9659002689921986,
                            -1.0530349272974702, -0.4303494072486586,
                            -2.320514821178775, -0.9180380156693098, 0, 0,
                            -2.2461981230126393, -0.6426031874148609,
                            -0.7055576174627515, -0.2131694024409814,
                            -1.710609728440938, -0.5164830567828097,
                            -0.008707182451020031, -0.6203842200394035,
                            0.0013149358808439494, -0.23428021728592624,
                            0.003194688530605789, -0.782859656115562,
                            -0.009337005913608098, -0.25906917355908676,
                            -4.087393428925091, -4.1652787777631675,
                            -8.02385959091948, -3.240512651984099,
                            -27.284932543550706, -26.892401706074804,
                            -82.43628495875033, -80.72759582717585]

    ###########################################################################
    # Testing pure-python and fortran versions of behler-neural on different number
    # of processes

    for fortran in [False, True]:
        for cores in range(1, 7):

            label = 'NonperiodFortran%s-%i' % (fortran, cores)

            calc = AMP(fingerprint=Behler(cutoff=6.5, Gs=Gs,),
                       regression=NeuralNetwork(hiddenlayers=hiddenlayers,
                                                weights=weights,
                                                scalings=scalings,
                                                activation='sigmoid',),
                       fortran=fortran,
                       label=label)

            calc.train(images=images, energy_goal=10.**10.,
                       force_goal=10.**10., force_coefficient=0.04,
                       cores=cores, read_fingerprints=False)

            assert (abs(calc.cost_function - 7144.30292363230) <
                    10.**(-5.)), \
                'The calculated value of cost function is wrong!'

            assert (abs(calc.energy_per_atom_rmse - 24.31472406476930) <
                    10.**(-7.)), \
                'The calculated value of energy per atom RMSE is wrong!'

            assert (abs(calc.force_rmse - 144.7113314827651) <
                    10 ** (-7)), \
                'The calculated value of force RMSE is wrong!'

            for _ in range(len(correct_der_cost_fxn)):
                assert(abs(calc.der_variables_cost_function[_] -
                           correct_der_cost_fxn[_] < 10 ** (-10))), \
                    'The calculated value of cost function derivative is \
                    wrong!'

###############################################################################
###############################################################################
# The test function for periodic systems


def periodic_test():

    ###########################################################################
    # Making the list of images

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

    for image in images:
        image.set_calculator(EMT())
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)

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

    ###########################################################################
    # Derivative of the cost function values

    correct_der_cost_fxn = [3.8310870400843917 * (10 ** (-13)),
                            1.2503659239313091 * (10 ** (-27)),
                            2.103510760408106 * (10 ** (-12)),
                            6.865305193847003 * (10 ** (-27)),
                            1.3378423843513887 * (10 ** (-13)),
                            4.366365241722989 * (10 ** (-28)),
                            -0.02045805446811484, -0.02045805446824862,
                            0.02045805446824862, 65.64757374227074,
                            0.016395651946408484, -88.40824381641114,
                            -0.01790386763283238, 88.34713720442579,
                            0.017894801448348468, 103.81964389461949,
                            -95.77552903208176, -98.14122995271147,
                            -8.302900534252412, -1.2604369815702425,
                            8.30289453040453, 1.2599968355841735,
                            -8.302846709076462, -1.2549471650455433,
                            28.325980890513435, 28.092259638399817,
                            -29.377059572489596, -11.237957825468813,
                            11.217481115669644, -87.08582317485761,
                            -20.705307849287813, -125.73267675714658,
                            -35.138861405305406]

    ###########################################################################
    # Testing pure-python and fortran versions of behler-neural on different
    # number of processes

    for fortran in [False, True]:
        for cores in range(1, 5):

            label = 'PeriodFortran%s-%i' % (fortran, cores)

            calc = AMP(fingerprint=Behler(cutoff=4., Gs=Gs,),
                       regression=NeuralNetwork(hiddenlayers=hiddenlayers,
                                                weights=weights,
                                                scalings=scalings,
                                                activation='tanh',),
                       fortran=fortran,
                       label=label)

            calc.train(images=images, energy_goal=10.**10.,
                       force_goal=10.**10, force_coefficient=0.04,
                       cores=cores, read_fingerprints=False)

            assert (abs(calc.cost_function - 8005.262570965399) <
                    10.**(-7.)), \
                'The calculated value of cost function is wrong!'

            assert (abs(calc.energy_per_atom_rmse - 43.73579809791985) <
                    10.**(-8.)), \
                'The calculated value of energy per atom RMSE is wrong!'

            assert (abs(calc.force_rmse - 137.44097112273843) <
                    10 ** (-8.)), \
                'The calculated value of force RMSE is wrong!'

            for _ in range(len(correct_der_cost_fxn)):
                assert(abs(calc.der_variables_cost_function[_] -
                           correct_der_cost_fxn[_] < 10 ** (-8))), \
                    'The calculated value of cost function derivative is \
                    wrong!'

    ###########################################################################

if __name__ == '__main__':
    non_periodic_test()
    periodic_test()
