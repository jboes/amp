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
from amp import Amp
from amp.descriptor import Behler
from amp.regression import NeuralNetwork
from amp import SimulatedAnnealing

###############################################################################
# The test function for non-periodic systems


def non_periodic_0th_bfgs_step_test():

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
    # Correct values

    correct_cost = 7144.810783950215
    correct_energy_rmse = 24.318837496017185
    correct_force_rmse = 144.70282475062052
    correct_der_cost_fxn = [0, 0, 0, 0, 0, 0, 0.01374139170953901,
                            0.36318423812749656, 0.028312691567496464,
                            0.6012336354445753, 0.9659002689921986,
                            -1.2897770059416218, -0.5718960935176884,
                            -2.6425667221503035, -1.1960399246712894,
                            0, 0, -2.7256379713943852, -0.9080181026559658,
                            -0.7739948323247023, -0.2915789426043727,
                            -2.05998290443513, -0.6156374289747903,
                            -0.0060865174621348985, -0.8296785483640939,
                            0.0008092646748983969, 0.041613027034688874,
                            0.003426469079592851, -0.9578004568876517,
                            -0.006281929608090211, -0.28835884773094056,
                            -4.2457774110285245, -4.317412094174614,
                            -8.02385959091948, -3.240512651984099,
                            -27.289862194996896, -26.8177742762254,
                            -82.45107056053345, -80.6816768350809]

    ###########################################################################
    # Testing pure-python and fortran versions of behler-neural on different
    # number of processes

    for global_search in [None, 'SA']:
        for fortran in [False, True]:
            for extend_variables in [False, True]:
                for data_format in ['db', 'json']:
                    for save_memory in [False]:
                        for cores in range(1, 7):

                            string = 'CuOPdbp/0/%s-%s-%s-%s-%s-%i'
                            label = string % (global_search, fortran,
                                              extend_variables, data_format,
                                              save_memory, cores)

                            if global_search is 'SA':
                                gs = \
                                    SimulatedAnnealing(temperature=10, steps=5)
                            elif global_search is None:
                                gs = None

                            print label

                            calc = Amp(descriptor=Behler(cutoff=6.5, Gs=Gs),
                                       regression=NeuralNetwork(
                                       hiddenlayers=hiddenlayers,
                                       weights=weights,
                                       scalings=scalings,
                                       activation='sigmoid',),
                                       fortran=fortran,
                                       label=label)

                            calc.train(images=images, energy_goal=10.**10.,
                                       force_goal=10.**10.,
                                       force_coefficient=0.04,
                                       cores=cores, data_format=data_format,
                                       save_memory=save_memory,
                                       global_search=gs,
                                       extend_variables=extend_variables)

                            assert (abs(calc.cost_function - correct_cost) <
                                    10.**(-5.)), \
                                'The calculated value of cost function is \
                                wrong!'

                            assert (abs(calc.energy_per_atom_rmse -
                                        correct_energy_rmse) <
                                    10.**(-10.)), \
                                'The calculated value of energy per atom RMSE \
                                is wrong!'

                            assert (abs(calc.force_rmse - correct_force_rmse) <
                                    10 ** (-7)), \
                                'The calculated value of force RMSE is wrong!'

                            for _ in range(len(correct_der_cost_fxn)):

                                assert(abs(calc.der_variables_cost_function[
                                    _] - correct_der_cost_fxn[_]) <
                                    10 ** (-9)), \
                                    'The calculated value of cost function \
                                derivative is wrong!'

                            dblabel = label
                            secondlabel = '_' + label

                            calc = Amp(descriptor=Behler(cutoff=6.5, Gs=Gs),
                                       regression=NeuralNetwork(
                                       hiddenlayers=hiddenlayers,
                                       weights=weights,
                                       scalings=scalings,
                                       activation='sigmoid',),
                                       fortran=fortran,
                                       label=secondlabel,
                                       dblabel=dblabel)

                            calc.train(images=images, energy_goal=10.**10.,
                                       force_goal=10.**10.,
                                       force_coefficient=0.04,
                                       cores=cores, data_format=data_format,
                                       save_memory=save_memory,
                                       global_search=gs,
                                       extend_variables=extend_variables)

                            assert (abs(calc.cost_function - correct_cost) <
                                    10.**(-5.)), \
                                'The calculated value of cost function is \
                                wrong!'

                            assert (abs(calc.energy_per_atom_rmse -
                                        correct_energy_rmse) <
                                    10.**(-10.)), \
                                'The calculated value of energy per atom RMSE \
                                is wrong!'

                            assert (abs(calc.force_rmse - correct_force_rmse) <
                                    10 ** (-7)), \
                                'The calculated value of force RMSE is wrong!'

                            for _ in range(len(correct_der_cost_fxn)):
                                assert(abs(calc.der_variables_cost_function[
                                    _] - correct_der_cost_fxn[_] <
                                    10 ** (-9))), \
                                    'The calculated value of cost function \
                                derivative is wrong!'

###############################################################################
###############################################################################
# The test function for non-periodic systems and 9th BFGS step


def non_periodic_9th_bfgs_step_test():

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

    correct_der_cost_fxn = [0., 0., 0., 0., 0., 0.,
                            -0.000016668027931985884, 0.0006312278605825839,
                            -0.000054513573108413894, -0.0003697706450902254,
                            -0.0020247662551898414, -
                            6.646409948513071 * (10**(-13.)),
                            2.0369167527005373 * (10**(-11.)),
                            3.3145443845297296 * (10**(-12.)),
                            4.446850673511179 * (10**(-11.)),
                            0., 0., 1.5235244261175764 * (10**(-12.)),
                            3.7457947528268445 * (10**(-11.)),
                            2.734091806654473 * (10**(-10.)),
                            2.5119878918499415 * (10**(-10.)),
                            2.73652126446101 * (10**(-10.)),
                            -27.852754072224172, 52.78595799044361,
                            -39.342065306237814, 75.16260335806507,
                            -11.72679516566031, 152.2254667650986,
                            -44.38897098262355, 119.50221928188513,
                            -13.151893574695276, -91.94222209180421,
                            -76.70272451429612, -4.495113286980851,
                            -0.0012900396572669472, -8.879496843201633,
                            -4.949600623373575 * (10**(-11.)),
                            -25.44508693568062, -30.92363203662]

    ###########################################################################
    # Testing pure-python and fortran versions of behler-neural on different
    # number of processes

    for global_search in [None, 'SA']:
        for fortran in [False, True]:
            for extend_variables in [False, True]:
                for data_format in ['db', 'json']:
                    for save_memory in [False]:
                        for cores in range(1, 7):

                            string = 'CuOPdbp/1/%s-%s-%s-%s-%s-%i'
                            label = string % (global_search, fortran,
                                              extend_variables, data_format,
                                              save_memory, cores)

                            if global_search is 'SA':
                                gs = \
                                    SimulatedAnnealing(temperature=10, steps=5)
                            elif global_search is None:
                                gs = None

                            print label

                            calc = Amp(descriptor=Behler(cutoff=6.5, Gs=Gs,),
                                       regression=NeuralNetwork(
                                       hiddenlayers=hiddenlayers,
                                       weights=weights,
                                       scalings=scalings,
                                       activation='sigmoid',),
                                       fortran=fortran,
                                       label=label)

                            calc.train(images=images, energy_goal=14.115,
                                       force_goal=144.007,
                                       force_coefficient=0.04,
                                       cores=cores, data_format=data_format,
                                       save_memory=save_memory,
                                       global_search=gs,
                                       extend_variables=extend_variables)

                            assert (abs(calc.cost_function -
                                        5143.710215976742) <
                                    10.**(-10.)), \
                                'The calculated value of cost function is \
                                wrong!'

                            assert (abs(calc.energy_per_atom_rmse -
                                        14.11476828359897) < 10.**(-7.)), \
                                'The calculated value of energy per atom RMSE \
                            is wrong!'

                            assert (abs(calc.force_rmse - 144.00654147430743) <
                                    10 ** (-10.)), \
                                'The calculated value of force RMSE is wrong!'

                            for _ in range(len(correct_der_cost_fxn)):
                                assert(abs(calc.der_variables_cost_function[
                                    _] - correct_der_cost_fxn[_] <
                                    10 ** (-10))), \
                                    'The calculated value of cost function \
                                derivative is wrong!'

                            dblabel = label
                            secondlabel = '_' + label

                            calc = Amp(descriptor=Behler(cutoff=6.5, Gs=Gs,),
                                       regression=NeuralNetwork(
                                       hiddenlayers=hiddenlayers,
                                       weights=weights,
                                       scalings=scalings,
                                       activation='sigmoid',),
                                       fortran=fortran,
                                       label=secondlabel,
                                       dblabel=dblabel)

                            calc.train(images=images, energy_goal=14.115,
                                       force_goal=144.007,
                                       force_coefficient=0.04,
                                       cores=cores, data_format=data_format,
                                       save_memory=save_memory,
                                       global_search=gs,
                                       extend_variables=extend_variables)

                            assert (abs(calc.cost_function -
                                        5143.710215976742) <
                                    10.**(-10.)), \
                                'The calculated value of cost function is \
                                wrong!'

                            assert (abs(calc.energy_per_atom_rmse -
                                        14.11476828359897) < 10.**(-10.)), \
                                'The calculated value of energy per atom RMSE \
                            is wrong!'

                            assert (abs(calc.force_rmse - 144.00654147430743) <
                                    10 ** (-10.)), \
                                'The calculated value of force RMSE is wrong!'

                            for _ in range(len(correct_der_cost_fxn)):
                                assert(abs(calc.der_variables_cost_function[
                                    _] - correct_der_cost_fxn[_] <
                                    10 ** (-10))), \
                                    'The calculated value of cost function \
                                derivative is wrong!'

###############################################################################
###############################################################################
# The test function for periodic systems and first BFGS step


def periodic_0th_bfgs_step_test():

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
    # Correct values

    correct_cost = 8004.292841472513
    correct_energy_rmse = 43.736001940333836
    correct_force_rmse = 137.4099476110887
    correct_der_cost_fxn = [0.0814166874813534, 0.03231235582927526,
                            0.04388650395741291, 0.017417514465933048,
                            0.0284312765975806, 0.011283700608821421,
                            0.09416957265766414, -0.12322258890997816,
                            0.12679918754162384, 63.5396007548815,
                            0.016247700195771732, -86.62639558745185,
                            -0.017777528287386473, 86.22415217678898,
                            0.017745913074805372, 104.58358033260711,
                            -96.7328020983672, -99.09843648854351,
                            -8.302880631971407, -1.2590007162073242,
                            8.3028773468822, 1.258759884181224,
                            -8.302866610677315, -1.2563833805673688,
                            28.324298392677846, 28.09315509472324,
                            -29.378744559315365, -11.247473567051799,
                            11.119951466671642, -87.08582317485761,
                            -20.93948523898559, -125.73267675714658,
                            -35.13852440758523]

    ###########################################################################
    # Testing pure-python and fortran versions of behler-neural on different
    # number of processes

    for global_search in [None, 'SA']:
        for fortran in [False, True]:
            for extend_variables in [False, True]:
                for data_format in ['db', 'json']:
                    for save_memory in [False]:
                        for cores in range(1, 5):

                            string = 'CuOPdbp/2/%s-%s-%s-%s-%s-%i'
                            label = string % (global_search, fortran,
                                              extend_variables, data_format,
                                              save_memory, cores)

                            if global_search is 'SA':
                                gs = \
                                    SimulatedAnnealing(temperature=10, steps=5)
                            elif global_search is None:
                                gs = None

                            print label

                            calc = Amp(descriptor=Behler(cutoff=4., Gs=Gs),
                                       regression=NeuralNetwork(
                                       hiddenlayers=hiddenlayers,
                                       weights=weights,
                                       scalings=scalings,
                                       activation='tanh',),
                                       fortran=fortran,
                                       label=label)

                            calc.train(images=images, energy_goal=10.**10.,
                                       force_goal=10.**10,
                                       force_coefficient=0.04,
                                       cores=cores, data_format=data_format,
                                       save_memory=save_memory,
                                       global_search=gs,
                                       extend_variables=extend_variables)

                            assert (abs(calc.cost_function - correct_cost) <
                                    10.**(-7.)), \
                                'The calculated value of cost function is \
                                wrong!'

                            assert (abs(calc.energy_per_atom_rmse -
                                        correct_energy_rmse) < 10.**(-10.)), \
                                'The calculated value of energy per atom RMSE \
                            is wrong!'

                            assert (abs(calc.force_rmse - correct_force_rmse) <
                                    10 ** (-8.)), \
                                'The calculated value of force RMSE is wrong!'

                            for _ in range(len(correct_der_cost_fxn)):
                                assert(abs(calc.der_variables_cost_function[
                                    _] -
                                    correct_der_cost_fxn[_]) < 10 ** (-8)), \
                                    'The calculated value of cost function \
                                   derivative is wrong!'

                            dblabel = label
                            secondlabel = '_' + label

                            calc = Amp(descriptor=Behler(cutoff=4., Gs=Gs),
                                       regression=NeuralNetwork(
                                       hiddenlayers=hiddenlayers,
                                       weights=weights,
                                       scalings=scalings,
                                       activation='tanh',),
                                       fortran=fortran,
                                       label=secondlabel,
                                       dblabel=dblabel)

                            calc.train(images=images, energy_goal=10.**10.,
                                       force_goal=10.**10,
                                       force_coefficient=0.04,
                                       cores=cores, data_format=data_format,
                                       save_memory=save_memory,
                                       global_search=gs,
                                       extend_variables=extend_variables)

                            assert (abs(calc.cost_function - correct_cost) <
                                    10.**(-7.)), \
                                'The calculated value of cost function is \
                                wrong!'

                            assert (abs(calc.energy_per_atom_rmse -
                                        correct_energy_rmse) < 10.**(-10.)), \
                                'The calculated value of energy per atom RMSE \
                            is wrong!'

                            assert (abs(calc.force_rmse - correct_force_rmse) <
                                    10 ** (-8.)), \
                                'The calculated value of force RMSE is wrong!'

                            for _ in range(len(correct_der_cost_fxn)):
                                assert(abs(calc.der_variables_cost_function[
                                    _] -
                                    correct_der_cost_fxn[_] < 10 ** (-8))), \
                                    'The calculated value of cost function \
                                   derivative is wrong!'

###############################################################################
###############################################################################
# The test function for periodic systems and second BFGS step


def periodic_2nd_bfgs_step_test():

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

    ###########################################################################

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

    correct_der_cost_fxn = [-1.7151227666400152 * (10**(-12)), 0.,
                            -9.417116231764654 * (10**(-12)), 0.,
                            -5.989328635891508 * (10**(-13)),
                            0., 0.0902086289117197,
                            0.09020862891230962, -0.09020862891230962,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            - 0.047003416005720916, - 0.14591097884496312,
                            0.04764923175651392, 0.14791632618466555,
                            0.04725045347471331, 0.1466784166125388,
                            - 0.010540739367591485, - 0.0011278318421869301,
                            - 0.014722965566521402, 5.528632458949722,
                            - 5.519009931047882, - 9.664941966089417,
                            6.6115858724733965, - 11.191620012897427,
                            -11.19114902124169]

    ###########################################################################
    # Testing pure-python and fortran versions of behler-neural on different
    # number of processes

    for global_search in [None, 'SA']:
        for fortran in [False, True]:
            for extend_variables in [False, True]:
                for data_format in ['db', 'json']:
                    for save_memory in [False]:
                        for cores in range(1, 5):

                            string = 'CuOPdbp/3/%s-%s-%s-%s-%s-%i'
                            label = string % (global_search, fortran,
                                              extend_variables, data_format,
                                              save_memory, cores)

                            if global_search is 'SA':
                                gs = \
                                    SimulatedAnnealing(temperature=10, steps=5)
                            elif global_search is None:
                                gs = None

                            print label

                            calc = Amp(descriptor=Behler(cutoff=4., Gs=Gs,),
                                       regression=NeuralNetwork(
                                       hiddenlayers=hiddenlayers,
                                       weights=weights,
                                       scalings=scalings,
                                       activation='tanh',),
                                       fortran=fortran,
                                       label=label)

                            calc.train(images=images, energy_goal=5.14,
                                       force_goal=136.10,
                                       force_coefficient=0.04,
                                       cores=cores, data_format=data_format,
                                       save_memory=save_memory,
                                       global_search=gs,
                                       extend_variables=extend_variables)

                            assert (abs(calc.cost_function -
                                        2301.384892179139) <
                                    10.**(-10.)), \
                                'The calculated value of cost function is \
                                wrong!'

                            assert (abs(calc.energy_per_atom_rmse -
                                        5.13539878419966) <
                                    10.**(-10.)), \
                                'The calculated value of energy per atom RMSE \
                                is wrong!'

                            assert (abs(calc.force_rmse - 136.08416299484367) <
                                    10 ** (-10.)), \
                                'The calculated value of force RMSE is wrong!'

                            for _ in range(len(correct_der_cost_fxn)):
                                assert(abs(calc.der_variables_cost_function[
                                    _] - correct_der_cost_fxn[_] <
                                    10 ** (-10.))), \
                                    'The calculated value of cost function \
                            derivative is wrong!'

                            dblabel = label
                            secondlabel = '_' + label

                            calc = Amp(descriptor=Behler(cutoff=4., Gs=Gs,),
                                       regression=NeuralNetwork(
                                       hiddenlayers=hiddenlayers,
                                       weights=weights,
                                       scalings=scalings,
                                       activation='tanh',),
                                       fortran=fortran,
                                       label=secondlabel,
                                       dblabel=dblabel)

                            calc.train(images=images, energy_goal=5.14,
                                       force_goal=136.10,
                                       force_coefficient=0.04,
                                       cores=cores, data_format=data_format,
                                       save_memory=save_memory,
                                       global_search=gs,
                                       extend_variables=extend_variables)

                            assert (abs(calc.cost_function -
                                        2301.384892179139) <
                                    10.**(-10.)), \
                                'The calculated value of cost function is \
                                wrong!'

                            assert (abs(calc.energy_per_atom_rmse -
                                        5.13539878419966) <
                                    10.**(-10.)), \
                                'The calculated value of energy per atom RMSE \
                                is wrong!'

                            assert (abs(calc.force_rmse - 136.08416299484367) <
                                    10 ** (-10.)), \
                                'The calculated value of force RMSE is wrong!'

                            for _ in range(len(correct_der_cost_fxn)):
                                assert(abs(calc.der_variables_cost_function[
                                    _] - correct_der_cost_fxn[_] <
                                    10 ** (-10.))), \
                                    'The calculated value of cost function \
                            derivative is wrong!'

###############################################################################
###############################################################################

if __name__ == '__main__':
    non_periodic_0th_bfgs_step_test()
#    non_periodic_9th_bfgs_step_test()
    periodic_0th_bfgs_step_test()
#    periodic_2nd_bfgs_step_test()
