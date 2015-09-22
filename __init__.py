""" This package contains different machine-learning calculators.

Developed by Andrew Peterson and Alireza Khorshidi (July 2015)
School of Engineering, Brown University, Providence, RI, USA, 02912
Andrew_Peterson@brown.edu
Alireza_Khorshidi@brown.edu

See the accompanying license file for details.
"""
###############################################################################

from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.parallel import paropen
import os
from ase import io
import numpy as np
import json
import tempfile
import warnings
from datetime import datetime
import multiprocessing as mp
import gc
from collections import OrderedDict
from scipy.optimize import fmin_bfgs as optimizer
from ase.calculators.neighborlist import NeighborList
from utilities import *
from descriptor import *
from regression import *
try:
    from amp import fmodules  # version 3 of fmodules
    fmodules_version = 3
except ImportError:
    fmodules = None

###############################################################################


class SimulatedAnnealing:

    """
    Class that implements simulated annealing algorithm for global search of
    variables. This algorithm is helpful to be used for pre-conditioning of the
    initial guess of variables for optimization of non-convex functions.

    :param initial_temp: Initial temperature which corresponds to initial
                         variance of the likelihood normal probability
                         distribution. Should take a value from 50 to 100.
    :type initial_temp: float
    :param steps: Number of search iterations.
    :type steps: int
    """
    ###########################################################################

    def __init__(self, initial_temp, steps):
        self.initial_temp = initial_temp
        self.steps = steps

    ###########################################################################

    def initialize(self, variables, log, costfxn):
        """
        Function to initialize this class with.

        :param variables: Calibrating variables.
        :type variables: list
        :param log: Write function at which to log data. Note this must be a
                    callable function.
        :type log: Logger object
        :param costfxn: Object of the CostFxnandDer class.
        :type costfxn: object
        """
        self.variables = variables
        self.log = log
        self.costfxn = costfxn
        self.log.tic('simulated_annealing')
        self.log('Simulated annealing started. ')

    ###########################################################################

    def get_variables(self,):
        """
        Function that samples from the space of variables according to
        simulated annealing algorithm.

        :returns: Best variables minimizing the cost function.
        """
        head1 = ('%4s %6s %6s %6s %6s %6s %6s %8s')
        self.log(head1 % ('step',
                          'temp',
                          'newcost',
                          'newlogp',
                          'oldlogp',
                          'pratio',
                          'rand',
                          'accpt?(%)'))
        self.log(head1 % ('=' * 4,
                          '=' * 6,
                          '=' * 6,
                          '=' * 6,
                          '=' * 6,
                          '=' * 6,
                          '=' * 6,
                          '=' * 8))
        variables = self.variables
        temp = self.initial_temp
        allvariables = [variables]

        calculate_gradient = False
        self.costfxn.param.regression._variables = variables

        if self.costfxn.fortran:
            task_args = (self.costfxn.param, calculate_gradient)
            (energy_square_error, force_square_error, _) = \
                self.costfxn._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_fortran,
                _args=task_args, len_of_variables=len(variables))
        else:
            task_args = (self.costfxn.reg, self.costfxn.param,
                         self.costfxn.sfp, self.costfxn.snl,
                         self.costfxn.energy_coefficient,
                         self.costfxn.force_coefficient,
                         self.costfxn.train_forces, len(variables),
                         calculate_gradient)
            (energy_square_error, force_square_error, _) = \
                self.costfxn._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_python,
                _args=task_args, len_of_variables=len(variables))

        square_error = \
            self.costfxn.energy_coefficient * energy_square_error + \
            self.costfxn.force_coefficient * force_square_error

        besterror = square_error
        bestvariables = variables

        logp = -square_error / temp

        accepted = 0

        for step in range(self.steps):
            _steps = np.random.rand(len(variables)) * 2. - 1.
            _steps *= 0.2
            newvariables = variables + _steps

            calculate_gradient = False
            self.costfxn.param.regression._variables = newvariables

            if self.costfxn.fortran:
                task_args = (self.costfxn.param, calculate_gradient)
                (energy_square_error, force_square_error, _) = \
                    self.costfxn._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_fortran,
                    _args=task_args, len_of_variables=len(variables))
            else:
                task_args = (self.costfxn.reg, self.costfxn.param,
                             self.costfxn.sfp, self.costfxn.snl,
                             self.costfxn.energy_coefficient,
                             self.costfxn.force_coefficient,
                             self.costfxn.train_forces, len(variables),
                             calculate_gradient)
                (energy_square_error, force_square_error, _) = \
                    self.costfxn._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_python,
                    _args=task_args, len_of_variables=len(variables))

            new_square_error = \
                self.costfxn.energy_coefficient * energy_square_error + \
                self.costfxn.force_coefficient * force_square_error

            newlogp = -new_square_error / temp

            pratio = np.exp(newlogp - logp)
            rand = np.random.rand()
            if rand < pratio:
                accept = True
                accepted += 1.
            else:
                accept = False
            line = ('%5s' ' %5.2f' ' %5.2f' ' %5.2f' ' %5.2f' ' %3.2f'
                    ' %3.2f' ' %5s')
            self.log(line % (step,
                             temp,
                             new_square_error,
                             newlogp,
                             logp,
                             pratio,
                             rand,
                             '%s(%i)' % (accept,
                                         int(accepted * 100 / (step + 1)))))
            if new_square_error < besterror:
                bestvariables = newvariables
                besterror = new_square_error
            if accept:
                variables = newvariables
                allvariables.append(newvariables)
                logp = newlogp

            if (accepted / (step + 1) < 0.5):
                temp += 0.0005 * temp
            else:
                temp -= 0.002 * temp

        self.log('Simulated annealing exited. ', toc='simulated_annealing')
        self.log('\n')

        return bestvariables

###############################################################################
###############################################################################
###############################################################################


class Amp(Calculator):

    """
    Atomistic Machine-Learning Potential (Amp) ASE calculator

    :param descriptor: Class representing local atomic environment. Can be
                        only None and Behler for now. Input arguments for
                        Behler are cutoff and Gs; for more information see
                        docstring for the class Behler.
    :type descriptor: object
    :param regression: Class representing the regression method. Can be only
                       NeuralNetwork for now. Input arguments for NeuralNetwork
                       are hiddenlayers, activation, weights, and scalings; for
                       more information see docstring for the class
                       NeuralNetwork.
    :type regression: object
    :param fingerprints_range: Range of fingerprints of each chemical species.
                               Should be fed as a dictionary of chemical
                               species and a list of minimum and maximun, e.g:

                               >>> fingerprints_range={"Pd": [0.31, 0.59], "O":[0.56, 0.72]}

    :type fingerprints_range: dict
    :param load: Path for loading an existing parameters of Amp calculator.
    :type load: str
    :param label: Default prefix/location used for all files.
    :type label: str
    :param dblabel: Optional separate prefix/location for database files,
                    including fingerprints, fingerprint derivatives, and
                    neighborlists. This file location can be shared between
                    calculator instances to avoid re-calculating redundant
                    information. If not supplied, just uses the value from
                    label.
    :type dblabel: str
    :param extrapolate: If True, allows for extrapolation, if False, does not
                        allow.
    :type extrapolate: bool
    :param fortran: If True, will use fortran modules, if False, will not.
    :type fortran: bool

    :raises: RuntimeError
    """
    implemented_properties = ['energy', 'forces']

    default_parameters = {
        'descriptor': Behler(),
        'regression': NeuralNetwork(),
        'fingerprints_range': None,
    }

    ###########################################################################

    def __init__(self, load=None, label=None, dblabel=None, extrapolate=True,
                 fortran=True, **kwargs):

        self.extrapolate = extrapolate
        self.fortran = fortran
        self.dblabel = dblabel
        if not dblabel:
            self.dblabel = label

        if self.fortran and not fmodules:
            raise RuntimeError('Not using fortran modules. '
                               'Either compile fmodules as described in the '
                               'README for improved performance, or '
                               'initialize calculator with fortran=False.')

        if self.fortran and fmodules:
            wrong_version = fmodules.check_version(version=fmodules_version)
            if wrong_version:
                raise RuntimeError('Fortran part is not updated. Recompile'
                                   'with f2py as described in the README. '
                                   'Correct version is %i.'
                                   % fmodules_version)

        # Reading parameters from existing file if any:
        if load:
            try:
                json_file = paropen(load, 'rb')
            except IOError:
                json_file = paropen(make_filename(load,
                                                  'trained-parameters.json'),
                                    'rb')

            parameters = load_parameters(json_file)

            kwargs = {}
            kwargs['fingerprints_range'] = parameters['fingerprints_range']
            if parameters['descriptor'] == 'Behler':
                kwargs['descriptor'] = \
                    Behler(cutoff=parameters['cutoff'],
                           Gs=parameters['Gs'],
                           fingerprints_tag=parameters['fingerprints_tag'],
                           fortran=fortran,)
            elif parameters['descriptor'] == 'None':
                kwargs['descriptor'] = None
                if parameters['no_of_atoms'] == 'None':
                    parameters['no_of_atoms'] = None
            else:
                raise RuntimeError('Descriptor is not recognized to Amp. '
                                   'User should add the descriptor under '
                                   'consideration.')

            if parameters['regression'] == 'NeuralNetwork':
                kwargs['regression'] = \
                    NeuralNetwork(hiddenlayers=parameters['hiddenlayers'],
                                  activation=parameters['activation'],
                                  variables=parameters['variables'],)
                if kwargs['descriptor'] is None:
                    kwargs['no_of_atoms'] = parameters['no_of_atoms']
            else:
                raise RuntimeError('Regression method is not recognized to '
                                   'Amp for loading parameters. User should '
                                   'add the regression method under '
                                   'consideration.')

        Calculator.__init__(self, label=label, **kwargs)

        param = self.parameters

        if param.descriptor is not None:
            self.fp = param.descriptor

        self.reg = param.regression

        self.reg.initialize(param, load)

    ###########################################################################

    def set(self, **kwargs):
        """
        Function to set parameters.
        """
        changed_parameters = Calculator.set(self, **kwargs)
        # FIXME. Decide whether to call reset. Decide if this is
        # meaningful in our implementation!
        if len(changed_parameters) > 0:
            self.reset()

    ###########################################################################

    def set_label(self, label):
        """
        Sets label, ensuring that any needed directories are made.

        :param label: Default prefix/location used for all files.
        :type label: str
        """
        Calculator.set_label(self, label)

        # Create directories for output structure if needed.
        if self.label:
            if (self.directory != os.curdir and
                    not os.path.isdir(self.directory)):
                os.makedirs(self.directory)

    ###########################################################################

    def initialize(self, atoms):
        """
        :param atoms: ASE atoms object.
        :type atoms: ASE dict
        """
        self.par = {}
        self.rc = 0.0
        self.numbers = atoms.get_atomic_numbers()
        self.forces = np.empty((len(atoms), 3))
        self.nl = NeighborList([0.5 * self.rc + 0.25] * len(atoms),
                               self_interaction=False)

    ###########################################################################

    def calculate(self, atoms, properties, system_changes):
        """
        Calculation of the energy of system and forces of all atoms.
        """
        Calculator.calculate(self, atoms, properties, system_changes)

        param = self.parameters
        if param.descriptor is None:  # pure atomic-coordinates scheme
            self.reg.initialize(param=param,
                                atoms=atoms)
        param = self.reg.ravel_variables()

        if param.regression._variables is None:
            raise RuntimeError("Calculator not trained; can't return "
                               'properties.')

        if 'numbers' in system_changes:
            self.initialize(atoms)

        self.nl.update(atoms)

        if param.descriptor is not None:  # fingerprinting scheme
            self.cutoff = param.descriptor.cutoff

            # FIXME: What is the difference between the two updates on the top
            # and bottom? Is the one on the top necessary? Where is self.nl
            #  coming from?

            # Update the neighborlist for making fingerprints. Used if atoms
            # position has changed.
            _nl = NeighborList(cutoffs=([self.cutoff / 2.] *
                                        len(atoms)),
                               self_interaction=False,
                               bothways=True,
                               skin=0.)
            _nl.update(atoms)

            self.fp.atoms = atoms
            self.fp._nl = _nl

            # If fingerprints_range is not available, it will raise an error.
            if param.fingerprints_range is None:
                raise RuntimeError('The keyword "fingerprints_range" is not '
                                   'available. It can be provided to the '
                                   'calculator either by introducing a JSON '
                                   'file, or by directly feeding the keyword '
                                   'to the calculator. If you do not know the '
                                   'values but still want to run the '
                                   'calculator, initialize it with '
                                   'fingerprints_range="auto".')

            # If fingerprints_range is not given either as a direct keyword or
            # as the josn file, but instead is given as 'auto', it will be
            # calculated here.
            if param.fingerprints_range == 'auto':
                warnings.warn('The values of "fingerprints_range" are not '
                              'given. The user is expected to understand what '
                              'is being done!')
                param.fingerprints_range = \
                    calculate_fingerprints_range(self.fp,
                                                 self.reg.elements,
                                                 self.fp.atoms,
                                                 _nl)
            # Deciding on whether it is exptrapoling or interpolating is
            # possible only when fingerprints_range is provided by the user.
            elif self.extrapolate is False:
                if compare_train_test_fingerprints(
                        self.fp,
                        self.fp.atoms,
                        param.fingerprints_range,
                        _nl) == 1:
                    raise ExtrapolateError('Trying to extrapolate, which'
                                           ' is not allowed. Change to '
                                           'extrapolate=True if this is'
                                           ' desired.')

    ##################################################################

        if properties == ['energy']:

            self.reg.reset_energy()
            self.energy = 0.0

            if param.descriptor is None:  # pure atomic-coordinates scheme

                input = (atoms.positions).ravel()
                self.energy = self.reg.get_energy(input,)

            else:  # fingerprinting scheme

                for atom in atoms:
                    index = atom.index
                    symbol = atom.symbol
                    n_indices, n_offsets = _nl.get_neighbors(index)
                    # for calculating fingerprints, summation runs over
                    # neighboring atoms of type I (either inside or outside
                    # the main cell)
                    n_symbols = [atoms[n_index].symbol
                                 for n_index in n_indices]
                    Rs = [atoms.positions[n_index] +
                          np.dot(n_offset, atoms.get_cell())
                          for n_index, n_offset in zip(n_indices, n_offsets)]
                    indexfp = self.fp.get_fingerprint(index, symbol,
                                                      n_symbols, Rs)
                    # fingerprints are scaled to [-1, 1] range
                    scaled_indexfp = [None] * len(indexfp)
                    count = 0
                    for _ in range(len(indexfp)):
                        if (param.fingerprints_range[symbol][_][1] -
                                param.fingerprints_range[symbol][_][0]) \
                                > (10.**(-8.)):
                            scaled_value = -1. + \
                                2. * (indexfp[_] - param.fingerprints_range[
                                    symbol][_][0]) / \
                                (param.fingerprints_range[symbol][_][1] -
                                 param.fingerprints_range[symbol][_][0])
                        else:
                            scaled_value = indexfp[_]
                        scaled_indexfp[count] = scaled_value
                        count += 1

                    atomic_amp_energy = self.reg.get_energy(scaled_indexfp,
                                                            index, symbol,)
                    self.energy += atomic_amp_energy

            self.results['energy'] = float(self.energy)

    ##################################################################

        if properties == ['forces']:

            self.reg.reset_energy()
            outputs = {}
            self.forces[:] = 0.0

            if param.descriptor is None:  # pure atomic-coordinates scheme

                input = (atoms.positions).ravel()
                _ = self.reg.get_energy(input,)
                for atom in atoms:
                    self_index = atom.index
                    self.reg.reset_forces()
                    for i in range(3):
                        _input = [0. for __ in range(3 * len(atoms))]
                        _input[3 * self_index + i] = 1.
                        force = self.reg.get_force(i, _input,)
                        self.forces[self_index][i] = force

            else:  # fingerprinting scheme

                # Neighborlists for all atoms are calculated.
                dict_nl = {}
                n_self_offsets = {}
                for self_atom in atoms:
                    self_index = self_atom.index
                    neighbor_indices, neighbor_offsets = \
                        _nl.get_neighbors(self_index)
                    n_self_indices = np.append(self_index, neighbor_indices)
                    if len(neighbor_offsets) == 0:
                        _n_self_offsets = [[0, 0, 0]]
                    else:
                        _n_self_offsets = np.vstack(([[0, 0, 0]],
                                                     neighbor_offsets))
                    dict_nl[self_index] = n_self_indices
                    n_self_offsets[self_index] = _n_self_offsets

                for atom in atoms:
                    index = atom.index
                    symbol = atom.symbol
                    n_indices, n_offsets = _nl.get_neighbors(index)
                    # for calculating fingerprints, summation runs over
                    # neighboring atoms of type I (either inside or outside
                    # the main cell)
                    n_symbols = [atoms[n_index].symbol
                                 for n_index in n_indices]
                    Rs = [atoms.positions[n_index] +
                          np.dot(n_offset, atoms.get_cell())
                          for n_index, n_offset in zip(n_indices, n_offsets)]
                    indexfp = self.fp.get_fingerprint(index, symbol,
                                                      n_symbols, Rs)
                    # fingerprints are scaled to [-1, 1] range
                    scaled_indexfp = [None] * len(indexfp)
                    count = 0
                    for _ in range(len(indexfp)):
                        if (param.fingerprints_range[symbol][_][1] -
                                param.fingerprints_range[symbol][_][0]) \
                                > (10.**(-8.)):
                            scaled_value = -1. + \
                                2. * (indexfp[_] - param.fingerprints_range[
                                    symbol][_][0]) / \
                                (param.fingerprints_range[symbol][_][1] -
                                 param.fingerprints_range[symbol][_][0])
                        else:
                            scaled_value = indexfp[_]
                        scaled_indexfp[count] = scaled_value
                        count += 1

                    __ = self.reg.get_energy(scaled_indexfp, index, symbol)

                for atom in atoms:
                    self_index = atom.index
                    n_self_indices = dict_nl[self_index]
                    _n_self_offsets = n_self_offsets[self_index]
                    n_self_symbols = [atoms[n_index].symbol
                                      for n_index in n_self_indices]
                    self.reg.reset_forces()
                    for i in range(3):
                        force = 0.
                        for n_symbol, n_index, n_offset in zip(
                                n_self_symbols,
                                n_self_indices,
                                _n_self_offsets):
                            # for calculating forces, summation runs over
                            # neighbor atoms of type II (within the main cell
                            # only)
                            if n_offset[0] == 0 and n_offset[1] == 0 and \
                                    n_offset[2] == 0:

                                neighbor_indices, neighbor_offsets = \
                                    _nl.get_neighbors(n_index)
                                neighbor_symbols = \
                                    [atoms[_index].symbol
                                     for _index in neighbor_indices]
                                Rs = [atoms.positions[_index] +
                                      np.dot(_offset, atoms.get_cell())
                                      for _index, _offset
                                      in zip(neighbor_indices,
                                             neighbor_offsets)]
                                # for calculating derivatives of fingerprints,
                                # summation runs over neighboring atoms of type
                                # I (either inside or outside the main cell)
                                der_indexfp = self.fp.get_der_fingerprint(
                                    n_index, n_symbol,
                                    neighbor_indices,
                                    neighbor_symbols,
                                    Rs, self_index, i)

                                # fingerprint derivatives are scaled
                                scaled_der_indexfp = [None] * len(der_indexfp)
                                count = 0
                                for _ in range(len(der_indexfp)):
                                    if (param.fingerprints_range[
                                        n_symbol][_][1] -
                                        param.fingerprints_range[
                                        n_symbol][_][0]) \
                                            > (10.**(-8.)):
                                        scaled_value = 2. * der_indexfp[_] / \
                                            (param.fingerprints_range[
                                                n_symbol][_][1] -
                                             param.fingerprints_range[
                                                n_symbol][_][0])
                                    else:
                                        scaled_value = der_indexfp[_]
                                    scaled_der_indexfp[count] = scaled_value
                                    count += 1

                                force += self.reg.get_force(i,
                                                            scaled_der_indexfp,
                                                            n_index, n_symbol,)

                        self.forces[self_index][i] = force

                del dict_nl, outputs, n_self_offsets, n_self_indices,
                n_self_symbols, _n_self_offsets, scaled_indexfp, indexfp

            self.results['forces'] = self.forces

    ###########################################################################

    def train(
            self,
            images,
            energy_goal=0.001,
            force_goal=0.005,
            overfitting_constraint=0.,
            force_coefficient=None,
            cores=None,
            optimizer=optimizer,
            read_fingerprints=True,
            overwrite=False,
            global_search=SimulatedAnnealing(initial_temp=70,
                                             steps=2000),
            perturb_variables=None):
        """
        Fits a variable set to the data, by default using the "fmin_bfgs"
        optimizer. The optimizer takes as input a cost function to reduce and
        an initial guess of variables and returns an optimized variable set.

        :param images: List of ASE atoms objects with positions, symbols,
                       energies, and forces in ASE format. This is the training
                       set of data. This can also be the path to an ASE
                       trajectory (.traj) or database (.db) file. Energies can
                       be obtained from any reference, e.g. DFT calculations.
        :type images: list or str
        :param energy_goal: Threshold energy per atom rmse at which simulation
                            is converged.
        :type energy_goal: float
        :param force_goal: Threshold force rmse at which simulation is
                           converged. The default value is in unit of eV/Ang.
                           If 'force_goal = None', forces will not be trained.
        :type force_goal: float
        :param cores: Number of cores to parallelize over. If not specified,
                      attempts to determine from environment.
        :type cores: int
        :param optimizer: The optimization object. The default is to use
                          scipy's fmin_bfgs, but any optimizer that behaves in
                          the same way will do.
        :type optimizer: object
        :param read_fingerprints: Determines whether or not the code should
                                  read fingerprints already calculated and
                                  saved in the script directory.
        :type read_fingerprints: bool
        :param overwrite: If a trained output file with the same name exists,
                          overwrite it.
        :type overwrite: bool
        :param global_search: Method for global search of initial variables.
                              Will ignore, if initial variables are already
                              given. For now, it can be either None, or
                              SimulatedAnnealing(initial_temp, steps).
        :type global_search: object
        :param perturb_variables: If not None, after training, variables
                                  will be perturbed by the amount specified,
                                  and plotted as pdf book. A typical value is
                                  0.01.
        :type perturb_variables: float
        """
        param = self.parameters
        filename = make_filename(self.label, 'trained-parameters.json')
        if (not overwrite) and os.path.exists(filename):
            raise IOError('File exists: %s.\nIf you want to overwrite,'
                          ' set overwrite=True or manually delete.'
                          % filename)

        self.overfitting_constraint = overfitting_constraint

        if force_goal is None:
            train_forces = False
            if not force_coefficient:
                force_coefficient = 0.
        else:
            train_forces = True
            if not force_coefficient:
                force_coefficient = (energy_goal / force_goal)**2.

        log = Logger(make_filename(self.label, 'train-log.txt'))

        log('Amp training started. ' + now() + '\n')
        if param.descriptor is None:  # pure atomic-coordinates scheme
            log('Local environment descriptor: None')
        else:  # fingerprinting scheme
            log('Local environment descriptor: ' +
                param.descriptor.__class__.__name__)
        log('Regression: ' + param.regression.__class__.__name__ + '\n')

        if not cores:
            cores = mp.cpu_count()
        log('Parallel processing over %i cores.\n' % cores)

        if isinstance(images, str):
            extension = os.path.splitext(images)[1]
            if extension == '.traj':
                images = io.Trajectory(images, 'r')
            elif extension == '.db':
                images = io.read(images)

        if param.descriptor is None:  # pure atomic-coordinates scheme
            param.no_of_atoms = len(images[0])
            for image in images:
                if len(image) != param.no_of_atoms:
                    raise RuntimeError('Number of atoms in different images '
                                       'is not the same. Try '
                                       'descriptor=Behler.')

        log('Training on %i images.' % len(images))

        # Images is converted to dictionary form; key is hash of image.
        log.tic()
        log('Hashing images...')
        dict_images = {}
        for image in images:
            key = hash_image(image)
            if key in dict_images.keys():
                log('Warning: Duplicate image (based on identical hash).'
                    ' Was this expected? Hash: %s' % key)
            dict_images[key] = image
        images = dict_images.copy()
        del dict_images
        hashs = sorted(images.keys())
        log(' %i unique images after hashing.' % len(hashs))
        log(' ...hashing completed.', toc=True)

        self.elements = set([atom.symbol for hash in hashs
                             for atom in images[hash]])
        self.elements = sorted(self.elements)

        msg = '%i unique elements included: ' % len(self.elements)
        msg += ', '.join(self.elements)
        log(msg)

        if param.descriptor is not None:  # fingerprinting scheme
            param = self.fp.log(log, param, self.elements)
        param = self.reg.log(log, param, self.elements, images)

        # "MultiProcess" object is initialized
        _mp = MultiProcess(self.fortran, no_procs=cores)

        if param.descriptor is None:  # pure atomic-coordinates scheme
            self.sfp = None
            snl = None
        else:  # fingerprinting scheme
            # Neighborlist for all images are calculated and saved
            log.tic()
            snl = SaveNeighborLists(param.descriptor.cutoff, hashs,
                                    images, self.dblabel, log, train_forces,
                                    read_fingerprints)

            gc.collect()

            # Fingerprints are calculated and saved
            self.sfp = SaveFingerprints(
                self.fp,
                self.elements,
                hashs,
                images,
                self.dblabel,
                train_forces,
                read_fingerprints,
                snl,
                log,
                _mp)

            gc.collect()

            # If fingerprints_range has not been loaded, it will take value
            # from the json file.
            if param.fingerprints_range is None:
                param.fingerprints_range = self.sfp.fingerprints_range

        # If you want to use only one core for feed-forward evaluation
        # uncomment this line: Re-initializing "Multiprocessing" object with
        # one core
#        _mp = MultiProcess(self.fortran, no_procs=1)

        costfxn = CostFxnandDer(
            self.reg,
            param,
            hashs,
            images,
            self.label,
            log,
            energy_goal,
            force_goal,
            train_forces,
            _mp,
            self.overfitting_constraint,
            force_coefficient,
            self.fortran,
            self.sfp,
            snl,)

        del hashs, images

        gc.collect()

        if (param.regression.global_search is True) and \
                (global_search is not None):
            log('\n' + 'Starting global search...')
            gs = global_search
            gs.initialize(param.regression._variables, log, costfxn)
            param.regression._variables = gs.get_variables()

        # saving initial parameters
        filename = make_filename(self.label, 'initial-parameters.json')
        save_parameters(filename, param)
        log('Initial parameters saved in file %s.' % filename)

        log.tic('optimize')
        log('\n' + 'Starting optimization of cost function...')
        log(' Energy goal: %.3e' % energy_goal)
        if train_forces:
            log(' Force goal: %.3e' % force_goal)
            log(' Cost function force coefficient: %f' % force_coefficient)
        else:
            log(' No force training.')
        log.tic()
        converged = False

        variables = param.regression._variables

        try:
            optimizer(f=costfxn.f, x0=variables,
                      fprime=costfxn.fprime,
                      gtol=10. ** -500.)

        except ConvergenceOccurred:
            converged = True

        if not converged:
            log('Saving checkpoint data.')
            filename = make_filename(self.label, 'parameters-checkpoint.json')
            save_parameters(filename, costfxn.param)
            log(' ...could not find parameters for the desired goal\n'
                'error. Least error parameters saved as checkpoint.\n'
                'Try it again or assign a larger value for "goal".',
                toc='optimize')
        else:
            param.regression._variables = costfxn.param.regression._variables
            self.reg.update_variables(param)
            log(' ...optimization completed successfully. Optimal '
                'parameters saved.', toc='optimize')
            filename = make_filename(self.label, 'trained-parameters.json')
            save_parameters(filename, param)

            self.cost_function = costfxn.cost_function
            self.energy_per_atom_rmse = costfxn.energy_per_atom_rmse
            self.force_rmse = costfxn.force_rmse
            self.der_variables_cost_function = \
                costfxn.der_variables_square_error

        # perturb variables and plot cost function
        if perturb_variables is not None:

            log.tic('perturb')
            log('\n' + 'Perturbing variables...')

            calculate_gradient = False
            energy_coefficient = 1.
            optimizedvariables = costfxn.param.regression._variables.copy()
            no_of_variables = len(optimizedvariables)
            optimizedcost = costfxn.cost_function
            zeros = np.zeros(no_of_variables)

            all_variables = []
            all_costs = []
            for _ in range(no_of_variables):
                log('variable %s out of %s' % (_, no_of_variables - 1))
                costs = []
                perturbance = zeros.copy()
                perturbance[_] -= perturb_variables
                perturbedvariables = optimizedvariables + perturbance

                param.regression._variables = perturbedvariables

                if self.fortran:
                    task_args = (param, calculate_gradient)
                    (energy_square_error,
                     force_square_error,
                     ___) = \
                        costfxn._mp.share_cost_function_task_between_cores(
                        task=_calculate_cost_function_fortran,
                        _args=task_args, len_of_variables=no_of_variables)
                else:
                    task_args = (self.reg, param, self.sfp, snl,
                                 energy_coefficient, force_coefficient,
                                 train_forces, no_of_variables,
                                 calculate_gradient)
                    (energy_square_error,
                     force_square_error,
                     ___) = \
                        costfxn._mp.share_cost_function_task_between_cores(
                        task=_calculate_cost_function_python,
                        _args=task_args, len_of_variables=no_of_variables)

                newcost = energy_coefficient * energy_square_error + \
                    force_coefficient * force_square_error
                costs.append(newcost)

                costs.append(optimizedcost)

                perturbance = zeros.copy()
                perturbance[_] += perturb_variables
                perturbedvariables = optimizedvariables + perturbance

                param.regression._variables = perturbedvariables

                if self.fortran:
                    task_args = (param, calculate_gradient)
                    (energy_square_error,
                     force_square_error,
                     ___) = \
                        costfxn._mp.share_cost_function_task_between_cores(
                        task=_calculate_cost_function_fortran,
                        _args=task_args, len_of_variables=no_of_variables)
                else:
                    task_args = (self.reg, param, self.sfp, snl,
                                 energy_coefficient, force_coefficient,
                                 train_forces, no_of_variables,
                                 calculate_gradient)
                    (energy_square_error,
                     force_square_error,
                     ___) = \
                        costfxn._mp.share_cost_function_task_between_cores(
                        task=_calculate_cost_function_python,
                        _args=task_args, len_of_variables=no_of_variables)

                newcost = energy_coefficient * energy_square_error + \
                    force_coefficient * force_square_error
                costs.append(newcost)

                all_variables.append([optimizedvariables[_] -
                                      perturb_variables,
                                      optimizedvariables[_],
                                      optimizedvariables[_] +
                                      perturb_variables])
                all_costs.append(costs)

            log('Plotting cost function vs perturbed variables...')

            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import rcParams
            from matplotlib import pyplot
            from matplotlib.backends.backend_pdf import PdfPages
            rcParams.update({'figure.autolayout': True})

            filename = make_filename(self.label, 'perturbed-parameters.pdf')
            with PdfPages(filename) as pdf:
                for _ in range(no_of_variables):
                    fig = pyplot.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(all_variables[_],
                            all_costs[_],
                            marker='o', linestyle='--', color='b',)
                    ax.set_xlabel('variable %s' % _)
                    ax.set_ylabel('cost function')
                    pdf.savefig(fig)
                    pyplot.close(fig)

            log(' ...perturbing variables completed.', toc='perturb')

        if not converged:
            raise TrainingConvergenceError()

###############################################################################
###############################################################################
###############################################################################


class MultiProcess:

    """
    Class to do parallel processing, using multiprocessing package which works
    on Python versions 2.6 and above.

    :param fortran: If True, allows for extrapolation, if False, does not
                    allow.
    :type fortran: bool
    :param no_procs: Number of processors.
    :type no_procs: int
    """
    ###########################################################################

    def __init__(self, fortran, no_procs):

        self.fortran = fortran
        self.no_procs = no_procs

    ###########################################################################

    def make_list_of_sub_images(self, hashs, images):
        """
        Two lists are made each with one entry per core. The entry of the first
        list contains list of hashes to be calculated by that core, and the
        entry of the second list contains dictionary of images to be calculated
        by that core.

        :param hashs: Unique keys, one key per image.
        :type hashs: list
        :param images: List of ASE atoms objects (the training set).
        :type images: list
        """
        quotient = int(len(hashs) / self.no_procs)
        remainder = len(hashs) - self.no_procs * quotient
        list_sub_hashes = [None] * self.no_procs
        list_sub_images = [None] * self.no_procs
        count0 = 0
        count1 = 0
        for _ in range(self.no_procs):
            if _ < remainder:
                len_sub_hashes = quotient + 1
            else:
                len_sub_hashes = quotient
            sub_hashes = [None] * len_sub_hashes
            sub_images = {}
            count2 = 0
            for j in range(len_sub_hashes):
                hash = hashs[count1]
                sub_hashes[count2] = hash
                sub_images[hash] = images[hash]
                count1 += 1
                count2 += 1
            list_sub_hashes[count0] = sub_hashes
            list_sub_images[count0] = sub_images
            count0 += 1

        self.list_sub_hashes = list_sub_hashes
        self.list_sub_images = list_sub_images

        del hashs, images, list_sub_hashes, list_sub_images, sub_hashes,
        sub_images

    ###########################################################################

    def share_fingerprints_task_between_cores(self, task, _args):
        """
        Fingerprint tasks are sent to cores for parallel processing.

        :param task: Function to be called on each process.
        :type task: function
        :param _args: Arguments to be fed to the function on each process.
        :type _args: function
        """
        args = {}
        for x in range(self.no_procs):
            sub_hashs = self.list_sub_hashes[x]
            sub_images = self.list_sub_images[x]
            args[x] = (x,) + (sub_hashs, sub_images,) + _args

        processes = [mp.Process(target=task, args=args[x])
                     for x in range(self.no_procs)]

        for x in range(self.no_procs):
            processes[x].start()

        for x in range(self.no_procs):
            processes[x].join()

        for x in range(self.no_procs):
            processes[x].terminate()

        del sub_hashs, sub_images

    ###########################################################################

    def ravel_images_data(self,
                          param,
                          sfp,
                          snl,
                          elements,
                          train_forces,
                          log,):
        """
        Reshape data of images into lists.

        :param param: ASE dictionary object.
        :type param: dict
        :param sfp: SaveFingerprints object.
        :type sfp: object
        :param snl: SaveNeighborLists object.
        :type snl: object
        :param elements: List if elements in images.
        :type elements: list
        :param train_forces: Determining whether forces are also trained or
                             not.
        :type train_forces: bool
        :param log: Write function at which to log data. Note this must be a
                    callable function.
        :type log: Logger object
        """
        log('Re-shaping images data to send to fortran90...')
        log.tic()

        self.train_forces = train_forces

        if param.descriptor is None:
            self.fingerprinting = False
        else:
            self.fingerprinting = True

        self.no_of_images = {}
        self.real_energies = {}
        for x in range(self.no_procs):
            self.no_of_images[x] = len(self.list_sub_hashes[x])
            self.real_energies[x] = \
                [self.list_sub_images[x][
                    hash].get_potential_energy(apply_constraint=False)
                    for hash in self.list_sub_hashes[x]]

        if self.fingerprinting:

            self.no_of_atoms_of_images = {}
            self.atomic_numbers_of_images = {}
            self.raveled_fingerprints_of_images = {}
            for x in range(self.no_procs):
                self.no_of_atoms_of_images[x] = \
                    [len(self.list_sub_images[x][hash])
                     for hash in self.list_sub_hashes[x]]
                self.atomic_numbers_of_images[x] = \
                    [atomic_numbers[atom.symbol]
                     for hash in self.list_sub_hashes[x]
                     for atom in self.list_sub_images[x][hash]]
                self.raveled_fingerprints_of_images[x] = \
                    ravel_fingerprints_of_images(self.list_sub_hashes[x],
                                                 self.list_sub_images[x],
                                                 sfp)
        else:
            self.atomic_positions_of_images = {}
            for x in range(self.no_procs):
                self.atomic_positions_of_images[x] = \
                    [self.list_sub_images[x][hash].positions.ravel()
                     for hash in self.list_sub_hashes[x]]

        if train_forces is True:

            self.real_forces = {}
            for x in range(self.no_procs):
                self.real_forces[x] = \
                    [self.list_sub_images[x][hash].get_forces(
                        apply_constraint=False)[index]
                     for hash in self.list_sub_hashes[x]
                     for index in range(len(self.list_sub_images[x][hash]))]

            if self.fingerprinting:
                self.list_of_no_of_neighbors = {}
                self.raveled_neighborlists = {}
                self.raveled_der_fingerprints = {}
                for x in range(self.no_procs):
                    (self.list_of_no_of_neighbors[x],
                     self.raveled_neighborlists[x],
                     self.raveled_der_fingerprints[x]) = \
                        ravel_neighborlists_and_der_fingerprints_of_images(
                        self.list_sub_hashes[x],
                        self.list_sub_images[x],
                        sfp,
                        snl)

        log(' ...data re-shaped.', toc=True)

    ###########################################################################

    def share_cost_function_task_between_cores(self, task, _args,
                                               len_of_variables):
        """
        Cost function and its derivatives of with respect to variables are
        calculated in parallel.

        :param task: Function to be called on each process.
        :type task: function
        :param _args: Arguments to be fed to the function on each process.
        :type _args: function
        :param len_of_variables: Number of variables.
        :type len_of_variables: int
        """
        queues = {}
        for x in range(self.no_procs):
            queues[x] = mp.Queue()

#        print "self.no_procs =", self.no_procs

        args = {}
        for x in range(self.no_procs):
            if self.fortran:
                args[x] = _args + (queues[x],)
            else:
                sub_hashs = self.list_sub_hashes[x]
#                print "sub_hashs =", sub_hashs
                sub_images = self.list_sub_images[x]
#                print "sub_images =", sub_images
                args[x] = (sub_hashs, sub_images,) + _args + (queues[x],)
#                print "args[x] =", args[x]

        energy_square_error = 0.
        force_square_error = 0.

#        print "len_of_variables =", len_of_variables

        der_variables_square_error = [0.] * len_of_variables

        processes = [mp.Process(target=task, args=args[x])
                     for x in range(self.no_procs)]

        for x in range(self.no_procs):
            if self.fortran:
                # data particular to each process is sent to fortran modules
                self.send_data_to_fortran(x,)
            processes[x].start()

        for x in range(self.no_procs):
            processes[x].join()

        for x in range(self.no_procs):
            processes[x].terminate()

        sub_energy_square_error = []
        sub_force_square_error = []
        sub_der_variables_square_error = []

        # Construct total square_error and derivative with respect to variables
        # from subprocesses
        results = {}
        for x in range(self.no_procs):
            results[x] = queues[x].get()
#            print "results[x] =", results[x]

        sub_energy_square_error = [results[x][0] for x in range(self.no_procs)]
        sub_force_square_error = [results[x][1] for x in range(self.no_procs)]
        sub_der_variables_square_error = [results[x][2]
                                          for x in range(self.no_procs)]

        for _ in range(len(sub_energy_square_error)):
            energy_square_error += sub_energy_square_error[_]
            force_square_error += sub_force_square_error[_]
            for j in range(len_of_variables):
                der_variables_square_error[j] += \
                    sub_der_variables_square_error[_][j]

        if not self.fortran:
            del sub_hashs, sub_images

        return (energy_square_error, force_square_error,
                der_variables_square_error)

    ###########################################################################

    def send_data_to_fortran(self, x,):
        """
        Function to send images data to fortran90 code.

        :param x: The number of process.
        :type x: int
        """
        fmodules.images_props.no_of_images = self.no_of_images[x]
        fmodules.images_props.real_energies = self.real_energies[x]

        if self.fingerprinting:
            fmodules.images_props.no_of_atoms_of_images = \
                self.no_of_atoms_of_images[x]
            fmodules.images_props.atomic_numbers_of_images = \
                self.atomic_numbers_of_images[x]
            fmodules.fingerprint_props.raveled_fingerprints_of_images = \
                self.raveled_fingerprints_of_images[x]
        else:
            fmodules.images_props.atomic_positions_of_images = \
                self.atomic_positions_of_images[x]

        if self.train_forces is True:

            fmodules.images_props.real_forces_of_images = \
                self.real_forces[x]

            if self.fingerprinting:
                fmodules.images_props.list_of_no_of_neighbors = \
                    self.list_of_no_of_neighbors[x]
                fmodules.images_props.raveled_neighborlists = \
                    self.raveled_neighborlists[x]
                fmodules.fingerprint_props.raveled_der_fingerprints = \
                    self.raveled_der_fingerprints[x]

###############################################################################
###############################################################################
###############################################################################


class SaveNeighborLists:

    """
    Neighborlists for all images with the given cutoff value are calculated
    and saved. As well as neighboring atomic indices, neighboring atomic
    offsets from the main cell are also saved. Offsets become important when
    dealing with periodic systems. Neighborlists are generally of two types:
    Type I which consists of atoms within the cutoff distance either in the
    main cell or in the adjacent cells, and Type II which consists of atoms in
    the main cell only and within the cutoff distance.

    :param cutoff: Cutoff radius, in Angstroms, around each atom.
    :type cutoff: float
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param label: Prefix name used for all files.
    :type label: str
    :param log: Write function at which to log data. Note this must be a
                callable function.
    :type log: Logger object
    :param train_forces: Determining whether forces are also trained or not.
    :type train_forces: bool
    :param read_fingerprints: Determines whether or not the code should read
                              fingerprints already calculated and saved in the
                              script directory.
    :type read_fingerprints: bool
    """
    ###########################################################################

    def __init__(self, cutoff, hashs, images, label, log, train_forces,
                 read_fingerprints):

        self.cutoff = cutoff
        self.images = images
        self.nl_data = {}

        if train_forces is True:
            log('Calculating neighborlist for each atom...')
            log.tic()
            if read_fingerprints:
                try:
                    filename = make_filename(label, 'neighborlists.json')
                    fp = paropen(filename, 'rb')
                    data = json.load(fp)
                except IOError:
                    log(' No saved neighborlist file found.')
                else:
                    for key1 in data.keys():
                        for key2 in data[key1]:
                            nl_value = data[key1][key2]
                            self.nl_data[(key1, int(key2))] = \
                                ([value[0] for value in nl_value],
                                 [value[1] for value in nl_value],)
                    log(' Saved neighborlist file %s loaded with %i entries.'
                        % (filename, len(data.keys())))
            else:
                log(' Neighborlists in the script directory (if any) will not '
                    'be used.')

            new_images = {}
            old_hashes = set([key[0] for key in self.nl_data.keys()])
            for hash in hashs:
                if hash not in old_hashes:
                    new_images[hash] = images[hash]

            log(' Calculating %i of %i neighborlists.'
                % (len(new_images), len(images)))

            if len(new_images) != 0:
                new_hashs = sorted(new_images.keys())
                for hash in new_hashs:
                    image = new_images[hash]
                    nl = NeighborList(cutoffs=([self.cutoff / 2.] *
                                               len(image)),
                                      self_interaction=False,
                                      bothways=True, skin=0.)
                    # FIXME: Is update necessary?
                    nl.update(image)
                    for self_atom in image:
                        self_index = self_atom.index
                        neighbor_indices, neighbor_offsets = \
                            nl.get_neighbors(self_index)
                        if len(neighbor_offsets) == 0:
                            n_self_offsets = [[0, 0, 0]]
                        else:
                            n_self_offsets = \
                                np.vstack(([[0, 0, 0]], neighbor_offsets))
                        n_self_indices = np.append(self_index,
                                                   neighbor_indices)
                        self.nl_data[(hash, self_index)] = \
                            (n_self_indices, n_self_offsets,)

                dict_data = OrderedDict()
                key0s = set([key[0] for key in self.nl_data.keys()])
                for key0 in key0s:
                    dict_data[key0] = {}
                for key in self.nl_data.keys():
                    dict_data[key[0]][key[1]] = self.nl_data[key]
                filename = make_filename(label, 'neighborlists.json')
                save_neighborlists(filename, dict_data)
                log(' ...neighborlists calculated and saved to %s.' %
                    filename, toc=True)

                del new_hashs, dict_data
            del new_images, old_hashes
        del images, self.images

###############################################################################
###############################################################################
###############################################################################


class SaveFingerprints:

    """
    Memory class to not recalculate fingerprints and their derivatives if
    not necessary. This could cause runaway memory usage; use with caution.

    :param fp: Fingerprint object.
    :type fp: object
    :param elements: List if elements in images.
    :type elements: list
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param label: Prefix name used for all files.
    :type label: str
    :param train_forces: Determining whether forces are also trained or not.
    :type train_forces: bool
    :param read_fingerprints: Determines whether or not the code should read
                              fingerprints already calculated and saved in the
                              script directory.
    :type read_fingerprints: bool
    :param snl: SaveNeighborLists object.
    :type snl: object
    :param log: Write function at which to log data. Note this must be a
                callable function.
    :type log: Logger object
    :param _mp: MultiProcess object.
    :type _mp: object
    """
    ###########################################################################

    def __init__(self,
                 fp,
                 elements,
                 hashs,
                 images,
                 label,
                 train_forces,
                 read_fingerprints,
                 snl,
                 log,
                 _mp):

        self.Gs = fp.Gs
        self.train_forces = train_forces
        self.fp_data = {}
        self.der_fp_data = {}

        log('Calculating atomic fingerprints...')
        log.tic()

        if read_fingerprints:
            try:
                filename = make_filename(label, 'fingerprints.json')
                fp = paropen(filename, 'rb')
                data = json.load(fp)
            except IOError:
                log('No saved fingerprint file found.')
            else:
                log.tic('read_fps')
                log(' Reading fingerprints from %s...' % filename)
                for key1 in data.keys():
                    for key2 in data[key1]:
                        fp_value = data[key1][key2]
                        self.fp_data[(key1, int(key2))] = \
                            [float(value) for value in fp_value]
                log(' ...fingerprints read',
                    toc='read_fps')
        else:
            log(' Pre-calculated fingerprints (if any) will not be used.')

        new_images = {}
        old_hashes = set([key[0] for key in self.fp_data.keys()])
        for hash in hashs:
            if hash not in old_hashes:
                new_images[hash] = images[hash]

        log(' Calculating %i of %i fingerprints. (%i exist in file.)'
            % (len(new_images), len(images), len(old_hashes)))

        if len(new_images) != 0:
            log.tic('calculate_fps')
            new_hashs = sorted(new_images.keys())
            # new images are shared between cores for fingerprint calculations
            _mp.make_list_of_sub_images(new_hashs, new_images)

            # Temporary files to hold child fingerprint calculations.
            childfiles = [tempfile.NamedTemporaryFile(prefix='fp-',
                                                      suffix='.json')
                          for _ in range(_mp.no_procs)]
            for _ in range(_mp.no_procs):
                log('  Processor %i calculations stored in file %s.'
                    % (_, childfiles[_].name))

            task_args = (fp, label, childfiles)
            _mp.share_fingerprints_task_between_cores(
                task=_calculate_fingerprints, _args=task_args)

            log(' Calculated %i new images.' % len(new_images),
                toc='calculate_fps')
            log.tic('read_fps')
            log(' Reading calculated child-fingerprints...')
            for f in childfiles:
                f.seek(0)
                data = json.load(f)
                for key1 in data.keys():
                    for key2 in data[key1]:
                        fp_value = data[key1][key2]
                        self.fp_data[(key1, int(key2))] = \
                            [float(value) for value in fp_value]
            log(' ...child-fingerprints are read.', toc='read_fps')

            del new_hashs

        dict_data = OrderedDict()
        keys1 = set([key[0] for key in self.fp_data.keys()])
        for key in keys1:
            dict_data[key] = {}
        for key in self.fp_data.keys():
            dict_data[key[0]][key[1]] = self.fp_data[key]
        if len(new_images) != 0:
            log.tic('save_fps')
            log(' Saving fingerprints...')
            filename = make_filename(label, 'fingerprints.json')
            save_fingerprints(filename, dict_data)
            log(' ...fingerprints saved to %s.' % filename,
                toc='save_fps')

        fingerprint_values = {}
        for element in elements:
            fingerprint_values[element] = {}
            for _ in range(len(self.Gs[element])):
                fingerprint_values[element][_] = []

        for hash in hashs:
            image = images[hash]
            for atom in image:
                for _ in range(len(self.Gs[atom.symbol])):
                    fingerprint_values[atom.symbol][_].append(
                        dict_data[hash][atom.index][_])

        fingerprints_range = OrderedDict()
        for element in elements:
            fingerprints_range[element] = \
                [[min(fingerprint_values[element][_]),
                  max(fingerprint_values[element][_])]
                 for _ in range(len(self.Gs[element]))]

        self.fingerprints_range = fingerprints_range

        del dict_data, keys1, new_images, old_hashes

        if train_forces is True:
            log('Calculating derivatives of atomic fingerprints '
                'with respect to coordinates...')
            log.tic('fp_forces')

            if read_fingerprints:
                try:
                    filename = make_filename(
                        label,
                        'fingerprint-derivatives.json')
                    fp = paropen(filename, 'rb')
                    data = json.load(fp)
                except IOError or ValueError:
                    log('Either no saved fingerprint-derivatives file found '
                        'or it cannot be read.')
                else:
                    log.tic('read_der_fps')
                    log(' Reading fingerprint derivatives from file %s' %
                        filename)
                    for key1 in data.keys():
                        for key2 in data[key1]:
                            key3 = json.loads(key2)
                            fp_value = data[key1][key2]
                            self.der_fp_data[(key1,
                                              (int(key3[0]),
                                               int(key3[1]),
                                               int(key3[2])))] = \
                                [float(value) for value in fp_value]
                    log(' ...fingerprint derivatives read.',
                        toc='read_der_fps')
            else:
                log(' Pre-calculated fingerprint derivatives (if any) will '
                    'not be used.')

            new_images = {}
            old_hashes = set([key[0] for key in self.der_fp_data.keys()])
            for hash in hashs:
                if hash not in old_hashes:
                    new_images[hash] = images[hash]

            log(' Calculating %i of %i fingerprint derivatives. '
                '(%i exist in file.)'
                % (len(new_images), len(images), len(old_hashes)))

            if len(new_images) != 0:
                log.tic('calculate_der_fps')
                new_hashs = sorted(new_images.keys())
                # new images are shared between cores for calculating
                # fingerprint derivatives
                _mp.make_list_of_sub_images(new_hashs, new_images)

                # Temporary files to hold child fingerprint calculations.
                childfiles = [tempfile.NamedTemporaryFile(prefix='fp-',
                                                          suffix='.json')
                              for _ in range(_mp.no_procs)]
                for _ in range(_mp.no_procs):
                    log('  Processor %i calculations stored in file %s.'
                        % (_, childfiles[_].name))

                task_args = (fp, snl, label, childfiles)
                _mp.share_fingerprints_task_between_cores(
                    task=_calculate_der_fingerprints,
                    _args=task_args)

                log(' Calculated %i new images.' % len(new_images),
                    toc='calculate_der_fps')

                log.tic('read_der_fps')
                log(' Reading child-fingerprint-derivatives...')
                for f in childfiles:
                    f.seek(0)
                    data = json.load(f)
                    for key1 in data.keys():
                        for key2 in data[key1]:
                            key3 = json.loads(key2)
                            fp_value = data[key1][key2]
                            self.der_fp_data[(key1,
                                              (int(key3[0]),
                                               int(key3[1]),
                                               int(key3[2])))] = \
                                [float(value) for value in fp_value]
                log(' ...child-fingerprint-derivatives are read.',
                    toc='read_der_fps')

                log.tic('save_der_fps')
                dict_data = OrderedDict()
                keys1 = set([key[0] for key in self.der_fp_data.keys()])
                for key in keys1:
                    dict_data[key] = {}
                for key in self.der_fp_data.keys():
                    dict_data[key[0]][key[1]] = self.der_fp_data[key]
                filename = make_filename(label, 'fingerprint-derivatives.json')
                save_der_fingerprints(filename, dict_data)
                log(' ...fingerprint derivatives calculated and saved to %s.'
                    % filename, toc='save_der_fps')

                del dict_data, keys1, new_hashs
            del new_images, old_hashes
        del images

        log(' ...all fingerprint operations complete.', toc=True)

###############################################################################
###############################################################################
###############################################################################


class CostFxnandDer:

    """
    Cost function and its derivative based on sum of squared deviations in
    energies and forces to be optimized by setting variables.

    :param reg: Regression object.
    :type reg: object
    :param param: ASE dictionary that contains cutoff and variables.
    :type param: dict
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: ASE atoms objects (the train set).
    :type images: dict
    :param label: Prefix used for all files.
    :type label: str
    :param log: Write function at which to log data. Note this must be a
                callable function.
    :type log: Logger object
    :param energy_goal: Threshold energy per atom rmse at which simulation is
                        converged.
    :type energy_goal: float
    :param force_goal: Threshold force rmse at which simulation is converged.
    :type force_goal: float
    :param train_forces: Determines whether or not forces should be trained.
    :type train_forces: bool
    :param _mp: Multiprocess object.
    :type _mp: object
    :param overfitting_constraint: The constant to constraint overfitting.
    :type overfitting_constraint: float
    :param force_coefficient: Multiplier of force RMSE in constructing the cost
                              function. This controls how tight force-fit is as
                              compared to energy fit. It also depends on force
                              and energy units. Working with eV and Angstrom,
                              0.04 seems to be a reasonable value.
    :type force_coefficient: float
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool
    :param sfp: SaveFingerprints object.
    :type sfp: object
    :param snl: SaveNeighborLists object.
    :type snl: object
    """
    ###########################################################################

    def __init__(self, reg, param, hashs, images, label, log,
                 energy_goal, force_goal, train_forces, _mp,
                 overfitting_constraint, force_coefficient, fortran, sfp=None,
                 snl=None,):

        self.reg = reg
        self.param = param
        self.label = label
        self.log = log
        self.energy_goal = energy_goal
        self.force_goal = force_goal
        self.sfp = sfp
        self.snl = snl
        self.train_forces = train_forces
        self.steps = 0
        self._mp = _mp
        self.overfitting_constraint = overfitting_constraint
        self.force_coefficient = force_coefficient
        self.fortran = fortran

        if param.descriptor is not None:  # pure atomic-coordinates scheme
            self.cutoff = param.descriptor.cutoff

        self.energy_convergence = False
        self.force_convergence = False
        if not self.train_forces:
            self.force_convergence = True

        self.energy_coefficient = 1.0

        self.no_of_images = len(hashs)

        # all images are shared between cores for feed-forward and
        # back-propagation calculations
        self._mp.make_list_of_sub_images(hashs, images)

        if self.fortran:
            # regression data is sent to fortran modules
            self.reg.send_data_to_fortran(param)
            # data common between processes is sent to fortran modules
            send_data_to_fortran(self.sfp,
                                 self.reg.elements,
                                 self.train_forces,
                                 self.energy_coefficient,
                                 self.force_coefficient,
                                 param,)
            self._mp.ravel_images_data(param,
                                       self.sfp,
                                       self.snl,
                                       self.reg.elements,
                                       self.train_forces,
                                       log,)
#            del self._mp.list_sub_images, self._mp.list_sub_hashes

        del hashs, images

        gc.collect()

    ###########################################################################

    def f(self, variables):
        """
        Function to calculate cost function.

        :param variables: Calibrating variables.
        :type variables: list
        """
        calculate_gradient = True
        log = self.log
        self.param.regression._variables = variables

        if self.fortran:
            task_args = (self.param, calculate_gradient)
            (energy_square_error,
             force_square_error,
             self.der_variables_square_error) = \
                self._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_fortran,
                _args=task_args, len_of_variables=len(variables))
        else:
            task_args = (self.reg, self.param, self.sfp, self.snl,
                         self.energy_coefficient, self.force_coefficient,
                         self.train_forces, len(variables), calculate_gradient)
            (energy_square_error,
             force_square_error,
             self.der_variables_square_error) = \
                self._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_python,
                _args=task_args, len_of_variables=len(variables))

        square_error = self.energy_coefficient * energy_square_error + \
            self.force_coefficient * force_square_error

        self.cost_function = square_error

        self.energy_per_atom_rmse = \
            np.sqrt(energy_square_error / self.no_of_images)
        self.force_rmse = np.sqrt(force_square_error / self.no_of_images)

        if self.steps == 0:
            if self.train_forces is True:
                head1 = ('%5s  %19s  %9s  %9s  %9s')
                log(head1 % ('', '', '',
                             ' (Energy',
                             ''))
                log(head1 % ('', '', 'Cost',
                             'per Atom)',
                             'Force'))
                log(head1 % ('Step', 'Time', 'Function',
                             'RMSE',
                             'RMSE'))
                log(head1 %
                    ('=' * 5, '=' * 19, '=' * 9, '=' * 9, '=' * 9))
            else:
                head1 = ('%3s  %16s %18s %10s')
                log(head1 % ('Step', 'Time', 'Cost',
                             '    RMS (Energy'))
                head2 = ('%3s  %28s %10s %12s')
                log(head2 % ('', '', 'Function',
                             'per Atom)'))
                head3 = ('%3s  %26s %10s %11s')
                log(head3 % ('', '', '',
                             '   Error'))
                log(head3 %
                    ('=' * 5, '=' * 26, '=' * 9, '=' * 10))

        self.steps += 1

        if self.train_forces is True:
            line = ('%5s' '  %19s' '  %9.3e' '  %9.3e' '  %9.3e')
            log(line % (self.steps - 1,
                        now(),
                        self.cost_function,
                        self.energy_per_atom_rmse,
                        self.force_rmse))
        else:
            line = ('%5s' ' %26s' ' %10.3e' ' %12.3e')
            log(line % (self.steps - 1,
                        now(),
                        self.cost_function,
                        self.energy_per_atom_rmse))

        if self.steps % 100 == 0:
            log('Saving checkpoint data.')
            filename = make_filename(
                self.label,
                'parameters-checkpoint.json')
            save_parameters(filename, self.param)

        if self.energy_per_atom_rmse < self.energy_goal and \
                self.energy_convergence is False:
            log('Energy convergence!')
            self.energy_convergence = True
        elif self.energy_per_atom_rmse > self.energy_goal and \
                self.energy_convergence is True:
            log('Energy unconverged!')
            self.energy_convergence = False

        if self.train_forces:
            if self.force_rmse < self.force_goal and \
                    self.force_convergence is False:
                log('Force convergence!')
                self.force_convergence = True
            elif self.force_rmse > self.force_goal and \
                    self.force_convergence is True:
                log('Force unconverged!')
                self.force_convergence = False

        if (self.energy_convergence and self.force_convergence):
            raise ConvergenceOccurred

        gc.collect()

        return self.cost_function

    ###########################################################################

    def fprime(self, variables):
        """
        Function to calculate derivative of cost function.

        :param variables: Calibrating variables.
        :type variables: list
        """
        if self.steps == 0:

            calculate_gradient = True
            self.param.regression._variables = variables

            if self.fortran:
                task_args = (self.param, calculate_gradient)
                (energy_square_error,
                 force_square_error,
                 self.der_variables_square_error) = \
                    self._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_fortran,
                    _args=task_args, len_of_variables=len(variables))
            else:
                task_args = (self.reg, self.param, self.sfp, self.snl,
                             self.energy_coefficient, self.force_coefficient,
                             self.train_forces, len(variables),
                             calculate_gradient)
                (energy_square_error,
                 force_square_error,
                 self.der_variables_square_error) = \
                    self._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_python,
                    _args=task_args, len_of_variables=len(variables))

        der_cost_function = self.der_variables_square_error

        der_cost_function = np.array(der_cost_function)

        return der_cost_function

###############################################################################
###############################################################################
###############################################################################


def _calculate_fingerprints(proc_no, hashes, images, fp, label, childfiles):
    """
    Function to be called on all processes simultaneously for calculating
    fingerprints.

    :param proc_no: Number of the process.
    :type proc_no: int
    :param hashes: Unique keys, one key per image.
    :type hashes: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param fp: Fingerprint object.
    :type fp: object
    :param label: Prefix name used for all files.
    :type label: str
    :param childfiles: Temporary files
    :type childfiles: file
    """
    fingerprints = {}
    for hash in hashes:
        fingerprints[hash] = {}
        atoms = images[hash]
        fp.initialize(atoms)
        _nl = NeighborList(cutoffs=([fp.cutoff / 2.] * len(atoms)),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        _nl.update(atoms)
        for atom in atoms:
            index = atom.index
            symbol = atom.symbol
            n_indices, n_offsets = _nl.get_neighbors(index)
            # for calculating fingerprints, summation runs over neighboring
            # atoms of type I (either inside or outside the main cell)
            n_symbols = [atoms[n_index].symbol for n_index in n_indices]
            Rs = [atoms.positions[n_index] +
                  np.dot(n_offset, atoms.get_cell())
                  for n_index, n_offset in zip(n_indices, n_offsets)]
            indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
            fingerprints[hash][index] = indexfp

    save_fingerprints(childfiles[proc_no], fingerprints)

    del hashes, images

###############################################################################


def _calculate_der_fingerprints(proc_no, hashes, images, fp,
                                snl, label, childfiles):
    """
    Function to be called on all processes simultaneously for calculating
    derivatives of fingerprints.

    :param proc_no: Number of the process.
    :type proc_no: int
    :param hashes: Unique keys, one key per image.
    :type hashes: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param fp: Fingerprint object.
    :type fp: object
    :param label: Prefix name used for all files.
    :type label: str
    :param childfiles: Temporary files
    :type childfiles: file
    """
    data = {}
    for hash in hashes:
        data[hash] = {}
        atoms = images[hash]
        fp.initialize(atoms)
        _nl = NeighborList(cutoffs=([fp.cutoff / 2.] *
                                    len(atoms)),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        _nl.update(atoms)
        for self_atom in atoms:
            self_index = self_atom.index
            n_self_indices = snl.nl_data[(hash, self_index)][0]
            n_self_offsets = snl.nl_data[(hash, self_index)][1]
            n_symbols = [atoms[n_index].symbol for n_index in n_self_indices]
            for n_symbol, n_index, n_offset in zip(n_symbols, n_self_indices,
                                                   n_self_offsets):
                # derivative of fingerprints are needed only with respect to
                # coordinates of atoms of type II (within the main cell only)
                if n_offset[0] == 0 and n_offset[1] == 0 and n_offset[2] == 0:
                    for i in range(3):
                        neighbor_indices, neighbor_offsets = \
                            _nl.get_neighbors(n_index)
                        # for calculating derivatives of fingerprints,
                        # summation runs over neighboring atoms of type I
                        # (either inside or outside the main cell)
                        neighbor_symbols = \
                            [atoms[
                                _index].symbol for _index in neighbor_indices]
                        Rs = [atoms.positions[_index] +
                              np.dot(_offset, atoms.get_cell())
                              for _index, _offset
                              in zip(neighbor_indices, neighbor_offsets)]
                        der_indexfp = fp.get_der_fingerprint(
                            n_index, n_symbol,
                            neighbor_indices,
                            neighbor_symbols,
                            Rs, self_index, i)

                        data[hash][(n_index, self_index, i)] = der_indexfp

    save_der_fingerprints(childfiles[proc_no], data)

    del hashes, images, data

###############################################################################


def _calculate_cost_function_fortran(param, calculate_gradient, queue):
    """
    Function to be called on all processes simultaneously for calculating cost
    function and its derivative with respect to variables in fortran.

    :param param: ASE dictionary.
    :type param: dict
    :param queue: multiprocessing queue.
    :type queue: object
    """
    variables = param.regression._variables

    (energy_square_error,
     force_square_error,
     der_variables_square_error) = \
        fmodules.share_cost_function_task_between_cores(
        variables=variables,
        len_of_variables=len(variables),
        calculate_gradient=calculate_gradient)

    queue.put([energy_square_error,
               force_square_error,
               der_variables_square_error])

###############################################################################


def _calculate_cost_function_python(hashes, images, reg, param, sfp,
                                    snl, energy_coefficient,
                                    force_coefficient, train_forces,
                                    len_of_variables, calculate_gradient,
                                    queue):
    """
    Function to be called on all processes simultaneously for calculating cost
    function and its derivative with respect to variables in python.

    :param hashes: Unique keys, one key per image.
    :type hashes: list
    :param images: ASE atoms objects (the train set).
    :type images: dict
    :param reg: Regression object.
    :type reg: object
    :param param: ASE dictionary that contains cutoff and variables.
    :type param: dict
    :param sfp: SaveFingerprints object.
    :type sfp: object
    :param snl: SaveNeighborLists object.
    :type snl: object
    :param energy_coefficient: Multiplier of energy per atom RMSE in
                               constructing the cost function.
    :type energy_coefficient: float
    :param force_coefficient: Multiplier of force RMSE in constructing the cost
                              function.
    :type force_coefficient: float
    :param train_forces: Determines whether or not forces should be trained.
    :type train_forces: bool
    :param len_of_variables: Number of calibrating variables.
    :type len_of_variables: int
    :param calculate_gradient: Determines whether or not gradient of the cost
                               function with respect to variables should also
                               be calculated.
    :type calculate_gradient: bool
    :param queue: multiprocessing queue.
    :type queue: object
    """
    der_variables_square_error = np.zeros(len_of_variables)

    energy_square_error = 0.
    force_square_error = 0.

    reg.update_variables(param)

    for hash in hashes:
        atoms = images[hash]
        real_energy = atoms.get_potential_energy(apply_constraint=False)
        real_forces = atoms.get_forces(apply_constraint=False)

        reg.reset_energy()
        amp_energy = 0.

        if param.descriptor is None:  # pure atomic-coordinates scheme

            input = (atoms.positions).ravel()
            amp_energy = reg.get_energy(input,)

        else:  # fingerprinting scheme

            for atom in atoms:
                index = atom.index
                symbol = atom.symbol
                indexfp = sfp.fp_data[(hash, index)]
                # fingerprints are scaled to [-1, 1] range
                scaled_indexfp = [None] * len(indexfp)
                count = 0
                for _ in range(len(indexfp)):
                    if (sfp.fingerprints_range[symbol][_][1] -
                            sfp.fingerprints_range[symbol][_][0]) > \
                            (10.**(-8.)):
                        scaled_value = \
                            -1. + 2. * (indexfp[_] -
                                        sfp.fingerprints_range[symbol][_][0]) \
                            / (sfp.fingerprints_range[symbol][_][1] -
                               sfp.fingerprints_range[symbol][_][0])
                    else:
                        scaled_value = indexfp[_]
                    scaled_indexfp[count] = scaled_value
                    count += 1
                atomic_amp_energy = reg.get_energy(scaled_indexfp,
                                                   index, symbol,)
                amp_energy += atomic_amp_energy

        energy_square_error += ((amp_energy - real_energy) ** 2.) / \
            (len(atoms) ** 2.)

        if calculate_gradient:

            if param.descriptor is None:  # pure atomic-coordinates scheme

                partial_der_variables_square_error = \
                    reg.get_variable_der_of_energy()
                der_variables_square_error += \
                    energy_coefficient * 2. * (amp_energy - real_energy) * \
                    partial_der_variables_square_error / (len(atoms) ** 2.)

            else:  # fingerprinting scheme

                for atom in atoms:
                    index = atom.index
                    symbol = atom.symbol
                    partial_der_variables_square_error =\
                        reg.get_variable_der_of_energy(index, symbol)
                    der_variables_square_error += \
                        energy_coefficient * 2. * (amp_energy - real_energy) \
                        * partial_der_variables_square_error / \
                        (len(atoms) ** 2.)

        if train_forces is True:

            real_forces = atoms.get_forces(apply_constraint=False)
            amp_forces = np.zeros((len(atoms), 3))
            for self_atom in atoms:
                self_index = self_atom.index
                reg.reset_forces()

                if param.descriptor is None:  # pure atomic-coordinates scheme

                    for i in range(3):
                        _input = [0. for __ in range(3 * len(atoms))]
                        _input[3 * self_index + i] = 1.
                        force = reg.get_force(i, _input,)
                        amp_forces[self_index][i] = force

                else:  # fingerprinting scheme

                    n_self_indices = snl.nl_data[(hash, self_index)][0]
                    n_self_offsets = snl.nl_data[(hash, self_index)][1]
                    n_symbols = [atoms[n_index].symbol
                                 for n_index in n_self_indices]

                    for n_symbol, n_index, n_offset in zip(n_symbols,
                                                           n_self_indices,
                                                           n_self_offsets):
                        # for calculating forces, summation runs over neighbor
                        # atoms of type II (within the main cell only)
                        if n_offset[0] == 0 and n_offset[1] == 0 and \
                                n_offset[2] == 0:
                            for i in range(3):
                                der_indexfp = sfp.der_fp_data[(hash,
                                                               (n_index,
                                                                self_index,
                                                                i))]
                                # fingerprint derivatives are scaled
                                scaled_der_indexfp = [None] * len(der_indexfp)
                                count = 0
                                for _ in range(len(der_indexfp)):
                                    if (sfp.fingerprints_range[
                                            n_symbol][_][1] -
                                            sfp.fingerprints_range
                                            [n_symbol][_][0]) > (10.**(-8.)):
                                        scaled_value = 2. * der_indexfp[_] / \
                                            (sfp.fingerprints_range
                                             [n_symbol][_][1] -
                                             sfp.fingerprints_range
                                             [n_symbol][_][0])
                                    else:
                                        scaled_value = der_indexfp[_]
                                    scaled_der_indexfp[count] = scaled_value
                                    count += 1

                                force = reg.get_force(i, scaled_der_indexfp,
                                                      n_index, n_symbol,)

                                amp_forces[self_index][i] += force

                for i in range(3):
                    force_square_error += \
                        ((1.0 / 3.0) * (amp_forces[self_index][i] -
                                        real_forces[self_index][i]) **
                         2.) / len(atoms)

                if calculate_gradient:

                    for i in range(3):
                        # pure atomic-coordinates scheme
                        if param.descriptor is None:

                            partial_der_variables_square_error = \
                                reg.get_variable_der_of_forces(self_index, i)
                            der_variables_square_error += \
                                force_coefficient * (2.0 / 3.0) * \
                                (- amp_forces[self_index][i] +
                                 real_forces[self_index][i]) * \
                                partial_der_variables_square_error \
                                / len(atoms)

                        else:  # fingerprinting scheme

                            for n_symbol, n_index, n_offset in \
                                    zip(n_symbols, n_self_indices,
                                        n_self_offsets):
                                if n_offset[0] == 0 and n_offset[1] == 0 and \
                                        n_offset[2] == 0:

                                    partial_der_variables_square_error = \
                                        reg.get_variable_der_of_forces(
                                            self_index,
                                            i,
                                            n_index,
                                            n_symbol,)
                                    der_variables_square_error += \
                                        force_coefficient * (2.0 / 3.0) * \
                                        (- amp_forces[self_index][i] +
                                         real_forces[self_index][i]) * \
                                        partial_der_variables_square_error \
                                        / len(atoms)

    del hashes, images

    queue.put([energy_square_error,
               force_square_error,
               der_variables_square_error])

###############################################################################


def calculate_fingerprints_range(fp, elements, atoms, nl):
    """
    Function to calculate fingerprints range.

    :param fp: Fingerprint object.
    :type fp: object
    :param elements: List of atom symbols.
    :type elements: list of str
    :param atoms: ASE atoms object.
    :type atoms: ASE dict
    :param nl: ASE NeighborList object.
    :type nl: object

    :returns: Range of fingerprints of elements.
    """
    fingerprint_values = {}
    for element in elements:
        fingerprint_values[element] = {}
        for i in range(len(fp.Gs[element])):
            fingerprint_values[element][i] = []

    for atom in atoms:
        index = atom.index
        symbol = atom.symbol
        n_indices, n_offsets = nl.get_neighbors(index)
        # for calculating fingerprints, summation runs over neighboring
        # atoms of type I (either inside or outside the main cell)
        n_symbols = [atoms[n_index].symbol for n_index in n_indices]
        Rs = [atoms.positions[n_index] +
              np.dot(n_offset, atoms.get_cell())
              for n_index, n_offset in zip(n_indices, n_offsets)]
        indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
        for _ in range(len(fp.Gs[symbol])):
            fingerprint_values[symbol][_].append(indexfp[_])

    fingerprints_range = {}
    for element in elements:
        fingerprints_range[element] = [None] * len(fp.Gs[element])
        count = 0
        for _ in range(len(fp.Gs[element])):
            if len(fingerprint_values[element][_]) > 0:
                minimum = min(fingerprint_values[element][_])
                maximum = max(fingerprint_values[element][_])
            else:
                minimum = -1.0
                maximum = -1.0
            fingerprints_range[element][count] = [minimum, maximum]
            count += 1

    return fingerprints_range

###############################################################################


def compare_train_test_fingerprints(fp, atoms, fingerprints_range, nl):
    """
    Function to compare train images with the test image and decide whether
    the prediction is interpolation or extrapolation.

    :param fp: Fingerprint object.
    :type fp: object
    :param atoms: ASE atoms object.
    :type atoms: ASE dict
    :param fingerprints_range: Range of fingerprints of each chemical species.
    :type fingerprints_range: dict
    :param nl: ASE NeighborList object.
    :type nl: object

    :returns: Zero for interpolation, and one for extrapolation.
    """
    compare_train_test_fingerprints = 0

    for atom in atoms:
        index = atom.index
        symbol = atom.symbol
        n_indices, n_offsets = nl.get_neighbors(index)
        # for calculating fingerprints, summation runs over neighboring
        # atoms of type I (either inside or outside the main cell)
        n_symbols = [atoms[n_index].symbol for n_index in n_indices]
        Rs = [atoms.positions[n_index] +
              np.dot(n_offset, atoms.get_cell())
              for n_index, n_offset in zip(n_indices, n_offsets)]
        indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
        for i in range(len(indexfp)):
            if indexfp[i] < fingerprints_range[symbol][i][0] or \
                    indexfp[i] > fingerprints_range[symbol][i][1]:
                compare_train_test_fingerprints = 1
                break
    return compare_train_test_fingerprints

###############################################################################


def interpolate_images(images, load, fortran=True):
    """
    Function to remove extrapolation images from the "images" set based on
    load data.

    :param images: List of ASE atoms objects with positions, symbols, energies,
                   and forces in ASE format. This can also be the path to an
                   ASE trajectory (.traj) or database (.db) file.
    :type images: list or str
    :param load: Path for loading an existing parameters of Amp calculator.
    :type load: str
    :param fortran: If True, will use fortran modules, if False, will not.
    :type fortran: bool

    :returns: Two dictionary of all images, and interpolated images,
              respectively.
    """
    if isinstance(images, str):
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = io.Trajectory(images, 'r')
        elif extension == '.db':
            images = io.read(images)

    # Images is converted to dictionary form; key is hash of image.
    dict_images = {}
    for image in images:
        key = hash_image(image)
        dict_images[key] = image
    images = dict_images.copy()
    del dict_images

    amp = Amp(load=load, fortran=fortran)
    param = amp.parameters
    fp = param.descriptor
    fingerprints_range = param.fingerprints_range
    # FIXME: this function should be extended to no fingerprints scheme.
    if fp is not None:  # fingerprinting scheme
        cutoff = fp.cutoff

    # Dictionary of interpolated images set is initialized
    interpolated_images = {}
    for hash, atoms in images.items():
        fp.atoms = atoms
        _nl = NeighborList(cutoffs=([cutoff / 2.] * len(atoms)),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        _nl.update(atoms)
        fp._nl = _nl
        compare_train_test_fingerprints = 0
        for atom in atoms:
            index = atom.index
            symbol = atom.symbol
            n_indices, n_offsets = _nl.get_neighbors(index)
            # for calculating fingerprints, summation runs over neighboring
            # atoms of type I (either inside or outside the main cell)
            n_symbols = [atoms[n_index].symbol for n_index in n_indices]
            Rs = [atoms.positions[n_index] +
                  np.dot(n_offset, atoms.get_cell())
                  for n_index, n_offset in zip(n_indices, n_offsets)]
            indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
            for i in range(len(indexfp)):
                if indexfp[i] < fingerprints_range[symbol][i][0] or \
                        indexfp[i] > fingerprints_range[symbol][i][1]:
                    compare_train_test_fingerprints = 1
                    break
        if compare_train_test_fingerprints == 0:
            interpolated_images[hash] = image

    return images, interpolated_images

###############################################################################


def send_data_to_fortran(sfp, elements, train_forces,
                         energy_coefficient, force_coefficient, param):
    """
    Function to send images data to fortran code. Is used just once.

    :param sfp: SaveFingerprints object.
    :type sfp: object
    :param elements: List of atom symbols.
    :type elements: list of str
    :param train_forces: Determines whether or not forces should be trained.
    :type train_forces: bool
    :param energy_coefficient: Multiplier of energy per atom RMSE in
                               constructing the cost function.
    :type energy_coefficient: float
    :param force_coefficient: Multiplier of force RMSE in constructing the cost
                              function.
    :type force_coefficient: float
    :param param: ASE dictionary that contains cutoff and variables.
    :type param: dict
    """
    if param.descriptor is None:
        fingerprinting = False
    else:
        fingerprinting = True

    if fingerprinting:
        no_of_elements = len(elements)
        elements_numbers = [atomic_numbers[elm] for elm in elements]
        min_fingerprints = \
            [[param.fingerprints_range[elm][_][0]
              for _ in range(len(param.fingerprints_range[elm]))]
             for elm in elements]
        max_fingerprints = [[param.fingerprints_range[elm][_][1]
                             for _
                             in range(len(param.fingerprints_range[elm]))]
                            for elm in elements]
        len_fingerprints_of_elements = [len(sfp.Gs[elm]) for elm in elements]
    else:
        no_of_atoms_of_image = param.no_of_atoms

    fmodules.images_props.energy_coefficient = energy_coefficient
    fmodules.images_props.force_coefficient = force_coefficient
    fmodules.images_props.train_forces = train_forces
    fmodules.images_props.fingerprinting = fingerprinting

    if fingerprinting:
        fmodules.images_props.no_of_elements = no_of_elements
        fmodules.images_props.elements_numbers = elements_numbers
        fmodules.fingerprint_props.min_fingerprints = min_fingerprints
        fmodules.fingerprint_props.max_fingerprints = max_fingerprints
        fmodules.fingerprint_props.len_fingerprints_of_elements = \
            len_fingerprints_of_elements
    else:
        fmodules.images_props.no_of_atoms_of_image = no_of_atoms_of_image

##############################################################################


def ravel_fingerprints_of_images(hashs, images, sfp):
    """
    Reshape fingerprints of all images into a matrix.

    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: ASE atoms objects (the train set).
    :type images: dict
    :param sfp: SaveFingerprints object.
    :type sfp: object
    """
    keys = [(hash, index) for hash in hashs
            for index in range(len(images[hash]))]

    raveled_fingerprints = [sfp.fp_data[key] for key in keys]

    del hashs, images, keys

    return raveled_fingerprints

###############################################################################


def ravel_neighborlists_and_der_fingerprints_of_images(hashs,
                                                       images, sfp, snl):
    """
    Reshape neighborlists and derivatives of fingerprints of all images into a
    matrix.

    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: ASE atoms objects (the train set).
    :type images: dict
    :param sfp: SaveFingerprints object.
    :type sfp: object
    :param snl: SaveNeighborLists object.
    :type snl: object
    """
    # Only neighboring atoms of type II (within the main cell) needs to be sent
    # to fortran for force training
    list_of_no_of_neighbors = []
    for hash in hashs:
        atoms = images[hash]
        for self_atom in atoms:
            self_index = self_atom.index
            n_self_offsets = snl.nl_data[(hash, self_index)][1]
            count = 0
            for n_offset in n_self_offsets:
                if n_offset[0] == 0 and n_offset[1] == 0 and n_offset[2] == 0:
                    count += 1
            list_of_no_of_neighbors.append(count)

    raveled_neighborlists = [n_index for hash in hashs
                             for self_atom in images[hash]
                             for n_index, n_offset in
                             zip(snl.nl_data[(hash, self_atom.index)][0],
                                 snl.nl_data[(hash, self_atom.index)][1])
                             if (n_offset[0] == 0 and n_offset[1] == 0 and
                                 n_offset[2] == 0)]

    raveled_der_fingerprints = \
        [sfp.der_fp_data[(hash, (n_index, self_atom.index, i))]
         for hash in hashs
         for self_atom in images[hash]
         for n_index, n_offset in
         zip(snl.nl_data[(hash, self_atom.index)][0],
             snl.nl_data[(hash, self_atom.index)][1])
         if (n_offset[0] == 0 and n_offset[1] == 0 and
             n_offset[2] == 0) for i in range(3)]

    del hashs, images

    return (list_of_no_of_neighbors,
            raveled_neighborlists,
            raveled_der_fingerprints)

###############################################################################


def now():
    """
    :returns: String of current time.
    """
    return datetime.now().isoformat().split('.')[0]

###############################################################################
