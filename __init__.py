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
from ase import io as aseio
import numpy as np
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
    from . import fmodules  # version 3 of fmodules
    fmodules_version = 3
except ImportError:
    fmodules = None

###############################################################################


class SimulatedAnnealing:

    """
    Class that implements simulated annealing algorithm for global search of
    variables. This algorithm is helpful to be used for pre-conditioning of the
    initial guess of variables for optimization of non-convex functions.

    :param temperature: Initial temperature which corresponds to initial
                         variance of the likelihood normal probability
                         distribution. Should take a value from 50 to 100.
    :type temperature: float
    :param steps: Number of search iterations.
    :type steps: int
    :param acceptance_criteria: A float in the range of zero to one.
                                Temperature will be controlled such that
                                acceptance rate meets this criteria.
    :type acceptance_criteria: float
    """
    ###########################################################################

    def __init__(self, temperature, steps, acceptance_criteria=0.5):
        self.temperature = temperature
        self.steps = steps
        self.acceptance_criteria = acceptance_criteria

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
        len_of_variables = len(variables)
        temp = self.temperature

        calculate_gradient = False
        self.costfxn.param.regression._variables = variables

        if self.costfxn.fortran:
            task_args = (self.costfxn.param, calculate_gradient)
            (energy_square_error, force_square_error, _) = \
                self.costfxn._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_fortran,
                _args=task_args, len_of_variables=len_of_variables)
        else:
            task_args = (self.costfxn.reg, self.costfxn.param,
                         self.costfxn.sfp, self.costfxn.snl,
                         self.costfxn.energy_coefficient,
                         self.costfxn.force_coefficient,
                         self.costfxn.train_forces, len_of_variables,
                         calculate_gradient)
            (energy_square_error, force_square_error, _) = \
                self.costfxn._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_python,
                _args=task_args, len_of_variables=len_of_variables)

        square_error = \
            self.costfxn.energy_coefficient * energy_square_error + \
            self.costfxn.force_coefficient * force_square_error

        allvariables = [variables]
        besterror = square_error
        bestvariables = variables

        accepted = 0

        step = 0
        while step < self.steps:

            # Calculating old log of probability
            logp = - square_error / temp

            # Calculating new log of probability
            _steps = np.random.rand(len_of_variables) * 2. - 1.
            _steps *= 0.2
            newvariables = variables + _steps
            calculate_gradient = False
            self.costfxn.param.regression._variables = newvariables
            if self.costfxn.fortran:
                task_args = (self.costfxn.param, calculate_gradient)
                (energy_square_error, force_square_error, _) = \
                    self.costfxn._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_fortran,
                    _args=task_args, len_of_variables=len_of_variables)
            else:
                task_args = (self.costfxn.reg, self.costfxn.param,
                             self.costfxn.sfp, self.costfxn.snl,
                             self.costfxn.energy_coefficient,
                             self.costfxn.force_coefficient,
                             self.costfxn.train_forces, len_of_variables,
                             calculate_gradient)
                (energy_square_error, force_square_error, _) = \
                    self.costfxn._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_python,
                    _args=task_args, len_of_variables=len_of_variables)
            new_square_error = \
                self.costfxn.energy_coefficient * energy_square_error + \
                self.costfxn.force_coefficient * force_square_error
            newlogp = - new_square_error / temp

            # Calculating probability ratio
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
                square_error = new_square_error

            # Changing temprature according to acceptance ratio
            if (accepted / (step + 1) < self.acceptance_criteria):
                temp += 0.0005 * temp
            else:
                temp -= 0.002 * temp
            step += 1

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
                                  activation=parameters['activation'],)
                kwargs['regression']._variables = parameters['variables']
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

        no_of_atoms = len(atoms)
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
                                        no_of_atoms),
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

                index = 0
                while index < no_of_atoms:
                    symbol = atoms[index].symbol
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
                    len_of_indexfp = len(indexfp)
                    # fingerprints are scaled to [-1, 1] range
                    scaled_indexfp = [None] * len_of_indexfp
                    count = 0
                    while count < len_of_indexfp:
                        if (param.fingerprints_range[symbol][count][1] -
                                param.fingerprints_range[symbol][count][0]) \
                                > (10.**(-8.)):
                            scaled_value = -1. + \
                                2. * (indexfp[count] -
                                      param.fingerprints_range[
                                      symbol][count][0]) / \
                                (param.fingerprints_range[symbol][count][1] -
                                 param.fingerprints_range[symbol][count][0])
                        else:
                            scaled_value = indexfp[count]
                        scaled_indexfp[count] = scaled_value
                        count += 1

                    atomic_amp_energy = self.reg.get_energy(scaled_indexfp,
                                                            index, symbol,)
                    self.energy += atomic_amp_energy
                    index += 1

            self.results['energy'] = float(self.energy)

    ##################################################################

        if properties == ['forces']:

            self.reg.reset_energy()
            outputs = {}
            self.forces[:] = 0.0

            if param.descriptor is None:  # pure atomic-coordinates scheme

                input = (atoms.positions).ravel()
                _ = self.reg.get_energy(input,)
                self_index = 0
                while self_index < no_of_atoms:
                    self.reg.reset_forces()
                    i = 0
                    while i < 3:
                        _input = [0.] * (3 * no_of_atoms)
                        _input[3 * self_index + i] = 1.
                        force = self.reg.get_force(i, _input,)
                        self.forces[self_index][i] = force
                        i += 1
                    self_index += 1

            else:  # fingerprinting scheme

                # Neighborlists for all atoms are calculated.
                dict_nl = {}
                n_self_offsets = {}
                self_index = 0
                while self_index < no_of_atoms:
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
                    self_index += 1

                index = 0
                while index < no_of_atoms:
                    symbol = atoms[index].symbol
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
                    len_of_indexfp = len(indexfp)
                    # fingerprints are scaled to [-1, 1] range
                    scaled_indexfp = [None] * len_of_indexfp
                    count = 0
                    while count < len_of_indexfp:
                        if (param.fingerprints_range[symbol][count][1] -
                                param.fingerprints_range[symbol][count][0]) \
                                > (10.**(-8.)):
                            scaled_value = -1. + \
                                2. * (indexfp[count] -
                                      param.fingerprints_range[
                                      symbol][count][0]) / \
                                (param.fingerprints_range[symbol][count][1] -
                                 param.fingerprints_range[symbol][count][0])
                        else:
                            scaled_value = indexfp[count]
                        scaled_indexfp[count] = scaled_value
                        count += 1

                    __ = self.reg.get_energy(scaled_indexfp, index, symbol)
                    index += 1

                self_index = 0
                while self_index < no_of_atoms:
                    n_self_indices = dict_nl[self_index]
                    _n_self_offsets = n_self_offsets[self_index]
                    n_self_symbols = [atoms[n_index].symbol
                                      for n_index in n_self_indices]
                    self.reg.reset_forces()
                    len_of_n_self_indices = len(n_self_indices)
                    i = 0
                    while i < 3:
                        force = 0.
                        n_count = 0
                        while n_count < len_of_n_self_indices:
                            n_symbol = n_self_symbols[n_count]
                            n_index = n_self_indices[n_count]
                            n_offset = _n_self_offsets[n_count]
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

                                len_of_der_indexfp = len(der_indexfp)

                                # fingerprint derivatives are scaled
                                scaled_der_indexfp = \
                                    [None] * len_of_der_indexfp
                                count = 0
                                while count < len_of_der_indexfp:
                                    if (param.fingerprints_range[
                                        n_symbol][count][1] -
                                        param.fingerprints_range[
                                        n_symbol][count][0]) \
                                            > (10.**(-8.)):
                                        scaled_value = 2. * \
                                            der_indexfp[count] / \
                                            (param.fingerprints_range[
                                                n_symbol][count][1] -
                                             param.fingerprints_range[
                                                n_symbol][count][0])
                                    else:
                                        scaled_value = der_indexfp[count]
                                    scaled_der_indexfp[count] = scaled_value
                                    count += 1

                                force += self.reg.get_force(i,
                                                            scaled_der_indexfp,
                                                            n_index, n_symbol,)
                            n_count += 1
                        self.forces[self_index][i] = force
                        i += 1
                    self_index += 1

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
            overwrite=False,
            data_format='db',
            global_search=SimulatedAnnealing(temperature=70,
                                             steps=2000),
            perturb_variables=None,
            extend_variables=True):
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
        :param overwrite: If a trained output file with the same name exists,
                          overwrite it.
        :type overwrite: bool
        :param data_format: Format of saved data. Can be either "json" or "db".
        :type data_format: str
        :param global_search: Method for global search of initial variables.
                              Will ignore, if initial variables are already
                              given. For now, it can be either None, or
                              SimulatedAnnealing(temperature, steps).
        :type global_search: object
        :param perturb_variables: If not None, after training, variables
                                  will be perturbed by the amount specified,
                                  and plotted as pdf book. A typical value is
                                  0.01.
        :type perturb_variables: float
        :param extend_variables: Determines whether or not the code should
                                 extend the number of variables if convergence
                                 does not happen.
        :type extend_variables: bool
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

        energy_coefficient = 1.

        log = Logger(make_filename(self.label, 'train-log.txt'))

        log('Amp training started. ' + now() + '\n')
        if param.descriptor is None:  # pure atomic-coordinates scheme
            log('Local environment descriptor: None')
        else:  # fingerprinting scheme
            log('Local environment descriptor: ' +
                param.descriptor.__class__.__name__)
        log('Regression: ' + param.regression.__class__.__name__ + '\n')

        if not cores:
            from utilities import count_allocated_cpus
            cores = count_allocated_cpus()
        log('Parallel processing over %i cores.\n' % cores)

        if isinstance(images, str):
            extension = os.path.splitext(images)[1]
            if extension == '.traj':
                images = aseio.Trajectory(images, 'r')
            elif extension == '.db':
                images = aseio.read(images)
        no_of_images = len(images)

        if param.descriptor is None:  # pure atomic-coordinates scheme
            param.no_of_atoms = len(images[0])
            count = 0
            while count < no_of_images:
                image = images[count]
                if len(image) != param.no_of_atoms:
                    raise RuntimeError('Number of atoms in different images '
                                       'is not the same. Try '
                                       'descriptor=Behler.')
                count += 1

        log('Training on %i images.' % no_of_images)

        # Images is converted to dictionary form; key is hash of image.
        log.tic()
        log('Hashing images...')
        dict_images = {}
        count = 0
        while count < no_of_images:
            image = images[count]
            key = hash_image(image)
            if key in dict_images.keys():
                log('Warning: Duplicate image (based on identical hash).'
                    ' Was this expected? Hash: %s' % key)
            dict_images[key] = image
            count += 1
        images = dict_images.copy()
        del dict_images
        hashs = sorted(images.keys())
        no_of_images = len(hashs)
        log(' %i unique images after hashing.' % no_of_images)
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

        # all images are shared between cores for feed-forward and
        # back-propagation calculations
        _mp.make_list_of_sub_images(no_of_images, hashs, images)

        io = IO(hashs, images,)  # utilities.IO object initialized.

        if param.descriptor is None:  # pure atomic-coordinates scheme
            self.sfp = None
            snl = None
        else:  # fingerprinting scheme
            # Neighborlist for all images are calculated and saved
            log.tic()
            snl = SaveNeighborLists(param.descriptor.cutoff, no_of_images,
                                    hashs, images, self.dblabel, log,
                                    train_forces, io, data_format)

            gc.collect()

            # Fingerprints are calculated and saved
            self.sfp = SaveFingerprints(self.fp, self.elements, no_of_images,
                                        hashs, images, self.dblabel,
                                        train_forces, snl, log, _mp, io,
                                        data_format)

            gc.collect()

            # If fingerprints_range has not been loaded, it will take value
            # from the json file.
            if param.fingerprints_range is None:
                param.fingerprints_range = self.sfp.fingerprints_range

        del hashs, images

        if self.fortran:
            # data common between processes is sent to fortran modules
            send_data_to_fortran(self.sfp,
                                 self.reg.elements,
                                 train_forces,
                                 energy_coefficient,
                                 force_coefficient,
                                 param,)
            _mp.ravel_images_data(param,
                                  self.sfp,
                                  snl,
                                  self.reg.elements,
                                  train_forces,
                                  log,)
#            del _mp.list_sub_images, _mp.list_sub_hashs

        costfxn = CostFxnandDer(
            self.reg,
            param,
            no_of_images,
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
        step = 0
        while not converged:
            if step > 0:
                param = self.reg.introduce_variables(log, param)
                costfxn = CostFxnandDer(
                    self.reg,
                    param,
                    no_of_images,
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

            variables = param.regression._variables

            try:
                optimizer(f=costfxn.f, x0=variables,
                          fprime=costfxn.fprime,
                          gtol=10. ** -500.)

            except ConvergenceOccurred:
                converged = True

            if extend_variables is False:
                break
            else:
                step += 1

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
            optimizedvariables = costfxn.param.regression._variables.copy()
            no_of_variables = len(optimizedvariables)
            optimizedcost = costfxn.cost_function
            zeros = np.zeros(no_of_variables)

            all_variables = []
            all_costs = []

            count = 0
            while count < no_of_variables:
                log('variable %s out of %s' % (count, no_of_variables - 1))
                costs = []
                perturbance = zeros.copy()
                perturbance[count] -= perturb_variables
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
                perturbance[count] += perturb_variables
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

                all_variables.append([optimizedvariables[count] -
                                      perturb_variables,
                                      optimizedvariables[count],
                                      optimizedvariables[count] +
                                      perturb_variables])
                all_costs.append(costs)
                count += 1

            log('Plotting cost function vs perturbed variables...')

            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import rcParams
            from matplotlib import pyplot
            from matplotlib.backends.backend_pdf import PdfPages
            rcParams.update({'figure.autolayout': True})

            filename = make_filename(self.label, 'perturbed-parameters.pdf')
            with PdfPages(filename) as pdf:
                count = 0
                while count < no_of_variables:
                    fig = pyplot.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(all_variables[count],
                            all_costs[count],
                            marker='o', linestyle='--', color='b',)
                    ax.set_xlabel('variable %s' % count)
                    ax.set_ylabel('cost function')
                    pdf.savefig(fig)
                    pyplot.close(fig)
                    count += 1

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

    def make_list_of_sub_images(self, no_of_images, hashs, images):
        """
        Two lists are made each with one entry per core. The entry of the first
        list contains list of hashs to be calculated by that core, and the
        entry of the second list contains dictionary of images to be calculated
        by that core.

        :param no_of_images: Total number of images.
        :type no_of_images: int
        :param hashs: Unique keys, one key per image.
        :type hashs: list
        :param images: List of ASE atoms objects (the training set).
        :type images: list
        """
        quotient = int(no_of_images / self.no_procs)
        remainder = no_of_images - self.no_procs * quotient
        list_sub_hashs = [None] * self.no_procs
        list_sub_images = [None] * self.no_procs
        count0 = 0
        count1 = 0
        while count0 < self.no_procs:
            if count0 < remainder:
                len_sub_hashs = quotient + 1
            else:
                len_sub_hashs = quotient
            sub_hashs = [None] * len_sub_hashs
            sub_images = {}
            count2 = 0
            while count2 < len_sub_hashs:
                hash = hashs[count1]
                sub_hashs[count2] = hash
                sub_images[hash] = images[hash]
                count1 += 1
                count2 += 1
            list_sub_hashs[count0] = sub_hashs
            list_sub_images[count0] = sub_images
            count0 += 1

        self.list_sub_hashs = list_sub_hashs
        self.list_sub_images = list_sub_images

        del hashs, images, list_sub_hashs, list_sub_images, sub_hashs,
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
        count = 0
        while count < self.no_procs:
            sub_hashs = self.list_sub_hashs[count]
            sub_images = self.list_sub_images[count]
            args[count] = (count,) + (sub_hashs, sub_images,) + _args
            count += 1

        processes = [mp.Process(target=task, args=args[_])
                     for _ in range(self.no_procs)]

        count = 0
        while count < self.no_procs:
            processes[count].start()
            count += 1

        count = 0
        while count < self.no_procs:
            processes[count].join()
            count += 1

        count = 0
        while count < self.no_procs:
            processes[count].terminate()
            count += 1

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
        x = 0
        while x < self.no_procs:
            self.no_of_images[x] = len(self.list_sub_hashs[x])
            self.real_energies[x] = \
                [self.list_sub_images[x][
                    hash].get_potential_energy(apply_constraint=False)
                    for hash in self.list_sub_hashs[x]]
            x += 1

        if self.fingerprinting:

            self.no_of_atoms_of_images = {}
            self.atomic_numbers_of_images = {}
            self.raveled_fingerprints_of_images = {}
            x = 0
            while x < self.no_procs:
                self.no_of_atoms_of_images[x] = \
                    [len(self.list_sub_images[x][hash])
                     for hash in self.list_sub_hashs[x]]
                self.atomic_numbers_of_images[x] = \
                    [atomic_numbers[atom.symbol]
                     for hash in self.list_sub_hashs[x]
                     for atom in self.list_sub_images[x][hash]]
                self.raveled_fingerprints_of_images[x] = \
                    ravel_fingerprints_of_images(self.list_sub_hashs[x],
                                                 self.list_sub_images[x],
                                                 sfp)
                x += 1
        else:
            self.atomic_positions_of_images = {}
            x = 0
            while x < self.no_procs:
                self.atomic_positions_of_images[x] = \
                    [self.list_sub_images[x][hash].positions.ravel()
                     for hash in self.list_sub_hashs[x]]
                x += 1

        if train_forces is True:

            self.real_forces = {}
            x = 0
            while x < self.no_procs:
                self.real_forces[x] = \
                    [self.list_sub_images[x][hash].get_forces(
                        apply_constraint=False)[index]
                     for hash in self.list_sub_hashs[x]
                     for index in range(len(self.list_sub_images[x][hash]))]
                x += 1

            if self.fingerprinting:
                self.list_of_no_of_neighbors = {}
                self.raveled_neighborlists = {}
                self.raveled_der_fingerprints = {}
                x = 0
                while x < self.no_procs:
                    (self.list_of_no_of_neighbors[x],
                     self.raveled_neighborlists[x],
                     self.raveled_der_fingerprints[x]) = \
                        ravel_neighborlists_and_der_fingerprints_of_images(
                        self.list_sub_hashs[x],
                        self.list_sub_images[x],
                        sfp,
                        snl)
                    x += 1

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
        x = 0
        while x < self.no_procs:
            queues[x] = mp.Queue()
            x += 1

        args = {}
        x = 0
        while x < self.no_procs:
            if self.fortran:
                args[x] = _args + (queues[x],)
            else:
                sub_hashs = self.list_sub_hashs[x]
                sub_images = self.list_sub_images[x]
                args[x] = (sub_hashs, sub_images,) + _args + (queues[x],)
            x += 1

        energy_square_error = 0.
        force_square_error = 0.

        der_variables_square_error = [0.] * len_of_variables

        processes = [mp.Process(target=task, args=args[_])
                     for _ in range(self.no_procs)]

        x = 0
        while x < self.no_procs:
            if self.fortran:
                # data particular to each process is sent to fortran modules
                self.send_data_to_fortran(x,)
            processes[x].start()
            x += 1

        x = 0
        while x < self.no_procs:
            processes[x].join()
            x += 1

        x = 0
        while x < self.no_procs:
            processes[x].terminate()
            x += 1

        sub_energy_square_error = []
        sub_force_square_error = []
        sub_der_variables_square_error = []

        # Construct total square_error and derivative with respect to variables
        # from subprocesses
        results = {}
        x = 0
        while x < self.no_procs:
            results[x] = queues[x].get()
            x += 1

        sub_energy_square_error = [results[_][0] for _ in range(self.no_procs)]
        sub_force_square_error = [results[_][1] for _ in range(self.no_procs)]
        sub_der_variables_square_error = [results[_][2]
                                          for _ in range(self.no_procs)]

        _ = 0
        while _ < self.no_procs:
            energy_square_error += sub_energy_square_error[_]
            force_square_error += sub_force_square_error[_]
            count = 0
            while count < len_of_variables:
                der_variables_square_error[count] += \
                    sub_der_variables_square_error[_][count]
                count += 1
            _ += 1

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
    :param no_of_images: Total number of images.
    :type no_of_images: int
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
    :param io: utilities.IO class for reading/saving data.
    :type io: object
    :param data_format: Format of saved data. Can be either "json" or "db".
    :type data_format: str
    """
    ###########################################################################

    def __init__(self, cutoff, no_of_images, hashs, images, label, log,
                 train_forces, io, data_format):

        self.cutoff = cutoff
        self.images = images
        self.nl_data = {}

        if train_forces is True:
            new_images = images
            log('Calculating neighborlist for each atom...')
            log.tic()
            if data_format is 'json':
                filename = make_filename(label, 'neighborlists.json')
            elif data_format is 'db':
                filename = make_filename(label, 'neighborlists.db')
            if not os.path.exists(filename):
                log(' No saved neighborlist file found.')
            else:
                old_hashs, self.nl_data = io.read(filename,
                                                  'neighborlists',
                                                  self.nl_data,
                                                  data_format)
                log(' Saved neighborlist file %s loaded with %i entries.'
                    % (filename, len(old_hashs)))
                new_images = {}
                count = 0
                while count < no_of_images:
                    hash = hashs[count]
                    if hash not in old_hashs:
                        new_images[hash] = images[hash]
                    count += 1
                del old_hashs

            log(' Calculating %i of %i neighborlists.'
                % (len(new_images), len(images)))

            if len(new_images) != 0:
                new_hashs = sorted(new_images.keys())
                no_of_new_images = len(new_hashs)
                count = 0
                while count < no_of_new_images:
                    hash = new_hashs[count]
                    image = new_images[hash]
                    no_of_atoms = len(image)
                    self.nl_data[hash] = {}
                    nl = NeighborList(cutoffs=([self.cutoff / 2.] *
                                               no_of_atoms),
                                      self_interaction=False,
                                      bothways=True, skin=0.)
                    # FIXME: Is update necessary?
                    nl.update(image)
                    self_index = 0
                    while self_index < no_of_atoms:
                        neighbor_indices, neighbor_offsets = \
                            nl.get_neighbors(self_index)
                        if len(neighbor_offsets) == 0:
                            n_self_offsets = [[0, 0, 0]]
                        else:
                            n_self_offsets = \
                                np.vstack(([[0, 0, 0]], neighbor_offsets))
                        n_self_indices = np.append(self_index,
                                                   neighbor_indices)
                        self.nl_data[hash][self_index] = \
                            (n_self_indices, n_self_offsets,)
                        self_index += 1
                    count += 1

                if data_format is 'json':
                    filename = make_filename(label, 'neighborlists.json')
                elif data_format is 'db':
                    filename = make_filename(label, 'neighborlists.db')
                io.save(filename, 'neighborlists', self.nl_data, data_format)
                log(' ...neighborlists calculated and saved to %s.' %
                    filename, toc=True)

                del new_hashs
            del new_images
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
    :param no_of_images: Total number of images.
    :type no_of_images: int
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param label: Prefix name used for all files.
    :type label: str
    :param train_forces: Determining whether forces are also trained or not.
    :type train_forces: bool
    :param snl: SaveNeighborLists object.
    :type snl: object
    :param log: Write function at which to log data. Note this must be a
                callable function.
    :type log: Logger object
    :param _mp: MultiProcess object.
    :type _mp: object
    :param io: utilities.IO class for reading/saving data.
    :type io: object
    :param data_format: Format of saved data. Can be either "json" or "db".
    :type data_format: str
    """
    ###########################################################################

    def __init__(self, fp, elements, no_of_images, hashs, images, label,
                 train_forces, snl, log, _mp, io, data_format):

        self.Gs = fp.Gs
        self.train_forces = train_forces
        self.fp_data = {}
        self.der_fp_data = {}
        new_images = images

        log('Calculating atomic fingerprints...')
        log.tic()
        if data_format is 'json':
            filename = make_filename(label, 'fingerprints.json')
        elif data_format is 'db':
            filename = make_filename(label, 'fingerprints.db')
        if not os.path.exists(filename):
            log('No saved fingerprint file found.')
        else:
            log.tic('read_fps')
            log(' Reading fingerprints from %s...' % filename)
            old_hashs, self.fp_data = io.read(filename, 'fingerprints',
                                              self.fp_data, data_format)
            log(' ...fingerprints read', toc='read_fps')
            new_images = {}

            count = 0
            while count < no_of_images:
                hash = hashs[count]
                if hash not in old_hashs:
                    new_images[hash] = images[hash]
                count += 1
            log(' Calculating %i of %i fingerprints. (%i exist in file.)'
                % (len(new_images), len(images), len(old_hashs)))
            del old_hashs

        if len(new_images) != 0:
            log.tic('calculate_fps')
            new_hashs = sorted(new_images.keys())
            no_of_new_images = len(new_hashs)
            # new images are shared between cores for fingerprint calculations
            _mp.make_list_of_sub_images(no_of_new_images, new_hashs,
                                        new_images)
            del new_hashs

            if data_format is 'json':
                # Temporary files to hold child fingerprint calculations.
                childfiles = [tempfile.NamedTemporaryFile(prefix='fp-',
                                                          suffix='.json')
                              for _ in range(_mp.no_procs)]
                _ = 0
                while _ < _mp.no_procs:
                    log('  Processor %i calculations stored in file %s.'
                        % (_, childfiles[_].name))
                    _ += 1

            elif data_format is 'db':
                filename = make_filename(label, 'fingerprints.db')
                childfiles = [filename] * _mp.no_procs

            task_args = (fp, label, childfiles, io, data_format)
            _mp.share_fingerprints_task_between_cores(
                task=_calculate_fingerprints, _args=task_args)

            log(' Calculated %i new images.' % no_of_new_images,
                toc='calculate_fps')

            log.tic('read_fps')
            log(' Reading calculated fingerprints...')
            if data_format is 'json':
                for filename in childfiles:
                    _, self.fp_data = io.read(filename, 'fingerprints',
                                              self.fp_data, data_format)
            elif data_format is 'db':
                filename = make_filename(label, 'fingerprints.db')
                _, self.fp_data = io.read(filename, 'fingerprints',
                                          self.fp_data, data_format)
            log(' ...fingerprints read.', toc='read_fps')

            if data_format is 'json':
                log.tic('save_fps')
                log(' Saving fingerprints...')
                filename = make_filename(label, 'fingerprints.json')
                io.save(filename, 'fingerprints', self.fp_data, data_format)
                log(' ...fingerprints saved to %s.' % filename, toc='save_fps')

        fingerprint_values = {}
        for element in elements:
            fingerprint_values[element] = {}
            len_of_fingerprint = len(self.Gs[element])
            _ = 0
            while _ < len_of_fingerprint:
                fingerprint_values[element][_] = []
                _ += 1

        count = 0
        while count < no_of_images:
            hash = hashs[count]
            image = images[hash]
            no_of_atoms = len(image)
            index = 0
            while index < no_of_atoms:
                symbol = image[index].symbol
                len_of_fingerprint = len(self.Gs[symbol])
                _ = 0
                while _ < len_of_fingerprint:
                    fingerprint_values[symbol][_].append(
                        self.fp_data[hash][index][_])
                    _ += 1
                index += 1
            count += 1
        del _

        fingerprints_range = OrderedDict()
        for element in elements:
            fingerprints_range[element] = \
                [[min(fingerprint_values[element][_]),
                  max(fingerprint_values[element][_])]
                 for _ in range(len(self.Gs[element]))]

        self.fingerprints_range = fingerprints_range

        del new_images, fingerprint_values

        if train_forces is True:
            new_images = images
            log('Calculating derivatives of atomic fingerprints '
                'with respect to coordinates...')
            log.tic('fp_forces')
            if data_format is 'json':
                filename = make_filename(label,
                                         'fingerprint-derivatives.json')
            elif data_format is 'db':
                filename = make_filename(label,
                                         'fingerprint-derivatives.db')
            if not os.path.exists(filename):
                log('Either no saved fingerprint-derivatives file found '
                    'or it cannot be read.')
            else:
                log.tic('read_der_fps')
                log(' Reading fingerprint derivatives from file %s' %
                    filename)
                old_hashs, self.der_fp_data = \
                    io.read(filename, 'fingerprint_derivatives',
                            self.der_fp_data, data_format)
                log(' ...fingerprint derivatives read.',
                    toc='read_der_fps')

                new_images = {}
                count = 0
                while count < no_of_images:
                    hash = hashs[count]
                    if hash not in old_hashs:
                        new_images[hash] = images[hash]
                    count += 1

                log(' Calculating %i of %i fingerprint derivatives. '
                    '(%i exist in file.)'
                    % (len(new_images), len(images), len(old_hashs)))
                del old_hashs

            if len(new_images) != 0:
                log.tic('calculate_der_fps')
                new_hashs = sorted(new_images.keys())
                no_of_new_images = len(new_hashs)
                # new images are shared between cores for calculating
                # fingerprint derivatives
                _mp.make_list_of_sub_images(no_of_new_images, new_hashs,
                                            new_images)
                del new_hashs

                if data_format is 'json':
                    # Temporary files to hold child fingerprint calculations.
                    childfiles = [tempfile.NamedTemporaryFile(prefix='fp-',
                                                              suffix='.json')
                                  for _ in range(_mp.no_procs)]
                    _ = 0
                    while _ < _mp.no_procs:
                        log('  Processor %i calculations stored in file %s.'
                            % (_, childfiles[_].name))
                        _ += 1

                elif data_format is 'db':
                    filename = make_filename(label,
                                             'fingerprint-derivatives.db')
                    childfiles = [filename] * _mp.no_procs

                task_args = (fp, snl, label, childfiles, io, data_format)
                _mp.share_fingerprints_task_between_cores(
                    task=_calculate_der_fingerprints,
                    _args=task_args)

                log(' Calculated %i new images.' % no_of_new_images,
                    toc='calculate_der_fps')

                log.tic('read_der_fps')
                log(' Reading calculated fingerprint-derivatives...')
                if data_format is 'json':
                    for filename in childfiles:
                        _, self.der_fp_data = \
                            io.read(filename,
                                    'fingerprint_derivatives',
                                    self.der_fp_data,
                                    data_format)
                elif data_format is 'db':
                    filename = make_filename(label,
                                             'fingerprint-derivatives.db')
                    _, self.der_fp_data = io.read(filename,
                                                  'fingerprint_derivatives',
                                                  self.der_fp_data,
                                                  data_format)

                log(' ...fingerprint-derivatives are read.',
                    toc='read_der_fps')

                if data_format is 'json':
                    log.tic('save_der_fps')
                    filename = make_filename(label,
                                             'fingerprint-derivatives.json')
                    io.save(filename, 'fingerprint_derivatives',
                            self.der_fp_data, data_format)
                    log(''' ...fingerprint-derivatives calculated and saved to
                    %s.''' % filename, toc='save_der_fps')

            del new_images
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
    :param no_of_images: Number of images.
    :type no_of_images: int
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

    def __init__(self, reg, param, no_of_images, label, log, energy_goal,
                 force_goal, train_forces, _mp, overfitting_constraint,
                 force_coefficient, fortran, sfp=None, snl=None,):

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
        self.no_of_images = no_of_images

        if param.descriptor is not None:  # pure atomic-coordinates scheme
            self.cutoff = param.descriptor.cutoff

        self.energy_convergence = False
        self.force_convergence = False
        if not self.train_forces:
            self.force_convergence = True

        self.energy_coefficient = 1.0

        if fortran:
            # regression data is sent to fortran modules
            self.reg.send_data_to_fortran(param)

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
        len_of_variables = len(variables)

        if self.fortran:
            task_args = (self.param, calculate_gradient)
            (energy_square_error,
             force_square_error,
             self.der_variables_square_error) = \
                self._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_fortran,
                _args=task_args, len_of_variables=len_of_variables)
        else:
            task_args = (self.reg, self.param, self.sfp, self.snl,
                         self.energy_coefficient, self.force_coefficient,
                         self.train_forces, len_of_variables,
                         calculate_gradient)
            (energy_square_error,
             force_square_error,
             self.der_variables_square_error) = \
                self._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_python,
                _args=task_args, len_of_variables=len_of_variables)

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
            len_of_variables = len(variables)

            if self.fortran:
                task_args = (self.param, calculate_gradient)
                (energy_square_error,
                 force_square_error,
                 self.der_variables_square_error) = \
                    self._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_fortran,
                    _args=task_args, len_of_variables=len_of_variables)
            else:
                task_args = (self.reg, self.param, self.sfp, self.snl,
                             self.energy_coefficient, self.force_coefficient,
                             self.train_forces, len_of_variables,
                             calculate_gradient)
                (energy_square_error,
                 force_square_error,
                 self.der_variables_square_error) = \
                    self._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_python,
                    _args=task_args, len_of_variables=len_of_variables)

        der_cost_function = self.der_variables_square_error

        der_cost_function = np.array(der_cost_function)

        return der_cost_function

###############################################################################
###############################################################################
###############################################################################


def _calculate_fingerprints(proc_no, hashs, images, fp, label, childfiles, io,
                            data_format):
    """
    Function to be called on all processes simultaneously for calculating
    fingerprints.

    :param proc_no: Number of the process.
    :type proc_no: int
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param fp: Fingerprint object.
    :type fp: object
    :param label: Prefix name used for all files.
    :type label: str
    :param childfiles: Temporary files
    :type childfiles: file
    :param io: utilities.IO class for reading/saving data.
    :type io: object
    :param data_format: Format of saved data. Can be either "json" or "db".
    :type data_format: str
    """
    fingerprints = {}
    no_of_images = len(hashs)
    count = 0
    while count < no_of_images:
        hash = hashs[count]
        fingerprints[hash] = {}
        atoms = images[hash]
        no_of_atoms = len(atoms)
        fp.initialize(atoms)
        _nl = NeighborList(cutoffs=([fp.cutoff / 2.] * len(atoms)),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        _nl.update(atoms)
        index = 0
        while index < no_of_atoms:
            symbol = atoms[index].symbol
            n_indices, n_offsets = _nl.get_neighbors(index)
            # for calculating fingerprints, summation runs over neighboring
            # atoms of type I (either inside or outside the main cell)
            n_symbols = [atoms[n_index].symbol for n_index in n_indices]
            Rs = [atoms.positions[n_index] +
                  np.dot(n_offset, atoms.get_cell())
                  for n_index, n_offset in zip(n_indices, n_offsets)]
            indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
            fingerprints[hash][index] = indexfp
            index += 1
        count += 1

    io.save(childfiles[proc_no], 'fingerprints', fingerprints, data_format)

    del hashs, images

###############################################################################


def _calculate_der_fingerprints(proc_no, hashs, images, fp,
                                snl, label, childfiles, io, data_format):
    """
    Function to be called on all processes simultaneously for calculating
    derivatives of fingerprints.

    :param proc_no: Number of the process.
    :type proc_no: int
    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects (the training set).
    :type images: list
    :param fp: Fingerprint object.
    :type fp: object
    :param label: Prefix name used for all files.
    :type label: str
    :param childfiles: Temporary files
    :type childfiles: file
    :param io: utilities.IO class for reading/saving data.
    :type io: object
    :param data_format: Format of saved data. Can be either "json" or "db".
    :type data_format: str
    """
    data = {}

    no_of_images = len(hashs)
    count = 0
    while count < no_of_images:
        hash = hashs[count]
        data[hash] = {}
        atoms = images[hash]
        no_of_atoms = len(atoms)
        fp.initialize(atoms)
        _nl = NeighborList(cutoffs=([fp.cutoff / 2.] * no_of_atoms),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        _nl.update(atoms)
        self_index = 0
        while self_index < no_of_atoms:
            n_self_indices = snl.nl_data[hash][self_index][0]
            n_self_offsets = snl.nl_data[hash][self_index][1]
            n_symbols = [atoms[n_index].symbol for n_index in n_self_indices]
            len_of_n_self_indices = len(n_self_indices)
            n_count = 0
            while n_count < len_of_n_self_indices:
                n_symbol = n_symbols[n_count]
                n_index = n_self_indices[n_count]
                n_offset = n_self_offsets[n_count]
                # derivative of fingerprints are needed only with respect to
                # coordinates of atoms of type II (within the main cell only)
                if n_offset[0] == 0 and n_offset[1] == 0 and n_offset[2] == 0:
                    i = 0
                    while i < 3:
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
                        i += 1
                n_count += 1
            self_index += 1
        count += 1

    io.save(childfiles[proc_no], 'fingerprint_derivatives', data, data_format)

    del hashs, images, data

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


def _calculate_cost_function_python(hashs, images, reg, param, sfp,
                                    snl, energy_coefficient,
                                    force_coefficient, train_forces,
                                    len_of_variables, calculate_gradient,
                                    queue):
    """
    Function to be called on all processes simultaneously for calculating cost
    function and its derivative with respect to variables in python.

    :param hashs: Unique keys, one key per image.
    :type hashs: list
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

    no_of_images = len(hashs)
    count0 = 0
    while count0 < no_of_images:
        hash = hashs[count0]
        atoms = images[hash]
        no_of_atoms = len(atoms)
        real_energy = atoms.get_potential_energy(apply_constraint=False)
        real_forces = atoms.get_forces(apply_constraint=False)

        reg.reset_energy()
        amp_energy = 0.

        if param.descriptor is None:  # pure atomic-coordinates scheme

            input = (atoms.positions).ravel()
            amp_energy = reg.get_energy(input,)

        else:  # fingerprinting scheme
            index = 0
            while index < no_of_atoms:
                symbol = atoms[index].symbol
                indexfp = sfp.fp_data[hash][index]
                # fingerprints are scaled to [-1, 1] range
                scaled_indexfp = [None] * len(indexfp)
                count = 0
                while count < len(indexfp):
                    if (sfp.fingerprints_range[symbol][count][1] -
                            sfp.fingerprints_range[symbol][count][0]) > \
                            (10.**(-8.)):
                        scaled_value = \
                            -1. + 2. * (indexfp[count] -
                                        sfp.fingerprints_range[
                                        symbol][count][0]) \
                            / (sfp.fingerprints_range[symbol][count][1] -
                               sfp.fingerprints_range[symbol][count][0])
                    else:
                        scaled_value = indexfp[count]
                    scaled_indexfp[count] = scaled_value
                    count += 1
                atomic_amp_energy = reg.get_energy(scaled_indexfp,
                                                   index, symbol,)
                amp_energy += atomic_amp_energy
                index += 1

        energy_square_error += ((amp_energy - real_energy) ** 2.) / \
            (no_of_atoms ** 2.)

        if calculate_gradient:

            if param.descriptor is None:  # pure atomic-coordinates scheme

                partial_der_variables_square_error = \
                    reg.get_variable_der_of_energy()
                der_variables_square_error += \
                    energy_coefficient * 2. * (amp_energy - real_energy) * \
                    partial_der_variables_square_error / (no_of_atoms ** 2.)

            else:  # fingerprinting scheme

                index = 0
                while index < no_of_atoms:
                    symbol = atoms[index].symbol
                    partial_der_variables_square_error =\
                        reg.get_variable_der_of_energy(index, symbol)
                    der_variables_square_error += \
                        energy_coefficient * 2. * (amp_energy - real_energy) \
                        * partial_der_variables_square_error / \
                        (no_of_atoms ** 2.)
                    index += 1

        if train_forces is True:

            real_forces = atoms.get_forces(apply_constraint=False)
            amp_forces = np.zeros((no_of_atoms, 3))
            self_index = 0
            while self_index < no_of_atoms:
                reg.reset_forces()

                if param.descriptor is None:  # pure atomic-coordinates scheme

                    i = 0
                    while i < 3:
                        _input = [0.] * (3 * no_of_atoms)
                        _input[3 * self_index + i] = 1.
                        force = reg.get_force(i, _input,)
                        amp_forces[self_index][i] = force
                        i += 1

                else:  # fingerprinting scheme

                    n_self_indices = snl.nl_data[hash][self_index][0]
                    n_self_offsets = snl.nl_data[hash][self_index][1]
                    n_symbols = [atoms[n_index].symbol
                                 for n_index in n_self_indices]
                    len_of_n_self_indices = len(n_self_indices)
                    n_count = 0
                    while n_count < len_of_n_self_indices:
                        n_symbol = n_symbols[n_count]
                        n_index = n_self_indices[n_count]
                        n_offset = n_self_offsets[n_count]
                        # for calculating forces, summation runs over neighbor
                        # atoms of type II (within the main cell only)
                        if n_offset[0] == 0 and n_offset[1] == 0 and \
                                n_offset[2] == 0:
                            i = 0
                            while i < 3:
                                der_indexfp = \
                                    sfp.der_fp_data[hash][(n_index,
                                                           self_index,
                                                           i)]
                                len_of_der_indexfp = len(der_indexfp)
                                # fingerprint derivatives are scaled
                                scaled_der_indexfp = \
                                    [None] * len_of_der_indexfp
                                count = 0
                                while count < len_of_der_indexfp:
                                    if (sfp.fingerprints_range[
                                            n_symbol][count][1] -
                                            sfp.fingerprints_range
                                            [n_symbol][count][0]) > \
                                            (10.**(-8.)):
                                        scaled_value = 2. * \
                                            der_indexfp[count] / \
                                            (sfp.fingerprints_range
                                             [n_symbol][count][1] -
                                             sfp.fingerprints_range
                                             [n_symbol][count][0])
                                    else:
                                        scaled_value = der_indexfp[count]
                                    scaled_der_indexfp[count] = scaled_value
                                    count += 1

                                force = reg.get_force(i, scaled_der_indexfp,
                                                      n_index, n_symbol,)

                                amp_forces[self_index][i] += force
                                i += 1
                        n_count += 1

                i = 0
                while i < 3:
                    force_square_error += \
                        ((1.0 / 3.0) * (amp_forces[self_index][i] -
                                        real_forces[self_index][i]) **
                         2.) / no_of_atoms
                    i += 1

                if calculate_gradient:

                    i = 0
                    while i < 3:
                        # pure atomic-coordinates scheme
                        if param.descriptor is None:

                            partial_der_variables_square_error = \
                                reg.get_variable_der_of_forces(self_index, i)
                            der_variables_square_error += \
                                force_coefficient * (2.0 / 3.0) * \
                                (- amp_forces[self_index][i] +
                                 real_forces[self_index][i]) * \
                                partial_der_variables_square_error \
                                / no_of_atoms

                        else:  # fingerprinting scheme

                            n_count = 0
                            while n_count < len_of_n_self_indices:
                                n_symbol = n_symbols[n_count]
                                n_index = n_self_indices[n_count]
                                n_offset = n_self_offsets[n_count]
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
                                        / no_of_atoms
                                n_count += 1
                        i += 1
                self_index += 1
        count0 += 1

    del hashs, images

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
        i = 0
        len_of_fingerprints = len(fp.Gs[element])
        while i < len_of_fingerprints:
            fingerprint_values[element][i] = []
            i += 1

    no_of_atoms = len(atoms)
    index = 0
    while index < no_of_atoms:
        symbol = atoms[index].symbol
        n_indices, n_offsets = nl.get_neighbors(index)
        # for calculating fingerprints, summation runs over neighboring
        # atoms of type I (either inside or outside the main cell)
        n_symbols = [atoms[n_index].symbol for n_index in n_indices]
        Rs = [atoms.positions[n_index] +
              np.dot(n_offset, atoms.get_cell())
              for n_index, n_offset in zip(n_indices, n_offsets)]
        indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
        len_of_fingerprints = len(fp.Gs[symbol])
        _ = 0
        while _ < len_of_fingerprints:
            fingerprint_values[symbol][_].append(indexfp[_])
            _ += 1
        index += 1

    fingerprints_range = {}
    for element in elements:
        len_of_fingerprints = len(fp.Gs[element])
        fingerprints_range[element] = [None] * len_of_fingerprints
        count = 0
        while count < len_of_fingerprints:
            if len(fingerprint_values[element][count]) > 0:
                minimum = min(fingerprint_values[element][count])
                maximum = max(fingerprint_values[element][count])
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

    no_of_atoms = len(atoms)
    index = 0
    while index < no_of_atoms:
        symbol = atoms[index].symbol
        n_indices, n_offsets = nl.get_neighbors(index)
        # for calculating fingerprints, summation runs over neighboring
        # atoms of type I (either inside or outside the main cell)
        n_symbols = [atoms[n_index].symbol for n_index in n_indices]
        Rs = [atoms.positions[n_index] +
              np.dot(n_offset, atoms.get_cell())
              for n_index, n_offset in zip(n_indices, n_offsets)]
        indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
        len_of_fingerprints = len(indexfp)
        i = 0
        while i < len_of_fingerprints:
            if indexfp[i] < fingerprints_range[symbol][i][0] or \
                    indexfp[i] > fingerprints_range[symbol][i][1]:
                compare_train_test_fingerprints = 1
                break
            i += 1
        index += 1
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
            images = aseio.Trajectory(images, 'r')
        elif extension == '.db':
            images = aseio.read(images)

    # Images is converted to dictionary form; key is hash of image.
    dict_images = {}
    no_of_images = len(images)
    count = 0
    while count < no_of_images:
        image = images[count]
        hash = hash_image(image)
        dict_images[hash] = image
        count += 1
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
    hashs = images.keys()
    count = 0
    while count < no_of_images:
        hash = hashs[count]
        atoms = images[hash]
        fp.atoms = atoms
        _nl = NeighborList(cutoffs=([cutoff / 2.] * len(atoms)),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        _nl.update(atoms)
        fp._nl = _nl
        compare_train_test_fingerprints = 0
        no_of_atoms = len(atoms)
        index = 0
        while index < no_of_atoms:
            symbol = atoms[index].symbol
            n_indices, n_offsets = _nl.get_neighbors(index)
            # for calculating fingerprints, summation runs over neighboring
            # atoms of type I (either inside or outside the main cell)
            n_symbols = [atoms[n_index].symbol for n_index in n_indices]
            Rs = [atoms.positions[n_index] +
                  np.dot(n_offset, atoms.get_cell())
                  for n_index, n_offset in zip(n_indices, n_offsets)]
            indexfp = fp.get_fingerprint(index, symbol, n_symbols, Rs)
            len_of_fingerprints = len(indexfp)
            i = 0
            while i < len_of_fingerprints:
                if indexfp[i] < fingerprints_range[symbol][i][0] or \
                        indexfp[i] > fingerprints_range[symbol][i][1]:
                    compare_train_test_fingerprints = 1
                    break
                i += 1
            index += 1
        if compare_train_test_fingerprints == 0:
            interpolated_images[hash] = image
        count += 1

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

    raveled_fingerprints = [sfp.fp_data[hash][index]
                            for hash in hashs
                            for index in range(len(images[hash]))]

    del hashs, images

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
    no_of_images = len(hashs)
    _ = 0
    while _ < no_of_images:
        hash = hashs[_]
        atoms = images[hash]
        no_of_atoms = len(atoms)
        self_index = 0
        while self_index < no_of_atoms:
            n_self_offsets = snl.nl_data[hash][self_index][1]
            len_of_n_self_offsets = len(n_self_offsets)
            count = 0
            n_count = 0
            while n_count < len_of_n_self_offsets:
                n_offset = n_self_offsets[n_count]
                if n_offset[0] == 0 and n_offset[1] == 0 and n_offset[2] == 0:
                    count += 1
                del n_offset
                n_count += 1
            list_of_no_of_neighbors.append(count)
            self_index += 1
        del hash
        _ += 1

    raveled_neighborlists = [n_index for hash in hashs
                             for self_atom in images[hash]
                             for n_index, n_offset in
                             zip(snl.nl_data[hash][self_atom.index][0],
                                 snl.nl_data[hash][self_atom.index][1])
                             if (n_offset[0] == 0 and n_offset[1] == 0 and
                                 n_offset[2] == 0)]

    raveled_der_fingerprints = \
        [sfp.der_fp_data[hash][(n_index, self_atom.index, i)]
         for hash in hashs
         for self_atom in images[hash]
         for n_index, n_offset in
         zip(snl.nl_data[hash][self_atom.index][0],
             snl.nl_data[hash][self_atom.index][1])
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
