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
import io
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
from fingerprint import *
from regression import *
try:
    from amp import fmodules  # version 1 of fmodules
    fmodules_version = 1
except ImportError:
    fmodules = None

###############################################################################


class AMP(Calculator):

    """
    Atomistic Machine-Learning Potential (AMP) ASE calculator
    Inputs:
        fingerprint: class representing local atomic environment. Can be only
                     Behler for now. Input arguments for Behler are cutoff, Gs,
                     and fingerprints_range; for more information see
                     docstring for the class Behler.
        regression: class representing the regression method. Can be only
                    NeuralNetwork for now. Input arguments for NeuralNetwork
                    are hiddenlayers, activation, weights, and scalings; for
                    more information see docstring for the class NeuralNetwork.
        fingerprints_range: range of fingerprints of each chemical species.
                            Should be fed as a dictionary of chemical species
                            and a list of minimum and maximun, e.g.
                            fingerprints_range={"Pd": [0.31, 0.59],
                                "O":[0.56, 0.72]}
        load: string
            load an existing (trained) AMP calculator from this path.
        label: string
            default prefix / location used for all files.
        dblabel: string
            optional separate prefix (location) for database files,
            including fingerprints, fingerprint derivatives, and
            neighborlists. This file location can be shared between
            calculator instances to avoid re-calculating redundant
            information. If not supplied, just uses the value from
            label.
        extrapolate: boolean
            If True, allows for extrapolation, if False, does not allow.
        fortran: boolean
            If True, will use the fortran subroutines, else will not.
    Output:
        energy: float
        forces: matrix of floats

    """

    implemented_properties = ['energy', 'forces']

    default_parameters = {
        'fingerprint': Behler(),
        'regression': NeuralNetwork(),
        'fingerprints_range': None,
    }

    #########################################################################

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
            if parameters['fingerprint'] == 'Behler':
                kwargs['fingerprint'] = \
                    Behler(cutoff=parameters['cutoff'],
                           Gs=parameters['Gs'],
                           fingerprints_tag=parameters['fingerprints_tag'],
                           fortran=fortran,)
            elif parameters['fingerprint'] == 'None':
                kwargs['fingerprint'] = None
                if parameters['no_of_atoms'] == 'None':
                    parameters['no_of_atoms'] = None
            else:
                raise RuntimeError('Fingerprinting scheme is not recognized '
                                   'to AMP for loading parameters. User '
                                   'should add the fingerprinting scheme '
                                   'under consideration.')

            if parameters['regression'] == 'NeuralNetwork':
                kwargs['regression'] = \
                    NeuralNetwork(hiddenlayers=parameters['hiddenlayers'],
                                  activation=parameters['activation'],
                                  variables=parameters['variables'],)
                if kwargs['fingerprint'] is None:
                    kwargs['no_of_atoms'] = parameters['no_of_atoms']
            else:
                raise RuntimeError('Regression method is not recognized to '
                                   'AMP for loading parameters. User should '
                                   'add the regression method under '
                                   'consideration.')

        Calculator.__init__(self, label=label, **kwargs)

        param = self.parameters

        if param.fingerprint is not None:
            self.fp = param.fingerprint

        self.reg = param.regression

        self.reg.initialize(param, load)

    #########################################################################

    def set(self, **kwargs):
        """Function to set parameters."""
        changed_parameters = Calculator.set(self, **kwargs)
        # FIXME. Decide whether to call reset. Decide if this is
        # meaningful in our implementation!
        if len(changed_parameters) > 0:
            self.reset()

    #########################################################################

    def set_label(self, label):
        """Sets label, ensuring that any needed directories are made."""

        Calculator.set_label(self, label)

        # Create directories for output structure if needed.
        if self.label:
            if (self.directory != os.curdir and
                    not os.path.isdir(self.directory)):
                os.makedirs(self.directory)

    #########################################################################

    def initialize(self, atoms):
        self.par = {}
        self.rc = 0.0
        self.numbers = atoms.get_atomic_numbers()
        self.forces = np.empty((len(atoms), 3))
        self.nl = NeighborList([0.5 * self.rc + 0.25] * len(atoms),
                               self_interaction=False)

    #########################################################################

    def calculate(self, atoms, properties, system_changes):
        """Calculation of the energy of the system and forces of all atoms."""
        Calculator.calculate(self, atoms, properties, system_changes)

        self.atoms = atoms
        param = self.parameters
        if param.fingerprint is None:  # pure atomic-coordinates scheme
            self.reg.initialize(param=param,
                                atoms=self.atoms)
        param = self.reg.ravel_variables()

        if param.regression.variables is None:
            raise RuntimeError("Calculator not trained; can't return "
                               'properties.')

        if 'numbers' in system_changes:
            self.initialize(atoms)

        self.nl.update(atoms)

        if param.fingerprint is not None:  # fingerprinting scheme
            self.cutoff = param.fingerprint.cutoff

            # FIXME: What is the difference between the two updates on the top
            # and bottom? Is the one on the top necessary? Where is self.nl
            #  coming from?
            self.update()

            self.fp.atoms = atoms
            self.fp._nl = self._nl

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
                                                 self._nl)
            # Deciding on whether it is exptrapoling or interpolating is
            # possible only when fingerprints_range is provided by the user.
            elif self.extrapolate is False:
                if compare_train_test_fingerprints(
                        self.fp,
                        self.fp.atoms,
                        param.fingerprints_range,
                        self._nl) == 1:
                    raise ExtrapolateError('Trying to extrapolate, which'
                                           ' is not allowed. Change to '
                                           'extrapolate=True if this is'
                                           ' desired.')

    ##################################################################

        if properties == ['energy']:

            self.reg.reset_energy()
            self.energy = 0.0

            if param.fingerprint is None:  # pure atomic-coordinates scheme

                input = (atoms.positions).ravel()
                self.energy = self.reg.get_energy(input,)

            else:  # fingerprinting scheme

                for atom in atoms:
                    index = atom.index
                    symbol = atom.symbol
                    n_indices, n_offsets = self._nl.get_neighbors(index)
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

            if param.fingerprint is None:  # pure atomic-coordinates scheme

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
                        self._nl.get_neighbors(self_index)
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
                    n_indices, n_offsets = self._nl.get_neighbors(index)
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
                                    self._nl.get_neighbors(n_index)
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

    #########################################################################

    def update(self):
        """Update the neighborlist for making fingerprint. Use if atoms
        position has changed."""

        self._nl = NeighborList(cutoffs=([self.cutoff / 2.] *
                                         len(self.atoms)),
                                self_interaction=False,
                                bothways=True,
                                skin=0.)
        self._nl.update(self.atoms)

    #########################################################################

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
            overwrite=False,):
        """Fits a variable set to the data, by default using the "fmin_bfgs"
        optimizer. The optimizer takes as input a cost function to reduce and
        an initial guess of variables and returns an optimized variable set.
        Inputs:
            images: list of ASE atoms objects
                    List of objects with positions, symbols, energies, and
                    forces in ASE format. This is the training set of data.
                    This can also be the path to an ASE trajectory (.traj)
                    or database (.db) file.  Energies can be obtained from
                    any reference, e.g. DFT calculations.
            energy_goal: threshold rms(energy per atom) error at which
                    simulation is converged.
                    The default value is in unit of eV.
            force_goal: threshold rmse per atom for forces at which simulation
                    is converged.  The default value is in unit of eV/Ang. If
                    'force_goal = None', forces will not be trained.
            overfitting_constraint: the constant to suppress overfitting.
                    A proper value for this constant is subtle and depends on
                    the train data set; a small value may not cure the
                    over-fitting issue, whereas a large value may cause
                    over-smoothness.
            force_coefficient: multiplier of force square error in
                    constructing the cost function. This controls how
                    significant force-fit is as compared to energy fit in
                    approaching convergence. It depends on force and energy
                    units. If not specified, guesses the value such that
                    energy and force contribute equally to the cost
                    function when they hit their converged values.
            cores: (int) number of cores to parallelize over
                    If not specified, attempts to determine from environment.
            optimizer: function
                    The optimization object. The default is to use scipy's
                    fmin_bfgs, but any optimizer that behaves in the same
                    way will do.
            read_fingerprints: (bool)
                    Determines whether or not the code should read fingerprints
                    already calculated and saved in the script directory.
            overwrite: (bool)
                    If a trained output file with the same name exists,
                    overwrite it.
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

        log('AMP training started. ' + now() + '\n')
        if param.fingerprint is None:  # pure atomic-coordinates scheme
            log('Local environment descriptor: None')
        else:  # fingerprinting scheme
            log('Local environment descriptor: ' +
                param.fingerprint.__class__.__name__)
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

        if param.fingerprint is None:  # pure atomic-coordinates scheme
            param.no_of_atoms = len(images[0])
            for image in images:
                if len(image) != param.no_of_atoms:
                    raise RuntimeError('Number of atoms in different images '
                                       'is not the same. Try '
                                       'fingerprint=Behler.')

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
        hash_keys = sorted(images.keys())
        log(' %i unique images after hashing.' % len(hash_keys))
        log(' ...hashing completed.', toc=True)

        self.elements = set([atom.symbol for hash_key in hash_keys
                             for atom in images[hash_key]])
        self.elements = sorted(self.elements)

        msg = '%i unique elements included: ' % len(self.elements)
        msg += ', '.join(self.elements)
        log(msg)

        if param.fingerprint is not None:  # fingerprinting scheme
            param = self.fp.log(log, param, self.elements)
        param = self.reg.log(log, param, self.elements, images)

        # "MultiProcess" object is initialized
        _mp = MultiProcess(self.fortran, no_procs=cores)

        if param.fingerprint is None:  # pure atomic-coordinates scheme
            self.sfp = None
            snl = None
        else:  # fingerprinting scheme
            # Neighborlist for all images are calculated and saved
            log.tic()
            snl = SaveNeighborLists(param.fingerprint.cutoff, hash_keys,
                                    images, self.dblabel, log, train_forces,
                                    read_fingerprints)

            gc.collect()

            # Fingerprints are calculated and saved
            self.sfp = SaveFingerprints(
                self.fp,
                self.elements,
                hash_keys,
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
            hash_keys,
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

        del hash_keys, images

        gc.collect()

        # saving initial parameters
        filename = make_filename(self.label, 'initial-parameters.json')
        save_parameters(filename, param)
        log('Initial parameters saved in file %s.' % filename)

        log('Starting optimization of cost function...')
        log(' Energy goal: %.3e' % energy_goal)
        if train_forces:
            log(' Force goal: %.3e' % force_goal)
            log(' Cost function force coefficient: %f' % force_coefficient)
        else:
            log(' No force training.')
        log.tic()
        converged = False

        variables = param.regression.variables

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
                toc=True)
            raise TrainingConvergenceError()

        param.regression.variables = costfxn.param.regression.variables
        self.reg.update_variables(param)

        log(' ...optimization completed successfully. Optimal '
            'parameters saved.', toc=True)
        filename = make_filename(self.label, 'trained-parameters.json')
        save_parameters(filename, param)

        self.cost_function = costfxn.cost_function
        self.energy_per_atom_rmse = costfxn.energy_per_atom_rmse
        self.force_rmse = costfxn.force_rmse
        self.der_variables_cost_function = costfxn.der_variables_square_error

###############################################################################
###############################################################################
###############################################################################


class MultiProcess:

    """Class to do parallel processing, using multiprocessing package which
    works on Python versions 2.6 and above.
    Inputs:
            fortran: boolean
                If True, will use the fortran subroutines, else won't.
            no_procs: number of processors
            """

    def __init__(self, fortran, no_procs):

        self.fortran = fortran
        self.no_procs = no_procs
        self.queues = {}
        for x in range(no_procs):
            self.queues[x] = mp.Queue()

    ##########################################################################

    def make_list_of_sub_images(self, hash_keys, images):
        """Two lists are made each with one entry per core. The entry of the
        first list contains list of hashes to be calculated by that core,
        and the entry of the second list contains dictionary of images to be
        calculated by that core."""

        quotient = int(len(hash_keys) / self.no_procs)
        remainder = len(hash_keys) - self.no_procs * quotient
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
                hash = hash_keys[count1]
                sub_hashes[count2] = hash
                sub_images[hash] = images[hash]
                count1 += 1
                count2 += 1
            list_sub_hashes[count0] = sub_hashes
            list_sub_images[count0] = sub_images
            count0 += 1

        self.list_sub_hashes = list_sub_hashes
        self.list_sub_images = list_sub_images

        del hash_keys, images, list_sub_hashes, list_sub_images, sub_hashes,
        sub_images

    ##########################################################################

    def share_fingerprints_task_between_cores(self, task, _args):
        """Fingerprints tasks are sent to cores for parallel processing"""

        args = {}
        for x in range(self.no_procs):
            sub_hash_keys = self.list_sub_hashes[x]
            sub_images = self.list_sub_images[x]
            args[x] = (x,) + (sub_hash_keys, sub_images,) + _args

        processes = [mp.Process(target=task, args=args[x])
                     for x in range(self.no_procs)]

        for x in range(self.no_procs):
            processes[x].start()

        for x in range(self.no_procs):
            processes[x].join()

        for x in range(self.no_procs):
            processes[x].terminate()

        del sub_hash_keys, sub_images

    ##########################################################################

    def share_cost_function_task_between_cores(self, task, _args,
                                               len_of_variables):
        """Derivatives of the cost function with respect to variables are
        calculated in parallel"""

        args = {}
        for x in range(self.no_procs):
            if self.fortran:
                args[x] = (x + 1,) + _args + (self.queues[x],)
            else:
                sub_hash_keys = self.list_sub_hashes[x]
                sub_images = self.list_sub_images[x]
                args[x] = (sub_hash_keys, sub_images,) + _args + \
                    (self.queues[x],)

        energy_square_error = 0.
        force_square_error = 0.

        der_variables_square_error = [0.] * len_of_variables

        processes = [mp.Process(target=task, args=args[x])
                     for x in range(self.no_procs)]

        for x in range(self.no_procs):
            processes[x].start()

        for x in range(self.no_procs):
            processes[x].join()

        for x in range(self.no_procs):
            processes[x].terminate()

        sub_energy_square_error = []
        sub_force_square_error = []
        sub_der_variables_square_error = []

#       Construct total square_error and derivative with respect to variables
#       from subprocesses
        results = {}
        for x in range(self.no_procs):
            results[x] = self.queues[x].get()

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
            del sub_hash_keys, sub_images

        return (energy_square_error, force_square_error,
                der_variables_square_error)

###############################################################################
###############################################################################
###############################################################################


class SaveNeighborLists:

    """Neighborlists for all images with the given cutoff value are calculated
    and saved. As well as neighboring atomic indices, neighboring atomic
    offsets from the main cell are also saved. Offsets become important when
    dealing with periodic systems. Neighborlists are generally of two types:
    Type I which consists of atoms within the cutoff distance either in the
    main cell or in the adjacent cells, and Type II which consists of atoms in
    the main cell only and within the cutoff distance.

        cutoff: cutoff radius, in Angstroms, around each atom

        hash_keys: unique keys for each of "images"

        images: list of ASE atoms objects (the training set)

        label: name used for all files.

        log: write function at which to log data. Note this must be a
           callable function

        train_forces:  boolean to representing whether forces are also trained
                        or not

        read_fingerprints: boolean to determines whether or not the code should
                            read fingerprints already calculated and saved in
                            the script directory.

    """

    ##########################################################################

    def __init__(self, cutoff, hash_keys, images, label, log, train_forces,
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
            for hash_key in hash_keys:
                if hash_key not in old_hashes:
                    new_images[hash_key] = images[hash_key]

            log(' Calculating %i of %i neighborlists.'
                % (len(new_images), len(images)))

            if len(new_images) != 0:
                new_hashs = sorted(new_images.keys())
                for hash_key in new_hashs:
                    image = new_images[hash_key]
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
                        self.nl_data[(hash_key, self_index)] = \
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

    """Memory class to not recalculate fingerprints and their derivatives if
    not necessary. This could cause runaway memory usage; use with caution.

        fp: fingerprint object

        elements: list if elements in images

        hash_keys: unique keys, one key per image

        images: list of ASE atoms objects (the training set)

        label: name used for all files

        train_forces:  boolean representing whether forces are also trained or
                        not

        read_fingerprints: boolean to determines whether or not the code should
                            read fingerprints already calculated and saved in
                            the script directory.

        snl: object of the class SaveNeighborLists

        log: write function at which to log data. Note this must be a
           callable function.

        _mp: "MultiProcess" object

    """

    ##########################################################################

    def __init__(self,
                 fp,
                 elements,
                 hash_keys,
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
        for hash_key in hash_keys:
            if hash_key not in old_hashes:
                new_images[hash_key] = images[hash_key]

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

        for hash_key in hash_keys:
            image = images[hash_key]
            for atom in image:
                for _ in range(len(self.Gs[atom.symbol])):
                    fingerprint_values[atom.symbol][_].append(
                        dict_data[hash_key][atom.index][_])

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
            for hash_key in hash_keys:
                if hash_key not in old_hashes:
                    new_images[hash_key] = images[hash_key]

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

    """Cost function and its derivative based on squared difference in energy,
    to be optimized in setting variables.

        reg: regression object

        param: dictionary that contains cutoff and variables

        hash_keys: unique keys, one key per image

        images: list of ASE atoms objects (the training set)

        label: name used for all files

        log: write function at which to log data. Note this must be a
           callable function.

        energy_goal: threshold rms(error per atom) at which simulation is
        converged

        force_goal: threshold rmse/atom at which simulation is converged

        train_forces:  boolean representing whether forces are also trained or
        not

        _mp: object of "MultiProcess" class

        overfitting_constraint: the constant to constraint overfitting

        force_coefficient: multiplier of force RMSE in constructing the cost
        function. This controls how tight force-fit is as compared to
        energy fit. It also depends on force and energy units. Working with
        eV and Angstrom, 0.04 seems to be a reasonable value.

        fortran: boolean
            If True, will use the fortran subroutines, else will not.

        sfp: object of the class SaveFingerprints, which contains all
           fingerprints

        snl: object of the class SaveNeighborLists

    """
    #########################################################################

    def __init__(self, reg, param, hash_keys, images, label, log,
                 energy_goal, force_goal, train_forces, _mp,
                 overfitting_constraint, force_coefficient, fortran, sfp=None,
                 snl=None,):

        self.reg = reg
        self.param = param
        self.hash_keys = hash_keys
        self.images = images
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

        if param.fingerprint is not None:  # pure atomic-coordinates scheme
            self.cutoff = param.fingerprint.cutoff

        self.energy_convergence = False
        self.force_convergence = False
        if not self.train_forces:
            self.force_convergence = True

        self.energy_coefficient = 1.0

        self.no_of_images = len(self.hash_keys)

        # all images are shared between cores for feed-forward and
        # back-propagation calculations
        self._mp.make_list_of_sub_images(self.hash_keys, self.images)

        if self.fortran:
            no_procs = self._mp.no_procs
            no_sub_images = [len(self._mp.list_sub_images[i])
                             for i in range(no_procs)]
            self.reg.send_data_to_fortran(param)
            send_data_to_fortran(self.hash_keys, self.images,
                                 self.sfp, self.snl,
                                 self.reg.elements,
                                 self.train_forces,
                                 self.energy_coefficient,
                                 self.force_coefficient,
                                 log,
                                 param,
                                 no_procs,
                                 no_sub_images)
            del self._mp.list_sub_images, self._mp.list_sub_hashes

        del self.hash_keys, hash_keys, self.images, images

    #########################################################################

    def f(self, variables):
        """function to calculate the cost function"""

        log = self.log
        self.param.regression.variables = variables

        if self.fortran:
            task_args = (self.param,)
            (energy_square_error,
             force_square_error,
             self.der_variables_square_error) = \
                self._mp.share_cost_function_task_between_cores(
                task=_calculate_cost_function_fortran,
                _args=task_args, len_of_variables=len(variables))
        else:
            task_args = (self.reg, self.param, self.sfp, self.snl,
                         self.energy_coefficient, self.force_coefficient,
                         self.train_forces, len(variables))
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

    #########################################################################

    def fprime(self, variables):
        """function to calculate derivative of the cost function"""

        if self.steps == 0:

            self.param.regression.variables = variables

            if self.fortran:
                task_args = (self.param,)
                (energy_square_error,
                 force_square_error,
                 self.der_variables_square_error) = \
                    self._mp.share_cost_function_task_between_cores(
                    task=_calculate_cost_function_fortran,
                    _args=task_args, len_of_variables=len(variables))
            else:
                task_args = (self.reg, self.param, self.sfp, self.snl,
                             self.energy_coefficient, self.force_coefficient,
                             self.train_forces, len(variables))
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


class ConvergenceOccurred(Exception):

    """Kludge to decide when scipy's optimizers are complete."""
    pass

###############################################################################
###############################################################################
###############################################################################


class TrainingConvergenceError(Exception):

    """Error to be raise if training does not converge."""
    pass

###############################################################################
###############################################################################
###############################################################################


class ExtrapolateError(Exception):

    """Error class in the case of extrapolation."""
    pass

###############################################################################
###############################################################################
###############################################################################


def _calculate_fingerprints(proc_no,
                            hashes,
                            images,
                            fp,
                            label,
                            childfiles):
    """wrapper function to be used in multiprocessing for calculating
    fingerprints."""

    fingerprints = {}
    for hash_key in hashes:
        fingerprints[hash_key] = {}
        atoms = images[hash_key]
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
            fingerprints[hash_key][index] = indexfp

    save_fingerprints(f=childfiles[proc_no], fingerprints=fingerprints)

    del hashes, images

###############################################################################


def _calculate_der_fingerprints(proc_no, hashes, images, fp,
                                snl, label, childfiles):
    """wrapper function to be used in multiprocessing for calculating
    derivatives of fingerprints."""

    data = {}
    for hash_key in hashes:
        data[hash_key] = {}
        atoms = images[hash_key]
        fp.initialize(atoms)
        _nl = NeighborList(cutoffs=([fp.cutoff / 2.] *
                                    len(atoms)),
                           self_interaction=False,
                           bothways=True,
                           skin=0.)
        _nl.update(atoms)
        for self_atom in atoms:
            self_index = self_atom.index
            n_self_indices = snl.nl_data[(hash_key, self_index)][0]
            n_self_offsets = snl.nl_data[(hash_key, self_index)][1]
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

                        data[hash_key][(n_index, self_index, i)] = der_indexfp

    save_der_fingerprints(childfiles[proc_no], data)

    del hashes, images, data

###############################################################################


def _calculate_cost_function_fortran(proc_no, param, queue):
    """wrapper function to be used in multiprocessing for calculating
    cost function and it's derivative with respect to variables"""

    variables = param.regression.variables

    (energy_square_error,
     force_square_error,
     der_variables_square_error) = \
        fmodules.share_cost_function_task_between_cores(proc=proc_no,
                                                        variables=variables,
                                                        len_of_variables=len(
                                                            variables))

    queue.put([energy_square_error,
               force_square_error,
               der_variables_square_error])

###############################################################################


def _calculate_cost_function_python(hashes, images, reg, param, sfp,
                                    snl, energy_coefficient,
                                    force_coefficient, train_forces,
                                    len_of_variables, queue):
    """wrapper function to be used in multiprocessing for calculating
    cost function and it's derivative with respect to variables."""

    der_variables_square_error = np.zeros(len_of_variables)

    energy_square_error = 0.
    force_square_error = 0.

    reg.update_variables(param)

    for hash_key in hashes:
        atoms = images[hash_key]
        real_energy = atoms.get_potential_energy(apply_constraint=False)
        real_forces = atoms.get_forces(apply_constraint=False)

        reg.reset_energy()
        amp_energy = 0.

        if param.fingerprint is None:  # pure atomic-coordinates scheme

            input = (atoms.positions).ravel()
            amp_energy = reg.get_energy(input,)

        else:  # fingerprinting scheme

            for atom in atoms:
                index = atom.index
                symbol = atom.symbol
                indexfp = sfp.fp_data[(hash_key, index)]
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

        if param.fingerprint is None:  # pure atomic-coordinates scheme

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
                    energy_coefficient * 2. * (amp_energy - real_energy) * \
                    partial_der_variables_square_error / (len(atoms) ** 2.)

        if train_forces is True:

            real_forces = atoms.get_forces(apply_constraint=False)
            amp_forces = np.zeros((len(atoms), 3))
            for self_atom in atoms:
                self_index = self_atom.index
                reg.reset_forces()

                if param.fingerprint is None:  # pure atomic-coordinates scheme

                    for i in range(3):
                        _input = [0. for __ in range(3 * len(atoms))]
                        _input[3 * self_index + i] = 1.
                        force = reg.get_force(i, _input,)
                        amp_forces[self_index][i] = force

                else:  # fingerprinting scheme

                    n_self_indices = snl.nl_data[(hash_key, self_index)][0]
                    n_self_offsets = snl.nl_data[(hash_key, self_index)][1]
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
                                der_indexfp = sfp.der_fp_data[(hash_key,
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

                for i in range(3):
                    # pure atomic-coordinates scheme
                    if param.fingerprint is None:

                        partial_der_variables_square_error = \
                            reg.get_variable_der_of_forces(self_index, i)
                        der_variables_square_error += \
                            force_coefficient * (2.0 / 3.0) * \
                            (- amp_forces[self_index][i] +
                             real_forces[self_index][i]) * \
                            partial_der_variables_square_error \
                            / len(atoms)

                    else:  # fingerprinting scheme

                        for n_symbol, n_index, n_offset in zip(n_symbols,
                                                               n_self_indices,
                                                               n_self_offsets):
                            if n_offset[0] == 0 and n_offset[1] == 0 and \
                                    n_offset[2] == 0:

                                partial_der_variables_square_error = \
                                    reg.get_variable_der_of_forces(self_index,
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
    """function to calculate fingerprints range.
    inputs:
        fp: object of fingerprint class
        elements: list if all elements of atoms
        atoms: ASE atom object
        nl: ASE NeighborList object
    output:
        fingerprints_range: range of fingerprints of atoms"""

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
    """function to compare train images with the test image and decide whether
    the prediction is interpolation or extrapolation.
    inputs:
        fp: object of CalculateFingerprints class
        atoms: ASE atom object
        fingerprints_range: range of fingerprints of the train images
        nl: ASE NeighborList object
    output:
        compare_train_test_fingerprints:
        integer: zero for interpolation, and one for extrapolation"""

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


def send_data_to_fortran(hash_keys, images, sfp, snl, elements, train_forces,
                         energy_coefficient, force_coefficient, log,
                         param, no_procs, no_sub_images):
    """Function to send images data to fortran90 code"""

    log('Re-shaping data to send to fortran90...')
    log.tic()

    if param.fingerprint is None:
        fingerprinting = False
    else:
        fingerprinting = True

    no_of_images = len(images)
    real_energies = \
        [images[hash_key].get_potential_energy(apply_constraint=False)
         for hash_key in hash_keys]

    if fingerprinting:
        no_of_elements = len(elements)
        elements_numbers = [atomic_numbers[elm] for elm in elements]
        no_of_atoms_of_images = [len(images[hash_key])
                                 for hash_key in hash_keys]
        atomic_numbers_of_images = \
            [atomic_numbers[atom.symbol]
             for hash_key in hash_keys
             for atom in images[hash_key]]
        min_fingerprints = \
            [[param.fingerprints_range[elm][_][0]
              for _ in range(len(param.fingerprints_range[elm]))]
             for elm in elements]
        max_fingerprints = [[param.fingerprints_range[elm][_][1]
                             for _
                             in range(len(param.fingerprints_range[elm]))]
                            for elm in elements]
        raveled_fingerprints_of_images = \
            ravel_fingerprints_of_images(hash_keys, images, sfp)
        len_fingerprints_of_elements = [len(sfp.Gs[elm]) for elm in elements]
    else:
        no_of_atoms_of_image = param.no_of_atoms
        atomic_positions_of_images = \
            [images[hash_key].positions.ravel()
             for hash_key in hash_keys]

    if train_forces is True:
        real_forces_of_images = []
        for hash_key in hash_keys:
            forces = images[hash_key].get_forces(apply_constraint=False)
            for index in range(len(images[hash_key])):
                real_forces_of_images.append(forces[index])
        if fingerprinting:
            (list_of_no_of_neighbors,
             raveled_neighborlists,
             raveled_der_fingerprints) = \
                ravel_neighborlists_and_der_fingerprints_of_images(hash_keys,
                                                                   images,
                                                                   sfp,
                                                                   snl)

    log(' ...data re-shaped.', toc=True)

    log('Sending data to fortran90...')
    log.tic()

    fmodules.images_props.energy_coefficient = energy_coefficient
    fmodules.images_props.force_coefficient = force_coefficient
    fmodules.images_props.no_of_images = no_of_images
    fmodules.images_props.real_energies = real_energies
    fmodules.images_props.train_forces = train_forces
    fmodules.images_props.fingerprinting = fingerprinting
    fmodules.images_props.no_procs = no_procs
    fmodules.images_props.no_sub_images = no_sub_images
    del real_energies, hash_keys, images

    if fingerprinting:
        fmodules.images_props.no_of_elements = no_of_elements
        fmodules.images_props.elements_numbers = elements_numbers
        fmodules.images_props.no_of_atoms_of_images = no_of_atoms_of_images
        fmodules.images_props.atomic_numbers_of_images = \
            atomic_numbers_of_images
        fmodules.fingerprint_props.min_fingerprints = min_fingerprints
        fmodules.fingerprint_props.max_fingerprints = max_fingerprints
        fmodules.fingerprint_props.raveled_fingerprints_of_images = \
            raveled_fingerprints_of_images
        fmodules.fingerprint_props.len_fingerprints_of_elements = \
            len_fingerprints_of_elements
        del min_fingerprints, max_fingerprints, raveled_fingerprints_of_images,
        no_of_atoms_of_images, atomic_numbers_of_images
    else:
        fmodules.images_props.atomic_positions_of_images = \
            atomic_positions_of_images
        del atomic_positions_of_images
        fmodules.images_props.no_of_atoms_of_image = no_of_atoms_of_image

    if train_forces is True:

        fmodules.images_props.real_forces_of_images = real_forces_of_images
        del real_forces_of_images

        if fingerprinting:
            fmodules.images_props.list_of_no_of_neighbors = \
                list_of_no_of_neighbors
            fmodules.images_props.raveled_neighborlists = raveled_neighborlists
            fmodules.fingerprint_props.raveled_der_fingerprints = \
                raveled_der_fingerprints
            del list_of_no_of_neighbors, raveled_neighborlists,
            raveled_der_fingerprints

    gc.collect()

    log(' ...data sent to fortran90.', toc=True)

##############################################################################


def ravel_fingerprints_of_images(hash_keys, images, sfp):
    """Reshape fingerprints of all images into a matrix"""

    keys = [(hash_key, index) for hash_key in hash_keys
            for index in range(len(images[hash_key]))]

    raveled_fingerprints = [sfp.fp_data[key] for key in keys]

    del hash_keys, images, keys

    return raveled_fingerprints

###############################################################################


def ravel_neighborlists_and_der_fingerprints_of_images(hash_keys,
                                                       images, sfp, snl):
    """Reshape neighborlists and derivatives of fingerprints of all images into
    a matrix"""

    # Only neighboring atoms of type II (within the main cell) needs to be sent
    # to fortran for force training
    list_of_no_of_neighbors = []
    for hash_key in hash_keys:
        atoms = images[hash_key]
        for self_atom in atoms:
            self_index = self_atom.index
            n_self_offsets = snl.nl_data[(hash_key, self_index)][1]
            count = 0
            for n_offset in n_self_offsets:
                if n_offset[0] == 0 and n_offset[1] == 0 and n_offset[2] == 0:
                    count += 1
            list_of_no_of_neighbors.append(count)

    raveled_neighborlists = [n_index for hash_key in hash_keys
                             for self_atom in images[hash_key]
                             for n_index, n_offset in
                             zip(snl.nl_data[(hash_key, self_atom.index)][0],
                                 snl.nl_data[(hash_key, self_atom.index)][1])
                             if (n_offset[0] == 0 and n_offset[1] == 0 and
                                 n_offset[2] == 0)]

    raveled_der_fingerprints = \
        [sfp.der_fp_data[(hash_key, (n_index, self_atom.index, i))]
         for hash_key in hash_keys
         for self_atom in images[hash_key]
         for n_index, n_offset in
         zip(snl.nl_data[(hash_key, self_atom.index)][0],
             snl.nl_data[(hash_key, self_atom.index)][1])
         if (n_offset[0] == 0 and n_offset[1] == 0 and
             n_offset[2] == 0) for i in range(3)]

    del hash_keys, images

    return (list_of_no_of_neighbors,
            raveled_neighborlists,
            raveled_der_fingerprints)

###############################################################################


def now():
    """Return string of current time."""
    return datetime.now().isoformat().split('.')[0]

###############################################################################
