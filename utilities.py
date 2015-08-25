#!/usr/bin/env python
"""This module contains utilities for use with various aspects of the
AMP calculators."""

###############################################################################

import numpy as np
import hashlib
import time
import os
import json
from ase import io
from ase.parallel import paropen

###############################################################################


def randomize_images(images, fraction=0.8):
    """Randomly assigns 'fraction' of the images to a training set and
    (1 - 'fraction') to a test set. Returns two lists of ASE images.
    'images' can either be a list of ASE images or a path to an ASE
    trajectory.
    """
    file_opened = False
    if type(images) == str:
        images = io.Trajectory(images, 'r')
        file_opened = True
    trainingsize = int(fraction * len(images))
    testsize = len(images) - trainingsize
    testindices = []
    while len(testindices) < testsize:
        next = np.random.randint(len(images))
        if next not in testindices:
            testindices.append(next)
    testindices.sort()
    trainindices = [index for index in range(len(images)) if index not in
                    testindices]
    train_images = [images[index] for index in trainindices]
    test_images = [images[index] for index in testindices]
    if file_opened:
        images.close()
    return train_images, test_images

###############################################################################


class ExtrapolateError(Exception):
    pass

###############################################################################


class UntrainedError(Exception):
    pass

###############################################################################


def hash_image(atoms):
    """Creates a unique signature for a particular ASE atoms object.
    This is used to check whether an image has been seen before.
    This is just an md5 hash of a string representation of the atoms
    object.
    """
    string = str(atoms.pbc)
    for number in atoms.cell.flatten():
        string += '%.15f' % number
    string += str(atoms.get_atomic_numbers())
    for number in atoms.get_positions().flatten():
        string += '%.15f' % number

    md5 = hashlib.md5(string)
    hash = md5.hexdigest()
    return hash

###############################################################################


class Logger:

    """Logger that can also deliver timing information.
    Initialize with the path to the file to write to.
    """

    def __init__(self, filename):
        self._f = paropen(filename, 'a')
        self._tics = {}

    def tic(self, label=None):
        """Start a timer. Optionally supply a label if you need to manage
        multiple timers."""
        if label:
            self._tics[label] = time.time()
        else:
            self._tic = time.time()

    def __call__(self, message, toc=None):
        """Writes <message> to the log file. <tic> is used to start a
        timer.  If you supply toc=True or toc=<label>, it will append
        timing information in minutes to the timer."""
        dt = ''
        if toc:
            if toc is True:
                tic = self._tic
            else:
                tic = self._tics[toc]
            dt = (time.time() - tic) / 60.
            dt = ' %.1f min.' % dt
        self._f.write(message + dt + '\n')
        self._f.flush()

###############################################################################


def count_allocated_cpus():
    """
    This function accesses the file provided by the batch management system to
    count the number of cores allocated to the current job. It is currently
    fully implemented and tested for PBS, while SLURM, SGE and LoadLeveler are
    not fully tested.
    """
    if 'PBS_NODEFILE' in os.environ.keys():
        ncores = len(open(os.environ['PBS_NODEFILE']).readlines())
    elif 'SLURM_JOB_NODELIST' in os.environ.keys():
        raise Warning('Functionality for SLURM is untested and might not '
                      'work.')
        ncores = len(open(os.environ['SLURM_JOB_NODELIST']).readlines())
    elif 'LOADL_PROCESSOR_LIST' in os.environ.keys():
        raise Warning('Functionality for LoadLeveler is untested and might '
                      'not work.')
        ncores = len(open(os.environ['LOADL_PROCESSOR_LIST']).readlines())
    elif 'PE_HOSTFILE' in os.environ.keys():
        raise Warning('Functionality for SGE is untested and might not work.')
        ncores = 0
        MACHINEFILE = open(os.environ['PE_HOSTFILE']).readlines()
        for line in MACHINEFILE:
            fields = string.split(line)
            nprocs = int(fields[1])
            ncores += nprocs
    else:
        raise NotImplementedError('Unsupported batch management system. '
                                  'Currently only PBS, SLURM, LoadLeveler '
                                  'and SGE are supported.')

    return ncores

###############################################################################


def names_of_allocated_nodes():
    """
    This function accesses the file provided by the batch management system to
    count the number of allocated to the current job, as well as to provide
    their names. It is currently fully implemented and tested for PBS, while
    SLURM, SGE and LoadLeveler are not fully tested.
    """
    if 'PBS_NODEFILE' in os.environ.keys():
        node_list = set(open(os.environ['PBS_NODEFILE']).readlines())
    elif 'SLURM_JOB_NODELIST' in os.environ.keys():
        raise Warning('Support for SLURM is untested and might not work.')
        node_list = set(open(os.environ['SLURM_JOB_NODELIST']).readlines())
    elif 'LOADL_PROCESSOR_LIST' in os.environ.keys():
        raise Warning('Support for LoadLeveler is untested and might not '
                      'work.')
        node_list = set(open(os.environ['LOADL_PROCESSOR_LIST']).readlines())
    elif 'PE_HOSTFILE' in os.environ.keys():
        raise Warning('Support for SGE is untested and might not work.')
        nodes = []
        MACHINEFILE = open(os.environ['PE_HOSTFILE']).readlines()
        for line in MACHINEFILE:
            # nodename = fields[0]
            # ncpus = fields[1]
            # queue = fields[2]
            # UNDEFINED = fields[3]
            fields = string.split(line)
            node = int(fields[0])
            nodes += node
        node_list = set(nodes)
    else:
        raise NotImplementedError('Unsupported batch management system. '
                                  'Currently only PBS and SLURM are '
                                  'supported.')

    return node_list, len(node_list)

###############################################################################


def save_neighborlists(filename, neighborlists):
    """Save neighborlists in json format."""

    new_dict = {}
    for key1 in neighborlists.keys():
        new_dict[key1] = {}
        for key2 in neighborlists[key1].keys():
            nl_value = neighborlists[key1][key2]
            new_dict[key1][key2] = [[nl_value[0][i],
                                     [j for j in nl_value[1][i]]]
                                    for i in range(len(nl_value[0]))]

    with paropen(filename, 'wb') as outfile:
        json.dump(new_dict, outfile)

###############################################################################


def save_fingerprints(f, fingerprints):
    """Save fingerprints in json format. f is either a file object or a
    path."""

    new_dict = {}
    for key1 in fingerprints.keys():
        new_dict[key1] = {}
        for key2 in fingerprints[key1].keys():
            fp_value = fingerprints[key1][key2]
            new_dict[key1][key2] = [value for value in fp_value]

    try:
        json.dump(new_dict, f)
        f.flush()
        return
    except AttributeError:
        with paropen(f, 'wb') as outfile:
            json.dump(new_dict, outfile)

###############################################################################


def save_der_fingerprints(f, der_fingerprints):
    """Save derivatives of fingerprints in json format. f is either a file
    object or a path."""

    new_dict = {}
    for key1 in der_fingerprints.keys():
        new_dict[key1] = {}
        for key2 in der_fingerprints[key1].keys():
            fp_value = der_fingerprints[key1][key2]
            new_dict[key1][str([key2[0], key2[1], key2[2]])] = [
                value for value in fp_value]

    try:
        json.dump(new_dict, f)
        f.flush()
        return
    except AttributeError:
        with paropen(f, 'wb') as outfile:
            json.dump(new_dict, outfile)

###############################################################################


def save_parameters(filename, param):
    """Save parameters in json format."""

    parameters = {}
    for key in param.keys():
        if (key != 'regression') and (key != 'fingerprint'):
            parameters[key] = param[key]

    if param.fingerprint is not None:
        parameters['Gs'] = param.fingerprint.Gs
        parameters['cutoff'] = param.fingerprint.cutoff
        parameters['fingerprints_tag'] = param.fingerprint.fingerprints_tag

    if param.fingerprint is None:
        parameters['fingerprint'] = 'None'
        parameters['no_of_atoms'] = param.regression.no_of_atoms
    elif param.fingerprint.__class__.__name__ == 'Behler':
        parameters['fingerprint'] = 'Behler'
    else:
        raise RuntimeError('Fingerprinting scheme is not recognized to AMP '
                           'for saving parameters. User should add the '
                           'fingerprinting scheme under consideration.')

    if param.regression.__class__.__name__ == 'NeuralNetwork':
        parameters['regression'] = 'NeuralNetwork'
        parameters['hiddenlayers'] = param.regression.hiddenlayers
        parameters['activation'] = param.regression.activation
    else:
        raise RuntimeError('Regression method is not recognized to AMP for '
                           'saving parameters. User should add the '
                           'regression method under consideration.')

    variables = []
    for _ in param.regression._variables:
        variables.append(_)
    parameters['variables'] = variables

    base_filename = os.path.splitext(filename)[0]
    export_filename = os.path.join(base_filename + '.json')

    with paropen(export_filename, 'wb') as outfile:
        json.dump(parameters, outfile)

###############################################################################


def load_parameters(json_file):
    """Reads parameters from JSON file."""

    parameters = json.load(json_file)

    return parameters

###############################################################################


def make_filename(label, base_filename):
    """Creates a filename from the label and the base_filename which should be
    a string"""

    if not label:
        filename = base_filename
    else:
        filename = os.path.join(label + '-' + base_filename)

    return filename

###############################################################################
