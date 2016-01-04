#!/usr/bin/env python

import numpy as np
import hashlib
import time
import os
import json
import sqlite3
from ase import io as aseio
from ase.parallel import paropen

###############################################################################


def randomize_images(images, fraction=0.8):
    """
    Randomly assigns 'fraction' of the images to a training set and
    (1 - 'fraction') to a test set. Returns two lists of ASE images.

    :param images: List of ASE atoms objects in ASE format. This can also be
                   the path to an ASE trajectory (.traj) or database (.db)
                   file.
    :type images: list or str
    :param fraction: Portion of train_images to all images.
    :type fraction: float

    :returns: Lists of train and test images.
    """
    file_opened = False
    if type(images) == str:
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = aseio.Trajectory(images, 'r')
        elif extension == '.db':
            images = aseio.read(images)
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


class FingerprintsError(Exception):

    """
    Error class in case the functional form of fingerprints has changed.
    """
    pass

###############################################################################


class ConvergenceOccurred(Exception):

    """
    Kludge to decide when scipy's optimizers are complete.
    """
    pass

###############################################################################


class TrainingConvergenceError(Exception):

    """
    Error to be raise if training does not converge.
    """
    pass

###############################################################################


class ExtrapolateError(Exception):

    """
    Error class in the case of extrapolation.
    """
    pass

###############################################################################


class UntrainedError(Exception):

    """
    Error class in the case of unsuccessful training.
    """
    pass

###############################################################################


def hash_image(atoms):
    """
    Creates a unique signature for a particular ASE atoms object.
    This is used to check whether an image has been seen before.
    This is just an md5 hash of a string representation of the atoms
    object.

    :param atoms: ASE atoms object.
    :type atoms: ASE dict

    :returns: Hash key of 'atoms'.
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

    """
    Logger that can also deliver timing information.

    :param filename: File object or path to the file to write to.
    :type filename: str
    """
    ###########################################################################

    def __init__(self, filename):
        self._f = paropen(filename, 'a')
        self._tics = {}

    ###########################################################################

    def tic(self, label=None):
        """
        Start a timer.

        :param label: Label for managing multiple timers.
        :type label: str
        """
        if label:
            self._tics[label] = time.time()
        else:
            self._tic = time.time()

    ###########################################################################

    def __call__(self, message, toc=None):
        """
        Writes message to the log file.

        :param message: Message to be written.
        :type message: str
        :param toc: tic is used to start a timer. If toc=True or toc=label, it
                    will append timing information in minutes to the timer.
        :type toc: bool or str
        """
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
    elif 'SLURM_NTASKS' in os.environ.keys():
        ncores = int(os.environ['SLURM_NTASKS'])
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
        import multiprocessing
        ncores = multiprocessing.cpu_count()

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


def save_parameters(filename, param):
    """
    Save parameters in json format.

    :param filename: Path to the file to write to.
    :type filename: str
    :param param: ASE dictionary object of parameters.
    :type param: dict
    """
    parameters = {}
    keys = param.keys()
    len_of_keys = len(keys)
    count = 0
    while count < len_of_keys:
        key = keys[count]
        if (key != 'regression') and (key != 'descriptor'):
            parameters[key] = param[key]
        count += 1

    if param.descriptor is not None:
        parameters['Gs'] = param.descriptor.Gs
        parameters['cutoff'] = param.descriptor.cutoff
        parameters['fingerprints_tag'] = param.descriptor.fingerprints_tag

    if param.descriptor is None:
        parameters['descriptor'] = 'None'
        parameters['no_of_atoms'] = param.regression.no_of_atoms
    elif param.descriptor.__class__.__name__ == 'Behler':
        parameters['descriptor'] = 'Behler'
    else:
        raise RuntimeError('Descriptor is not recognized to Amp for saving '
                           'parameters. User should add the descriptor under '
                           ' consideration.')

    if param.regression.__class__.__name__ == 'NeuralNetwork':
        parameters['regression'] = 'NeuralNetwork'
        parameters['hiddenlayers'] = param.regression.hiddenlayers
        parameters['activation'] = param.regression.activation
    else:
        raise RuntimeError('Regression method is not recognized to Amp for '
                           'saving parameters. User should add the '
                           'regression method under consideration.')

    parameters['variables'] = list(param.regression._variables)

    base_filename = os.path.splitext(filename)[0]
    export_filename = os.path.join(base_filename + '.json')

    with paropen(export_filename, 'wb') as outfile:
        json.dump(parameters, outfile)

###############################################################################


def load_parameters(json_file):

    parameters = json.load(json_file)

    return parameters

###############################################################################


def make_filename(label, base_filename):
    """
    Creates a filename from the label and the base_filename which should be
    a string.

    :param label: Prefix.
    :type label: str
    :param base_filename: Basic name of the file.
    :type base_filename: str
    """
    if not label:
        filename = base_filename
    else:
        filename = os.path.join(label + '-' + base_filename)

    return filename

###############################################################################


class IO:

    """
    Class that save and read neighborlists, fingerprints, and their
    derivatives from either json or db files.

    :param hashs: Unique keys, one key per image.
    :type hashs: list
    :param images: List of ASE atoms objects with positions, symbols, energies,
                   and forces in ASE format. This is the training set of data.
                   This can also be the path to an ASE trajectory (.traj) or
                   database (.db) file. Energies can be obtained from any
                   reference, e.g. DFT calculations.
    :type images: list or str
    """
    ###########################################################################

    def __init__(self, hashs, images):
        self.hashs = hashs
        self.images = images

    ###########################################################################

    def save(self, filename, data_type, data, data_format):
        """
        Saves data.

        :param filename: Name of the file to save data to or read data from.
        :type filename: str
        :param data_type: Can be either 'neighborlists', 'fingerprints', or
                          'fingerprint-derivatives'.
        :type data_type: str
        :param data: Data to be saved.
        :type data: dict
        :param data_format: Format of saved data. Can be either "json" or
                            "db".
        :type data_format: str
        """
        if data_type is 'neighborlists':

            hashs = data.keys()
            no_of_images = len(hashs)
            if data_format is 'json':
                # Reformatting data for saving
                new_dict = {}
                count = 0
                while count < no_of_images:
                    hash = hashs[count]
                    image = self.images[hash]
                    new_dict[hash] = {}
                    no_of_atoms = len(image)
                    index = 0
                    while index < no_of_atoms:
                        nl_value = data[hash][index]
                        new_dict[hash][index] = [[nl_value[0][i],
                                                  list(nl_value[1][i])]
                                                 for i in
                                                 range(len(nl_value[0]))]
                        index += 1
                    count += 1
                with paropen(filename, 'wb') as outfile:
                    json.dump(new_dict, outfile)
                del new_dict

            elif data_format is 'db':
                conn = sqlite3.connect(filename)
                c = conn.cursor()
                # Create table
                c.execute('''CREATE TABLE IF NOT EXISTS neighborlists
                (image text, atom integer, neighbor_atom integer,
                xoffset integer, yoffset integer, zoffset integer)''')
                count = 0
                while count < no_of_images:
                    hash = hashs[count]
                    image = self.images[hash]
                    no_of_atoms = len(image)
                    index = 0
                    while index < no_of_atoms:
                        value0 = data[hash][index][0]
                        value1 = data[hash][index][1]
                        len_of_neighbors = len(value0)
                        _ = 0
                        while _ < len_of_neighbors:
                            value = value1[_]
                            # Insert a row of data
                            row = (hash, index, value0[_], value[0],
                                   value[1], value[2])
                            c.execute('''INSERT INTO neighborlists VALUES
                            (?, ?, ?, ?, ?, ?)''', row)
                            _ += 1
                        index += 1
                    count += 1
                # Save (commit) the changes
                conn.commit()
                conn.close()

        elif data_type is 'fingerprints':

            if data_format is 'json':
                try:
                    json.dump(data, filename)
                    filename.flush()
                    return
                except AttributeError:
                    with paropen(filename, 'wb') as outfile:
                        json.dump(data, outfile)

            elif data_format is 'db':
                conn = sqlite3.connect(filename)
                c = conn.cursor()
                # Create table
                c.execute('''CREATE TABLE IF NOT EXISTS fingerprints
                (image text, atom integer, value real)''')
                hashs = data.keys()
                no_of_images = len(hashs)
                count = 0
                while count < no_of_images:
                    hash = hashs[count]
                    image = self.images[hash]
                    no_of_atoms = len(image)
                    index = 0
                    while index < no_of_atoms:
                        value = data[hash][index]
                        len_of_fingerprints = len(value)
                        _ = 0
                        while _ < len_of_fingerprints:
                            # Insert a row of data
                            row = (hash, index, value[_])
                            c.execute('''INSERT INTO fingerprints VALUES
                            (?, ?, ?)''', row)
                            _ += 1
                        index += 1
                    count += 1
                # Save (commit) the changes
                conn.commit()
                conn.close()

        elif data_type is 'fingerprint_derivatives':

            hashs = data.keys()
            no_of_images = len(hashs)
            if data_format is 'json':
                new_dict = {}
                count0 = 0
                while count0 < no_of_images:
                    hash = hashs[count0]
                    new_dict[hash] = {}
                    pair_atom_keys = data[hash].keys()
                    len_of_pair_atom_keys = len(pair_atom_keys)
                    count1 = 0
                    while count1 < len_of_pair_atom_keys:
                        pair_atom_key = pair_atom_keys[count1]
                        new_dict[hash][str(pair_atom_key)] = \
                            data[hash][pair_atom_key]
                        count1 += 1
                    count0 += 1
                try:
                    json.dump(new_dict, filename)
                    filename.flush()
                    return
                except AttributeError:
                    with paropen(filename, 'wb') as outfile:
                        json.dump(new_dict, outfile)
                del new_dict

            elif data_format is 'db':
                conn = sqlite3.connect(filename)
                c = conn.cursor()
                # Create table
                c.execute('''CREATE TABLE IF NOT EXISTS fingerprint_derivatives
                (image text, atom integer, neighbor_atom integer,
                direction integer, value real)''')
                count0 = 0
                while count0 < no_of_images:
                    hash = hashs[count0]
                    pair_atom_keys = data[hash].keys()
                    len_of_pair_atom_keys = len(pair_atom_keys)
                    count1 = 0
                    while count1 < len_of_pair_atom_keys:
                        pair_atom_key = pair_atom_keys[count1]
                        n_index = pair_atom_key[0]
                        self_index = pair_atom_key[1]
                        i = pair_atom_key[2]
                        value = data[hash][pair_atom_key]
                        len_of_value = len(value)
                        _ = 0
                        while _ < len_of_value:
                            # Insert a row of data
                            row = (hash, self_index, n_index, i, value[_])
                            c.execute('''INSERT INTO fingerprint_derivatives
                            VALUES (?, ?, ?, ?, ?)''', row)
                            _ += 1
                        count1 += 1
                    count0 += 1

                # Save (commit) the changes
                conn.commit()
                conn.close()

        del data

    ###########################################################################

    def read(self, filename, data_type, data, data_format):
        """
        Reads data.

        :param filename: Name of the file to save data to or read data from.
        :type filename: str
        :param data_type: Can be either 'neighborlists', 'fingerprints', or
                          'fingerprint-derivatives'.
        :type data_type: str
        :param data: Data to be read.
        :type data: dict
        :param data_format: Format of saved data. Can be either "json" or
                            "db".
        :type data_format: str
        """
        hashs = []
        if data_type is 'neighborlists':

            if data_format is 'json':
                fp = paropen(filename, 'rb')
                loaded_data = json.load(fp)
                hashs = loaded_data.keys()
                no_of_images = len(hashs)
                count = 0
                while count < no_of_images:
                    hash = hashs[count]
                    data[hash] = {}
                    image = self.images[hash]
                    no_of_atoms = len(image)
                    index = 0
                    while index < no_of_atoms:
                        nl_value = loaded_data[hash][str(index)]
                        nl_indices = [value[0] for value in nl_value]
                        nl_offsets = [value[1] for value in nl_value]
                        data[hash][index] = (nl_indices, nl_offsets,)
                        index += 1
                    count += 1

            elif data_format is 'db':
                with sqlite3.connect(filename) as conn:
                    c = conn.cursor()
                    c.execute("SELECT * FROM neighborlists")
                    _hash = None
                    while True:
                        row = c.fetchone()
                        if row is None:
                            break
                        hash, index, n_index, xoffset, yoffset, zoffset = row
                        if hash != _hash:
                            data[hash] = {}
                            _hash = hash
                            _index = None
                        if index != _index:
                            data[hash][index] = [[], []]
                            _index = index
                        data[hash][index][0] += [n_index]
                        data[hash][index][1] += [[xoffset, yoffset, zoffset]]
                hashs = data.keys()

        elif data_type is 'fingerprints':

            if data_format is 'json':
                if isinstance(filename, str):
                    fp = paropen(filename, 'rb')
                    loaded_data = json.load(fp)
                else:
                    filename.seek(0)
                    loaded_data = json.load(filename)
                hashs = loaded_data.keys()
                no_of_images = len(hashs)
                count = 0
                while count < no_of_images:
                    hash = hashs[count]
                    data[hash] = {}
                    image = self.images[hash]
                    no_of_atoms = len(image)
                    index = 0
                    while index < no_of_atoms:
                        fp_value = loaded_data[hash][str(index)]
                        data[hash][index] = \
                            [float(value) for value in fp_value]
                        index += 1
                    count += 1

            elif data_format is 'db':
                with sqlite3.connect(filename) as conn:
                    c = conn.cursor()
                    c.execute("SELECT * FROM fingerprints")
                    _hash = None
                    while True:
                        row = c.fetchone()
                        if row is None:
                            break
                        hash, index, fp = row
                        if hash != _hash:
                            data[hash] = {}
                            _hash = hash
                            _index = None
                        if index != _index:
                            data[hash][index] = []
                            _index = index
                        data[hash][index] += [fp]
                hashs = data.keys()

        elif data_type is 'fingerprint_derivatives':

            if data_format is 'json':
                if isinstance(filename, str):
                    fp = paropen(filename, 'rb')
                    loaded_data = json.load(fp)
                else:
                    filename.seek(0)
                    loaded_data = json.load(filename)
                hashs = loaded_data.keys()
                no_of_images = len(hashs)
                count0 = 0
                while count0 < no_of_images:
                    hash = hashs[count0]
                    data[hash] = {}
                    image = self.images[hash]
                    pair_atom_keys = loaded_data[hash].keys()
                    len_of_pair_atom_keys = len(pair_atom_keys)
                    count1 = 0
                    while count1 < len_of_pair_atom_keys:
                        pair_atom_key = pair_atom_keys[count1]
                        fp_value = loaded_data[hash][pair_atom_key]
                        data[hash][eval(pair_atom_key)] = \
                            [float(value) for value in fp_value]
                        count1 += 1
                    count0 += 1

            elif data_format is 'db':
                with sqlite3.connect(filename) as conn:
                    c = conn.cursor()
                    c.execute("SELECT * FROM fingerprint_derivatives")
                    _hash = None
                    while True:
                        row = c.fetchone()
                        if row is None:
                            break
                        hash, self_index, n_index, i, fp_der = row
                        index = (n_index, self_index, i)
                        if hash != _hash:
                            data[hash] = {}
                            _hash = hash
                            _index = None
                        if index != _index:
                            data[hash][index] = []
                            _index = index
                        data[hash][index] += [fp_der]
                hashs = data.keys()

        return hashs, data

###############################################################################
