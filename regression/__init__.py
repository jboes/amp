#!/usr/bin/env python
"""
Script that contains different regression methods.
"""

import numpy as np
from collections import OrderedDict

###############################################################################


class NeuralNetwork:

    """
    Class that implements a basic feed-forward neural network.
    Parameters:
        hiddenlayers: dictionary of chemical element symbols and architectures
                    of their corresponding hidden layers of the conventional
                    neural network. Note that for each atom, number of nodes in
                    the input layer is always equal to the number of symmetry
                    functions (G), and the number of nodes in the output
                    layer is always one. For example,
                    hiddenlayers = {"O":(3,5), "Au":(5,6)} means that for "O"
                    we have two hidden layers, the first one with three nodes
                    and the second one having five nodes.
        activation: string to assign the type of activation funtion. "linear"
                    refers to linear function, "tanh" refers to tanh function,
                    and "sigmoid" refers to sigmoid function.
        weights: dictionary of symbols and dictionaries of arrays of weights;
                   each symbol and related dictionary corresponds to one
                   element and one conventional neural network.  The weights
                   dictionary for the above example has dimensions
                   weights = {"O": {1: np.array(149,3), 2: np.array(4,5),
                   3: np.array(6,1)}, "Au": {1: np.array(148,5),
                   2: np.array(6,6), 3: np.array(7,1)}. The arrays are set up
                   to connect node i in the previous layer with node j in the
                   current layer with indices w[i,j]. There are n+1 rows
                   (i values), with the last row being the bias, and m columns
                   (j values). If weights is not given, the arrays will be
                   randomly generated from values between -0.5 and 0.5.
        scalings: dictionary of variables for slope and intercept for each
                    element. They are used in order to remove the final value
                    from the range that the activation function would
                    otherwise be locked in. For example,
                      scalings={"Pd":{"slope":2, "intercept":1},
                                "O":{"slope":3, "intercept":4}}
        variables (internal): list of variables. weights and scalings can be
                              fed in the form of a list

    **NOTE: The dimensions of the weight matrix should be consistent with
            hiddenlayers.**
    """

    #########################################################################

    def __init__(self):

        self.default_parameters = {'hiddenlayers': (5, 5),
                                   'activation': 'tanh',
                                   'weights': None,
                                   'scalings': None,
                                   'variables': None,
                                   }

    #########################################################################

    def initialize(self, param, fortran, load):
        """
        Inputs:
        param: dictionary that contains weights, scalings, activation and
                hiddenlayers
        fortran: boolean
                If True, will use the fortran subroutines, else will not.
        load: string
            load an existing (trained) BPNeural calculator from this path.
        """

        self.fortran = fortran

        self.weights = param['weights']
        self.scalings = param['scalings']

        self.activation = param['activation']
        # Checking that the activation function is given correctly:
        if self.activation not in ['linear', 'tanh', 'sigmoid']:
            raise NotImplementedError('Unknown activation function; '
                                      'activation must be one of '
                                      '"linear", "tanh", or "sigmoid".')

        if param['Gs'] is not None:

            self.elements = sorted(param['Gs'].keys())

            # If hiddenlayers is fed by the user in the tuple format,
            # it will now be converted to a dictionary.
            if isinstance(param['hiddenlayers'], tuple):
                hiddenlayers = {}
                for element in self.elements:
                    hiddenlayers[element] = param['hiddenlayers']
                param['hiddenlayers'] = hiddenlayers

            self.hiddenlayers = param['hiddenlayers']

            self.hiddensizes = {}
            for element in self.elements:
                structure = self.hiddenlayers[element]
                if isinstance(structure, str):
                    structure = structure.split('-')
                elif isinstance(structure, int):
                    structure = [structure]
                else:
                    structure = list(structure)
                hiddensizes = [int(part) for part in structure]
                self.hiddensizes[element] = hiddensizes

            self.ravel = _RavelVariables(param, self.elements)

            if load is not None:
                self.weights, self.scalings = \
                    self.ravel.to_dicts(param['variables'])

        # Checking the compatibility of the forms of Gs, hiddenlayers and
        # weights.
        string1 = 'Gs and weights are not compatible.'
        string2 = 'hiddenlayers and weights are not compatible.'
        if 'hiddenlayers' in param.keys():
            if isinstance(param['hiddenlayers'], dict):
                self.elements = sorted(param['hiddenlayers'].keys())
                for element in self.elements:
                    if param['weights'] is not None:
                        if 'Gs' in param.keys():
                            if np.shape(param['weights'][element][1])[0] \
                                    != len(param['Gs'][element]) + 1:
                                raise RuntimeError(string1)
                        if isinstance(param['hiddenlayers'][element], int):
                            if np.shape(param['weights'][element][1])[1] \
                                    != param['hiddenlayers'][element]:
                                raise RuntimeError(string2)
                        else:
                            if np.shape(param['weights'][element][1])[1] \
                                    != param['hiddenlayers'][element][0]:
                                raise RuntimeError(string2)
                            for _ in range(2, len(param['hiddenlayers']
                                                  [element]) + 1):
                                if (np.shape(param['weights']
                                             [element][_])[0] !=
                                        param['hiddenlayers'][
                                        element][_ - 2] + 1 or
                                        np.shape(param['weights']
                                                 [element][_])[1] !=
                                        param['hiddenlayers']
                                        [element][_ - 1]):
                                    raise RuntimeError(string2)
        del string1
        del string2

        self.param = param

    #########################################################################

    def ravel_variables(self):
        """Wrapper function for raveling weights and scalings."""

        if (self.param['variables'] is None) and self.weights:
            self.param['variables'] = \
                self.ravel.to_vector(self.weights, self.scalings)

        return self.param

    #########################################################################

    def reset_energy(self):
        """Resets local variables corresponding to energy."""

        self.o = {}
        self.D = {}
        self.delta = {}
        self.ohat = {}

    #########################################################################

    def reset_forces(self):
        """Resets local variables corresponding to forces."""

        self.der_coordinates_o = {}
        self.der_coordinates_weights_atomic_output = {}

    #########################################################################

    def update_variables(self, param):
        """Updating variables."""

        self.param['variables'] = param['variables']
        self.weights, self.scalings = \
            self.ravel.to_dicts(self.param['variables'])

        self.W = {}
        for element in self.elements:
            weight = self.weights[element]
            self.W[element] = {}
            for j in range(len(weight)):
                self.W[element][j + 1] = np.delete(weight[j + 1], -1, 0)

        if self.zero_weights is None:
            self.zero_weights = {}
            for key1 in self.weights.keys():
                self.zero_weights[key1] = {}
                for key2 in self.weights[key1].keys():
                    self.zero_weights[key1][key2] = \
                        np.zeros(shape=np.shape(self.weights[key1][key2]))

            self.zero_scalings = {}
            for key1 in self.scalings.keys():
                self.zero_scalings[key1] = {}
                for key2 in self.scalings[key1].keys():
                    self.zero_scalings[key1][key2] = 0.

    #########################################################################

    def get_output(self, index, symbol, indexfp):
        """Given fingerprints of the indexed atom and its symbol, outputs of
        the neural network are calculated."""

        self.o[index] = {}
        hiddensizes = self.hiddensizes[symbol]
        weight = self.weights[symbol]
        o = {}  # node values
        layer = 1  # input layer
        net = {}  # excitation
        ohat = {}
        temp = np.zeros((1, len(indexfp) + 1))
        for _ in range(len(indexfp)):
            temp[0, _] = indexfp[_]
        temp[0, len(indexfp)] = 1.0
        ohat[0] = temp
        net[1] = np.dot(ohat[0], weight[1])
        if self.activation == 'linear':
            o[1] = net[1]  # linear activation
        elif self.activation == 'tanh':
            o[1] = np.tanh(net[1])  # tanh activation
        elif self.activation == 'sigmoid':  # sigmoid activation
            o[1] = 1. / (1. + np.exp(-net[1]))
        temp = np.zeros((1, np.shape(o[1])[1] + 1))
        for _ in range(np.shape(o[1])[1]):
            temp[0, _] = o[1][0, _]
        temp[0, np.shape(o[1])[1]] = 1.0
        ohat[1] = temp
        for hiddensize in hiddensizes[1:]:
            layer += 1
            net[layer] = np.dot(ohat[layer - 1], weight[layer])
            if self.activation == 'linear':
                o[layer] = net[layer]  # linear activation
            elif self.activation == 'tanh':
                o[layer] = np.tanh(net[layer])  # tanh activation
            elif self.activation == 'sigmoid':
                # sigmoid activation
                o[layer] = 1. / (1. + np.exp(-net[layer]))
            temp = np.zeros((1, np.size(o[layer]) + 1))
            for _ in range(np.size(o[layer])):
                temp[0, _] = o[layer][0, _]
            temp[0, np.size(o[layer])] = 1.0
            ohat[layer] = temp
        layer += 1  # output layer
        net[layer] = np.dot(ohat[layer - 1], weight[layer])
        if self.activation == 'linear':
            o[layer] = net[layer]  # linear activation
        elif self.activation == 'tanh':
            o[layer] = np.tanh(net[layer])  # tanh activation
        elif self.activation == 'sigmoid':
            # sigmoid activation
            o[layer] = 1. / (1. + np.exp(-net[layer]))

        del hiddensizes, weight, ohat, net

        atomic_amp_energy = self.scalings[symbol]['slope'] * \
            float(o[layer]) + self.scalings[symbol]['intercept']

        self.o[index] = o
        temp = np.zeros((1, len(indexfp)))
        for _ in range(len(indexfp)):
            temp[0, _] = indexfp[_]
        self.o[index][0] = temp

        return atomic_amp_energy

    #########################################################################

    def get_force(self, n_index, n_symbol, i, der_indexfp):
        """Feed-forward for calculating forces."""

        hiddensizes = self.hiddensizes[n_symbol]
        weight = self.weights[n_symbol]
        der_o = {}  # node values
        der_o[0] = der_indexfp
        layer = 0  # input layer
        for hiddensize in hiddensizes[0:]:
            layer += 1
            temp = np.dot(np.matrix(der_o[layer - 1]),
                          np.delete(weight[layer], -1, 0))
            der_o[layer] = []
            for j in range(np.size(self.o[n_index][layer])):
                if self.activation == 'linear':  # linear function
                    der_o[layer].append(float(temp[0, j]))
                elif self.activation == 'sigmoid':  # sigmoid function
                    der_o[layer].append(float(self.o[n_index][layer][0, j] *
                                              (1. -
                                               self.o[n_index][layer][0, j])) *
                                        float(temp[0, j]))
                elif self.activation == 'tanh':  # tanh function
                    der_o[layer].append(
                        float(1. - self.o[n_index][layer][0, j] *
                              self.o[n_index][layer][0, j]) *
                        float(temp[0, j]))
        layer += 1  # output layer
        temp = np.dot(np.matrix(der_o[layer - 1]),
                      np.delete(weight[layer], -1, 0))
        if self.activation == 'linear':  # linear function
            der_o[layer] = float(temp)
        elif self.activation == 'sigmoid':  # sigmoid function
            der_o[layer] = float(self.o[n_index][layer] *
                                 (1. - self.o[n_index][layer]) * temp)
        elif self.activation == 'tanh':  # tanh function
            der_o[layer] = float((1. - self.o[n_index][layer] *
                                  self.o[n_index][layer]) * temp)

        force = float(-(self.scalings[n_symbol]['slope'] * der_o[layer]))

        der_o[layer] = [der_o[layer]]

        self.der_coordinates_o[(n_index, i)] = der_o

        return force

    #########################################################################

    def get_variable_der_of_energy(self, index, symbol):
        """Calculates energy square error with respect to variables."""

        partial_der_variables_square_error = np.zeros(self.ravel.count)

        partial_der_weights_square_error, partial_der_scalings_square_error = \
            self.ravel.to_dicts(partial_der_variables_square_error)

        N = len(self.o[index]) - 2
        self.D[index] = {}
        for k in range(1, N + 2):
            # calculating D matrix
            self.D[index][k] = \
                np.zeros(shape=(np.size(self.o[index][k]),
                                np.size(self.o[index][k])))
            for j in range(np.size(self.o[index][k])):
                if self.activation == 'linear':  # linear
                    self.D[index][k][j, j] = 1.
                elif self.activation == 'sigmoid':  # sigmoid
                    self.D[index][k][j, j] = \
                        float(self.o[index][k][0, j]) * \
                        float((1. - self.o[index][k][0, j]))
                elif self.activation == 'tanh':  # tanh
                    self.D[index][k][j, j] = \
                        float(1. - self.o[index][k][0, j] *
                              self.o[index][k][0, j])
        # Calculating delta
        self.delta[index] = {}
        # output layer
        self.delta[index][N + 1] = self.D[index][N + 1]
        # hidden layers
        for k in range(N, 0, -1):
            self.delta[index][k] = np.dot(self.D[index][k],
                                          np.dot(self.W[symbol][k + 1],
                                                 self.delta[index][k + 1]))
        # Calculating ohat
        self.ohat[index] = {}
        for k in range(1, N + 2):
            self.ohat[index][k - 1] = \
                np.zeros(shape=(1, np.size(self.o[index][k - 1]) + 1))
            for j in range(np.size(self.o[index][k - 1])):
                self.ohat[index][k - 1][0, j] = self.o[index][k - 1][0, j]
            self.ohat[index][k - 1][0, np.size(self.o[index][k - 1])] = 1.0

        partial_der_scalings_square_error[symbol]['intercept'] = 1.
        partial_der_scalings_square_error[symbol]['slope'] = \
            float(self.o[index][N + 1])

        for k in range(1, N + 2):
            partial_der_weights_square_error[symbol][k] = \
                float(self.scalings[symbol]['slope']) * \
                np.dot(np.matrix(self.ohat[index][k - 1]).T,
                       np.matrix(self.delta[index][k]).T)

        partial_der_variables_square_error = \
            self.ravel.to_vector(partial_der_weights_square_error,
                                 partial_der_scalings_square_error)

        return partial_der_variables_square_error

    #########################################################################

    def calculate_variable_der_of_forces(self, self_index, n_index,
                                         n_symbol, i):
        """Calculates force square error with respect to variables."""

        N = len(self.o[n_index]) - 2

        der_coordinates_D = {}
        for k in range(1, N + 2):
            # Calculating coordinate derivative of D matrix
            der_coordinates_D[k] = \
                np.zeros(shape=(np.size(self.o[n_index][k]),
                                np.size(self.o[n_index][k])))
            for j in range(np.size(self.o[n_index][k])):
                if self.activation == 'linear':  # linear
                    der_coordinates_D[k][j, j] = 0.
                elif self.activation == 'tanh':  # tanh
                    der_coordinates_D[k][j, j] = \
                        - 2. * self.o[n_index][k][0, j] * \
                        self.der_coordinates_o[(n_index, i)][k][j]
                elif self.activation == 'sigmoid':  # sigmoid
                    der_coordinates_D[k][j, j] = self.der_coordinates_o[
                        (n_index, i)][k][j] \
                        - 2. * self.o[n_index][k][0, j] * \
                        self.der_coordinates_o[
                        (n_index, i)][k][j]
        # Calculating coordinate derivative of delta
        der_coordinates_delta = {}
        # output layer
        der_coordinates_delta[N + 1] = der_coordinates_D[N + 1]
        # hidden layers
        temp1 = {}
        temp2 = {}
        for k in range(N, 0, -1):
            temp1[k] = np.dot(self.W[n_symbol][k + 1],
                              self.delta[n_index][k + 1])
            temp2[k] = np.dot(self.W[n_symbol][k + 1],
                              der_coordinates_delta[k + 1])
            der_coordinates_delta[k] = \
                np.dot(der_coordinates_D[k], temp1[k]) + \
                np.dot(self.D[n_index][k], temp2[k])
        # Calculating coordinate derivative of ohat and
        # coordinates weights derivative of atomic_output
        der_coordinates_ohat = {}
        self.der_coordinates_weights_atomic_output[(n_index, i)] = {}
        for k in range(1, N + 2):
            der_coordinates_ohat[k - 1] = []
            for j in range(len(self.der_coordinates_o[(n_index, i)][k - 1])):
                der_coordinates_ohat[k - 1].append(
                    self.der_coordinates_o[(n_index, i)][k - 1][j])
            der_coordinates_ohat[k - 1].append(0.)
            self.der_coordinates_weights_atomic_output[(n_index, i)][k] = \
                np.dot(np.matrix(der_coordinates_ohat[k - 1]).T,
                       np.matrix(self.delta[n_index][k]).T) + \
                np.dot(np.matrix(self.ohat[n_index][k - 1]).T,
                       np.matrix(der_coordinates_delta[k]).T)

    #########################################################################

    def get_variable_der_of_forces(self, self_index,
                                   n_index, n_symbol, i):
        """Adds up partial variable derivatives of forces calculated by
        calculate_variable_der_of_forces."""

        partial_der_variables_square_error = np.zeros(self.ravel.count)

        partial_der_weights_square_error, partial_der_scalings_square_error = \
            self.ravel.to_dicts(partial_der_variables_square_error)

#        partial_der_weights_square_error = self.zero_weights
#        partial_der_scalings_square_error = self.zero_scalings

        N = len(self.o[n_index]) - 2
        for k in range(1, N + 2):
            partial_der_weights_square_error[n_symbol][k] = \
                float(self.scalings[n_symbol]['slope']) * \
                self.der_coordinates_weights_atomic_output[(n_index, i)][k]
        partial_der_scalings_square_error[n_symbol]['slope'] = \
            self.der_coordinates_o[(n_index, i)][N + 1][0]

        partial_der_variables_square_error = \
            self.ravel.to_vector(partial_der_weights_square_error,
                                 partial_der_scalings_square_error)

        return partial_der_variables_square_error

    #########################################################################

    def log(self, log, param, elements, images):
        """Logs and makes variables if not already exist."""

        self.zero_weights = None
        self.elements = elements

        # If hiddenlayers is fed by the user in the tuple format,
        # it will now be converted to a dictionary.
        if isinstance(param['hiddenlayers'], tuple):
            hiddenlayers = {}
            for element in self.elements:
                hiddenlayers[element] = param['hiddenlayers']
            param['hiddenlayers'] = hiddenlayers

        self.hiddenlayers = param['hiddenlayers']

        self.hiddensizes = {}
        for element in self.elements:
            structure = self.hiddenlayers[element]
            if isinstance(structure, str):
                structure = structure.split('-')
            elif isinstance(structure, int):
                structure = [structure]
            else:
                structure = list(structure)
            hiddensizes = [int(part) for part in structure]
            self.hiddensizes[element] = hiddensizes

        self.ravel = _RavelVariables(param, self.elements)

        log('Hidden-layer structure:')
        for item in param['hiddenlayers'].items():
            log(' %2s: %s' % item)

        # If weights are not given, generates random weights
        if not (self.weights or param['variables']):
            log('Initializing with random weights.')
            self.weights = make_weight_matrices(param['Gs'],
                                                self.elements,
                                                self.hiddenlayers,
                                                self.activation)
        else:
            log('Initial weights already present.')
        # If scalings are not given, generates random scalings
        if not (self.scalings or param['variables']):
            log('Initializing with random scalings.')
            self.scalings = make_scalings_matrices(self.elements,
                                                   images,
                                                   self.activation)
        else:
            log('Initial scalings already present.')

        if param['variables'] is None:
            param['variables'] = \
                self.ravel.to_vector(self.weights, self.scalings)
            param.pop('weights')
            param.pop('scalings')

        return param

###############################################################################
###############################################################################
###############################################################################


def make_weight_matrices(Gs, elements, hiddenlayers, activation):
    """Makes random weight matrices from variables according to the convention
    used in Behler (J. Chem. Physc.: atom-centered symmetry functions for
    constructing ...)."""

    if activation == 'linear':
        weight_range = 0.3
    else:
        weight_range = 3.

    weight = {}
    nn_structure = {}
    for element in sorted(elements):
        if isinstance(hiddenlayers[element], int):
            nn_structure[element] = ([len(Gs[element])] +
                                     [hiddenlayers[element]] +
                                     [1])
        else:
            nn_structure[element] = (
                [len(Gs[element])] +
                [layer for layer in hiddenlayers[element]] + [1])
        weight[element] = {}
        normalized_weight_range = weight_range / len(Gs[element])
        weight[element][1] = np.random.random((len(Gs[element]) + 1,
                                               nn_structure[element][1])) * \
            normalized_weight_range - \
            normalized_weight_range / 2.
        for layer in range(len(list(nn_structure[element])) - 3):
            normalized_weight_range = weight_range / \
                nn_structure[element][layer + 1]
            weight[element][layer + 2] = np.random.random(
                (nn_structure[element][layer + 1] + 1,
                 nn_structure[element][layer + 2])) * \
                normalized_weight_range - normalized_weight_range / 2.
        normalized_weight_range = weight_range / nn_structure[element][-2]
        weight[element][len(list(nn_structure[element])) - 1] = \
            np.random.random((nn_structure[element][-2] + 1, 1)) \
            * normalized_weight_range - normalized_weight_range / 2.
        for _ in range(len(weight[element])):  # biases
            size = weight[element][_ + 1][-1].size
            for jj in range(size):
                weight[element][_ + 1][-1][jj] = 0.
    return weight

###############################################################################


def make_scalings_matrices(elements, images, activation):
    """Makes initial scaling matrices, such that the range of activation
    is scaled to the range of actual energies."""

    max_act_energy = max(image.get_potential_energy()
                         for hash_key, image in images.items())
    min_act_energy = min(image.get_potential_energy()
                         for hash_key, image in images.items())

    for hash_key, image in images.items():
        if image.get_potential_energy() == \
                max_act_energy:
            no_atoms_of_max_act_energy = len(image)
        if image.get_potential_energy() == \
                min_act_energy:
            no_atoms_of_min_act_energy = len(image)

    max_act_energy_per_atom = max_act_energy / no_atoms_of_max_act_energy
    min_act_energy_per_atom = min_act_energy / no_atoms_of_min_act_energy

    scaling = {}
    for element in elements:
        scaling[element] = {}
        if activation == 'sigmoid':  # sigmoid activation function
            scaling[element]['intercept'] = min_act_energy_per_atom
            scaling[element]['slope'] = (max_act_energy_per_atom -
                                         min_act_energy_per_atom)
        elif activation == 'tanh':  # tanh activation function
            scaling[element]['intercept'] = (max_act_energy_per_atom +
                                             min_act_energy_per_atom) / 2.
            scaling[element]['slope'] = (max_act_energy_per_atom -
                                         min_act_energy_per_atom) / 2.
        elif activation == 'linear':  # linear activation function
            scaling[element]['intercept'] = (max_act_energy_per_atom +
                                             min_act_energy_per_atom) / 2.
            scaling[element]['slope'] = (10. ** (-10.)) * \
                                        (max_act_energy_per_atom -
                                         min_act_energy_per_atom) / 2.
    del images

    return scaling

###############################################################################
###############################################################################
###############################################################################


class _RavelVariables:

    """Class to ravel and unravel variable values into a single vector.
    This is used for feeding into the optimizer. Feed in a list of
    dictionaries to initialize the shape of the transformation. Note no
    data is saved in the class; each time it is used it is passed either
    the dictionaries or vector.
        param: dictionary that contains hiddenlayers and Gs.
        elements: list of elements
    """

    ##########################################################################

    def __init__(self, param, elements):

        self.count = 0
        self.weightskeys = []
        self.scalingskeys = []

        for element in elements:
            structure = param['hiddenlayers'][element]
            if isinstance(structure, str):
                structure = structure.split('-')
            elif isinstance(structure, int):
                structure = [structure]
            else:
                structure = list(structure)
            hiddensizes = [int(part) for part in structure]
            for layer in range(1, len(hiddensizes) + 2):
                if layer == 1:
                    shape = (len(param['Gs'][element]) + 1, hiddensizes[0])
                elif layer == (len(hiddensizes) + 1):
                    shape = (hiddensizes[layer - 2] + 1, 1)
                else:
                    shape = (
                        hiddensizes[layer - 2] + 1, hiddensizes[layer - 1])
                size = shape[0] * shape[1]
                self.weightskeys.append({'key1': element,
                                         'key2': layer,
                                         'shape': shape,
                                         'size': size})
                self.count += size

        for element in elements:
            self.scalingskeys.append({'key1': element,
                                      'key2': 'intercept'})
            self.scalingskeys.append({'key1': element,
                                      'key2': 'slope'})
            self.count += 2

    #########################################################################

    def to_vector(self, weights, scalings):
        """Puts the weights and scalings embedded dictionaries into a single
        vector and returns it. The dictionaries need to have the identical
        structure to those it was initialized with."""

        vector = np.zeros(self.count)
        count = 0
        for k in sorted(self.weightskeys):
            lweights = np.array(weights[k['key1']][k['key2']]).ravel()
            vector[count:(count + lweights.size)] = lweights
            count += lweights.size
        for k in sorted(self.scalingskeys):
            vector[count] = scalings[k['key1']][k['key2']]
            count += 1
        return vector

    #########################################################################

    def to_dicts(self, vector):
        """Puts the vector back into weights and scalings dictionaries of the
        form initialized. vector must have same length as the output of
        unravel."""

        assert len(vector) == self.count
        count = 0
        weights = OrderedDict()
        scalings = OrderedDict()
        for k in sorted(self.weightskeys):
            if k['key1'] not in weights.keys():
                weights[k['key1']] = OrderedDict()
            matrix = vector[count:count + k['size']]
            matrix = np.array(matrix).flatten()
            matrix = np.matrix(matrix.reshape(k['shape']))
            weights[k['key1']][k['key2']] = matrix
            count += k['size']
        for k in sorted(self.scalingskeys):
            if k['key1'] not in scalings.keys():
                scalings[k['key1']] = OrderedDict()
            scalings[k['key1']][k['key2']] = vector[count]
            count += 1
        return weights, scalings

    #########################################################################

    def calculate_weights_norm_and_der(self, weights):
        """Calculates the norm of weights as well as the vector of its
        derivative to constratint overfitting."""

        count = 0
        weights_norm = 0.
        der_of_weights_norm = np.zeros(self.count)
        for k in sorted(self.weightskeys):
            weight = weights[k['key1']][k['key2']]
            lweights = np.array(weight).ravel()
            # there is no overfitting constraint on the values of biases
            for i in range(np.shape(weight)[1]):
                lweights[- (i + 1)] = 0.
            for i in range(len(lweights)):
                weights_norm += lweights[i] ** 2.
            der_of_weights_norm[count:(count + lweights.size)] = \
                2. * lweights
            count += lweights.size
        for k in sorted(self.scalingskeys):
            # there is no overfitting constraint on the values of scalings
            der_of_weights_norm[count] = 0.
            count += 1
        return weights_norm, der_of_weights_norm

###############################################################################
###############################################################################
###############################################################################
