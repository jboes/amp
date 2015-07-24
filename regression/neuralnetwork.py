#!/usr/bin/env python
"""
Script that contains neural network regression method.

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

    def __init__(self, hiddenlayers=(5, 5), activation='tanh', weights=None,
                 scalings=None, variables=None):

        self.hiddenlayers = hiddenlayers
        self.weights = weights
        self.scalings = scalings
        self.variables = variables

        self.activation = activation
        # Checking that the activation function is given correctly:
        if self.activation not in ['linear', 'tanh', 'sigmoid']:
            raise NotImplementedError('Unknown activation function; '
                                      'activation must be one of '
                                      '"linear", "tanh", or "sigmoid".')

        # Switch to check whether number of atoms are available. Used in the
        # pure atomic-coordinates scheme.
        self.switch = False
        self.no_of_atoms = None

    #########################################################################

    def initialize(self, param, load=None, atoms=None):
        """
        Checks compatibility between fingerprint output dimension and
        regression input dimension.
        """

        self.param = param

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme

            if load is not None:
                self.no_of_atoms = param.no_of_atoms

            if (self.no_of_atoms is not None) and (atoms is not None):
                assert (self.no_of_atoms == len(atoms)), \
                    'Number of atoms in the system should be %i.' \
                    % self.no_of_atoms

            if self.switch is False:

                if atoms is not None:
                    self.no_of_atoms = len(atoms)

                if self.no_of_atoms is not None:

                    self.ravel = \
                        _RavelVariables(hiddenlayers=self.hiddenlayers,
                                        no_of_atoms=self.no_of_atoms)
                    structure = self.hiddenlayers
                    if isinstance(structure, str):
                        structure = structure.split('-')
                    elif isinstance(structure, int):
                        structure = [structure]
                    else:
                        structure = list(structure)
                    self.hiddensizes = [int(part) for part in structure]

                    if load is not None:
                        self.weights, self.scalings = \
                            self.ravel.to_dicts(self.variables)

                    # Checking the compatibility of the forms of coordinates,
                    #  hiddenlayers and weights:
                    string1 = 'number of atoms and weights are not compatible.'
                    string2 = 'hiddenlayers and weights are not compatible.'

                    if self.weights is not None:
                        if np.shape(self.weights[1])[0] != \
                                3 * self.no_of_atoms + 1:
                            raise RuntimeError(string1)
                        if np.shape(self.weights[1])[1] != self.hiddensizes[0]:
                            raise RuntimeError(string2)
                        for _ in range(2, len(self.hiddensizes) + 1):
                            if np.shape(self.weights[_])[0] != \
                                    self.hiddensizes[_ - 2] + 1 or \
                                    np.shape(self.weights[_])[1] != \
                                    self.hiddensizes[_ - 1]:
                                raise RuntimeError(string2)
                    del string1
                    del string2

                    self.switch = True

        else:  # fingerprinting scheme

            Gs = self.param.fingerprint.Gs

            if Gs is not None:

                self.elements = sorted(Gs.keys())

                # If hiddenlayers is fed by the user in the tuple format,
                # it will now be converted to a dictionary.
                if isinstance(self.hiddenlayers, tuple):
                    hiddenlayers = {}
                    for element in self.elements:
                        hiddenlayers[element] = self.hiddenlayers
                    self.hiddenlayers = hiddenlayers

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

                self.ravel = _RavelVariables(hiddenlayers=self.hiddenlayers,
                                             elements=self.elements,
                                             Gs=Gs)

                if load is not None:
                    self.weights, self.scalings = \
                        self.ravel.to_dicts(self.variables)

            # Checking the compatibility of the forms of Gs, hiddenlayers and
            # weights.
            string1 = 'Gs and weights are not compatible.'
            string2 = 'hiddenlayers and weights are not compatible.'
            if isinstance(self.hiddenlayers, dict):
                self.elements = sorted(self.hiddenlayers.keys())
                for element in self.elements:
                    if self.weights is not None:
                        if Gs is not None:
                            if np.shape(self.weights[element][1])[0] \
                                    != len(Gs[element]) + 1:
                                raise RuntimeError(string1)
                        if isinstance(self.hiddenlayers[element], int):
                            if np.shape(self.weights[element][1])[1] \
                                    != self.hiddenlayers[element]:
                                raise RuntimeError(string2)
                        else:
                            if np.shape(self.weights[element][1])[1] \
                                    != self.hiddenlayers[element][0]:
                                raise RuntimeError(string2)
                            for _ in range(2, len(self.hiddenlayers
                                                  [element]) + 1):
                                if (np.shape(self.weights
                                             [element][_])[0] !=
                                        self.hiddenlayers[
                                        element][_ - 2] + 1 or
                                        np.shape(self.weights
                                                 [element][_])[1] !=
                                        self.hiddenlayers
                                        [element][_ - 1]):
                                    raise RuntimeError(string2)
            del string1
            del string2

    #########################################################################

    def ravel_variables(self):
        """Wrapper function for raveling weights and scalings."""

        if (self.param.regression.variables is None) and self.weights:
            self.param.regression.variables = \
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

    #########################################################################

    def update_variables(self, param):
        """Updating variables."""

        self.variables = param.regression.variables
        self.weights, self.scalings = \
            self.ravel.to_dicts(self.variables)

        self.W = {}

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme
            weight = self.weights
            for j in range(len(weight)):
                self.W[j + 1] = np.delete(weight[j + 1], -1, 0)
        else:  # fingerprinting scheme
            for element in self.elements:
                weight = self.weights[element]
                self.W[element] = {}
                for j in range(len(weight)):
                    self.W[element][j + 1] = np.delete(weight[j + 1], -1, 0)

    #########################################################################

    def get_output(self, input, index=None, symbol=None,):
        """Given fingerprints of the indexed atom and its symbol, outputs of
        the neural network are calculated."""

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme
            self.o = {}
            hiddensizes = self.hiddensizes
            weight = self.weights
        else:  # fingerprinting scheme
            self.o[index] = {}
            hiddensizes = self.hiddensizes[symbol]
            weight = self.weights[symbol]

        o = {}  # node values
        layer = 1  # input layer
        net = {}  # excitation
        ohat = {}
        temp = np.zeros((1, len(input) + 1))
        for _ in range(len(input)):
            temp[0, _] = input[_]
        temp[0, len(input)] = 1.0
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

        temp = np.zeros((1, len(input)))
        for _ in range(len(input)):
            temp[0, _] = input[_]

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme

            amp_energy = self.scalings['slope'] * \
                float(o[layer]) + self.scalings['intercept']
            self.o = o
            self.o[0] = temp
            return amp_energy

        else:  # fingerprinting scheme
            atomic_amp_energy = self.scalings[symbol]['slope'] * \
                float(o[layer]) + self.scalings[symbol]['intercept']
            self.o[index] = o
            self.o[index][0] = temp
            return atomic_amp_energy

    #########################################################################

    def get_force(self, i, der_indexfp, n_index=None, n_symbol=None,):
        """Feed-forward for calculating forces."""

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme
            o = self.o
            hiddensizes = self.hiddensizes
            weight = self.weights
        else:  # fingerprinting scheme
            o = self.o[n_index]
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
            for j in range(np.size(o[layer])):
                if self.activation == 'linear':  # linear function
                    der_o[layer].append(float(temp[0, j]))
                elif self.activation == 'sigmoid':  # sigmoid function
                    der_o[layer].append(float(o[layer][0, j] *
                                              (1. -
                                               o[layer][0, j])) *
                                        float(temp[0, j]))
                elif self.activation == 'tanh':  # tanh function
                    der_o[layer].append(
                        float(1. - o[layer][0, j] *
                              o[layer][0, j]) *
                        float(temp[0, j]))
        layer += 1  # output layer
        temp = np.dot(np.matrix(der_o[layer - 1]),
                      np.delete(weight[layer], -1, 0))
        if self.activation == 'linear':  # linear function
            der_o[layer] = float(temp)
        elif self.activation == 'sigmoid':  # sigmoid function
            der_o[layer] = float(o[layer] *
                                 (1. - o[layer]) * temp)
        elif self.activation == 'tanh':  # tanh function
            der_o[layer] = float((1. - o[layer] *
                                  o[layer]) * temp)

        der_o[layer] = [der_o[layer]]

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme
            self.der_coordinates_o[i] = der_o
            force = float(-(self.scalings['slope'] * der_o[layer][0]))
        else:  # fingerprinting scheme
            self.der_coordinates_o[(n_index, i)] = der_o
            force = float(-(self.scalings[n_symbol]['slope'] *
                            der_o[layer][0]))

        return force

    #########################################################################

    def get_variable_der_of_energy(self, index=None, symbol=None):
        """Returns the derivative of energy square error with respect to
        variables."""

        partial_der_variables_square_error = np.zeros(self.ravel.count)

        partial_der_weights_square_error, partial_der_scalings_square_error = \
            self.ravel.to_dicts(partial_der_variables_square_error)

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme
            o = self.o
            W = self.W
        else:  # fingerprinting scheme
            o = self.o[index]
            W = self.W[symbol]

        N = len(o) - 2  # number of hiddenlayers
        D = {}
        for k in range(1, N + 2):
            D[k] = np.zeros(shape=(np.size(o[k]), np.size(o[k])))
            for j in range(np.size(o[k])):
                if self.activation == 'linear':  # linear
                    D[k][j, j] = 1.
                elif self.activation == 'sigmoid':  # sigmoid
                    D[k][j, j] = float(o[k][0, j]) * \
                        float((1. - o[k][0, j]))
                elif self.activation == 'tanh':  # tanh
                    D[k][j, j] = float(1. - o[k][0, j] * o[k][0, j])
        # Calculating delta
        delta = {}
        # output layer
        delta[N + 1] = D[N + 1]
        # hidden layers
        for k in range(N, 0, -1):  # backpropagate starting from output layer
            delta[k] = np.dot(D[k], np.dot(W[k + 1], delta[k + 1]))
        # Calculating ohat
        ohat = {}
        for k in range(1, N + 2):
            ohat[k - 1] = \
                np.zeros(shape=(1, np.size(o[k - 1]) + 1))
            for j in range(np.size(o[k - 1])):
                ohat[k - 1][0, j] = o[k - 1][0, j]
            ohat[k - 1][0, np.size(o[k - 1])] = 1.0

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme
            partial_der_scalings_square_error['intercept'] = 1.
            partial_der_scalings_square_error['slope'] = float(o[N + 1])
            for k in range(1, N + 2):
                partial_der_weights_square_error[k] = \
                    float(self.scalings['slope']) * \
                    np.dot(np.matrix(ohat[k - 1]).T, np.matrix(delta[k]).T)
        else:  # fingerprinting scheme
            partial_der_scalings_square_error[symbol]['intercept'] = 1.
            partial_der_scalings_square_error[symbol]['slope'] = \
                float(o[N + 1])
            for k in range(1, N + 2):
                partial_der_weights_square_error[symbol][k] = \
                    float(self.scalings[symbol]['slope']) * \
                    np.dot(np.matrix(ohat[k - 1]).T, np.matrix(delta[k]).T)
        partial_der_variables_square_error = \
            self.ravel.to_vector(partial_der_weights_square_error,
                                 partial_der_scalings_square_error)

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme
            self.D = D
            self.delta = delta
            self.ohat = ohat
        else:  # fingerprinting scheme
            self.D[index] = D
            self.delta[index] = delta
            self.ohat[index] = ohat

        return partial_der_variables_square_error

    #########################################################################

    def get_variable_der_of_forces(self, self_index, i,
                                   n_index=None, n_symbol=None,):
        """Returns the derivative of force square error with respect to
        variables."""

        partial_der_variables_square_error = np.zeros(self.ravel.count)

        partial_der_weights_square_error, partial_der_scalings_square_error = \
            self.ravel.to_dicts(partial_der_variables_square_error)

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme
            o = self.o
            der_coordinates_o = self.der_coordinates_o[i]
            W = self.W
            delta = self.delta
            ohat = self.ohat
            D = self.D
        else:  # fingerprinting scheme
            o = self.o[n_index]
            der_coordinates_o = self.der_coordinates_o[(n_index, i)]
            W = self.W[n_symbol]
            delta = self.delta[n_index]
            ohat = self.ohat[n_index]
            D = self.D[n_index]

        N = len(o) - 2
        der_coordinates_D = {}
        for k in range(1, N + 2):
            # Calculating coordinate derivative of D matrix
            der_coordinates_D[k] = \
                np.zeros(shape=(np.size(o[k]), np.size(o[k])))
            for j in range(np.size(o[k])):
                if self.activation == 'linear':  # linear
                    der_coordinates_D[k][j, j] = 0.
                elif self.activation == 'tanh':  # tanh
                    der_coordinates_D[k][j, j] = \
                        - 2. * o[k][0, j] * der_coordinates_o[k][j]
                elif self.activation == 'sigmoid':  # sigmoid
                    der_coordinates_D[k][j, j] = der_coordinates_o[k][j] - \
                        2. * o[k][0, j] * der_coordinates_o[k][j]
        # Calculating coordinate derivative of delta
        der_coordinates_delta = {}
        # output layer
        der_coordinates_delta[N + 1] = der_coordinates_D[N + 1]
        # hidden layers
        temp1 = {}
        temp2 = {}
        for k in range(N, 0, -1):
            temp1[k] = np.dot(W[k + 1], delta[k + 1])
            temp2[k] = np.dot(W[k + 1], der_coordinates_delta[k + 1])
            der_coordinates_delta[k] = \
                np.dot(der_coordinates_D[k], temp1[k]) + np.dot(D[k], temp2[k])
        # Calculating coordinate derivative of ohat and
        # coordinates weights derivative of atomic_output
        der_coordinates_ohat = {}
        der_coordinates_weights_atomic_output = {}
        for k in range(1, N + 2):
            der_coordinates_ohat[k - 1] = []
            for j in range(len(der_coordinates_o[k - 1])):
                der_coordinates_ohat[k - 1].append(
                    der_coordinates_o[k - 1][j])
            der_coordinates_ohat[k - 1].append(0.)
            der_coordinates_weights_atomic_output[k] = \
                np.dot(np.matrix(der_coordinates_ohat[k - 1]).T,
                       np.matrix(delta[k]).T) + \
                np.dot(np.matrix(ohat[k - 1]).T,
                       np.matrix(der_coordinates_delta[k]).T)

        if self.param.fingerprint is None:  # pure atomic-coordinates scheme
            for k in range(1, N + 2):
                partial_der_weights_square_error[k] = \
                    float(self.scalings['slope']) * \
                    der_coordinates_weights_atomic_output[k]
            partial_der_scalings_square_error['slope'] = \
                der_coordinates_o[N + 1][0]
        else:  # fingerprinting scheme
            for k in range(1, N + 2):
                partial_der_weights_square_error[n_symbol][k] = \
                    float(self.scalings[n_symbol]['slope']) * \
                    der_coordinates_weights_atomic_output[k]
            partial_der_scalings_square_error[n_symbol]['slope'] = \
                der_coordinates_o[N + 1][0]

        partial_der_variables_square_error = \
            self.ravel.to_vector(partial_der_weights_square_error,
                                 partial_der_scalings_square_error)

        return partial_der_variables_square_error

    #########################################################################

    def log(self, log, param, elements, images):
        """Logs and makes variables if not already exist."""

        self.elements = elements

        if param.fingerprint is None:  # pure atomic-coordinates scheme

            self.no_of_atoms = param.no_of_atoms
            self.hiddenlayers = param.regression.hiddenlayers

            structure = self.hiddenlayers
            if isinstance(structure, str):
                structure = structure.split('-')
            elif isinstance(structure, int):
                structure = [structure]
            else:
                structure = list(structure)
            hiddensizes = [int(part) for part in structure]
            self.hiddensizes = hiddensizes

            self.ravel = _RavelVariables(
                hiddenlayers=self.hiddenlayers,
                elements=self.elements,
                no_of_atoms=self.no_of_atoms)

        else:  # fingerprinting scheme

            # If hiddenlayers is fed by the user in the tuple format,
            # it will now be converted to a dictionary.
            if isinstance(param.regression.hiddenlayers, tuple):
                hiddenlayers = {}
                for element in self.elements:
                    hiddenlayers[element] = param.regression.hiddenlayers
                param.regression.hiddenlayers = hiddenlayers

            self.hiddenlayers = param.regression.hiddenlayers

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

            self.ravel = _RavelVariables(hiddenlayers=self.hiddenlayers,
                                         elements=self.elements,
                                         Gs=param.fingerprint.Gs)

        log('Hidden-layer structure:')
        if param.fingerprint is None:  # pure atomic-coordinates scheme
            log(' %s' % str(self.hiddenlayers))
        else:  # fingerprinting scheme
            for item in self.hiddenlayers.items():
                log(' %2s: %s' % item)

        # If weights are not given, generates random weights
        if not (self.weights or self.variables):
            log('Initializing with random weights.')
            if param.fingerprint is None:  # pure atomic-coordinates scheme
                self.weights = make_weight_matrices(self.hiddenlayers,
                                                    self.activation,
                                                    self.no_of_atoms)
            else:  # fingerprinting scheme
                self.weights = make_weight_matrices(self.hiddenlayers,
                                                    self.activation,
                                                    None,
                                                    param.fingerprint.Gs,
                                                    self.elements,)

        else:
            log('Initial weights already present.')
        # If scalings are not given, generates random scalings
        if not (self.scalings or self.variables):
            log('Initializing with random scalings.')

            if param.fingerprint is None:  # pure atomic-coordinates scheme
                self.scalings = make_scalings_matrices(images,
                                                       self.activation,)
            else:  # fingerprinting scheme
                self.scalings = make_scalings_matrices(images,
                                                       self.activation,
                                                       self.elements,)
        else:
            log('Initial scalings already present.')

        if self.variables is None:
            param.regression.variables = \
                self.ravel.to_vector(self.weights, self.scalings)

        return param

###############################################################################
###############################################################################
###############################################################################


def make_weight_matrices(hiddenlayers, activation, no_of_atoms=None, Gs=None,
                         elements=None):
    """Makes random weight matrices from variables."""

    if activation == 'linear':
        weight_range = 0.3
    else:
        weight_range = 3.

    weight = {}
    nn_structure = {}

    if no_of_atoms is not None:  # pure atomic-coordinates scheme

        if isinstance(hiddenlayers, int):
            nn_structure = ([3 * no_of_atoms] + [hiddenlayers] + [1])
        else:
            nn_structure = (
                [3 * no_of_atoms] +
                [layer for layer in hiddenlayers] + [1])
        weight = {}
        normalized_weight_range = weight_range / (3 * no_of_atoms)
        weight[1] = np.random.random((3 * no_of_atoms + 1,
                                      nn_structure[1])) * \
            normalized_weight_range - \
            normalized_weight_range / 2.
        for layer in range(len(list(nn_structure)) - 3):
            normalized_weight_range = weight_range / \
                nn_structure[layer + 1]
            weight[layer + 2] = np.random.random(
                (nn_structure[layer + 1] + 1,
                 nn_structure[layer + 2])) * \
                normalized_weight_range - normalized_weight_range / 2.
        normalized_weight_range = weight_range / nn_structure[-2]
        weight[len(list(nn_structure)) - 1] = \
            np.random.random((nn_structure[-2] + 1, 1)) \
            * normalized_weight_range - normalized_weight_range / 2.
        for _ in range(len(weight)):  # biases
            size = weight[_ + 1][-1].size
            for jj in range(size):
                weight[_ + 1][-1][jj] = 0.

    else:

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
                                                   nn_structure[
                                                   element][1])) * \
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


def make_scalings_matrices(images, activation, elements=None):
    """Makes initial scaling matrices, such that the range of activation
    is scaled to the range of actual energies."""

    max_act_energy = max(image.get_potential_energy(apply_constraint=False)
                         for hash_key, image in images.items())
    min_act_energy = min(image.get_potential_energy(apply_constraint=False)
                         for hash_key, image in images.items())

    for hash_key, image in images.items():
        if image.get_potential_energy(apply_constraint=False) == \
                max_act_energy:
            no_atoms_of_max_act_energy = len(image)
        if image.get_potential_energy(apply_constraint=False) == \
                min_act_energy:
            no_atoms_of_min_act_energy = len(image)

    max_act_energy_per_atom = max_act_energy / no_atoms_of_max_act_energy
    min_act_energy_per_atom = min_act_energy / no_atoms_of_min_act_energy

    scaling = {}

    if elements is None:  # pure atomic-coordinates scheme

        scaling = {}
        if activation == 'sigmoid':  # sigmoid activation function
            scaling['intercept'] = min_act_energy_per_atom
            scaling['slope'] = (max_act_energy_per_atom -
                                min_act_energy_per_atom)
        elif activation == 'tanh':  # tanh activation function
            scaling['intercept'] = (max_act_energy_per_atom +
                                    min_act_energy_per_atom) / 2.
            scaling['slope'] = (max_act_energy_per_atom -
                                min_act_energy_per_atom) / 2.
        elif activation == 'linear':  # linear activation function
            scaling['intercept'] = (max_act_energy_per_atom +
                                    min_act_energy_per_atom) / 2.
            scaling['slope'] = (10. ** (-10.)) * \
                (max_act_energy_per_atom -
                 min_act_energy_per_atom) / 2.

    else:  # fingerprinting scheme

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
    """

    ##########################################################################

    def __init__(self, hiddenlayers, elements=None, Gs=None, no_of_atoms=None):

        self.no_of_atoms = no_of_atoms
        self.count = 0
        self.weightskeys = []
        self.scalingskeys = []

        if self.no_of_atoms is None:  # fingerprinting scheme

            for element in elements:
                structure = hiddenlayers[element]
                if isinstance(structure, str):
                    structure = structure.split('-')
                elif isinstance(structure, int):
                    structure = [structure]
                else:
                    structure = list(structure)
                hiddensizes = [int(part) for part in structure]
                for layer in range(1, len(hiddensizes) + 2):
                    if layer == 1:
                        shape = (len(Gs[element]) + 1, hiddensizes[0])
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

        else:  # pure atomic-coordinates scheme

            structure = hiddenlayers
            if isinstance(structure, str):
                structure = structure.split('-')
            elif isinstance(structure, int):
                structure = [structure]
            else:
                structure = list(structure)
            hiddensizes = [int(part) for part in structure]
            for layer in range(1, len(hiddensizes) + 2):
                if layer == 1:
                    shape = (3 * no_of_atoms + 1, hiddensizes[0])
                elif layer == (len(hiddensizes) + 1):
                    shape = (hiddensizes[layer - 2] + 1, 1)
                else:
                    shape = (
                        hiddensizes[layer - 2] + 1, hiddensizes[layer - 1])
                size = shape[0] * shape[1]
                self.weightskeys.append({'key': layer,
                                         'shape': shape,
                                         'size': size})
                self.count += size

            self.scalingskeys.append({'key': 'intercept'})
            self.scalingskeys.append({'key': 'slope'})
            self.count += 2

    #########################################################################

    def to_vector(self, weights, scalings):
        """Puts the weights and scalings embedded dictionaries into a single
        vector and returns it. The dictionaries need to have the identical
        structure to those it was initialized with."""

        vector = np.zeros(self.count)
        count = 0
        for k in sorted(self.weightskeys):
            if self.no_of_atoms is None:  # fingerprinting scheme
                lweights = np.array(weights[k['key1']][k['key2']]).ravel()
            else:  # pure atomic-coordinates scheme
                lweights = (np.array(weights[k['key']])).ravel()
            vector[count:(count + lweights.size)] = lweights
            count += lweights.size
        for k in sorted(self.scalingskeys):
            if self.no_of_atoms is None:  # fingerprinting scheme
                vector[count] = scalings[k['key1']][k['key2']]
            else:  # pure atomic-coordinates scheme
                vector[count] = scalings[k['key']]
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
            if self.no_of_atoms is None:  # fingerprinting scheme
                if k['key1'] not in weights.keys():
                    weights[k['key1']] = OrderedDict()
            matrix = vector[count:count + k['size']]
            matrix = np.array(matrix).flatten()
            matrix = np.matrix(matrix.reshape(k['shape']))
            if self.no_of_atoms is None:  # fingerprinting scheme
                weights[k['key1']][k['key2']] = matrix
            else:  # pure atomic-coordinates scheme
                weights[k['key']] = matrix
            count += k['size']
        for k in sorted(self.scalingskeys):
            if self.no_of_atoms is None:  # fingerprinting scheme
                if k['key1'] not in scalings.keys():
                    scalings[k['key1']] = OrderedDict()
                scalings[k['key1']][k['key2']] = vector[count]
            else:  # pure atomic-coordinates scheme
                scalings[k['key']] = vector[count]
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
            if self.no_of_atoms is None:  # fingerprinting scheme
                weight = weights[k['key1']][k['key2']]
            else:  # pure atomic-coordinates scheme
                weight = weights[k['key']]
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
