#!/usr/bin/env python
"""
Script that contains neural network regression method.
"""
import numpy as np
from collections import OrderedDict
try:
    from .. import fmodules
except ImportError:
    fmodules = None

###############################################################################


class NeuralNetwork:

    """
    Class that implements a basic feed-forward neural network.

    :param hiddenlayers: Dictionary of chemical element symbols and
                         architectures of their corresponding hidden layers of
                         the conventional neural network. Number of nodes of
                         last layer is always one corresponding to energy.
                         However, number of nodes of first layer is equal to
                         three times number of atoms in the system in the case
                         of no descriptor, and is equal to length of symmetry
                         functions of the descriptor. Can be fed as:

                         >>> hiddenlayers = (3, 2,)

                         for example, in which a neural network with two hidden
                         layers, the first one having three nodes and the
                         second one having two nodes is assigned (to the whole
                         atomic system in the no descriptor case, and to each
                         chemical element in the fingerprinting scheme).
                         In the fingerprinting scheme, neural network for each
                         species can be assigned seperately, as:

                         >>> hiddenlayers = {"O":(3,5), "Au":(5,6)}

                         for example.
    :type hiddenlayers: dict
    :param activation: Assigns the type of activation funtion. "linear" refers
                       to linear function, "tanh" refers to tanh function, and
                       "sigmoid" refers to sigmoid function.
    :type activation: str
    :param weights: In the case of no descriptor, keys correspond to layers
                    and values are two dimensional arrays of network weight.
                    In the fingerprinting scheme, keys correspond to chemical
                    elements and values are dictionaries with layer keys and
                    network weight two dimensional arrays as values. Arrays are
                    set up to connect node i in the previous layer with node j
                    in the current layer with indices w[i,j]. The last value
                    for index i corresponds to bias. If weights is not given,
                    arrays will be randomly generated.
    :type weights: dict
    :param scalings: In the case of no descriptor, keys are "intercept" and
                     "slope" and values are real numbers. In the fingerprinting
                     scheme, keys correspond to chemical elements and values
                     are dictionaries with "intercept" and "slope" keys and
                     real number values. If scalings is not given, it will be
                     randomly generated.
    :type scalings: dict

    .. note:: Dimensions of weight two dimensional arrays should be consistent
              with hiddenlayers.

    :raises: RuntimeError, NotImplementedError
    """
    ###########################################################################

    def __init__(self, hiddenlayers=(1, 1), activation='tanh', weights=None,
                 scalings=None,):

        self.hiddenlayers = hiddenlayers
        self._weights = weights
        self._scalings = scalings

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
        self.global_search = False
        self._variables = None

    ###########################################################################

    def initialize(self, param, load=None, atoms=None):
        """
        Loads parameters if fed by a json file. Also checks compatibility
        between dimensions of first layer and hidden layers with weights.

        :param param: Object containing symmetry function's (if any) and
                      regression's properties.
        :type param: ASE calculator's Parameters class
        :param load: Path for loading an existing Amp calculator.
        :type load: str
        :param atoms: Only used for the case of no descriptor.
        :type atoms: ASE atoms object.
        """
        self.param = param

        if self.param.descriptor is None:  # pure atomic-coordinates scheme

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

                    structure = self.hiddenlayers
                    if isinstance(structure, str):
                        structure = structure.split('-')
                    elif isinstance(structure, int):
                        structure = [structure]
                    else:
                        structure = list(structure)
                    self.hiddenlayers = [int(part) for part in structure]

                    self.ravel = \
                        _RavelVariables(hiddenlayers=self.hiddenlayers,
                                        no_of_atoms=self.no_of_atoms)

                    if load is not None:
                        self._weights, self._scalings = \
                            self.ravel.to_dicts(self._variables)

                    # Checking the compatibility of the forms of coordinates,
                    #  hiddenlayers and weights:
                    string1 = 'number of atoms and weights are not compatible.'
                    string2 = 'hiddenlayers and weights are not compatible.'

                    if self._weights is not None:
                        if np.shape(self._weights[1])[0] != \
                                3 * self.no_of_atoms + 1:
                            raise RuntimeError(string1)
                        if np.shape(self._weights[1])[1] != \
                                self.hiddenlayers[0]:
                            raise RuntimeError(string2)
                        for _ in range(2, len(self.hiddenlayers) + 1):
                            if np.shape(self._weights[_])[0] != \
                                    self.hiddenlayers[_ - 2] + 1 or \
                                    np.shape(self._weights[_])[1] != \
                                    self.hiddenlayers[_ - 1]:
                                raise RuntimeError(string2)
                    del string1
                    del string2

                    self.switch = True

        else:  # fingerprinting scheme

            Gs = self.param.descriptor.Gs

            if Gs is not None:

                self.elements = sorted(Gs.keys())

                # If hiddenlayers is fed by the user in the tuple format,
                # it will now be converted to a dictionary.
                if isinstance(self.hiddenlayers, tuple):
                    hiddenlayers = {}
                    for element in self.elements:
                        hiddenlayers[element] = self.hiddenlayers
                    self.hiddenlayers = hiddenlayers

                for element in self.elements:
                    structure = self.hiddenlayers[element]
                    if isinstance(structure, str):
                        structure = structure.split('-')
                    elif isinstance(structure, int):
                        structure = [structure]
                    else:
                        structure = list(structure)
                    hiddenlayers = [int(part) for part in structure]
                    self.hiddenlayers[element] = hiddenlayers

                self.ravel = _RavelVariables(hiddenlayers=self.hiddenlayers,
                                             elements=self.elements,
                                             Gs=Gs)

                if load is not None:
                    self._weights, self._scalings = \
                        self.ravel.to_dicts(self._variables)

            # Checking the compatibility of the forms of Gs, hiddenlayers and
            # weights.
            string1 = 'Gs and weights are not compatible.'
            string2 = 'hiddenlayers and weights are not compatible.'
            if isinstance(self.hiddenlayers, dict):
                self.elements = sorted(self.hiddenlayers.keys())
                for element in self.elements:
                    if self._weights is not None:
                        if Gs is not None:
                            if np.shape(self._weights[element][1])[0] \
                                    != len(Gs[element]) + 1:
                                raise RuntimeError(string1)
                        if isinstance(self.hiddenlayers[element], int):
                            if np.shape(self._weights[element][1])[1] \
                                    != self.hiddenlayers[element]:
                                raise RuntimeError(string2)
                        else:
                            if np.shape(self._weights[element][1])[1] \
                                    != self.hiddenlayers[element][0]:
                                raise RuntimeError(string2)
                            for _ in range(2, len(self.hiddenlayers
                                                  [element]) + 1):
                                if (np.shape(self._weights
                                             [element][_])[0] !=
                                        self.hiddenlayers[
                                        element][_ - 2] + 1 or
                                        np.shape(self._weights
                                                 [element][_])[1] !=
                                        self.hiddenlayers
                                        [element][_ - 1]):
                                    raise RuntimeError(string2)
            del string1
            del string2

    ###########################################################################

    def ravel_variables(self):
        """
        Wrapper function for raveling weights and scalings into a list.

        :returns: param: Object containing regression's properties.
        """
        if (self.param.regression._variables is None) and self._weights:
            self.param.regression._variables = \
                self.ravel.to_vector(self._weights, self._scalings)

        return self.param

    ###########################################################################

    def reset_energy(self):
        """
        Resets local variables corresponding to energy.
        """
        self.o = {}
        self.D = {}
        self.delta = {}
        self.ohat = {}

    ###########################################################################

    def reset_forces(self):
        """
        Resets local variables corresponding to forces.
        """
        self.der_coordinates_o = {}

    ###########################################################################

    def update_variables(self, param):
        """
        Updates variables.

        :param param: Object containing regression's properties.
        :type param: ASE calculator's Parameters class
        """
        self._variables = param.regression._variables
        self._weights, self._scalings = \
            self.ravel.to_dicts(self._variables)

        self.W = {}

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            weight = self._weights
            for j in range(len(weight)):
                self.W[j + 1] = np.delete(weight[j + 1], -1, 0)
        else:  # fingerprinting scheme
            for element in self.elements:
                weight = self._weights[element]
                self.W[element] = {}
                for j in range(len(weight)):
                    self.W[element][j + 1] = np.delete(weight[j + 1], -1, 0)

    ###########################################################################

    def introduce_variables(self, log, param):
        """
        Introducing new variables.

        :param log: Write function at which to log data. Note this must be a
                    callable function.
        :type log: Logger object
        :param param: Object containing regression's properties.
        :type param: ASE calculator's Parameters class
        """
        log('Introducing new hidden-layer nodes...')

        self._weights, self._scalings = \
            self.ravel.to_dicts(param.regression._variables)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            for j in range(1, len(self._weights) + 1):
                shape = np.shape(self._weights[j])
                if j == 1:
                    self._weights[j] = \
                        np.insert(self._weights[j],
                                  shape[1],
                                  shape[0] * [0],
                                  1)
                elif j == len(self._weights):
                    self._weights[j] = \
                        np.insert(self._weights[j],
                                  -1,
                                  shape[1] * [0],
                                  0)
                else:
                    self._weights[j] = \
                        np.insert(self._weights[j],
                                  shape[1],
                                  shape[0] * [0],
                                  1)
                    self._weights[j] = \
                        np.insert(self._weights[j],
                                  -1,
                                  (shape[1] + 1) * [0],
                                  0)

        else:  # fingerprinting scheme
            for element in self.elements:
                for j in range(1, len(self._weights[element]) + 1):
                    shape = np.shape(self._weights[element][j])
                    if j == 1:
                        self._weights[element][j] = \
                            np.insert(self._weights[element][j],
                                      shape[1],
                                      shape[0] * [0],
                                      1)
                    elif j == len(self._weights[element]):
                        self._weights[element][j] = \
                            np.insert(self._weights[element][j],
                                      -1,
                                      shape[1] * [0],
                                      0)
                    else:
                        self._weights[element][j] = \
                            np.insert(self._weights[element][j],
                                      shape[1],
                                      shape[0] * [0],
                                      1)
                        self._weights[element][j] = \
                            np.insert(self._weights[element][j],
                                      -1,
                                      (shape[1] + 1) * [0],
                                      0)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            for _ in range(len(self.hiddenlayers)):
                self.hiddenlayers[_] += 1
            self.ravel = _RavelVariables(hiddenlayers=self.hiddenlayers,
                                         no_of_atoms=self.no_of_atoms)
        else:  # fingerprinting scheme
            for element in self.elements:
                for _ in range(len(self.hiddenlayers[element])):
                    self.hiddenlayers[element][_] += 1
            self.ravel = _RavelVariables(hiddenlayers=self.hiddenlayers,
                                         elements=self.elements,
                                         Gs=param.descriptor.Gs)

        self._variables = \
            self.ravel.to_vector(self._weights, self._scalings)

        param.regression._variables = self._variables

        log('Hidden-layer structure:')
        if param.descriptor is None:  # pure atomic-coordinates scheme
            log(' %s' % str(self.hiddenlayers))
        else:  # fingerprinting scheme
            for item in self.hiddenlayers.items():
                log(' %2s: %s' % item)

        param.regression.hiddenlayers = self.hiddenlayers
        self.hiddenlayers = self.hiddenlayers

        return param

    ###########################################################################

    def get_energy(self, input, index=None, symbol=None,):
        """
        Given input to the neural network, output (which corresponds to energy)
        is calculated.

        :param index: Index of the atom for which atomic energy is calculated
                      (only used in the fingerprinting scheme)
        :type index: int
        :param symbol: Index of the atom for which atomic energy is calculated
                       (only used in the fingerprinting scheme)
        :type symbol: str

        :returns: float -- energy
        """
        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            self.o = {}
            hiddenlayers = self.hiddenlayers
            weight = self._weights
        else:  # fingerprinting scheme
            self.o[index] = {}
            hiddenlayers = self.hiddenlayers[symbol]
            weight = self._weights[symbol]

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
        for hiddenlayer in hiddenlayers[1:]:
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

        del hiddenlayers, weight, ohat, net

        temp = np.zeros((1, len(input)))
        for _ in range(len(input)):
            temp[0, _] = input[_]

        if self.param.descriptor is None:  # pure atomic-coordinates scheme

            amp_energy = self._scalings['slope'] * \
                float(o[layer]) + self._scalings['intercept']
            self.o = o
            self.o[0] = temp
            return amp_energy

        else:  # fingerprinting scheme
            atomic_amp_energy = self._scalings[symbol]['slope'] * \
                float(o[layer]) + self._scalings[symbol]['intercept']
            self.o[index] = o
            self.o[index][0] = temp
            return atomic_amp_energy

    ###########################################################################

    def get_force(self, i, der_indexfp, n_index=None, n_symbol=None,):
        """
        Given derivative of input to the neural network, derivative of output
        (which corresponds to forces) is calculated.

        :param i: Direction of force.
        :type i: int
        :param der_indexfp: List of derivatives of inputs
        :type der_indexfp: list
        :param n_index: Index of the neighbor atom which force is acting at.
                        (only used in the fingerprinting scheme)
        :type n_index: int
        :param n_symbol: Symbol of the neighbor atom which force is acting at.
                         (only used in the fingerprinting scheme)
        :type n_symbol: str

        :returns: float -- force
        """
        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            o = self.o
            hiddenlayers = self.hiddenlayers
            weight = self._weights
        else:  # fingerprinting scheme
            o = self.o[n_index]
            hiddenlayers = self.hiddenlayers[n_symbol]
            weight = self._weights[n_symbol]

        der_o = {}  # node values
        der_o[0] = der_indexfp
        layer = 0  # input layer
        for hiddenlayer in hiddenlayers[0:]:
            layer += 1
            temp = np.dot(np.matrix(der_o[layer - 1]),
                          np.delete(weight[layer], -1, 0))
            der_o[layer] = [None] * np.size(o[layer])
            count = 0
            for j in range(np.size(o[layer])):
                if self.activation == 'linear':  # linear function
                    der_o[layer][count] = float(temp[0, j])
                elif self.activation == 'sigmoid':  # sigmoid function
                    der_o[layer][count] = float(temp[0, j]) * \
                        float(o[layer][0, j] * (1. - o[layer][0, j]))
                elif self.activation == 'tanh':  # tanh function
                    der_o[layer][count] = float(temp[0, j]) * \
                        float(1. - o[layer][0, j] * o[layer][0, j])
                count += 1
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

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            self.der_coordinates_o[i] = der_o
            force = float(-(self._scalings['slope'] * der_o[layer][0]))
        else:  # fingerprinting scheme
            self.der_coordinates_o[(n_index, i)] = der_o
            force = float(-(self._scalings[n_symbol]['slope'] *
                            der_o[layer][0]))

        return force

    ###########################################################################

    def get_variable_der_of_energy(self, index=None, symbol=None):
        """
        Returns the derivative of energy square error with respect to
        variables.

        :param index: Index of the atom for which atomic energy is calculated
                      (only used in the fingerprinting scheme)
        :type index: int
        :param symbol: Index of the atom for which atomic energy is calculated
                       (only used in the fingerprinting scheme)
        :type symbol: str

        :returns: list of float -- the value of the derivative of energy square
                                   error with respect to variables.
        """
        partial_der_variables_square_error = np.zeros(self.ravel.count)

        partial_der_weights_square_error, partial_der_scalings_square_error = \
            self.ravel.to_dicts(partial_der_variables_square_error)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
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

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            partial_der_scalings_square_error['intercept'] = 1.
            partial_der_scalings_square_error['slope'] = float(o[N + 1])
            for k in range(1, N + 2):
                partial_der_weights_square_error[k] = \
                    float(self._scalings['slope']) * \
                    np.dot(np.matrix(ohat[k - 1]).T, np.matrix(delta[k]).T)
        else:  # fingerprinting scheme
            partial_der_scalings_square_error[symbol]['intercept'] = 1.
            partial_der_scalings_square_error[symbol]['slope'] = \
                float(o[N + 1])
            for k in range(1, N + 2):
                partial_der_weights_square_error[symbol][k] = \
                    float(self._scalings[symbol]['slope']) * \
                    np.dot(np.matrix(ohat[k - 1]).T, np.matrix(delta[k]).T)
        partial_der_variables_square_error = \
            self.ravel.to_vector(partial_der_weights_square_error,
                                 partial_der_scalings_square_error)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            self.D = D
            self.delta = delta
            self.ohat = ohat
        else:  # fingerprinting scheme
            self.D[index] = D
            self.delta[index] = delta
            self.ohat[index] = ohat

        return partial_der_variables_square_error

    ###########################################################################

    def get_variable_der_of_forces(self, self_index, i,
                                   n_index=None, n_symbol=None,):
        """
        Returns the derivative of force square error with respect to variables.

        :param self_index: Index of the center atom.
        :type self_index: int
        :param i: Direction of force.
        :type i: int
        :param n_index: Index of the neighbor atom which force is acting at.
                        (only used in the fingerprinting scheme)
        :type n_index: int
        :param n_symbol: Symbol of the neighbor atom which force is acting at.
                         (only used in the fingerprinting scheme)
        :type n_symbol: str

        :returns: list of float -- the value of the derivative of force square
                                   error with respect to variables.
        """
        partial_der_variables_square_error = np.zeros(self.ravel.count)

        partial_der_weights_square_error, partial_der_scalings_square_error = \
            self.ravel.to_dicts(partial_der_variables_square_error)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
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
            der_coordinates_ohat[k - 1] = \
                [None] * (1 + len(der_coordinates_o[k - 1]))
            count = 0
            for j in range(len(der_coordinates_o[k - 1])):
                der_coordinates_ohat[k - 1][count] = \
                    der_coordinates_o[k - 1][j]
                count += 1
            der_coordinates_ohat[k - 1][count] = 0.
            der_coordinates_weights_atomic_output[k] = \
                np.dot(np.matrix(der_coordinates_ohat[k - 1]).T,
                       np.matrix(delta[k]).T) + \
                np.dot(np.matrix(ohat[k - 1]).T,
                       np.matrix(der_coordinates_delta[k]).T)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            for k in range(1, N + 2):
                partial_der_weights_square_error[k] = \
                    float(self._scalings['slope']) * \
                    der_coordinates_weights_atomic_output[k]
            partial_der_scalings_square_error['slope'] = \
                der_coordinates_o[N + 1][0]
        else:  # fingerprinting scheme
            for k in range(1, N + 2):
                partial_der_weights_square_error[n_symbol][k] = \
                    float(self._scalings[n_symbol]['slope']) * \
                    der_coordinates_weights_atomic_output[k]
            partial_der_scalings_square_error[n_symbol]['slope'] = \
                der_coordinates_o[N + 1][0]
        partial_der_variables_square_error = \
            self.ravel.to_vector(partial_der_weights_square_error,
                                 partial_der_scalings_square_error)

        return partial_der_variables_square_error

    ###########################################################################

    def log(self, log, param, elements, images):
        """
        Prints out in the log file and generates variables if do not already
        exist.

        :param log: Write function at which to log data. Note this must be a
                    callable function.
        :type log: Logger object
        :param param: Object containing regression's properties.
        :type param: ASE calculator's Parameters class
        :param elements: List of atom symbols.
        :type elements: list of str
        :param images: ASE atoms objects (the training set).
        :type images: dict

        :returns: Object containing regression's properties.
        """
        self.elements = elements

        if param.descriptor is None:  # pure atomic-coordinates scheme

            self.no_of_atoms = param.no_of_atoms
            self.hiddenlayers = param.regression.hiddenlayers

            structure = self.hiddenlayers
            if isinstance(structure, str):
                structure = structure.split('-')
            elif isinstance(structure, int):
                structure = [structure]
            else:
                structure = list(structure)
            hiddenlayers = [int(part) for part in structure]
            self.hiddenlayers = hiddenlayers

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

            for element in self.elements:
                structure = self.hiddenlayers[element]
                if isinstance(structure, str):
                    structure = structure.split('-')
                elif isinstance(structure, int):
                    structure = [structure]
                else:
                    structure = list(structure)
                hiddenlayers = [int(part) for part in structure]
                self.hiddenlayers[element] = hiddenlayers

            self.ravel = _RavelVariables(hiddenlayers=self.hiddenlayers,
                                         elements=self.elements,
                                         Gs=param.descriptor.Gs)

        log('Hidden-layer structure:')
        if param.descriptor is None:  # pure atomic-coordinates scheme
            log(' %s' % str(self.hiddenlayers))
        else:  # fingerprinting scheme
            for item in self.hiddenlayers.items():
                log(' %2s: %s' % item)

        # If weights are not given, generates random weights
        if not (self._weights or self._variables):
            self.global_search = True
            log('Initializing with random weights.')
            if param.descriptor is None:  # pure atomic-coordinates scheme
                self._weights = make_weight_matrices(self.hiddenlayers,
                                                     self.activation,
                                                     self.no_of_atoms)
            else:  # fingerprinting scheme
                self._weights = make_weight_matrices(self.hiddenlayers,
                                                     self.activation,
                                                     None,
                                                     param.descriptor.Gs,
                                                     self.elements,)

        else:
            log('Initial weights already present.')
        # If scalings are not given, generates random scalings
        if not (self._scalings or self._variables):
            log('Initializing with random scalings.')

            if param.descriptor is None:  # pure atomic-coordinates scheme
                self._scalings = make_scalings_matrices(images,
                                                        self.activation,)
            else:  # fingerprinting scheme
                self._scalings = make_scalings_matrices(images,
                                                        self.activation,
                                                        self.elements,)
        else:
            log('Initial scalings already present.')

        if self._variables is None:
            param.regression._variables = \
                self.ravel.to_vector(self._weights, self._scalings)

        return param

    ###########################################################################

    def send_data_to_fortran(self, param):
        """
        Sends regression data to fortran.

        :param param: Object containing symmetry function's (if any) and
                      regression's properties.
        :type param: ASE calculator's Parameters class
        """
        if param.descriptor is None:
            fingerprinting = False
        else:
            fingerprinting = True

        if fingerprinting:
            no_layers_of_elements = \
                [3 if isinstance(param.regression.hiddenlayers[elm], int)
                 else (len(param.regression.hiddenlayers[elm]) + 2)
                 for elm in self.elements]
            nn_structure = OrderedDict()
            for elm in self.elements:
                if isinstance(param.regression.hiddenlayers[elm], int):
                    nn_structure[elm] = ([len(param.descriptor.Gs[elm])] +
                                         [param.regression.hiddenlayers[elm]] +
                                         [1])
                else:
                    nn_structure[elm] = ([len(param.descriptor.Gs[elm])] +
                                         [layer for layer in
                                          param.regression.hiddenlayers[elm]] +
                                         [1])

            no_nodes_of_elements = [nn_structure[elm][_]
                                    for elm in self.elements
                                    for _ in range(len(nn_structure[elm]))]

        else:
            no_layers_of_elements = []
            if isinstance(param.regression.hiddenlayers, int):
                no_layers_of_elements = [3]
            else:
                no_layers_of_elements = \
                    [len(param.regression.hiddenlayers) + 2]
            if isinstance(param.regression.hiddenlayers, int):
                nn_structure = ([3 * param.no_of_atoms] +
                                [param.regression.hiddenlayers] + [1])
            else:
                nn_structure = ([3 * param.no_of_atoms] +
                                [layer for layer in
                                 param.regression.hiddenlayers] + [1])
            no_nodes_of_elements = [nn_structure[_]
                                    for _ in range(len(nn_structure))]

        fmodules.regression.no_layers_of_elements = no_layers_of_elements
        fmodules.regression.no_nodes_of_elements = no_nodes_of_elements
        if param.regression.activation == 'tanh':
            activation_signal = 1
        elif param.regression.activation == 'sigmoid':
            activation_signal = 2
        elif param.regression.activation == 'linear':
            activation_signal = 3
        fmodules.regression.activation_signal = activation_signal

###############################################################################
###############################################################################
###############################################################################


def make_weight_matrices(hiddenlayers, activation, no_of_atoms=None, Gs=None,
                         elements=None):
    """
    Generates random weight arrays from variables.

    :param hiddenlayers: Dictionary of chemical element symbols and
                         architectures of their corresponding hidden layers of
                         the conventional neural network. Number of nodes of
                         last layer is always one corresponding to energy.
                         However, number of nodes of first layer is equal to
                         three times number of atoms in the system in the case
                         of no descriptor, and is equal to length of symmetry
                         functions in the fingerprinting scheme. Can be fed as:

                         >>> hiddenlayers = (3, 2,)

                         for example, in which a neural network with two hidden
                         layers, the first one having three nodes and the
                         second one having two nodes is assigned (to the whole
                         atomic system in the case of no descriptor, and to
                         each chemical element in the fingerprinting scheme).
                         In the fingerprinting scheme, neural network for each
                         species can be assigned seperately, as:

                         >>> hiddenlayers = {"O":(3,5), "Au":(5,6)}

                         for example.
    :type hiddenlayers: dict
    :param activation: Assigns the type of activation funtion. "linear" refers
                       to linear function, "tanh" refers to tanh function, and
                       "sigmoid" refers to sigmoid function.
    :type activation: str
    :param no_of_atoms: Number of atoms in atomic systems; used only in the
                        case of no descriptor.
    :type no_of_atoms: int
    :param Gs: Dictionary of symbols and lists of dictionaries for making
               symmetry functions. Either auto-genetrated, or given in the
               following form, for example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

               Used in the fingerprinting scheme only.
    :type Gs: dict
    :param elements: List of atom symbols; used in the fingerprinting scheme
                     only.
    :type elements: list of str

    :returns: weights
    """
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
            for __ in range(size):
                weight[_ + 1][-1][__] = 0.

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
                for __ in range(size):
                    weight[element][_ + 1][-1][__] = 0.

    return weight

###############################################################################


def make_scalings_matrices(images, activation, elements=None):
    """
    Generates initial scaling matrices, such that the range of activation
    is scaled to the range of actual energies.

    :param images: ASE atoms objects (the training set).
    :type images: dict
    :param activation: Assigns the type of activation funtion. "linear" refers
                       to linear function, "tanh" refers to tanh function, and
                       "sigmoid" refers to sigmoid function.
    :type activation: str
    :param elements: List of atom symbols; used in the fingerprinting scheme
                     only.
    :type elements: list of str

    :returns: scalings
    """
    max_act_energy = max(image.get_potential_energy(apply_constraint=False)
                         for hash, image in images.items())
    min_act_energy = min(image.get_potential_energy(apply_constraint=False)
                         for hash, image in images.items())

    for hash, image in images.items():
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

    """
    Class to ravel and unravel weight and scaling values into a single vector.
    This is used for feeding into the optimizer. Feed in a list of
    dictionaries to initialize the shape of the transformation. Note that no
    data is saved in the class; each time it is used it is passed either
    the dictionaries or vector.

    :param hiddenlayers: Dictionary of chemical element symbols and
                        architectures of their corresponding hidden layers of
                        the conventional neural network.
    :type hiddenlayers: dict
    :param elements: List of atom symbols; used in the fingerprinting scheme
                     only.
    :type elements: list of str
    :param Gs: Dictionary of symbols and lists of dictionaries for making
               symmetry functions. Either auto-genetrated, or given in the
               following form, for example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

               Used in the fingerprinting scheme only.
    :type Gs: dict
    :param no_of_atoms: Number of atoms in atomic systems; used only in the
                        case of no descriptor.
    :type no_of_atoms: int
    """
    ###########################################################################

    def __init__(self, hiddenlayers, elements=None, Gs=None, no_of_atoms=None):

        self.no_of_atoms = no_of_atoms
        self.count = 0
        self._weightskeys = []
        self._scalingskeys = []

        if self.no_of_atoms is None:  # fingerprinting scheme

            for element in elements:
                for layer in range(1, len(hiddenlayers[element]) + 2):
                    if layer == 1:
                        shape = \
                            (len(Gs[element]) + 1, hiddenlayers[element][0])
                    elif layer == (len(hiddenlayers[element]) + 1):
                        shape = (hiddenlayers[element][layer - 2] + 1, 1)
                    else:
                        shape = (
                            hiddenlayers[element][layer - 2] + 1,
                            hiddenlayers[element][layer - 1])
                    size = shape[0] * shape[1]
                    self._weightskeys.append({'key1': element,
                                              'key2': layer,
                                              'shape': shape,
                                              'size': size})
                    self.count += size

            for element in elements:
                self._scalingskeys.append({'key1': element,
                                           'key2': 'intercept'})
                self._scalingskeys.append({'key1': element,
                                           'key2': 'slope'})
                self.count += 2

        else:  # pure atomic-coordinates scheme

            for layer in range(1, len(hiddenlayers) + 2):
                if layer == 1:
                    shape = (3 * no_of_atoms + 1, hiddenlayers[0])
                elif layer == (len(hiddenlayers) + 1):
                    shape = (hiddenlayers[layer - 2] + 1, 1)
                else:
                    shape = (
                        hiddenlayers[layer - 2] + 1, hiddenlayers[layer - 1])
                size = shape[0] * shape[1]
                self._weightskeys.append({'key': layer,
                                          'shape': shape,
                                          'size': size})
                self.count += size

            self._scalingskeys.append({'key': 'intercept'})
            self._scalingskeys.append({'key': 'slope'})
            self.count += 2

    ###########################################################################

    def to_vector(self, weights, scalings):
        """
        Puts the weights and scalings embedded dictionaries into a single
        vector and returns it. The dictionaries need to have the identical
        structure to those it was initialized with.

        :param weights: In the case of no descriptor, keys correspond to
                        layers and values are two dimensional arrays of network
                        weight. In the fingerprinting scheme, keys correspond
                        to chemical elements and values are dictionaries with
                        layer keys and network weight two dimensional arrays as
                        values. Arrays are set up to connect node i in the
                        previous layer with node j in the current layer with
                        indices w[i,j]. The last value for index i corresponds
                        to bias. If weights is not given, arrays will be
                        randomly generated.
        :type weights: dict
        :param scalings: In the case of no descriptor, keys are "intercept"
                         and "slope" and values are real numbers. In the
                         fingerprinting scheme, keys correspond to chemical
                         elements and values are dictionaries with "intercept"
                         and "slope" keys and real number values. If scalings
                         is not given, it will be randomly generated.
        :type scalings: dict

        :returns: List of variables
        """
        vector = np.zeros(self.count)
        count = 0
        for k in sorted(self._weightskeys):
            if self.no_of_atoms is None:  # fingerprinting scheme
                lweights = np.array(weights[k['key1']][k['key2']]).ravel()
            else:  # pure atomic-coordinates scheme
                lweights = (np.array(weights[k['key']])).ravel()
            vector[count:(count + lweights.size)] = lweights
            count += lweights.size
        for k in sorted(self._scalingskeys):
            if self.no_of_atoms is None:  # fingerprinting scheme
                vector[count] = scalings[k['key1']][k['key2']]
            else:  # pure atomic-coordinates scheme
                vector[count] = scalings[k['key']]
            count += 1
        return vector

    ###########################################################################

    def to_dicts(self, vector):
        """
        Puts the vector back into weights and scalings dictionaries of the
        form initialized. vector must have same length as the output of
        unravel.

        :param vector: List of variables.
        :type vector: list

        :returns: weights and scalings
        """
        assert len(vector) == self.count
        count = 0
        weights = OrderedDict()
        scalings = OrderedDict()
        for k in sorted(self._weightskeys):
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
        for k in sorted(self._scalingskeys):
            if self.no_of_atoms is None:  # fingerprinting scheme
                if k['key1'] not in scalings.keys():
                    scalings[k['key1']] = OrderedDict()
                scalings[k['key1']][k['key2']] = vector[count]
            else:  # pure atomic-coordinates scheme
                scalings[k['key']] = vector[count]
            count += 1
        return weights, scalings

    ###########################################################################

    def calculate_weights_norm_and_der(self, weights):
        """
        Calculates norm of weights as well as the vector of its derivative
        for the use of constratinting overfitting.

        :param weights: In the case of no descriptor, keys correspond to
                        layers and values are two dimensional arrays of network
                        weight. In the fingerprinting scheme, keys correspond
                        to chemical elements and values are dictionaries with
                        layer keys and network weight two dimensional arrays as
                        values. Arrays are set up to connect node i in the
                        previous layer with node j in the current layer with
                        indices w[i,j]. The last value for index i corresponds
                        to bias. If weights is not given, arrays will be
                        randomly generated.
        :type weights: dict

        :returns: norm of weights (float) and variable derivative of norm of
                  of weights (list of float).
        """
        count = 0
        weights_norm = 0.
        der_of_weights_norm = np.zeros(self.count)
        for k in sorted(self._weightskeys):
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
        for k in sorted(self._scalingskeys):
            # there is no overfitting constraint on the values of scalings
            der_of_weights_norm[count] = 0.
            count += 1
        return weights_norm, der_of_weights_norm

###############################################################################
###############################################################################
###############################################################################
