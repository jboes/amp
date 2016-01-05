import numpy as np
from ase.data import atomic_numbers
from amp.utilities import FingerprintsError
try:
    from amp import fmodules
except ImportError:
    fmodules = None

###############################################################################


class Behler:

    """
    Class that calculates Behler fingerprints.

    :param cutoff: Radius above which neighbor interactions are ignored.
                   Default is 6.5 Angstroms.
    :type cutoff: float
    :param Gs: Dictionary of symbols and lists of dictionaries for making
               symmetry functions. Either auto-genetrated, or given in the
               following form, for example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

    :type Gs: dict
    :param fingerprints_tag: An internal tag for identifying the functional
                             form of fingerprints used in the code.
    :type fingerprints_tag: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :raises: FingerprintsError, NotImplementedError
    """
    ###########################################################################

    def __init__(self, cutoff=6.5, Gs=None, fingerprints_tag=1, fortran=True,):

        self.cutoff = cutoff
        self.Gs = Gs
        self.fingerprints_tag = fingerprints_tag
        self.fortran = fortran

        self.no_of_element_fingerprints = {}
        if Gs is not None:
            for element in Gs.keys():
                self.no_of_element_fingerprints[element] = len(Gs[element])

        # Checking if the functional forms of fingerprints in the train set
        # is the same as those of the current version of the code:
        if self.fingerprints_tag != 1:
            raise FingerprintsError('Functional form of fingerprints has been '
                                    'changed. Re-train you train images set, '
                                    'and use the new variables.')

    ###########################################################################

    def initialize(self, atoms):
        """
        Initializing atoms object.

        :param atoms: ASE atoms object.
        :type atoms: ASE dict
        """
        self.atoms = atoms

    ###########################################################################

    def get_fingerprint(self, index, symbol, n_symbols, Rs):
        """
        Returns the fingerprint of symmetry function values for atom
        specified by its index and symbol. n_symbols and Rs are lists of
        neighbors' symbols and Cartesian positions, respectively. Will
        automaticallyupdate if atom positions has changed; you don't need to
        call update() unless you are specifying a new atoms object.

        :param index: Index of the center atom.
        :type index: int
        :param symbol: Symbol of the center atom.
        :type symbol: str
        :param n_symbols: List of neighbors' symbols.
        :type n_symbols: list of str
        :param Rs: List of Cartesian atomic positions.
        :type Rs: list of list of float

        :returns: list of float -- the symmetry function values for atom
                                   specified by its index and symbol.
        """
        home = self.atoms[index].position

        len_of_symmetries = len(self.Gs[symbol])
        fingerprint = [None] * len_of_symmetries
        count = 0
        while count < len_of_symmetries:
            G = self.Gs[symbol][count]

            if G['type'] == 'G2':
                ridge = calculate_G2(n_symbols, Rs, G['element'], G['eta'],
                                     self.cutoff, home, self.fortran)
            elif G['type'] == 'G4':
                ridge = calculate_G4(n_symbols, Rs, G['elements'], G['gamma'],
                                     G['zeta'], G['eta'], self.cutoff, home,
                                     self.fortran)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])
            fingerprint[count] = ridge
            count += 1

        return fingerprint

    ###########################################################################

    def get_der_fingerprint(self, index, symbol, n_indices, n_symbols, Rs,
                            m, i):
        """
        Returns the value of the derivative of G for atom with index and
        symbol with respect to coordinate x_{i} of atom index m. n_indices,
        n_symbols and Rs are lists of neighbors' indices, symbols and Cartesian
        positions, respectively.

        :param index: Index of the center atom.
        :type index: int
        :param symbol: Symbol of the center atom.
        :type symbol: str
        :param n_indices: List of neighbors' indices.
        :type n_indices: list of int
        :param n_symbols: List of neighbors' symbols.
        :type n_symbols: list of str
        :param Rs: List of Cartesian atomic positions.
        :type Rs: list of list of float
        :param m: Index of the pair atom.
        :type m: int
        :param i: Direction of the derivative; is an integer from 0 to 2.
        :type i: int

        :returns: list of float -- the value of the derivative of the
                                   fingerprints for atom with index and symbol
                                   with respect to coordinate x_{i} of atom
                                   index m.
        """

        len_of_symmetries = len(self.Gs[symbol])
        Rindex = self.atoms.positions[index]
        der_fingerprint = [None] * len_of_symmetries
        count = 0
        while count < len_of_symmetries:
            G = self.Gs[symbol][count]
            if G['type'] == 'G2':
                ridge = calculate_der_G2(
                    n_indices,
                    n_symbols,
                    Rs,
                    G['element'],
                    G['eta'],
                    self.cutoff,
                    index,
                    Rindex,
                    m,
                    i,
                    self.fortran)
            elif G['type'] == 'G4':
                ridge = calculate_der_G4(
                    n_indices,
                    n_symbols,
                    Rs,
                    G['elements'],
                    G['gamma'],
                    G['zeta'],
                    G['eta'],
                    self.cutoff,
                    index,
                    Rindex,
                    m,
                    i,
                    self.fortran)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])

            der_fingerprint[count] = ridge
            count += 1

        return der_fingerprint

    ###########################################################################

    def log(self, log, param, elements):
        """
        Generates symmetry functions if do not exist, and prints out in the log
        file.

        :param log: Write function at which to log data. Note this must be a
                    callable function.
        :type log: Logger object
        :param param: Object containing descriptor properties.
        :type param: ASE calculator's Parameters class
        :param elements: List of atom symbols.
        :type elements: list of str

        :returns: Object containing descriptor properties.
        """
        # If Gs is not given, generates symmetry functions
        if not param.descriptor.Gs:
            param.descriptor.Gs = make_symmetry_functions(elements)
            param.descriptor.no_of_element_fingerprints = {}
            for element in param.descriptor.Gs.keys():
                param.descriptor.no_of_element_fingerprints[element] = \
                    len(param.descriptor.Gs[element])

        log('Symmetry functions for each element:')
        for _ in param.descriptor.Gs.keys():
            log(' %2s: %i' % (_, len(param.descriptor.Gs[_])))

        return param

###############################################################################
###############################################################################
###############################################################################


def calculate_G2(symbols, Rs, G_element, eta, cutoff, home, fortran):
    """
    Calculate G2 symmetry function. Ideally this will not be used but
    will be a template for how to build the fortran version (and serves as
    a slow backup if the fortran one goes uncompiled).

    :param symbols: List of symbols of all atoms.
    :type symbols: list of str
    :param Rs: List of Cartesian atomic positions.
    :type Rs: list of list of float
    :param G_element: Symmetry functions of the center atom.
    :type G_element: dict
    :param eta: Parameter of Behler symmetry functions.
    :type eta: float
    :param cutoff: Radius above which neighbor interactions are ignored.
    :type cutoff: float
    :param home: Index of the center atom.
    :type home: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :returns: float -- G2 fingerprint.
    """
    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_g2(numbers=numbers, rs=Rs,
                                          g_number=G_number, g_eta=eta,
                                          cutoff=cutoff, home=home)
    else:
        ridge = 0.  # One aspect of a fingerprint :)
        len_of_symbols = len(symbols)
        count = 0
        while count < len_of_symbols:
            symbol = symbols[count]
            R = Rs[count]
            if symbol == G_element:
                Rij = np.linalg.norm(R - home)
                ridge += (np.exp(-eta * (Rij ** 2.) / (cutoff ** 2.)) *
                          cutoff_fxn(Rij, cutoff))
            count += 1

    return ridge

###############################################################################


def calculate_G4(symbols, Rs, G_elements, gamma, zeta, eta, cutoff, home,
                 fortran):
    """
    Calculate G4 symmetry function. Ideally this will not be used but
    will be a template for how to build the fortran version (and serves as
    a slow backup if the fortran one goes uncompiled).

    :param symbols: List of symbols of neighboring atoms.
    :type symbols: list of str
    :param Rs: List of Cartesian atomic positions of neighboring atoms.
    :type Rs: list of list of float
    :param G_elements: Symmetry functions of the center atom.
    :type G_elements: dict
    :param gamma: Parameter of Behler symmetry functions.
    :type gamma: float
    :param zeta: Parameter of Behler symmetry functions.
    :type zeta: float
    :param eta: Parameter of Behler symmetry functions.
    :type eta: float
    :param cutoff: Radius above which neighbor interactions are ignored.
    :type cutoff: float
    :param home: Index of the center atom.
    :type home: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :returns: float -- G4 fingerprint.
    """

    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_g4(numbers=numbers, rs=Rs,
                                          g_numbers=G_numbers, g_gamma=gamma,
                                          g_zeta=zeta, g_eta=eta,
                                          cutoff=cutoff, home=home)
    else:
        ridge = 0.
        counts = range(len(symbols))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([symbols[j], symbols[k]])
                if els != G_elements:
                    continue
                Rij_ = Rs[j] - home
                Rij = np.linalg.norm(Rij_)
                Rik_ = Rs[k] - home
                Rik = np.linalg.norm(Rik_)
                Rjk = np.linalg.norm(Rs[j] - Rs[k])
                cos_theta_ijk = np.dot(Rij_, Rik_) / Rij / Rik
                term = (1. + gamma * cos_theta_ijk) ** zeta
                term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                               (cutoff ** 2.))
                term *= (1. / 3.) * (cutoff_fxn(Rij, cutoff) +
                                     cutoff_fxn(Rik, cutoff) +
                                     cutoff_fxn(Rjk, cutoff))
                ridge += term
        ridge *= 2. ** (1. - zeta)

    return ridge

###############################################################################


def make_symmetry_functions(elements):
    """
    Makes symmetry functions as in Nano Letters function by Artrith.
    Elements is a list of the elements, as in ["C", "O", "H", "Cu"].
    G[0] = {"type":"G2", "element": "O", "eta": 0.0009}
    G[40] = {"type":"G4", "elements": ["O", "Au"], "eta": 0.0001,
    "gamma": 1.0, "zeta": 1.0}

    If G (a list) is fed in, this will add to it and return an expanded
    version. If not, it will create a new one.

    :param elements: List of symbols of all atoms.
    :type elements: list of str

    :returns: dict of lists -- symmetry functions if not given by the user.
    """
    G = {}
    for element0 in elements:

        # Radial symmetry functions.
        etas = [0.05, 4., 20., 80.]
        _G = [{'type': 'G2', 'element': element, 'eta': eta}
              for eta in etas
              for element in elements]

        # Angular symmetry functions.
        etas = [0.005]
        zetas = [1., 4.]
        gammas = [+1., -1.]
        for eta in etas:
            for zeta in zetas:
                for gamma in gammas:
                    for i1, el1 in enumerate(elements):
                        for el2 in elements[i1:]:
                            els = sorted([el1, el2])
                            _G.append({'type': 'G4',
                                       'elements': els,
                                       'eta': eta,
                                       'gamma': gamma,
                                       'zeta': zeta})
        G[element0] = _G
    return G

###############################################################################


def cutoff_fxn(Rij, Rc):
    """
    Cosine cutoff function in Parinello-Behler method.

    :param Rc: Radius above which neighbor interactions are ignored.
    :type Rc: float
    :param Rij: Distance between pair atoms.
    :type Rij: float

    :returns: float -- the vaule of the cutoff function.
    """
    if Rij > Rc:
        return 0.
    else:
        return 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)

###############################################################################


def der_cutoff_fxn(Rij, Rc):
    """
    Derivative of the Cosine cutoff function.

    :param Rc: Radius above which neighbor interactions are ignored.
    :type Rc: float
    :param Rij: Distance between pair atoms.
    :type Rij: float

    :returns: float -- the vaule of derivative of the cutoff function.
    """
    if Rij > Rc:
        return 0.
    else:
        return -0.5 * np.pi / Rc * np.sin(np.pi * Rij / Rc)

###############################################################################


def Kronecker_delta(i, j):
    """
    Kronecker delta function.

    :param i: First index of Kronecker delta.
    :type i: int
    :param j: Second index of Kronecker delta.
    :type j: int

    :returns: int -- the value of the Kronecker delta.
    """
    if i == j:
        return 1.
    else:
        return 0.

###############################################################################


def der_position_vector(a, b, m, i):
    """
    Returns the derivative of the position vector R_{ab} with respect to
        x_{i} of atomic index m.

    :param a: Index of the first atom.
    :type a: int
    :param b: Index of the second atom.
    :type b: int
    :param m: Index of the atom force is acting on.
    :type m: int
    :param i: Direction of force.
    :type i: int

    :returns: list of float -- the derivative of the position vector R_{ab}
                               with respect to x_{i} of atomic index m.
    """
    der_position_vector = [None, None, None]
    der_position_vector[0] = (Kronecker_delta(m, a) - Kronecker_delta(m, b)) \
        * Kronecker_delta(0, i)
    der_position_vector[1] = (Kronecker_delta(m, a) - Kronecker_delta(m, b)) \
        * Kronecker_delta(1, i)
    der_position_vector[2] = (Kronecker_delta(m, a) - Kronecker_delta(m, b)) \
        * Kronecker_delta(2, i)

    return der_position_vector

###############################################################################


def der_position(m, n, Rm, Rn, l, i):
    """
    Returns the derivative of the norm of position vector R_{mn} with
        respect to x_{i} of atomic index l.

    :param m: Index of the first atom.
    :type m: int
    :param n: Index of the second atom.
    :type n: int
    :param Rm: Position of the first atom.
    :type Rm: float
    :param Rn: Position of the second atom.
    :type Rn: float
    :param l: Index of the atom force is acting on.
    :type l: int
    :param i: Direction of force.
    :type i: int

    :returns: list of float -- the derivative of the norm of position vector
                               R_{mn} with respect to x_{i} of atomic index l.
    """
    Rmn = np.linalg.norm(Rm - Rn)
    # mm != nn is necessary for periodic systems
    if l == m and m != n:
        der_position = (Rm[i] - Rn[i]) / Rmn
    elif l == n and m != n:
        der_position = -(Rm[i] - Rn[i]) / Rmn
    else:
        der_position = 0.
    return der_position

###############################################################################


def der_cos_theta(a, j, k, Ra, Rj, Rk, m, i):
    """
    Returns the derivative of Cos(theta_{ajk}) with respect to
        x_{i} of atomic index m.

    :param a: Index of the center atom.
    :type a: int
    :param j: Index of the first atom.
    :type j: int
    :param k: Index of the second atom.
    :type k: int
    :param Ra: Position of the center atom.
    :type Ra: float
    :param Rj: Position of the first atom.
    :type Rj: float
    :param Rk: Position of the second atom.
    :type Rk: float
    :param m: Index of the atom force is acting on.
    :type m: int
    :param i: Direction of force.
    :type i: int

    :returns: float -- derivative of Cos(theta_{ajk}) with respect to x_{i}
                       of atomic index m.
    """
    Raj_ = Ra - Rj
    Raj = np.linalg.norm(Raj_)
    Rak_ = Ra - Rk
    Rak = np.linalg.norm(Rak_)
    der_cos_theta = 1. / \
        (Raj * Rak) * np.dot(der_position_vector(a, j, m, i), Rak_)
    der_cos_theta += +1. / \
        (Raj * Rak) * np.dot(Raj_, der_position_vector(a, k, m, i))
    der_cos_theta += -1. / \
        ((Raj ** 2.) * Rak) * np.dot(Raj_, Rak_) * \
        der_position(a, j, Ra, Rj, m, i)
    der_cos_theta += -1. / \
        (Raj * (Rak ** 2.)) * np.dot(Raj_, Rak_) * \
        der_position(a, k, Ra, Rk, m, i)
    return der_cos_theta

###############################################################################


def calculate_der_G2(n_indices, symbols, Rs, G_element, eta, cutoff, a, Ra,
                     m, i, fortran):
    """
    Calculates coordinate derivative of G2 symmetry function for atom at
    index a and position Ra with respect to coordinate x_{i} of atom index
    m.

    :param n_indices: List of int of neighboring atoms.
    :type n_indices: list of int
    :param symbols: List of symbols of neighboring atoms.
    :type symbols: list of str
    :param Rs: List of Cartesian atomic positions of neighboring atoms.
    :type Rs: list of list of float
    :param G_element: Symmetry functions of the center atom.
    :type G_element: dict
    :param eta: Parameter of Behler symmetry functions.
    :type eta: float
    :param cutoff: Radius above which neighbor interactions are ignored.
    :type cutoff: float
    :param a: Index of the center atom.
    :type a: int
    :param Ra: Position of the center atom.
    :type Ra: float
    :param m: Index of the atom force is acting on.
    :type m: int
    :param i: Direction of force.
    :type i: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :returns: float -- coordinate derivative of G2 symmetry function for atom
                       at index a and position Ra with respect to coordinate
                       x_{i} of atom index m.
    """
    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_der_g2(n_indices=list(n_indices),
                                              numbers=numbers, rs=Rs,
                                              g_number=G_number,
                                              g_eta=eta, cutoff=cutoff,
                                              aa=a, home=Ra, mm=m,
                                              ii=i)
    else:
        ridge = 0.  # One aspect of a fingerprint :)

        len_of_symbols = len(symbols)
        count = 0
        while count < len_of_symbols:
            symbol = symbols[count]
            Rj = Rs[count]
            n_index = n_indices[count]
            if symbol == G_element:
                Raj = np.linalg.norm(Ra - Rj)
                term1 = (-2. * eta * Raj * cutoff_fxn(Raj, cutoff) /
                         (cutoff ** 2.) +
                         der_cutoff_fxn(Raj, cutoff))
                term2 = der_position(a, n_index, Ra, Rj, m, i)
                ridge += np.exp(- eta * (Raj ** 2.) / (cutoff ** 2.)) * \
                    term1 * term2
            count += 1
    return ridge

###############################################################################


def calculate_der_G4(n_indices, symbols, Rs, G_elements, gamma, zeta, eta,
                     cutoff, a, Ra, m, i, fortran):
    """
    Calculates coordinate derivative of G4 symmetry function for atom at
    index a and position Ra with respect to coordinate x_{i} of atom index m.

    :param n_indices: List of int of neighboring atoms.
    :type n_indices: list of int
    :param symbols: List of symbols of neighboring atoms.
    :type symbols: list of str
    :param Rs: List of Cartesian atomic positions of neighboring atoms.
    :type Rs: list of list of float
    :param G_elements: Symmetry functions of the center atom.
    :type G_elements: dict
    :param gamma: Parameter of Behler symmetry functions.
    :type gamma: float
    :param zeta: Parameter of Behler symmetry functions.
    :type zeta: float
    :param eta: Parameter of Behler symmetry functions.
    :type eta: float
    :param cutoff: Radius above which neighbor interactions are ignored.
    :type cutoff: float
    :param a: Index of the center atom.
    :type a: int
    :param Ra: Position of the center atom.
    :type Ra: float
    :param m: Index of the atom force is acting on.
    :type m: int
    :param i: Direction of force.
    :type i: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :returns: float -- coordinate derivative of G4 symmetry function for atom
                       at index a and position Ra with respect to coordinate
                       x_{i} of atom index m.
    """
    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_der_g4(n_indices=list(n_indices),
                                              numbers=numbers, rs=Rs,
                                              g_numbers=G_numbers,
                                              g_gamma=gamma,
                                              g_zeta=zeta, g_eta=eta,
                                              cutoff=cutoff, aa=a,
                                              home=Ra, mm=m,
                                              ii=i)
    else:
        ridge = 0.
        counts = range(len(symbols))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([symbols[j], symbols[k]])
                if els != G_elements:
                    continue
                Rj = Rs[j]
                Rk = Rs[k]
                Raj_ = Rs[j] - Ra
                Raj = np.linalg.norm(Raj_)
                Rak_ = Rs[k] - Ra
                Rak = np.linalg.norm(Rak_)
                Rjk_ = Rs[j] - Rs[k]
                Rjk = np.linalg.norm(Rjk_)
                cos_theta_ajk = np.dot(Raj_, Rak_) / Raj / Rak
                c1 = (1. + gamma * cos_theta_ajk)
                c2 = cutoff_fxn(Raj, cutoff)
                c3 = cutoff_fxn(Rak, cutoff)
                c4 = cutoff_fxn(Rjk, cutoff)
                if zeta == 1:
                    term1 = \
                        np.exp(- eta * (Raj ** 2. + Rak ** 2. + Rjk ** 2.) /
                               (cutoff ** 2.))
                else:
                    term1 = c1 ** (zeta - 1.) * \
                        np.exp(- eta * (Raj ** 2. + Rak ** 2. + Rjk ** 2.) /
                               (cutoff ** 2.))
                term2 = (1. / 3.) * (c2 + c3 + c4)
                term3 = der_cos_theta(a, n_indices[j], n_indices[k], Ra, Rj,
                                      Rk, m, i)
                term4 = gamma * zeta * term3
                term5 = der_position(a, n_indices[j], Ra, Rj, m, i)
                term4 += -2. * c1 * eta * Raj * term5 / (cutoff ** 2.)
                term6 = der_position(a, n_indices[k], Ra, Rk, m, i)
                term4 += -2. * c1 * eta * Rak * term6 / (cutoff ** 2.)
                term7 = der_position(n_indices[j], n_indices[k], Rj, Rk, m, i)
                term4 += -2. * c1 * eta * Rjk * term7 / (cutoff ** 2.)
                term2 = term2 * term4
                term8 = c1 * (1. / 3.) * der_cutoff_fxn(Raj, cutoff) * term5
                term9 = c1 * (1. / 3.) * der_cutoff_fxn(Rak, cutoff) * term6
                term10 = c1 * (1. / 3.) * der_cutoff_fxn(Rjk, cutoff) * term7
                term11 = term2 + term8 + term9 + term10
                term = term1 * term11
                ridge += term
        ridge *= 2. ** (1. - zeta)

    return ridge

###############################################################################
