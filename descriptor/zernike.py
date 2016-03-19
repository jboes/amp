import numpy as np
from numpy import cos, sqrt
from ase.data import atomic_numbers
from amp.utilities import FingerprintsError
from scipy.special import sph_harm
try:  # for scipy v <= 0.90
    from scipy import factorial as fac
except ImportError:
    try:  # for scipy v >= 0.10
        from scipy.misc import factorial as fac
    except ImportError:  # for newer version of scipy
        from scipy.special import factorial as fac

###############################################################################


class Zernike:

    """
    Class that calculates Zernike fingerprints.

    :param cutoff: Radius above which neighbor interactions are ignored.
                   Default is 6.5 Angstroms.
    :type cutoff: float
    :param Gs: Dictionary of symbols and lists of dictionaries for making
               symmetry functions. Either auto-genetrated, or given in the
               following form, for example:

               >>> Gs = {"Au": {"Au": 3., "O": 2.}, "O": {"Au": 5., "O": 10.}}

    :type Gs: dict
    :param nmax: Maximum degree of Zernike polynomials that will be included in
                 the fingerprint vector.
    :type nmax: integer
    :param fingerprints_tag: An internal tag for identifying the functional
                             form of fingerprints used in the code.
    :type fingerprints_tag: int

    :raises: FingerprintsError, NotImplementedError
    """
    ###########################################################################

    def __init__(self, cutoff=6.5, Gs=None, nmax=1,
                 fingerprints_tag=1,):

        self.cutoff = cutoff
        self.Gs = Gs
        self.nmax = nmax
        self.fingerprints_tag = fingerprints_tag

        self.no_of_element_fingerprints = {}
        if Gs is not None:
            for element in Gs.keys():
                no_of_element_fps = 0
                if isinstance(nmax, dict):

                    for n in range(nmax[element] + 1):
                        for l in range(n + 1):
                            if (n - l) % 2 == 0:
                                no_of_element_fps += 1
                else:
                    for n in range(nmax + 1):
                        for l in range(n + 1):
                            if (n - l) % 2 == 0:
                                no_of_element_fps += 1

                self.no_of_element_fingerprints[element] = no_of_element_fps

        self.factorial = []
        for _ in range(2 * nmax + 2):
            self.factorial += [float(fac(0.5 * _))]

        # Checking if the functional forms of fingerprints in the train set
        # is the same as those of the current version of the code:
        if self.fingerprints_tag != 1:
            raise FingerprintsError('Functional form of fingerprints has been '
                                    'changed. Re-train you train images set, '
                                    'and use the new variables.')

    ###########################################################################

    def initialize(self, fortran, atoms):
        """
        Initializing atoms object.

        :param fortran: If True, will use the fortran subroutines, else will
                        not.
        :type fortran: bool
        :param atoms: ASE atoms object.
        :type atoms: ASE dict
        """
        self.fortran = fortran
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
        len_of_neighbors = len(Rs)

        fingerprint = []
        for n in range(self.nmax + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    norm = 0.
                    for m in range(l + 1):
                        omega = 0.
                        for n_index in range(len_of_neighbors):
                            neighbor = Rs[n_index]
                            n_symbol = n_symbols[n_index]
                            x = (neighbor[0] - home[0]) / self.cutoff
                            y = (neighbor[1] - home[1]) / self.cutoff
                            z = (neighbor[2] - home[2]) / self.cutoff

                            rho = np.linalg.norm([x, y, z])

                            if rho > 0.:
                                theta = np.arccos(z / rho)
                            else:
                                theta = 0.

                            if x < 0.:
                                phi = np.pi + np.arctan(y / x)
                            elif 0. < x and y < 0.:
                                phi = 2 * np.pi + np.arctan(y / x)
                            elif 0. < x and 0. <= y:
                                phi = np.arctan(y / x)
                            elif x == 0. and 0. < y:
                                phi = 0.5 * np.pi
                            elif x == 0. and y < 0.:
                                phi = 1.5 * np.pi
                            else:
                                phi = 0.

                            ZZ = self.Gs[symbol][n_symbol] * \
                                calculate_R(n, l, rho, self.factorial) * \
                                sph_harm(m, l, phi, theta) * \
                                cutoff_fxn(rho * self.cutoff, self.cutoff)

                            # sum over neighbors
                            omega += np.conjugate(ZZ)
                        # sum over m values
                        if m == 0:
                            norm += omega * np.conjugate(omega)
                        else:
                            norm += 2. * omega * np.conjugate(omega)

                    fingerprint.append(norm.real)

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

        raise RuntimeError('Zernike descriptor does not work with '
                           'force training yet. Either turn off force '
                           'training by "force_goal=None", or use Gaussian '
                           'descriptor by "descriptor=Gaussian()".')

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

        log('Cutoff radius: %.3f' % self.cutoff)
        log('Maximum degree of Zernike polynomials: %.1f' % self.nmax)
        # If Gs is not given, generates symmetry functions
        if not param.descriptor.Gs:
            param.descriptor.Gs = make_symmetry_functions(elements)
            param.descriptor.no_of_element_fingerprints = {}
            for element in elements:
                no_of_element_fps = 0
                if isinstance(self.nmax, dict):
                    for n in range(self.nmax[element] + 1):
                        for l in range(n + 1):
                            if (n - l) % 2 == 0:
                                no_of_element_fps += 1
                else:
                    for n in range(self.nmax + 1):
                        for l in range(n + 1):
                            if (n - l) % 2 == 0:
                                no_of_element_fps += 1

                param.descriptor.no_of_element_fingerprints[element] = \
                    no_of_element_fps

        log('Number of descriptors for each element:')
        for element in elements:
            log(' %2s: %s' % (element,
                              str(param.descriptor.no_of_element_fingerprints[
                                  element])))

        log('Symmetry coefficients for each element:')
        for _ in param.descriptor.Gs.keys():
            log(' %2s: %s' % (_, str(param.descriptor.Gs[_])))

        return param

###############################################################################
###############################################################################
###############################################################################


def calculate_R(n, l, rho, factorial):
    """
    Calculates R_{n}^{l}(rho) according to the last equation of wikipedia.
    """

    if (n - l) % 2 != 0:
        return 0
    else:
        value = 0.
        k = (n - l) / 2
        term1 = sqrt(2. * n + 3.)

        for s in range(k + 1):
            b1 = binomial(k, s, factorial)
            b2 = binomial(n - s - 1 + 1.5, k, factorial)
            value += ((-1) ** s) * b1 * b2 * (rho ** (n - 2. * s))

        value *= term1
        return value

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
        return 0.5 * (cos(np.pi * Rij / Rc) + 1.)

###############################################################################


def binomial(n, k, factorial):
    """Returns C(n,k) = n!/(k!(n-k)!)."""

    assert n >= 0 and k >= 0 and n >= k, \
        'n and k should be non-negative integers with n >= k.'

    c = factorial[int(2 * n)] / \
        (factorial[int(2 * k)] * factorial[int(2 * (n - k))])
    return c

###############################################################################


def make_symmetry_functions(elements):
    """
    Makes symmetry functions.

    :param elements: List of symbols of all atoms.
    :type elements: list of str

    :returns: dict of lists -- symmetry functions if not given by the user.
    """
    G = {}
    for element0 in elements:
        G[element0] = {}
        for element in elements:
            G[element0][element] = atomic_numbers[element]

    return G

###############################################################################
