import numpy as np
from numpy import sin, cos, sqrt, exp
from ase.data import atomic_numbers
from amp.utilities import FingerprintsError
import math
try:
    from amp import fmodules
except ImportError:
    fmodules = None

###############################################################################


class SphericalHarmonics:

    """
    Class that calculates spherical harmonic fingerprints.

    :param cutoff: Radius above which neighbor interactions are ignored.
                   Default is 6.5 Angstroms.
    :type cutoff: float
    :param Gs: Dictionary of symbols and lists of dictionaries for making
               symmetry functions. Either auto-genetrated, or given in the
               following form, for example:

               >>> Gs = {"Au": {"Au": 3., "O": 2.}, "O": {"Au": 5., "O": 10.}}

    :type Gs: dict
    :param j: Maximum degree of spherical harmonics that will be included in
              the fingerprint vector.
    :type j: integer or half-integer
    :param fingerprints_tag: An internal tag for identifying the functional
                             form of fingerprints used in the code.
    :type fingerprints_tag: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :raises: FingerprintsError, NotImplementedError
    """
    ###########################################################################

    def __init__(self, cutoff=6.5, Gs=None, jmax=1,
                 fingerprints_tag=1, fortran=True,):

        self.cutoff = cutoff
        self.Gs = Gs
        self.jmax = jmax
        self.fingerprints_tag = fingerprints_tag
        self.fortran = fortran

        self.no_of_element_fingerprints = {}
        if Gs is not None:
            for element in Gs.keys():
                no_of_element_fps = 0
                if isinstance(jmax, dict):
                    for _2j1 in range(int(2 * jmax[element]) + 1):
                        for j in range(int(min(_2j1, jmax[element])) + 1):
                            no_of_element_fps += 1
                else:
                    for _2j1 in range(int(2 * jmax) + 1):
                        for j in range(int(min(_2j1, jmax)) + 1):
                            no_of_element_fps += 1
                self.no_of_element_fingerprints[element] = no_of_element_fps

        self.factorial = [1]
        for _ in range(int(3. * self.jmax) + 2):
            if _ > 0:
                self.factorial += [_ * self.factorial[_ - 1]]

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
        rs = []
        psis = []
        thetas = []
        phis = []
        len_of_neighbors = len(Rs)
        for n_index in range(len_of_neighbors):
            neighbor = Rs[n_index]
            x = neighbor[0] - home[0]
            y = neighbor[1] - home[1]
            z = neighbor[2] - home[2]
            r = np.linalg.norm(neighbor - home)
            if r > 10.**(-10.):

                psi = np.arccos(sqrt(1. - (r / self.cutoff)**2.))
                theta = np.arccos(z / (self.cutoff * sin(psi)))
                if abs((z / (self.cutoff * sin(psi))) - 1.0) < 10.**(-8.):
                    theta = 0.0
                elif abs((z / (self.cutoff * sin(psi))) + 1.0) < 10.**(-8.):
                    theta = np.pi
                phi = np.arctan(y / x)
                if x < 0.:
                    phi += np.pi
                elif x > 0. and y < 0.:
                    phi += 2. * np.pi

                if math.isnan(phi):
                    if y < 0.:
                        phi = 1.5 * np.pi
                    else:
                        phi = 0.5 * np.pi

                rs += [r]
                psis += [psi]
                thetas += [theta]
                phis += [phi]

        fingerprint = []
        for _2j1 in range(int(2 * self.jmax) + 1):
            j1 = 0.5 * _2j1
            j2 = 0.5 * _2j1
            for j in range(int(min(_2j1, self.jmax)) + 1):
                value = calculate_B(j1, j2, 1.0 * j, self.Gs[symbol],
                                    self.cutoff, self.factorial, n_symbols,
                                    rs, psis, thetas, phis, self.fortran)
                value = value.real
                fingerprint.append(value)

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

        raise RuntimeError('SphericalHarmonics descriptor does not work with '
                           'force training yet. Either turn off force '
                           'training by "force_goal=None", or use Behler '
                           'descriptor by "descriptor=Behler()".')

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
        log('Maximum degree of spherical harmonics: %.1f' % self.jmax)
        # If Gs is not given, generates symmetry functions
        if not param.descriptor.Gs:
            param.descriptor.Gs = make_symmetry_functions(elements)
            param.descriptor.no_of_element_fingerprints = {}
            for element in elements:
                no_of_element_fps = 0
                if isinstance(self.jmax, dict):
                    for _2j1 in range(int(2 * self.jmax[element]) + 1):
                        for j in range(int(min(_2j1, self.jmax[element])) + 1):
                            no_of_element_fps += 1
                else:
                    for _2j1 in range(int(2 * self.jmax) + 1):
                        for j in range(int(min(_2j1, self.jmax)) + 1):
                            no_of_element_fps += 1
                param.descriptor.no_of_element_fingerprints[element] = \
                    no_of_element_fps

        log('Symmetry functions for each element:')
        for _ in param.descriptor.Gs.keys():
            log(' %2s: %s' % (_, str(param.descriptor.Gs[_])))

        return param

###############################################################################
###############################################################################
###############################################################################


def calculate_B(j1, j2, j, G_element, cutoff, factorial, n_symbols, rs, psis,
                thetas, phis, fortran):
    """
    Calculates bi-spectrum B_{j1, j2, j} according to Eq. (5) of "Gaussian
    Approximation Potentials: The Accuracy of Quantum Mechanics, without the
    Electrons", Phys. Rev. Lett. 104, 136403.
    """

    mvals = m_values(j)
    B = 0.
    for m in mvals:
        for mp in mvals:
            c = calculate_c(j, mp, m, G_element, cutoff, factorial, n_symbols,
                            rs, psis, thetas, phis, fortran)
            m1bound = min(j1, m + j2)
            mp1bound = min(j1, mp + j2)
            m1 = max(-j1, m - j2)
            while m1 < (m1bound + 0.5):
                mp1 = max(-j1, mp - j2)
                while mp1 < (mp1bound + 0.5):
                    c1 = calculate_c(j1, mp1, m1, G_element, cutoff, factorial,
                                     n_symbols, rs, psis, thetas, phis,
                                     fortran)
                    c2 = calculate_c(j2, mp - mp1, m - m1, G_element, cutoff,
                                     factorial, n_symbols, rs, psis, thetas,
                                     phis, fortran)
                    B += CG(j1, m1, j2, m - m1, j, m, factorial) * \
                        CG(j1, mp1, j2, mp - mp1, j, mp, factorial) * \
                        np.conjugate(c) * c1 * c2
                    mp1 += 1.
                m1 += 1.

    return B

###############################################################################


def calculate_c(j, mp, m, G_element, cutoff, factorial, n_symbols, rs, psis,
                thetas, phis, fortran):
    """
    Calculates c^{j}_{m'm} according to Eq. (4) of "Gaussian Approximation
    Potentials: The Accuracy of Quantum Mechanics, without the Electrons",
    Phys. Rev. Lett. 104, 136403
    """

    value = 0.
    for n_symbol, r, psi, theta, phi in zip(n_symbols, rs, psis, thetas, phis):

        value += G_element[n_symbol] * \
            np.conjugate(U(j, m, mp, psi, theta, phi, factorial)) * \
            cutoff_fxn(r, cutoff)

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


def m_values(j):
    """Returns a list of m values for a given j."""

    assert j >= 0, '2*j should be a non-negative integer.'

    return [j - i for i in range(int(2 * j + 1))]

###############################################################################


def binomial(n, k, factorial):
    """Returns C(n,k) = n!/(k!(n-k)!)."""

    assert n >= 0 and k >= 0 and n >= k, \
        'n and k should be non-negative integers.'
    c = factorial[int(n)] / (factorial[int(k)] * factorial[int(n - k)])
    return c

###############################################################################


def WignerD(j, m, mp, alpha, beta, gamma, factorial):
    """Returns the Wigner-D matrix. alpha, beta, and gamma are the Euler
    angles."""

    result = 0
    if abs(beta - np.pi / 2.) < 10.**(-10.):
        # Varshalovich Eq. (5), Section 4.16, Page 113.
        # j, m, and mp here are J, M, and M', respectively, in Eq. (5).
        for k in range(int(2 * j + 1)):
            if k > j + mp or k > j - m:
                break
            elif k < mp - m:
                continue
            result += (-1)**k * binomial(j + mp, k, factorial) * \
                binomial(j - mp, k + m - mp, factorial)

        result *= (-1)**(m - mp) * \
            sqrt(float(factorial[int(j + m)] * factorial[int(j - m)]) /
                 float((factorial[int(j + mp)] * factorial[int(j - mp)]))) / \
            2.**j
        result *= exp(-1j * m * alpha) * exp(-1j * mp * gamma)

    else:
        # Varshalovich Eq. (10), Section 4.16, Page 113.
        # m, mpp, and mp here are M, m, and M', respectively, in Eq. (10).
        mvals = m_values(j)
        for mpp in mvals:
            # temp1 = WignerD(j, m, mpp, 0, np.pi/2, 0) = d(j, m, mpp, np.pi/2)
            temp1 = 0.
            for k in range(int(2 * j + 1)):
                if k > j + mpp or k > j - m:
                    break
                elif k < mpp - m:
                    continue
                temp1 += (-1)**k * binomial(j + mpp, k, factorial) * \
                    binomial(j - mpp, k + m - mpp, factorial)
            temp1 *= (-1)**(m - mpp) * \
                sqrt(float(factorial[int(j + m)] * factorial[int(j - m)]) /
                     float((factorial[int(j + mpp)] *
                            factorial[int(j - mpp)]))) / 2.**j

            # temp2 = WignerD(j, mpp, mp, 0, np.pi/2, 0) = d(j, mpp, mp,
            # np.pi/2)
            temp2 = 0.
            for k in range(int(2 * j + 1)):
                if k > j - mp or k > j - mpp:
                    break
                elif k < - mp - mpp:
                    continue
                temp2 += (-1)**k * binomial(j - mp, k, factorial) * \
                    binomial(j + mp, k + mpp + mp, factorial)
            temp2 *= (-1)**(mpp + mp) * \
                sqrt(float(factorial[int(j + mpp)] * factorial[int(j - mpp)]) /
                     float((factorial[int(j - mp)] *
                            factorial[int(j + mp)]))) / 2.**j

            result += temp1 * exp(-1j * mpp * beta) * temp2

        # Empirical normalization factor so results match Varshalovich
        # Tables 4.3-4.12
        # Note that this exact normalization does not follow from the
        # above equations
        result *= (1j**(2 * j - m - mp)) * ((-1)**(2 * m))
        result *= exp(-1j * m * alpha) * exp(-1j * mp * gamma)

    return result

###############################################################################


def U(j, m, mp, omega, theta, phi, factorial):
    """
    Calculates rotation matrix U_{MM'}^{J} in terms of rotation angle omega as
    well as rotation axis angles theta and phi, according to Varshalovich,
    Eq. (3), Section 4.5, Page 81. j, m, mp, and mpp here are J, M, M', and M''
    in Eq. (3).
    """

    result = 0.
    mvals = m_values(j)
    for mpp in mvals:
        result += WignerD(j, m, mpp, phi, theta, -phi, factorial) * \
            exp(- 1j * mpp * omega) * \
            WignerD(j, mpp, mp, phi, -theta, -phi, factorial)
    return result


###############################################################################


def CG(a, alpha, b, beta, c, gamma, factorial):
    """Clebsch-Gordan coefficient C_{a alpha b beta}^{c gamma} is calculated
    acoording to the expression given in Varshalovich Eq. (3), Section 8.2,
    Page 238."""

    if int(2. * a) != 2. * a or int(2. * b) != 2. * b or int(2. * c) != 2. * c:
        raise ValueError("j values must be integer or half integer")
    if int(2. * alpha) != 2. * alpha or int(2. * beta) != 2. * beta or \
            int(2. * gamma) != 2. * gamma:
        raise ValueError("m values must be integer or half integer")

    if alpha + beta - gamma != 0.:
        return 0.
    else:
        minimum = min(a + b - c, a - b + c, -a + b + c, a + b + c + 1.,
                      a - abs(alpha), b - abs(beta), c - abs(gamma))
        if minimum < 0.:
            return 0.
        else:
            sqrtarg = \
                factorial[int(a + alpha)] * \
                factorial[int(a - alpha)] * \
                factorial[int(b + beta)] * \
                factorial[int(b - beta)] * \
                factorial[int(c + gamma)] * \
                factorial[int(c - gamma)] * \
                (2. * c + 1.) * \
                factorial[int(a + b - c)] * \
                factorial[int(a - b + c)] * \
                factorial[int(-a + b + c)] / \
                factorial[int(a + b + c + 1.)]

            sqrtres = sqrt(sqrtarg)

            zmin = max(a + beta - c, b - alpha - c, 0.)
            zmax = min(b + beta, a - alpha, a + b - c)
            sumres = 0.
            for z in range(int(zmin), int(zmax) + 1):
                value = \
                    factorial[int(z)] * \
                    factorial[int(a + b - c - z)] * \
                    factorial[int(a - alpha - z)] * \
                    factorial[int(b + beta - z)] * \
                    factorial[int(c - b + alpha + z)] * \
                    factorial[int(c - a - beta + z)]
                sumres += (-1.)**z / value

            result = sqrtres * sumres

            return result

###############################################################################


def exact_Wigner_d(j, m, mp, beta):
    """Returns exact expressions of d_{MM'}^{j}(beta) according to Tables 4.3,
    4.4, 4.5, and 4.6 of Varshalovich, Page 119."""

    if j == 0.5:
        if m == 0.5:
            if mp == 0.5:
                value = cos(0.5 * beta)
            elif mp == -0.5:
                value = -sin(0.5 * beta)

        elif m == -0.5:
            if mp == 0.5:
                value = sin(0.5 * beta)
            elif mp == -0.5:
                value = cos(0.5 * beta)

    elif j == 1.0:
        if m == 1.0:
            if mp == 1.0:
                value = (1.0 + cos(beta)) / 2.0
            elif mp == 0.0:
                value = -sin(beta) / sqrt(2.0)
            elif mp == -1.0:
                value = (1.0 - cos(beta)) / 2.0

        elif m == 0.0:
            if mp == 1.0:
                value = sin(beta) / sqrt(2.0)
            elif mp == 0.0:
                value = cos(beta)
            elif mp == -1.0:
                value = -sin(beta) / sqrt(2.0)

        elif m == -1.0:
            if mp == 1.0:
                value = (1.0 - cos(beta)) / 2.0
            elif mp == 0.0:
                value = sin(beta) / sqrt(2.0)
            elif mp == -1.0:
                value = (1.0 + cos(beta)) / 2.0

    elif j == 1.5:
        if m == 1.5:
            if mp == 1.5:
                value = (cos(0.5 * beta)) ** 3.0
            elif mp == 0.5:
                value = -sqrt(3.0) * sin(0.5 * beta) * (cos(0.5 * beta))**2.0
            elif mp == -0.5:
                value = sqrt(3.0) * (sin(0.5 * beta))**2.0 * cos(0.5 * beta)
            elif mp == -1.5:
                value = -(sin(0.5 * beta))**3.0

        elif m == 0.5:
            if mp == 1.5:
                value = sqrt(3.0) * sin(0.5 * beta) * (cos(0.5 * beta))**2.0
            elif mp == 0.5:
                value = cos(0.5 * beta) * (3.0 * (cos(0.5 * beta))**2.0 - 2.0)
            elif mp == -0.5:
                value = sin(0.5 * beta) * (3.0 * (sin(0.5 * beta))**2.0 - 2.0)
            elif mp == -1.5:
                value = sqrt(3.0) * (sin(0.5 * beta))**2.0 * cos(0.5 * beta)

        elif m == -0.5:
            if mp == 1.5:
                value = sqrt(3.0) * (sin(0.5 * beta))**2.0 * cos(0.5 * beta)
            elif mp == 0.5:
                value = -sin(0.5 * beta) * (3.0 * (sin(0.5 * beta))**2.0 - 2.0)
            elif mp == -0.5:
                value = cos(0.5 * beta) * (3.0 * (cos(0.5 * beta))**2.0 - 2.0)
            elif mp == -1.5:
                value = -sqrt(3.0) * sin(0.5 * beta) * (cos(0.5 * beta))**2.0

        elif m == -1.5:
            if mp == 1.5:
                value = (sin(0.5 * beta))**3.0
            elif mp == 0.5:
                value = sqrt(3.0) * (sin(0.5 * beta))**2.0 * cos(0.5 * beta)
            elif mp == -0.5:
                value = sqrt(3.0) * sin(0.5 * beta) * (cos(0.5 * beta))**2.0
            elif mp == -1.5:
                value = (cos(0.5 * beta))**3.0

    elif j == 2.0:
        if m == 2.0:
            if mp == 2.0:
                value = (1.0 + cos(beta))**2.0 / 4.0
            elif mp == 1.0:
                value = -sin(beta) * (1.0 + cos(beta)) / 2.0
            elif mp == 0.0:
                value = 0.5 * sqrt(3.0 / 2.0) * (sin(beta))**2.0
            elif mp == -1.0:
                value = -sin(beta) * (1.0 - cos(beta)) / 2.0
            elif mp == -2.0:
                value = (1.0 - cos(beta))**2.0 / 4.0

        elif m == 1.0:
            if mp == 2.0:
                value = sin(beta) * (1.0 + cos(beta)) / 2.0
            elif mp == 1.0:
                value = (2.0 * (cos(beta))**2.0 + cos(beta) - 1.0) / 2.0
            elif mp == 0.0:
                value = -sqrt(3.0 / 2.0) * sin(beta) * cos(beta)
            elif mp == -1.0:
                value = -(2.0 * (cos(beta))**2.0 - cos(beta) - 1.0) / 2.0
            elif mp == -2.0:
                value = -sin(beta) * (1.0 - cos(beta)) / 2.0

        elif m == 0.0:
            if mp == 2.0:
                value = 0.5 * sqrt(3.0 / 2.0) * (sin(beta))**2.0
            elif mp == 1.0:
                value = sqrt(3.0 / 2.0) * sin(beta) * cos(beta)
            elif mp == 0.0:
                value = (3.0 * (cos(beta))**2.0 - 1.0) / 2.0
            elif mp == -1.0:
                value = -sqrt(3.0 / 2.0) * sin(beta) * cos(beta)
            elif mp == -2.0:
                value = 0.5 * sqrt(3.0 / 2.0) * (sin(beta))**2.0

        elif m == -1.0:
            if mp == 2.0:
                value = sin(beta) * (1.0 - cos(beta)) / 2.0
            elif mp == 1.0:
                value = -(2.0 * (cos(beta))**2.0 - cos(beta) - 1.0) / 2.0
            elif mp == 0.0:
                value = sqrt(3.0 / 2.0) * sin(beta) * cos(beta)
            elif mp == -1.0:
                value = (2.0 * (cos(beta))**2.0 + cos(beta) - 1.0) / 2.0
            elif mp == -2.0:
                value = -sin(beta) * (1.0 + cos(beta)) / 2.0

        elif m == -2.0:
            if mp == 2.0:
                value = (1.0 - cos(beta))**2.0 / 4.0
            elif mp == 1.0:
                value = sin(beta) * (1.0 - cos(beta)) / 2.0
            elif mp == 0.0:
                value = 0.5 * sqrt(3.0 / 2.0) * (sin(beta))**2.0
            elif mp == -1.0:
                value = sin(beta) * (1.0 + cos(beta)) / 2.0
            elif mp == -2.0:
                value = (1.0 + cos(beta))**2.0 / 4.0

    return value

######################################################################


def exact_U(j, m, mp, omega, theta, phi):
    """Returns exact expressions of U_{MM'}^{j}(omega, theta, phi) according to
    Tables 4.23-26 of Varshalovich, Page 127."""

    if j == 0.:
        if m == 0.:
            if mp == 0.:
                value = 1.

    if j == 0.5:
        if m == 0.5:
            if mp == 0.5:
                value = cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta)
            elif mp == -0.5:
                value = -1j * sin(omega / 2.) * sin(theta) * exp(-1j * phi)

        elif m == -0.5:
            if mp == 0.5:
                value = -1j * sin(omega / 2.) * sin(theta) * exp(1j * phi)
            elif mp == -0.5:
                value = cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta)

    elif j == 1.0:
        if m == 1.0:
            if mp == 1.0:
                value = (
                    cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**2.
            elif mp == 0.0:
                value = -1j * sqrt(2.) * sin(omega / 2.) * sin(theta) * \
                    exp(-1j * phi) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))
            elif mp == -1.0:
                value = -(sin(omega / 2.) * sin(theta) * exp(-1j * phi))**2.

        elif m == 0.0:
            if mp == 1.0:
                value = -1j * sqrt(2.) * sin(omega / 2.) * sin(theta) * \
                    exp(1j * phi) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))
            elif mp == 0.0:
                value = 1 - 2. * (sin(omega / 2.)**2.) * (sin(theta)**2.)
            elif mp == -1.0:
                value = -1j * sqrt(2.) * sin(omega / 2.) * sin(theta) * \
                    exp(-1j * phi) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))

        elif m == -1.0:
            if mp == 1.0:
                value = -(sin(omega / 2.) * sin(theta) * exp(1j * phi))**2.
            elif mp == 0.0:
                value = -1j * sqrt(2.) * sin(omega / 2.) * sin(theta) * \
                    exp(1j * phi) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))
            elif mp == -1.0:
                value = \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**2.

    elif j == 1.5:
        if m == 1.5:
            if mp == 1.5:
                value = (
                    cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**3.
            elif mp == 0.5:
                value = -1j * sqrt(3.) * sin(omega / 2.) * sin(theta) * \
                    exp(-1j * phi) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**2.
            elif mp == -0.5:
                value = - sqrt(3.) * (sin(omega / 2.) * sin(theta) *
                                      exp(-1j * phi))**2. * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))
            elif mp == -1.5:
                value = 1j * \
                    (sin(omega / 2.) * sin(theta) * exp(-1j * phi))**3.

        elif m == 0.5:
            if mp == 1.5:
                value = -1j * sqrt(3.) * sin(omega / 2.) * sin(theta) * \
                    exp(1j * phi) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**2.
            elif mp == 0.5:
                value = \
                    (1. - 3. * (sin(omega / 2.)**2.) * (sin(theta)**2.)) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))
            elif mp == -0.5:
                value = -1j * sin(omega / 2.) * sin(theta) * exp(-1j * phi) * \
                    (2. - 3. * (sin(omega / 2.)**2.) * (sin(theta)**2.))
            elif mp == -1.5:
                value = - sqrt(3.) * \
                    (sin(omega / 2.) * sin(theta) * exp(-1j * phi))**2. * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))

        elif m == -0.5:
            if mp == 1.5:
                value = - sqrt(3.) * \
                    ((sin(omega / 2.) * sin(theta) * exp(1j * phi))**2.) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))
            elif mp == 0.5:
                value = -1j * sin(omega / 2.) * sin(theta) * exp(1j * phi) * \
                    (2. - 3. * (sin(omega / 2.)**2.) * (sin(theta)**2.))
            elif mp == -0.5:
                value = \
                    (1. - 3. * (sin(omega / 2.)**2.) * (sin(theta)**2.)) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))
            elif mp == -1.5:
                value = -1j * sqrt(3.) * sin(omega / 2.) * sin(theta) * \
                    exp(-1j * phi) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**2.

        elif m == -1.5:
            if mp == 1.5:
                value = 1j * (sin(omega / 2.) * sin(theta) * exp(1j * phi))**3.
            elif mp == 0.5:
                value = - sqrt(3.) * \
                    ((sin(omega / 2.) * sin(theta) * exp(1j * phi))**2.) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))
            elif mp == -0.5:
                value = -1j * sqrt(3.) * sin(omega / 2.) * sin(theta) * \
                    exp(1j * phi) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**2.
            elif mp == -1.5:
                value = \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**3.

    elif j == 2.0:
        if m == 2.0:
            if mp == 2.0:
                value = (
                    cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**4.
            elif mp == 1.0:
                value = - 2. * 1j * sin(omega / 2.) * sin(theta) * \
                    exp(-1j * phi) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**3.
            elif mp == 0.0:
                value = - sqrt(6) * \
                    ((sin(omega / 2.) * sin(theta) * exp(-1j * phi))**2.) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**2.
            elif mp == -1.0:
                value = 2. * 1j * (sin(omega / 2.) * sin(theta) *
                                   exp(-1j * phi))**3. * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))
            elif mp == -2.0:
                value = (sin(omega / 2.) * sin(theta) * exp(-1j * phi))**4.

        elif m == 1.0:
            if mp == 2.0:
                value = - 2. * 1j * sin(omega / 2.) * sin(theta) * \
                    exp(1j * phi) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**3.
            elif mp == 1.0:
                value = \
                    (1. - 4. * (sin(omega / 2.)**2.) * (sin(theta)**2.)) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**2.
            elif mp == 0.0:
                value = -1j * sqrt(6) * sin(omega / 2.) * sin(theta) * \
                    exp(-1j * phi) * \
                    (1. - 2. * (sin(omega / 2.)**2.) * (sin(theta)**2.)) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))
            elif mp == -1.0:
                value = - ((sin(omega / 2.) * sin(theta) *
                            exp(-1j * phi))**2.) * \
                    (3. - 4. * (sin(omega / 2.)**2.) * (sin(theta)**2.))
            elif mp == -2.0:
                value = 2. * 1j * \
                    (sin(omega / 2.) * sin(theta) * exp(-1j * phi))**3. * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))

        elif m == 0.0:
            if mp == 2.0:
                value = - sqrt(6) * \
                    (sin(omega / 2.) * sin(theta) * exp(1j * phi))**2. * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))**2.
            elif mp == 1.0:
                value = -1j * sqrt(6) * sin(omega / 2.) * sin(theta) * \
                    exp(1j * phi) * \
                    (1 - 2. * (sin(omega / 2.)**2.) * (sin(theta)**2.)) * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))
            elif mp == 0.0:
                value = 1. - 6 * (sin(omega / 2.)**2.) * (sin(theta)**2.) * \
                    (1 - (sin(omega / 2.)**2.) * (sin(theta)**2.))
            elif mp == -1.0:
                value = -1j * sqrt(6) * sin(omega / 2.) * sin(theta) * \
                    exp(-1j * phi) * \
                    (1. - 2. * (sin(omega / 2.)**2.) * (sin(theta)**2.)) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))
            elif mp == -2.0:
                value = - sqrt(6) * (sin(omega / 2.) * sin(theta) *
                                     exp(-1j * phi))**2. * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**2.

        elif m == -1.0:
            if mp == 2.0:
                value = 2. * 1j * \
                    (sin(omega / 2.) * sin(theta) * exp(1j * phi))**3. * \
                    (cos(omega / 2.) - 1j * sin(omega / 2.) * cos(theta))
            elif mp == 1.0:
                value = \
                    - (sin(omega / 2.) * sin(theta) * exp(1j * phi))**2. * \
                    (3. - 4. * (sin(omega / 2.)**2.) * (sin(theta)**2.))
            elif mp == 0.0:
                value = -1j * sqrt(6) * sin(omega / 2.) * sin(theta) * \
                    exp(1j * phi) * \
                    (1. - 2. * (sin(omega / 2.)**2.) * (sin(theta)**2.)) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))
            elif mp == -1.0:
                value = \
                    (1. - 4. * (sin(omega / 2.)**2.) * (sin(theta)**2.)) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**2.
            elif mp == -2.0:
                value = - 2. * 1j * sin(omega / 2.) * sin(theta) * \
                    exp(-1j * phi) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**3.

        elif m == -2.0:
            if mp == 2.0:
                value = (sin(omega / 2.) * sin(theta) * exp(1j * phi))**4.
            elif mp == 1.0:
                value = 2. * 1j * \
                    (sin(omega / 2.) * sin(theta) * exp(1j * phi))**3. * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))
            elif mp == 0.0:
                value = - sqrt(6) * \
                    (sin(omega / 2.) * sin(theta) * exp(1j * phi))**2. * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**2.
            elif mp == -1.0:
                value = - 2. * 1j * sin(omega / 2.) * sin(theta) * \
                    exp(1j * phi) * \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**3.
            elif mp == -2.0:
                value = \
                    (cos(omega / 2.) + 1j * sin(omega / 2.) * cos(theta))**4.
    return value

###############################################################################


def compare_exact_numerical_WignerD():
    """Function to compare numerical value of D_{MM'}^{j}(alpha, beta, gamma)
    obtained by the function WignerD with the exact value obtained from
    exact_Wigner_d according to Varshalovich, Tables 4.3-6, Page 119."""

    angles = [0., 0.15 * np.pi, 0.35 * np.pi, 0.5 * np.pi, 0.65 * np.pi,
              0.8 * np.pi, np.pi, 1.15 * np.pi, 1.3 * np.pi,
              1.5 * np.pi, 1.65 * np.pi, 1.8 * np.pi, 2. * np.pi]

    degrees = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5],
               [0.5, -0.5, 0.5],
               [1., 1., 1.], [1., 1., 0.], [1., 1., -1.], [1., 0., 1.],
               [1., 0., 0.], [1., 0., -1.], [1., -1., 1.], [1., -1., 0.],
               [1., -1., -1.], [1.5, 1.5, 1.5], [1.5, 1.5, 0.5],
               [1.5, 1.5, -0.5], [1.5, 1.5, -1.5], [1.5, 0.5, 1.5],
               [1.5, 0.5, 0.5], [1.5, 0.5, -0.5], [1.5, 0.5, -1.5],
               [1.5, -0.5, 1.5], [1.5, -0.5, 0.5], [1.5, -0.5, -0.5],
               [1.5, -0.5, -1.5], [1.5, -1.5, 1.5], [1.5, -1.5, 0.5],
               [1.5, -1.5, -0.5], [1.5, -1.5, -1.5],
               [2., 2., 2.], [2., 2., 1.], [2., 2., 0.], [2., 2., -1.],
               [2., 2., -2.], [2., 1., 2.], [2., 1., 1.], [2., 1., 0.],
               [2., 1., -1.], [2., 1., -2.], [2., 0., 2.], [2., 0., 1.],
               [2., 0., 0.], [2., 0., -1.], [2., 0., -2.], [2., -1., 2.],
               [2., -1., 1.], [2., -1., 0.], [2., -1., -1.], [2., -1., -2.],
               [2., -2., 2.], [2., -2., 1.], [2., -2., 0.], [2., -2., -1.],
               [2., -2., -2.]]

    factorial = [1]
    for _ in range(int(2. * 2.) + 2):
        if _ > 0:
            factorial += [_ * factorial[_ - 1]]

    for alpha in angles:
        for beta in angles:
            for gamma in angles:
                for [jj, m, mp] in degrees:
                    DD = WignerD(jj, m, mp, alpha, beta, gamma, factorial)
                    dd = exact_Wigner_d(jj, m, mp, beta)
                    assert (abs(DD.real -
                                (exp(-1j * m * alpha) * dd *
                                 exp(-1j * mp * gamma)).real) < 10.**(-14.))
                    assert (abs(DD.imag -
                                (exp(-1j * m * alpha) * dd *
                                 exp(-1j * mp * gamma)).imag) < 10.**(-14.))

###############################################################################


def compare_exact_numerical_U():
    """Function to compare numerical value of U_{MM'}^{j}(omega, theta, phi)
    obtained by the function U with the exact value obtained from
    exact_U according to Varshalovich, Tables 4.23-26, Page 127."""

    angles = [0., 0.15 * np.pi, 0.35 * np.pi, 0.5 * np.pi, 0.65 * np.pi,
              0.8 * np.pi, np.pi, 1.15 * np.pi, 1.3 * np.pi,
              1.5 * np.pi, 1.65 * np.pi, 1.8 * np.pi, 2. * np.pi]

    degrees = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5],
               [0.5, -0.5, 0.5],
               [1., 1., 1.], [1., 1., 0.], [1., 1., -1.], [1., 0., 1.],
               [1., 0., 0.], [1., 0., -1.], [1., -1., 1.], [1., -1., 0.],
               [1., -1., -1.], [1.5, 1.5, 1.5], [1.5, 1.5, 0.5],
               [1.5, 1.5, -0.5], [1.5, 1.5, -1.5], [1.5, 0.5, 1.5],
               [1.5, 0.5, 0.5], [1.5, 0.5, -0.5], [1.5, 0.5, -1.5],
               [1.5, -0.5, 1.5], [1.5, -0.5, 0.5], [1.5, -0.5, -0.5],
               [1.5, -0.5, -1.5], [1.5, -1.5, 1.5], [1.5, -1.5, 0.5],
               [1.5, -1.5, -0.5], [1.5, -1.5, -1.5],
               [2., 2., 2.], [2., 2., 1.], [2., 2., 0.], [2., 2., -1.],
               [2., 2., -2.], [2., 1., 2.], [2., 1., 1.], [2., 1., 0.],
               [2., 1., -1.], [2., 1., -2.], [2., 0., 2.], [2., 0., 1.],
               [2., 0., 0.], [2., 0., -1.], [2., 0., -2.], [2., -1., 2.],
               [2., -1., 1.], [2., -1., 0.], [2., -1., -1.], [2., -1., -2.],
               [2., -2., 2.], [2., -2., 1.], [2., -2., 0.], [2., -2., -1.],
               [2., -2., -2.]]

    factorial = [1]
    for _ in range(int(2. * 2.) + 2):
        if _ > 0:
            factorial += [_ * factorial[_ - 1]]

    for omega in angles:
        for theta in angles:
            for phi in angles:
                for [jj, m, mp] in degrees:
                    UU = U(jj, m, mp, omega, theta, phi, factorial)
                    exactU = exact_U(jj, m, mp, omega, theta, phi)
                    assert (abs(UU.real - exactU.real) < 10.**(-14.))
                    assert (abs(UU.imag - exactU.imag) < 10.**(-14.))

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

# compare_exact_numerical_WignerD()
# compare_exact_numerical_U()
