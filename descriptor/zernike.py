import numpy as np
from numpy import cos, sqrt
from ase.data import atomic_numbers
from ..utilities import FingerprintsError
from scipy.special import sph_harm
try:  # for scipy v <= 0.90
    from scipy import factorial as fac
except ImportError:
    try:  # for scipy v >= 0.10
        from scipy.misc import factorial as fac
    except ImportError:  # for newer version of scipy
        from scipy.special import factorial as fac
try:
    from .. import fmodules
except ImportError:
    fmodules = None


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
        for _ in range(4 * nmax + 3):
            self.factorial += [float(fac(0.5 * _))]

        # Checking if the functional forms of fingerprints in the train set
        # is the same as those of the current version of the code:
        if self.fingerprints_tag != 1:
            raise FingerprintsError('Functional form of fingerprints has been '
                                    'changed. Re-train you train images set, '
                                    'and use the new variables.')

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
                        c_nlm = 0.
                        for n_index in range(len_of_neighbors):
                            neighbor = Rs[n_index]
                            n_symbol = n_symbols[n_index]
                            x = (neighbor[0] - home[0]) / self.cutoff
                            y = (neighbor[1] - home[1]) / self.cutoff
                            z = (neighbor[2] - home[2]) / self.cutoff
                            rho = np.linalg.norm([x, y, z])

                            if self.fortran:
                                Z_nlm = fmodules.calculate_z(n=n, l=l, m=m,
                                                             x=x, y=y, z=z,
                                                             factorial=self.factorial,
                                                             length=len(self.factorial))
                                Z_nlm = self.Gs[symbol][n_symbol] * Z_nlm * \
                                    cutoff_fxn(rho * self.cutoff, self.cutoff)
                            else:
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

                                Z_nlm = self.Gs[symbol][n_symbol] * \
                                    calculate_R(n, l, rho, self.factorial) * \
                                    sph_harm(m, l, phi, theta) * \
                                    cutoff_fxn(rho * self.cutoff, self.cutoff)

                            # Alternative ways to calculate Z_nlm
#                            Z_nlm = self.Gs[symbol][n_symbol] * \
#                                calculate_Z(n, l, m, x, y, z, self.factorial) * \
#                                cutoff_fxn(rho * self.cutoff, self.cutoff)
#                            Z_nlm = self.Gs[symbol][n_symbol] * \
#                                calculate_Z2(n, l, m, x, y, z) * \
#                                cutoff_fxn(rho * self.cutoff, self.cutoff)

                            # sum over neighbors
                            c_nlm += np.conjugate(Z_nlm)
                        # sum over m values
                        if m == 0:
                            norm += c_nlm * np.conjugate(c_nlm)
                        else:
                            norm += 2. * c_nlm * np.conjugate(c_nlm)

                    fingerprint.append(norm.real)

        return fingerprint

    def get_der_fingerprint(self, index, symbol, n_indices, n_symbols, Rs,
                            p, q):
        """
        Returns the value of the derivative of G for atom with index and
        symbol with respect to coordinate x_{q} of atom index p. n_indices,
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
        :param p: Index of the pair atom.
        :type p: int
        :param q: Direction of the derivative; is an integer from 0 to 2.
        :type q: int

        :returns: list of float -- the value of the derivative of the
                                   fingerprints for atom with index and symbol
                                   with respect to coordinate x_{q} of atom
                                   index p.
        """
        home = self.atoms[index].position

        fingerprint_prime = []
        for n in range(self.nmax + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:

                    if self.fortran:  # fortran version; faster
                        G_numbers = [self.Gs[symbol][elm] for elm in n_symbols]
                        numbers = [atomic_numbers[elm] for elm in n_symbols]
                        if len(Rs) == 0:
                            norm_prime = 0.
                        else:
                            norm_prime = \
                                fmodules.calculate_zernike_prime(n=n,
                                                                 l=l,
                                                                 n_length=len(
                                                                     n_indices),
                                                                 n_indices=list(
                                                                     n_indices),
                                                                 numbers=numbers,
                                                                 rs=Rs,
                                                                 g_numbers=G_numbers,
                                                                 cutoff=self.cutoff,
                                                                 indexx=index,
                                                                 home=home,
                                                                 p=p,
                                                                 q=q,
                                                                 fac_length=len(
                                                                     self.factorial),
                                                                 factorial=self.factorial)
                    else:
                        norm_prime = 0.
                        for m in range(l + 1):
                            c_nlm = 0.
                            c_nlm_prime = 0.
                            for n_index, n_symbol, neighbor in zip(n_indices,
                                                                   n_symbols,
                                                                   Rs):
                                x = (neighbor[0] - home[0]) / self.cutoff
                                y = (neighbor[1] - home[1]) / self.cutoff
                                z = (neighbor[2] - home[2]) / self.cutoff
                                rho = np.linalg.norm([x, y, z])

                                _Z_nlm = calculate_Z(n, l, m,
                                                     x, y, z,
                                                     self.factorial)
                                # Calculates Z_nlm
                                Z_nlm = _Z_nlm * \
                                    cutoff_fxn(rho * self.cutoff, self.cutoff)

                                # Calculates Z_nlm_prime
                                Z_nlm_prime = _Z_nlm * \
                                    cutoff_fxn_prime(
                                        rho * self.cutoff, self.cutoff) * \
                                    der_position(
                                        index, n_index, home, neighbor, p, q)
                                if (Kronecker(n_index, p) -
                                   Kronecker(index, p)) == 1:
                                    Z_nlm_prime += \
                                        cutoff_fxn(rho * self.cutoff,
                                                   self.cutoff) * \
                                        calculate_Z_prime(n, l, m,
                                                          x, y, z, q,
                                                          self.factorial) / \
                                        self.cutoff
                                elif (Kronecker(n_index, p) -
                                      Kronecker(index, p)) == -1:
                                    Z_nlm_prime -= \
                                        cutoff_fxn(rho * self.cutoff,
                                                   self.cutoff) * \
                                        calculate_Z_prime(n, l, m,
                                                          x, y, z, q,
                                                          self.factorial) / \
                                        self.cutoff

                                # sum over neighbors
                                c_nlm += self.Gs[symbol][
                                    n_symbol] * np.conjugate(Z_nlm)
                                c_nlm_prime += self.Gs[symbol][
                                    n_symbol] * np.conjugate(Z_nlm_prime)
                            # sum over m values
                            if m == 0:
                                norm_prime += 2. * c_nlm * \
                                    np.conjugate(c_nlm_prime)
                            else:
                                norm_prime += 4. * c_nlm * \
                                    np.conjugate(c_nlm_prime)

                    fingerprint_prime.append(norm_prime.real)
        return fingerprint_prime

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

# Extra functions


def calculate_q(nu, k, l, factorial):
    """
    Calculates q_{kl}^{nu} according to the unnumbered equation afer Eq. (7) of
    "3D Zernike Descriptors for Content Based Shape Retrieval", Computer-Aided
    Design 36 (2004) 1047-1062.
    """
    result = ((-1) ** (k + nu)) * sqrt((2. * l + 4. * k + 3.) / 3.) * \
        binomial(k, nu, factorial) * \
        binomial(2. * k, k, factorial) * \
        binomial(2. * (k + l + nu) + 1., 2. * k, factorial) / \
        binomial(k + l + nu, k, factorial) / (2. ** (2. * k))

    return result


def calculate_Z(n, l, m, x, y, z, factorial):
    """
    Calculates Z_{nl}^{m}(x, y, z) according to the unnumbered equation afer
    Eq. (11) of "3D Zernike Descriptors for Content Based Shape Retrieval",
    Computer-Aided Design 36 (2004) 1047-1062.
    """

    value = 0.
    term1 = sqrt((2. * l + 1.) * factorial[int(2 * (l + m))] *
                 factorial[int(2 * (l - m))]) / factorial[int(2 * l)]
    term2 = 2. ** (-m)

    k = int((n - l) / 2.)
    for nu in range(k + 1):
        q = calculate_q(nu, k, l, factorial)
        for alpha in range(nu + 1):
            b1 = binomial(nu, alpha, factorial)
            for beta in range(nu - alpha + 1):
                b2 = binomial(nu - alpha, beta, factorial)
                term3 = q * b1 * b2
                for u in range(m + 1):
                    term4 = ((-1.)**(m - u)) * binomial(m, u, factorial) * \
                        (1j**u)
                    for mu in range(int((l - m) / 2.) + 1):
                        term5 = ((-1.)**mu) * (2.**(-2. * mu)) * \
                            binomial(l, mu, factorial) * \
                            binomial(l - mu, m + mu, factorial)
                        for eta in range(mu + 1):
                            r = 2. * (eta + alpha) + u
                            s = 2. * (mu - eta + beta) + m - u
                            t = 2. * (nu - alpha - beta - mu) + l - m
                            value += term3 * term4 * term5 * \
                                binomial(mu, eta, factorial) * \
                                (x ** r) * (y ** s) * (z ** t)
    term6 = (1j) ** m
    value = term1 * term2 * term6 * value
    value = value / sqrt(4. * np.pi / 3.)

    return value


def calculate_Z2(n, l, m, x, y, z):
    """
    The second implementation of Z_{nl}^{m}(x, y, z).
    """

    value1 = 0.
    k1 = (n - l) / 2
    for nu in range(k1 + 1):
        value1 += (x ** 2. + y ** 2. + z ** 2.) ** (
            nu + 0.5 * l) * calculate_q(nu, k1, l)

    term1 = sqrt((2. * l + 1.) * float(fac(l + m)) * float(fac(l - m))) / \
        float(fac(l))
    term2 = 2. ** (-m)
    term3 = (1j * x - y) ** m
    term4 = z ** (l - m)

    k2 = int((l - m) / 2.)
    value2 = 0.
    for mu in range(k2 + 1):
        term5 = binomial2(l, mu)
        term6 = binomial2(l - mu, m + mu)
        term7 = ((-1.) ** mu) * \
            ((x ** 2. + y ** 2.) ** mu) / ((4 * (z ** 2.)) ** mu)
        value2 += term5 * term6 * term7

    value2 *= term1 * term2 * term3 * term4
    value2 *= (1j) ** m

    value = value1 * value2

    value = value / (x ** 2. + y ** 2. + z ** 2.) ** (0.5 * l)

    value *= sqrt(3. / (4. * np.pi))

    return value


def calculate_R(n, l, rho, factorial):
    """
    Calculates R_{n}^{l}(rho) according to the last equation of wikipedia.
    """
    if (n - l) % 2 != 0:
        return 0
    else:
        value = 0.
        k = (n - l) / 2
        term1 = np.sqrt(2. * n + 3.)

        for s in xrange(k + 1):
            b1 = binomial(k, s, factorial)
            b2 = binomial(n - s - 1 + 1.5, k, factorial)
            value += ((-1) ** s) * b1 * b2 * (rho ** (n - 2. * s))

        value *= term1
        return value


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


def cutoff_fxn_prime(Rij, Rc):
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


def binomial(n, k, factorial):
    """Returns C(n,k) = n!/(k!(n-k)!)."""

    assert n >= 0 and k >= 0 and n >= k, \
        'n and k should be non-negative integers with n >= k.'
    c = factorial[int(2 * n)] / \
        (factorial[int(2 * k)] * factorial[int(2 * (n - k))])
    return c


def binomial2(n, k):
    """Returns C(n,k) = n!/(k!(n-k)!)."""

    assert n >= 0 and k >= 0 and n >= k, \
        'n and k should be non-negative integers.'

    c = float(fac(n)) / (float(fac(k)) * float(fac(n - k)))
    return c


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


def der_position(m, n, Rm, Rn, l, i):
    """
    Returns the derivative dR_{mn}/dR_{l,i} of the norm of position vector
    R_{mn} with respect to x_{i} of atomic index l.

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


def Kronecker(i, j):
    """
    Kronecker delta function.

    :param i: First index of Kronecker delta.
    :type i: int
    :param j: Second index of Kronecker delta.
    :type j: int

    :returns: int -- the value of the Kronecker delta.
    """
    if i == j:
        return 1
    else:
        return 0


def calculate_Z_prime(n, l, m, x, y, z, p, factorial):
    """
    Calculates dZ_{nl}^{m}(x, y, z)/dR_{p} according to the unnumbered equation
    afer Eq. (11) of "3D Zernike Descriptors for Content Based Shape
    Retrieval", Computer-Aided Design 36 (2004) 1047-1062.
    """

    value = 0.
    term1 = sqrt((2. * l + 1.) * factorial[int(2 * (l + m))] *
                 factorial[int(2 * (l - m))]) / factorial[int(2 * l)]
    term2 = 2. ** (-m)

    k = int((n - l) / 2.)
    for nu in range(k + 1):
        q = calculate_q(nu, k, l, factorial)
        for alpha in range(nu + 1):
            b1 = binomial(nu, alpha, factorial)
            for beta in range(nu - alpha + 1):
                b2 = binomial(nu - alpha, beta, factorial)
                term3 = q * b1 * b2
                for u in range(m + 1):
                    term4 = ((-1.)**(m - u)) * binomial(
                        m, u, factorial) * (1j**u)
                    for mu in range(int((l - m) / 2.) + 1):
                        term5 = ((-1.)**mu) * (2.**(-2. * mu)) * \
                            binomial(l, mu, factorial) * \
                            binomial(l - mu, m + mu, factorial)
                        for eta in range(mu + 1):
                            r = 2 * (eta + alpha) + u
                            s = 2 * (mu - eta + beta) + m - u
                            t = 2 * (nu - alpha - beta - mu) + l - m
                            coefficient = term3 * term4 * \
                                term5 * binomial(mu, eta, factorial)
                            if p == 0:
                                if r != 0:
                                    value += coefficient * r * \
                                        (x ** (r - 1)) * (y ** s) * (z ** t)
                            elif p == 1:
                                if s != 0:
                                    value += coefficient * s * \
                                        (x ** r) * (y ** (s - 1)) * (z ** t)
                            elif p == 2:
                                if t != 0:
                                    value += coefficient * t * \
                                        (x ** r) * (y ** s) * (z ** (t - 1))
    term6 = (1j) ** m
    value = term1 * term2 * term6 * value
    value = value / sqrt(4. * np.pi / 3.)

    return value
