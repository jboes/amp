#!/usr/bin/env python
"""This test checks rotation and translation invariance of descriptor schemes.
Fingerprints both before and after a random rotation (+ translation) are
calculated and compared."""

import numpy as np
from numpy import sin, cos
from ase import Atom, Atoms
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from amp import Amp
from amp.descriptor import Behler, SphericalHarmonics, Zernike
import json
from ase.parallel import paropen
import shutil
import random

###############################################################################


def rotate_atom(x, y, z, phi, theta, psi):
    """Rotate atom in three dimensions."""

    rotation_matrix = [
        [cos(theta) * cos(psi),
         cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi),
         sin(phi) * sin(psi) - cos(phi) * sin(theta) * cos(psi)],
        [-cos(theta) * sin(psi),
         cos(phi) * cos(psi) - sin(phi) * sin(theta) * sin(psi),
         sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)],
        [sin(theta),
         -sin(phi) * cos(theta),
         cos(phi) * cos(theta)]
    ]

    [[xprime], [yprime], [zprime]] = np.dot(rotation_matrix, [[x], [y], [z]])

    return (xprime, yprime, zprime)

###############################################################################


def test():

    for descriptor in [Behler(), SphericalHarmonics(jmax=2.), Zernike(nmax=5)]:

        # Non-rotated atomic configuration
        atoms = Atoms([Atom('Pt', (0., 0., 0.)),
                       Atom('Pt', (0., 0., 1.)),
                       Atom('Pt', (0., 2., 1.))])

        atoms.set_constraint(FixAtoms(indices=[0]))
        atoms.set_calculator(EMT())
        atoms.get_potential_energy()
        atoms.get_forces(apply_constraint=False)

        calc = Amp(descriptor=descriptor,
                   fortran=False,
                   label='rotation_test/before_rot/')

        calc.train(images=[atoms],
                   force_goal=None,
                   energy_goal=10.**10.,
                   extend_variables=False,
                   global_search=None,
                   data_format='json')

    ###########################################################################

        # Randomly Rotated (and translated) atomic configuration
        rot = [random.random(), random.random(), random.random()]
        for i in range(1, len(atoms)):
            (atoms[i].x,
             atoms[i].y,
             atoms[i].z) = rotate_atom(atoms[i].x,
                                       atoms[i].y,
                                       atoms[i].z,
                                       rot[0] * np.pi,
                                       rot[1] * np.pi,
                                       rot[2] * np.pi)
        disp = [random.random(), random.random(), random.random()]
        for atom in atoms:
            atom.x += disp[0]
            atom.y += disp[1]
            atom.z += disp[2]

        calc = Amp(descriptor=descriptor,
                   fortran=False,
                   label='rotation_test/after_rot/')

        calc.train(images=[atoms],
                   force_goal=None,
                   energy_goal=10.**10.,
                   extend_variables=False,
                   global_search=None,
                   data_format='json')

    ###########################################################################

        fp1 = paropen('rotation_test/before_rot/-fingerprints.json', 'rb')
        nonrotated_data = json.load(fp1)

        fp2 = paropen('rotation_test/after_rot/-fingerprints.json', 'rb')
        rotated_data = json.load(fp2)

        for hash1, hash2 in zip(nonrotated_data.keys(), rotated_data.keys()):
            for index in nonrotated_data[hash1]:
                for _ in range(len(nonrotated_data[hash1][index])):
                    assert abs(nonrotated_data[hash1][index][_] -
                               rotated_data[hash2][index][_]) < 10.**(-7.)

        shutil.rmtree('rotation_test/before_rot')
        shutil.rmtree('rotation_test/after_rot')

###############################################################################

if __name__ == '__main__':
    test()
