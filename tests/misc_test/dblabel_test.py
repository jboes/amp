#!/usr/bin/env python
"""Tests that the dblabel command works at all. This test can be
combined with the read_write_test.py at some point."""

from ase.structure import molecule
from amp import Amp
from ase.calculators.emt import EMT
from ase import Atoms
import os


def make_training_images():
    atoms = molecule('CH4')
    atoms.set_calculator(EMT())
    atoms.get_potential_energy(apply_constraint=False)

    images = [atoms]

    atoms = Atoms(atoms)
    atoms.set_calculator(EMT())
    atoms[3].z += 0.5

    atoms.get_potential_energy(apply_constraint=False)

    images += [atoms]
    return images


def test():
    pwd = os.getcwd()
    testdir = 'dblabel_test'
    os.mkdir(testdir)
    os.chdir(testdir)
    images = make_training_images()
    calc = Amp(label='out', dblabel='db')
    calc.train(images=images, global_search=None)

    os.chdir(pwd)

if __name__ == '__main__':
    test()
