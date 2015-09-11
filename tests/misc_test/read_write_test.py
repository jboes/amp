#!/usr/bin/env python

import os

from ase.structure import molecule
from ase.calculators.emt import EMT
from ase import Atoms
from amp import Amp

###############################################################################


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

###############################################################################


def test():
    pwd = os.getcwd()
    testdir = 'read_write_test'
    os.mkdir(testdir)
    os.chdir(testdir)

    images = make_training_images()

    calc = Amp(label='calc')
    calc.train(images)

    # Test that we cannot overwrite. (Strange code here
    # because we *want* it to raise an exception...)
    try:
        calc.train(images)
    except IOError:
        pass
    else:
        raise RuntimeError('Code allowed to overwrite!')

    # Test that we can manually overwrite.
    calc.train(images, overwrite=True)

    # New directory calculator.
    calc = Amp(label='testdir/calc')
    calc.train(images)

    # Open existing, save under new name.
    calc = Amp(load='calc',
                    label='calc2')
    calc.train(images)

    # Change label and re-train
    calc.set_label('calc_new/calc')
    calc.train(images)

    # Open existing without specifying new name.
    calc = Amp(load='calc')
    calc.train(images)

    os.chdir(pwd)

###############################################################################

if __name__ == '__main__':
    test()
