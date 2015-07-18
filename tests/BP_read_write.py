#!/usr/bin/env python

import os

from ase.structure import molecule
from ase.calculators.emt import EMT
from ase import Atoms
from amp import AMP

###############################################################################


def make_training_images():
    atoms = molecule('CH4')
    atoms.set_calculator(EMT())
    atoms.get_potential_energy()

    images = [atoms]

    atoms = Atoms(atoms)
    atoms.set_calculator(EMT())
    atoms[3].z += 0.5

    atoms.get_potential_energy()

    images += [atoms]
    return images

###############################################################################


def test_read_write():
    pwd = os.getcwd()
    testdir = 'BP_read_write'
    os.mkdir(testdir)
    os.chdir(testdir)

    images = make_training_images()

    calc = AMP(label='test_writing')
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
    calc = AMP(label='testdir/calc')
    calc.train(images)

    # Open existing, save under new name.
    calc = AMP(load='test_writing',
                    label='test_writing2')
    calc.train(images)

    # Change label and re-train
    calc.set_label('test_writing_new/calc')
    calc.train(images)

    # Open existing without specifying new name.
    calc = AMP(load='test_writing')
    calc.train(images)

    os.chdir(pwd)

###############################################################################

if __name__ == '__main__':
    test_read_write()
