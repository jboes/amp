#!/usr/bin/env python

import os

from ase.structure import molecule
from ase.calculators.emt import EMT
from ase import Atoms
from amp import Amp
from amp.regression import NeuralNetwork
from ase.io import PickleTrajectory

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
    ff = PickleTrajectory('images.traj', 'w')
    for image in images:
        ff.write(image)

    for data_format in ['db', 'json']:

        calc = Amp(label='calc', regression=NeuralNetwork(hiddenlayers=(5, 5)))
        calc.train(images, global_search=None, extend_variables=False,
                   data_format=data_format)

        # Test that we cannot overwrite. (Strange code here
        # because we *want* it to raise an exception...)
        try:
            calc.train(images, global_search=None, extend_variables=False,
                       data_format=data_format)
        except IOError:
            pass
        else:
            raise RuntimeError('Code allowed to overwrite!')

        # Test that we can manually overwrite.
        calc.train(images, overwrite=True, global_search=None,
                   extend_variables=False, data_format=data_format)

        # New directory calculator.
        calc = Amp(label='testdir/calc',
                   regression=NeuralNetwork(hiddenlayers=(5, 5)))
        calc.train(images, global_search=None, extend_variables=False,
                   data_format=data_format)

        # Open existing, save under new name.
        calc = Amp(load='calc',
                        label='calc2')
        calc.train(images, global_search=None, extend_variables=False,
                   data_format=data_format)

        # Change label and re-train
        calc.set_label('calc_new/calc')
        calc.train(images, global_search=None, extend_variables=False,
                   data_format=data_format)

        # Open existing without specifying new name.
        calc = Amp(load='calc')
        calc.train(images, global_search=None, extend_variables=False,
                   data_format=data_format)

        os.chdir(pwd)

###############################################################################

if __name__ == '__main__':
    test()
