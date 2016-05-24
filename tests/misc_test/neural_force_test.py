#!/usr/bin/env python
"""Test of the Gaussian-neural scheme of Amp with force training. Randomly
generates data with the EMT potential in MD simulations. Both trains and tests
getting energy out of the calculator. Shows results for both interpolation and
extrapolation."""

import os
from ase.calculators.emt import EMT
from ase.lattice.surface import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms
from amp import Amp
from amp.regression import NeuralNetwork
from amp.utilities import randomize_images

###############################################################################


def generate_data(count):
    """Generates test or training data with a simple MD simulation."""
    atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
    adsorbate = Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
                       Atom('Cu', atoms[7].position + (0., 0., 5.))])
    atoms.extend(adsorbate)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    atoms.set_calculator(EMT())
    MaxwellBoltzmannDistribution(atoms, 300. * units.kB)
    dyn = VelocityVerlet(atoms, dt=1. * units.fs)
    newatoms = atoms.copy()
    newatoms.set_calculator(EMT())
    newatoms.get_potential_energy(apply_constraint=False)
    images = [newatoms]
    for step in range(count):
        dyn.run(5)
        newatoms = atoms.copy()
        newatoms.set_calculator(EMT())
        newatoms.get_potential_energy(apply_constraint=False)
        images.append(newatoms)
    return images

###############################################################################


def test_none():
    label = 'force_test'
    if not os.path.exists(label):
        os.mkdir(label)

    print('Generating data.')
    all_images = generate_data(4)
    train_images, test_images = randomize_images(all_images)

    print('Training none-neural network.')
    calc1 = Amp(descriptor=None,
                label=os.path.join(label, 'none'),
                regression=NeuralNetwork(hiddenlayers=(5, 5)))
    calc1.train(train_images, energy_goal=0.01, force_goal=0.05,
                global_search=None)

    print('Testing none-neural network.')
    energies1 = []
    for image in all_images:
        energies1.append(calc1.get_potential_energy(atoms=image))

    print('Verify making new calc works.')
    params = calc1.todict()
    calc2 = Amp(**params)
    energies2 = []
    for image in all_images:
        energies2.append(calc2.get_potential_energy(atoms=image))
    assert energies1 == energies2

    print('Verifying can move an atom and get new energy.')
    image = all_images[0]
    image.set_calculator(calc2)
    e1 = image.get_potential_energy(apply_constraint=False)
    f1 = image.get_forces(apply_constraint=False)
    image[0].x += 0.5  # perturb
    e2 = image.get_potential_energy(apply_constraint=False)
    f2 = image.get_forces(apply_constraint=False)
    assert e1 != e2
    assert not (f1 == f2).all()

###############################################################################


def test_gaussian():
    label = 'force_test'
    if not os.path.exists(label):
        os.mkdir(label)

    print('Generating data.')
    all_images = generate_data(4)
    train_images, test_images = randomize_images(all_images)

    print('Training Gaussian-neural network.')
    calc1 = Amp(label=os.path.join(label, 'Gaussian'),
                regression=NeuralNetwork(hiddenlayers=(5, 5)))
    calc1.train(train_images, energy_goal=0.01, force_goal=0.05)

    print('Testing Gaussian-neural network.')
    energies1 = []
    for image in all_images:
        energies1.append(calc1.get_potential_energy(atoms=image))

    print('Verify making new calc works.')
    params = calc1.todict()
    calc2 = Amp(**params)
    energies2 = []
    for image in all_images:
        energies2.append(calc2.get_potential_energy(atoms=image))
    assert energies1 == energies2

    print('Verifying can move an atom and get new energy.')
    image = all_images[0]
    image.set_calculator(calc2)
    e1 = image.get_potential_energy(apply_constraint=False)
    f1 = image.get_forces(apply_constraint=False)
    image[0].x += 0.5  # perturb
    e2 = image.get_potential_energy(apply_constraint=False)
    f2 = image.get_forces(apply_constraint=False)
    assert e1 != e2
    assert not (f1 == f2).all()

###############################################################################

if __name__ == '__main__':
    test_gaussian()
    test_none()
