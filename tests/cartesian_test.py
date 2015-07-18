#!/usr/bin/env python

"""Test of the cartesian neural network calculator. Randomly generates data
with the EMT potential in MD simulations. Both trains and tests getting
energy out of the calculator. Shows results for both interpolation and
extrapolation."""

###############################################################################

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import for headless test.
from matplotlib import pyplot as plt
from ase.calculators.emt import EMT
from ase.lattice.surface import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms
from scipy.optimize import fmin
import multiprocessing as mp
from amp.utilities import randomize_images
from amp import AMP

cores = mp.cpu_count()

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


def testCartesian():
    label = 'cartesian_test'
    if not os.path.exists(label):
        os.mkdir(label)
    fig = plt.figure(figsize=(5., 10.))

    # Case I: Interpolation.
    ax = fig.add_subplot(211)
    print('Generating data.')
    all_images = generate_data(10)
    train_images, test_images = randomize_images(all_images)

    print('Training network.')
    calc1 = AMP(fingerprint=None,
                fortran=False,
                label=os.path.join(label, 'calc1'))

    calc1.train(train_images, energy_goal=0.001,
                force_goal=0.005, cores=cores)

    print('Testing network.')

    pred_energies_int_train = []
    act_energies_int_train = []
    pred_energies_int_test = []
    act_energies_int_test = []

    for image in train_images:
        pred_energy = calc1.get_potential_energy(atoms=image)
        pred_energies_int_train.append(pred_energy)
        act_energy = image.get_potential_energy(apply_constraint=False)
        act_energies_int_train.append(act_energy)
        ax.plot(act_energy, pred_energy, 'b.')

    for image in test_images:
        pred_energy = calc1.get_potential_energy(atoms=image)
        pred_energies_int_test.append(pred_energy)
        act_energy = image.get_potential_energy(apply_constraint=False)
        act_energies_int_test.append(act_energy)
        ax.plot(act_energy, pred_energy, 'r.')

    # Perform a linear regression on the data points of the form y = a * x
    # where a is the estimator

    def cost_function(a):
        SSR = 0.  # Initialize sum of squared residuals
        for x, y in zip(act_energies_int_train, pred_energies_int_train):
            ypred = a * x
            SSR += (ypred - y) ** 2
        return SSR

    a = fmin(func=cost_function, x0=0.6)  # get OLS estimator

    print ('\nThe value of estimator for the training set in interpolation '
           'mode is %4.2f\n' % a)

    # Plot the regression line for the training set
    xs = np.linspace(min(act_energies_int_train),
                     max(act_energies_int_train), 100)
    y_smooth = [a * x for x in xs]
    ax.plot(xs, y_smooth, 'b-')

    def cost_function(a):
        SSR = 0.  # Initialize sum of squared residuals
        for x, y in zip(act_energies_int_test, pred_energies_int_test):
            ypred = a * x
            SSR += (ypred - y) ** 2
        return SSR

    a = fmin(func=cost_function, x0=0.6)  # get OLS estimator

    print ('\nThe value of the estimator for the test set in interpolation '
           'mode is %4.2f\n' % a)

    # Plot the regression line for the test set
    xs = np.linspace(min(act_energies_int_test),
                     max(act_energies_int_test), 100)
    y_smooth = [a * x for x in xs]
    ax.plot(xs, y_smooth, 'r-')
    ref = np.linspace(0., max(max(act_energies_int_test),
                              max(act_energies_int_train)), 100)
    ax.plot(ref, ref, 'm--')

    ax.set_xlabel('Actual energy, eV')
    ax.set_ylabel('Predicted energy, eV')
    ax.set_title('Interpolate')
    plt.grid()

    # Case II: Extrapolation
    ax = fig.add_subplot(212)
    print('Generating data.')
    train_images = generate_data(10)
    test_images = generate_data(10)

    print('Training network.')
    calc2 = AMP(fingerprint=None,
                fortran=False,
                label=os.path.join(label, 'calc2'))

    calc2.train(train_images, energy_goal=0.001,
                force_goal=0.005, cores=cores)

    print('Testing network.')

    pred_energies_ext_train = []
    act_energies_ext_train = []
    act_energies_ext_test = []
    pred_energies_ext_test = []

    for image in train_images:
        pred_energy = calc2.get_potential_energy(atoms=image)
        pred_energies_ext_train.append(pred_energy)
        act_energy = image.get_potential_energy(apply_constraint=False)
        act_energies_ext_train.append(act_energy)
        ax.plot(act_energy, pred_energy, 'b.')

    for image in test_images:
        pred_energy = calc2.get_potential_energy(atoms=image)
        pred_energies_ext_test.append(pred_energy)
        act_energy = image.get_potential_energy(apply_constraint=False)
        act_energies_ext_test.append(act_energy)
        ax.plot(act_energy, pred_energy, 'r.')

    # Perform a linear regression on the data points of the form y = a * x
    # where a is the estimator

    def cost_function(a):
        SSR = 0.  # Initialize sum of squared residuals
        for x, y in zip(act_energies_ext_train, pred_energies_ext_train):
            ypred = a * x
            SSR += (ypred - y) ** 2
        return SSR

    a = fmin(func=cost_function, x0=0.6)  # get OLS estimator

    print ('\nThe value of estimator for the training set in extrapolation '
           'mode is %4.2f\n' % a)

    # Plot the regression line for the training set
    xs = np.linspace(min(act_energies_ext_train),
                     max(act_energies_ext_train), 100)
    y_smooth = [a * x for x in xs]
    ax.plot(xs, y_smooth, 'b-')

    def cost_function(a):
        SSR = 0.  # Initialize sum of squared residuals
        for x, y in zip(act_energies_ext_test, pred_energies_ext_test):
            ypred = a * x
            SSR += (ypred - y) ** 2
        return SSR

    a = fmin(func=cost_function, x0=0.6)  # get OLS estimator

    print ('\nThe value of the estimator for the test set in extrapolation '
           'mode is %4.2f\n' % a)

    # Plot the regression line for the test set
    xs = np.linspace(min(act_energies_ext_test),
                     max(act_energies_ext_test), 100)
    y_smooth = [a * x for x in xs]
    ax.plot(xs, y_smooth, 'r-')
    ref = np.linspace(0., max(max(act_energies_ext_test),
                              max(act_energies_ext_train)), 100)
    ax.plot(ref, ref, 'm--')

    ax.set_xlabel('Actual energy, eV')
    ax.set_ylabel('Predicted energy, eV')
    ax.set_title('Extrapolate')
    plt.grid()

    fig.savefig(os.path.join(label, 'cartesian-test.pdf'))

###############################################################################

if __name__ == '__main__':
    testCartesian()
