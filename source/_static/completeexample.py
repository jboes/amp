from ase.lattice.surface import fcc110
from ase import Atom, Atoms
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase import io

from amp.utilities import randomize_images
from amp import Amp
from amp.descriptor import *
from amp.regression import *

###############################################################################


def test():

    # Generate atomic system to create test data.
    atoms = fcc110('Cu', (2, 2, 2), vacuum=7.)
    adsorbate = Atoms([Atom('H', atoms[7].position + (0., 0., 2.)),
                       Atom('H', atoms[7].position + (0., 0., 5.))])
    atoms.extend(adsorbate)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    calc = EMT()  # cheap calculator
    atoms.set_calculator(calc)

    # Run some molecular dynamics to generate data.
    trajectory = io.Trajectory('data.traj', 'w', atoms=atoms)
    MaxwellBoltzmannDistribution(atoms, temp=300. * units.kB)
    dynamics = VelocityVerlet(atoms, dt=1. * units.fs)
    dynamics.attach(trajectory)
    for step in range(50):
        dynamics.run(5)
    trajectory.close()

    # Train the calculator.
    train_images, test_images = randomize_images('data.traj')

    calc = Amp(descriptor=Behler(),
               regression=NeuralNetwork())
    calc.train(train_images, energy_goal=0.001, force_goal=None)

    # Plot and test the predictions.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot

    fig, ax = pyplot.subplots()

    for image in train_images:
        actual_energy = image.get_potential_energy()
        predicted_energy = calc.get_potential_energy(image)
        ax.plot(actual_energy, predicted_energy, 'b.')

    for image in test_images:
        actual_energy = image.get_potential_energy()
        predicted_energy = calc.get_potential_energy(image)
        ax.plot(actual_energy, predicted_energy, 'r.')

    ax.set_xlabel('Actual energy, eV')
    ax.set_ylabel('Amp energy, eV')

    fig.savefig('parityplot.png')

###############################################################################

if __name__ == '__main__':
    test()