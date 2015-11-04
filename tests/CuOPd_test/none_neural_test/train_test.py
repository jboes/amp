"""
Exact none-neural scheme cost function, energy per atom RMSE and force RMSE
for five different configurations have been calculated in Mathematica. This
script checks the values calculated by the code during training with and
without fortran modules and also on different number of cores.

"""
###############################################################################

import numpy as np
from collections import OrderedDict
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from amp import Amp
from amp.regression import NeuralNetwork

###############################################################################
# Making the list of images


def generate_images():
    '''Generates five images with varying configurations'''

    images = []
    for step in range(5):
        image = Atoms([Atom('Pd', (0., 0., 0.)), Atom('O', (0., 2., 0.)),
                       Atom('Pd', (0., 0., 3.)), Atom('Pd', (1., 0., 0.))])
        if step > 0:
            image[step - 1].position =  \
                image[step - 1].position + np.array([0., 1., 1.])

        image.set_calculator(EMT())
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)
        images.append(image)

    return images

###############################################################################
# Correct energies obtained from Mathematica

energies = [
    4.867131012402346,
    4.867131012402346,
    4.991184749099429,
    4.976000413790214,
    4.980739289325503]

square_error_energy = [
    3199.0527662215736,
    24.61269249131737,
    3280.5799195475247,
    3237.61554804185,
    9.554481244618035]

square_error_forces = [
    104497.24820961579,
    819.5410428375304,
    104480.43562362246,
    104527.37890557194,
    647.0618234348974]

square_error = [square_error_energy[_] + 0.04 * square_error_forces[_]
                for _ in range(5)]

cost_function = 0.
for _ in square_error:
    cost_function += _

SumSquareErrorEnergy = 0.
for _ in square_error_energy:
    SumSquareErrorEnergy += _

SumSquareErrorForces = 0.
for _ in square_error_forces:
    SumSquareErrorForces += _

energy_per_atom_rmse = np.sqrt(SumSquareErrorEnergy / 5)
force_rmse = np.sqrt(SumSquareErrorForces / 5)

###############################################################################
# The test function


def test():

    ###########################################################################
    # Parameters

    weights = OrderedDict([(1, np.array([[1., 2.5],
                                         [0., 1.5],
                                         [0., -1.5],
                                         [3., 9.],
                                         [1., -2.5],
                                         [2., 3.],
                                         [2., 2.5],
                                         [3., 0.],
                                         [-3.5, 1.],
                                         [5., 3.],
                                         [-2., 2.5],
                                         [-4., 4.],
                                         [0., 0.]])),
                           (2, np.array([[1.],
                                         [2.],
                                         [0.]])),
                           (3, np.array([[3.5],
                                         [0.]]))])

    scalings = OrderedDict([('intercept', 3.),
                            ('slope', 2.)])

    images = generate_images()

    ###########################################################################
    # Testing pure-python and fortran versions of behler-neural on different
    # number of processes

    for fortran in [False, True]:
        for data_format in ['db', 'json']:
            for cores in range(1, 7):

                label = 'traintest/%s-%s-%i' % (fortran, data_format, cores)

                calc = Amp(descriptor=None,
                           regression=NeuralNetwork(hiddenlayers=(2, 1),
                                                    activation='tanh',
                                                    weights=weights,
                                                    scalings=scalings,),
                           fortran=fortran,
                           label=label,)

                calc.train(images=images, energy_goal=10.**10.,
                           force_goal=10.**10., force_coefficient=0.04,
                           cores=cores, data_format=data_format)

                # Check for consistency between the two models
                assert (abs(calc.cost_function - cost_function) <
                        10.**(-5.)), \
                    'The calculated value of cost function is wrong!'
                assert (abs(calc.energy_per_atom_rmse - energy_per_atom_rmse) <
                        10.**(-5.)), \
                    'The calculated value of energy per atom RMSE is wrong!'
                assert (abs(calc.force_rmse - force_rmse) <
                        10 ** (-5)), \
                    'The calculated value of force RMSE is wrong!'

###############################################################################

if __name__ == '__main__':
    test()
