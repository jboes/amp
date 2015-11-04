"""
This script creates a list of three images. It then calculates Behler- Neural
scheme cost function, energy per atom RMSE and force RMSE of different
combinations of images with and without fortran modules on different number of
cores, and check consistency between them.

"""

###############################################################################

import numpy as np
from ase.calculators.emt import EMT
from collections import OrderedDict
from ase import Atoms, Atom
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

scalings = OrderedDict([('intercept', 3.), ('slope', 2.)])


###############################################################################
# Testing pure-python and fortran versions of non-neural on different
# number of processes and different number of images

def test():

    images = generate_images()

    count = 0
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
                           label=label)

                calc.train(images=images, energy_goal=10.**10.,
                           force_goal=10.**10., cores=cores,
                           read_fingerprints=False, data_format=data_format)

                if count == 0:
                    reference_cost_function = calc.cost_function
                    reference_energy_rmse = calc.energy_per_atom_rmse
                    reference_force_rmse = calc.force_rmse
                    ref_cost_fxn_variable_derivatives = \
                        calc.der_variables_cost_function
                else:
                    assert (abs(calc.cost_function -
                                reference_cost_function) < 10.**(-20.)), \
                        '''Cost function value for %r fortran, %r data
                            format, and %i cores is not consistent with
                            the value of python version on single
                            core.''' % (fortran, data_format, cores)

                    assert (abs(calc.energy_per_atom_rmse -
                                reference_energy_rmse) < 10.**(-20.)), \
                        '''Energy rmse value for %r fortran, %r data format,
                        and %i cores is not consistent with the value of
                        python version on single
                        core.''' % (fortran, data_format, cores)

                    assert (abs(calc.force_rmse -
                                reference_force_rmse) < 10.**(-20.)), \
                        '''Force rmse value for %r fortran, %r data format,
                        and %i cores is not consistent with the value of
                        python version on single
                        core.''' % (fortran, data_format, cores)

                    for _ in range(len(ref_cost_fxn_variable_derivatives)):
                        assert (calc.der_variables_cost_function[_] -
                                ref_cost_fxn_variable_derivatives[_] <
                                10.**(-10.))
                        '''Derivative of the cost function for %r fortran,
                        %r data format, and %i cores is not consistent with
                        the value of python version on single
                        core.''' % (fortran, data_format, cores)

                count = count + 1

###############################################################################

if __name__ == '__main__':
    test()
