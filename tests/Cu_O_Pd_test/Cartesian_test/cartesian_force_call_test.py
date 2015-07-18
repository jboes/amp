'''This simple script aims to check the forces for 5 configurations of Pd and O
atoms from CartesianNeural against their analytic analogs obtained from
mathematica scripts. This ensures satisfactory reliability of CartesianNeural
calculator'''

###############################################################################

import numpy as np
from collections import OrderedDict
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from amp import AMP
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

correct_predicted_energies = [
    2.8790052147097365,
    2.94996846154781,
    2.8379853804487674,
    2.93903176594631,
    2.8268641232578524]

correct_predicted_forces = \
    [np.array([[-0.028253803095123684, -0.06103335054650668,
                -0.013714970478412424],
               [-0.0673165965697658, 0.00706784362836426,
                0.03526128502374155],
               [-0.048364830343516386, -
                   0.03924170302550931, -0.028696070608096536],
               [0.0794422977490082, 0.03389515124106801,
                0.021709058290306244]]),
     np.array([[-0.01754641897084717, -0.04718648173035798,
                -0.018246318875359786],
               [-0.04610785668285803, 0.01678800945537331,
                   0.023269268101432896],
               [-0.04501990433085196, -0.028358666265857417,
                -0.01813490477232701],
               [0.06955036840504043, 0.01937071705830063,
                0.012577130063697805]]),
     np.array([[-0.030376877394300457, -0.06153773426936673,
                -0.010467643029520957],
               [-0.07048318721804682, 0.002147116482564936,
                   0.03730807825175713],
               [-0.04541050390105909, -
                   0.040436679167356084, -0.030714386093558595],
               [0.07652333304760882, 0.037180468399823786,
                0.023738197761539202]]),
     np.array([[-0.020581570904470325, -0.04436801298465096,
                -0.009894376204021112],
               [-0.048994382489288925, 0.00502581986042367,
                   0.02567261715470015],
               [-0.035083138770274434, -
                   0.028546246867670034, -0.02090063452755298],
               [0.057669824148646104, 0.024707655544067442,
                0.01582299072428838]]),
     np.array([[-0.023773130487652807, -0.05968048950332527,
                -0.020266110113729405],
               [-0.0604999681587194, 0.017067698006297854,
                0.030899010708997435],
               [-0.05413435425217735, -
                0.03659584739101289, -0.024426738768848506],
               [0.08497470748207635, 0.02701376900193143,
                0.017454739872069417]])]

###############################################################################
# The test function


def Cartesian_force_call_test():

    ###########################################################################
    # Parameters

    weights = \
        OrderedDict([(1, np.array([[0.14563579, 0.19176385],
                                   [-0.01991609, 0.35873379],
                                   [-0.27988951, 0.03490866],
                                   [0.19195185, 0.43116313],
                                   [0.41035737, 0.02617128],
                                   [-0.13235187, -0.23112657],
                                   [-0.29065111, 0.23865951],
                                   [0.05854897, 0.24249052],
                                   [0.13660673, 0.19288898],
                                   [0.31894165, -0.41831075],
                                   [-0.23522261, -0.24009372],
                                   [-0.14450575, -0.15275409],
                                   [0., 0.]])),
                     (2, np.array([[-0.27415999],
                                   [0.28538579],
                                   [0.]])),
                     (3, np.array([[0.32147131],
                                   [0.]]))])

    scalings = OrderedDict([('intercept', 3.),
                            ('slope', 2.)])

    images = generate_images()

    ###########################################################################
    # Testing pure-python and fortran versions of CartesianNeural on different
    # number of processes

    for fortran in [False]:  # change to [False, True] when fortran subroutines
        # added.

        calc = AMP(fingerprint=None,
                   regression=NeuralNetwork(hiddenlayers=(2, 1),
                                            weights=weights,
                                            scalings=scalings,
                                            activation='tanh'),
                   fortran=fortran,)

        predicted_energies = [calc.get_potential_energy(image) for image in
                              images]

        for image_no in range(len(predicted_energies)):
            assert (abs(predicted_energies[image_no] -
                        correct_predicted_energies[image_no]) <
                    5 * 10.**(-10.)), \
                'The calculated energy of image %i is wrong!' % (image_no + 1)

        predicted_forces = [calc.get_forces(image) for image in images]

        for image_no in range(len(predicted_forces)):
            for index in range(np.shape(predicted_forces[image_no])[0]):
                for direction in range(3):
                    assert (abs(predicted_forces[image_no][index][direction] -
                                correct_predicted_forces[image_no][index]
                                [direction]) < 5 * 10.**(-10.)), \
                        'The calculated %i force of atom %i of image %i is' \
                        'wrong!' % (direction, index, image_no + 1)

###############################################################################

if __name__ == '__main__':
    Cartesian_force_call_test()
