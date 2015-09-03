.. _UseAmp:

==================================
How to use Amp?
==================================

Amp can look at the atomic system in two different schemes. In the first scheme, the
atomic system is looked at as a whole. The consequent potential energy surface approximation is valid only for systems
with the same number of atoms and identities. In other words the learned potential is size-dependent. On the other hand,
in the second scheme, Amp looks at local atomic environments, ending up with a learned potential which is size-independent,
and can be simultaneously used for systems with different sizes.
The user should first determine which scheme is of interest. 

----------------------------------
Initializing Amp
----------------------------------

The calculator as well as descriptors and regressions should be first imported by::

   >>> from amp import Amp
   >>> from amp.descriptor import *
   >>> from amp.regression import *

Then Amp is initiated with a descriptor and a regression algorithm, e.g.::

   >>> calc = Amp(descriptor=Behler()
		       regression=NeuralNetwork())

The values of arguments shown above are the default values in the current release of Amp. A size-dependent scheme can be
simply taken by ``descriptor=None``. Optional arguments for initializing Amp can be reviewed at :ref:`main` methods.

----------------------------------
Starting from old parameters
----------------------------------

The parameters of the calculator are saved after training so that the calculator can be re-established for future
calculations. If the previously trained parameter file is named 'old.json', it can be introduced to Amp to take those
values with something like::

   >>> calc = Amp(load='old.json', label='new')

The label ('new') is used as a prefix for any output from use of this calculator. In general, the calculator is written
to not overwrite your old settings, but it is a good idea to have a different name for this label to avoid accidentally
overwriting your carefully trained calculator's files!

----------------------------------
Training Amp
----------------------------------

Training a new calculator instance occurs with the train method, which is given a list of train images and desired
values of root mean square error (rmse) for forces and atomic energy as below::

   >>> calc.train(images, energy_goal=0.001, force_goal=0.005)

Calling this method will generate output files where you can watch the progress. Note that this is in general a
computationally-intensive process! These two parameters as well as other optional parameters are described at.                  

In general, we plan to make the ASE database the recommended data type. However, this is a `work in progress <https://wiki.fysik.dtu.dk/ase/ase/db/db.html>`_ over at ASE.

----------------------------------
Growing your train set
----------------------------------

Say you have a nicely trained calculator that you built around a train set of 10,000 images, but you decide you want
to add another 1,000 images to the train set. Rather than starting training from scratch, it can be faster to train
from the previous optimum. This is fairly simple and can be accomplished as::

   >>> calc = Amp(load='old_calc', label='new_calc')
   >>> calc.train(all_images)

In this case, 'all_images' contains all 11,000 images (the old set **and** the new set).

If you are training on a large set of images, a building-up strategy can be effective. For example, to train on 100,000
images, you could first train on 10,000 images, then add 10,000 images to the set and re-train from the previous
parameters, etc. If the images are nicely randomized, this can give you some confidence that you are not training inside
too shallow of a basin. The first images set, which you are starting from, had better be representative of the verity of
all images you will add later.

----------------------------------
Using trained Amp
----------------------------------

The trained calculator can now be used just like any other ASE calculator, e.g.::

   >>> atoms = ...
   >>> atoms.set_calculator(calc)
   >>> energy = atoms.get_potential_energy()

where 'atoms' is an atomic configuration not necessarily included in the train set.

----------------------------------
Complete example
----------------------------------

The script below shows an example of training Amp. This script first creates some sample data using a molecular dynamics
simulation with the cheap EMT calculator in ASE. The images are then randomly divided into "train" and "test" data,
such that cross-validation can be performed. Then Amp is trained on the train data. After this, the predictions of the
Amp are compared to the parent data for both the train and test set. This is shown in a parity plot, which is saved to
disk.

.. literalinclude:: _static/completeexample.py

Note that most of the script is just creating the train set or analyzing the results -- the actual Amp initialization
and training takes place on just two lines in the middle of the script. The figure that was produced is shown below. As
both the generated data and the initial guess of the training parameters are random, this will look different each time
this script is run.

   .. image:: _static/parityplot.png
      :scale: 07 %
      :align: center
