.. _parallelcomputing:

==================================
Parallel Computing
==================================

As the number of atoms corresponding to data points (images) or the number of data points
themselves in the train data set increases, serial computing takes a long time, and thus,
resorting to parallel computing becomes inevitable. Two common approaches of parallel
computing are either to parallelize over multiple threads or to parallelize over multiple
processes. In the former approach, threads usually have access to the same memory area,
which can lead to conflicts in case of improper synchronization. But in the second approach, separate memory
locations are allocated to each spawned process which avoids any possible interference
between sub-tasks. Amp takes the second approach via "multiprocessing", a built-in module
in Python 2.6 (and above) standard library. For details about multiprocessing module, the
reader is directed to `multiprocessing documentation <https://docs.python.org/2/library/multiprocessing.html>`_.
Only local concurrency (multi-computing over local cores) is possible in the current release
of Amp. Amp starts with serial computing, but when arriving to calculating atomic
fingerprints and their derivatives, multiple processes are spawned each doing a sub-task on
part of the atomic configuration, and writing the results in temporary JSON files.
Temporary files are next unified and saved in the script directory. If Fortran modules are
utilized, data of all images including real energies and forces, atomic identities, and
calculated values of fingerprints (as well as their derivatives and neighbor lists if
force-training) are then reshaped to rectangular arrays. Portions of the reshaped data
corresponding to each process are sent to the copy of Fortran modules at each core. Partial
cost function and its derivative with respect to variables are calculated on each core, and
the calculated values are returned back to the main python code via multiprocessing "Queues".
The main python code then adds up partial values. Amp automatically finds the number of
requested cores, prints it out at the top of "train-log.txt" script, and makes use of all of
them. A schematic of how parallel computing is designed in Amp is shown in the below figure.
When descriptor is None, no fingerprint operation is carried out.

.. image:: _static/mp.png
   :scale: 05 %
   :align: center
