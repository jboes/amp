.. _main:

Main
=================

.. autoclass:: amp.AMP
   :members:

.. autoclass:: amp.MultiProcess
   :members:

.. autoclass:: amp.SaveNeighborLists
   :members:

.. autoclass:: amp.SaveFingerprints
   :members:

.. autoclass:: amp.CostFxnandDer
   :members:

.. autoexception:: amp.ConvergenceOccurred
   :members:

.. autoexception:: amp.TrainingConvergenceError
   :members:

.. autoexception:: amp.ExtrapolateError
   :members:

.. autofunction:: amp._calculate_fingerprints


.. autofunction:: amp._calculate_der_fingerprints

.. autofunction:: amp._calculate_cost_function_fortran

.. autofunction:: amp._calculate_cost_function_python

.. autofunction:: amp.calculate_fingerprints_range

.. autofunction:: amp.compare_train_test_fingerprints

.. autofunction:: amp.interpolate_images

.. autofunction:: amp.send_data_to_fortran

.. autofunction:: amp.ravel_fingerprints_of_images

.. autofunction:: amp.ravel_neighborlists_and_der_fingerprints_of_images

.. autofunction:: amp.now