# Amp: Atomistic Machine-learning Potentials#

Developed by Andrew Peterson & Alireza Khorshidi, Brown University School of Engineering. *Amp* allows for the modular representation of the potential energy surface with descriptor and regression methods of choice for the user.

This project lives at:
https://bitbucket.org/andrewpeterson/amp

Documentation lives at:
http://amp.readthedocs.org

(This project was formerly known as "Neural". The last stable version of Neural can be found at https://bitbucket.org/andrewpeterson/neural)


License
=======

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


Installation
==================================

AMP is python-based and is designed to integrate closely with the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_ (ASE).
In its most basic form, it has few requirements:

* Python, version 2.7 is recommended
* ASE
* NumPy (>= 1.9 for saving data in ".db" database format)
* SciPy

----------------------------------
Install ASE
----------------------------------

We always test against the latest version (svn checkout) of ASE, but slightly older versions (>=3.9.0) are likely to work
as well. Follow the instructions at the `ASE <https://wiki.fysik.dtu.dk/ase/download.html>`_ website. ASE itself depends
upon python with the standard numeric and scientific packages. Verify that you have working versions of
`NumPy <http://numpy.org>`_ and `SciPy <http://scipy.org>`_. We also recommend `matplotlib <http://matplotlib.org>`_ in
order to generate plots.

----------------------------------
Check out the code
----------------------------------

As a relatively new project, it may be preferable to use the development version rather than "stable" releases, as improvements are constantly being made and features added.
We run daily unit tests to make sure that our development code works as intended.
We recommend checking out the latest version of the code via `the project's bitbucket
page <https://bitbucket.org/andrewpeterson/amp/>`_. If you use git, check out the code with::

   $ cd ~/path/to/my/codes
   $ git clone git@bitbucket.org:andrewpeterson/amp.git

where you should replace '~/path/to/my/codes' with wherever you would like the code to be located on your computer.
If you do not use git, just download the code as a zip file from the project's
`download <https://bitbucket.org/andrewpeterson/amp/downloads>`_ page, and extract it into '~/path/to/my/codes'. Please make sure that the folder '~/path/to/my/codes/amp' includes the script '__init__.py' as well as the folders 'descriptor', 'regression', ... 
At the download page, you can also find historical numbered releases.

----------------------------------
Set the environment
----------------------------------

You need to let your python version know about the existence of the amp module. Add the following line to your '.bashrc'
(or other appropriate spot), with the appropriate path substituted for '~/path/to/my/codes'::

   $ export PYTHONPATH=~/path/to/my/codes:$PYTHONPATH

You can check that this works by starting python and typing the below command, verifying that the location listed from
the second command is where you expect::

   >>> import amp
   >>> print(amp.__file__)

----------------------------------
Recommended step: Fortran modules
----------------------------------

The code is designed to work in pure python, which makes it is easier to read, develop, and debug. However, it will be
annoyingly slow unless you compile the associated fortran modules which speed up some crucial parts of the code. The
compilation of Fortran codes and integration with the python parts is accomplished with the command 'f2py', which is
part of NumPy. A Fortran compiler will also be necessary on your system; a reasonable open-source option is GNU Fortran,
or gfortran. This complier will generate Fortran modules (.mod). gfortran will also be used by f2py to generate
extension module 'fmodules.so' on Linux or 'fmodules.pyd' on Windows. In order to prepare the extension module take the
following steps:

* Compile regression Fortran subroutines inside the regression folder by::

   $ cd ~/path/to/my/codes/regression/
   $ gfortran -c neuralnetwork.f90

* Move the module 'regression.mod' created in the last step, to the parent directory by::

   $ mv regression.mod ../

* Compile the main Fortran subroutines in the parent directory in companian with the descriptor and regression subroutines
  by something like::

   $ f2py -c -m fmodules main.f90 descriptor/gaussian.f90 regression/neuralnetwork.f90

or on a Windows machine by::

   $ f2py -c -m fmodules main.f90 descriptor/gaussian.f90 regression/neuralnetwork.f90 --fcompiler=gnu95 --compiler=mingw32

If you update the code and your fmodules extension is not updated, an exception will be raised, telling you
to re-compile.

----------------------------------
Recommended step: Run the tests
----------------------------------

We include tests in the package to ensure that it still runs as intended as we continue our development; we run these
tests on the latest build every night to try to keep bugs out. It is a good idea to run these tests after you install the
package to see if your installation is working. The tests are in the folder `tests`; they are designed to run with
`nose <https://nose.readthedocs.org/>`_. If you have nose installed, run the commands below::

   $ mkdir /tmp/amptests
   $ cd /tmp/amptests
   $ nosetests ~/path/to/my/codes/amp/tests
