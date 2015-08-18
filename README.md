# Amp: Machine learning for atomistics #

Developed by Andrew Peterson & Alireza Khorshidi, Brown University School of Engineering

(This project was formerly known as "Neural". The last stable version of Neural can be found at https://bitbucket.org/andrewpeterson/neural)


This project lives at:
https://bitbucket.org/andrewpeterson/amp

Documentation lives at:
http://machine-learning-for-atomistics.readthedocs.org

*Amp* (*A*tomistic *m*achine-learning *p*otentials) allows for the modular representation of the potential energy surface with fingerprinting and regression methods of choice for the user.

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

Recommended step: Compiling Fortran modules:
============================================

AMP works in pure python, however, it will be annoyingly slow unless
the associated Fortran 90 modules are compiled which will speed up
several parts of the codes. The compilation of the Fortran 90 code
and integration with the python parts is accomplished with f2py,
which is part of NumPy. A Fortran 90 compiler will also be necessary
on the system; a reasonable open-source option is GNU Fortran, or
gfortran. This compiler will generate Fortran modules (.mod).
gfortran will also be used by f2py to generate extension module
fmodules.so on Linux or fmodules.pyd on Windows. In order to
prepare the extension module the following steps need to be taken:

Compile regression Fortran subroutines inside the regression
folder by:

$ cd ~/path/to/my/codes/regression

$ gfortran -c neuralnetwork.f90

Move the module ``regression.mod'' created in the last step, to the parent directory
by:

$ mv regression.mod ../regression.mod

Go to the parent directory and compile the main Fortran subroutines in companion with the descriptor and regression subroutines
by something like:

$ cd ../

$ f2py -c -m fmodules main.f90 descriptor/behler.f90 regression/neuralnetwork.f90

or on a Windows machine by:

$ f2py -c -m fmodules main.f90 descriptor/behler.f90 regression/neuralnetwork.f90 --fcompiler=gnu95 --compiler=mingw32

If the version of fmodules.f90 is not updated, an exception
will be raised which tells user which version number should be
employed.

