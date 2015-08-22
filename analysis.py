"""
Tools for analysis of output. Currently only contains a tool to make a
plot of convergence.

This can be imported or run directly as a module as

    python -m neural.analysis <file-to-analyze>

"""

###############################################################################

import numpy as np
import os
from ase import io
from amp import AMP
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

###############################################################################


class ConvergencePlot:

    """
    Creates plot analyzing the convergence behavior.
    Only works on BPNeural log files at the moment.
    
    :param logfile: Write function at which log data exists.
    :type log: Logger object

    """
    ###########################################################################

    def __init__(self, logfile):


        f = open(logfile, 'r')
        lines = f.read().splitlines()
        f.close()

        # Get number of images.
        for line in lines:
            if 'unique images after hashing.' in line:
                no_images = float(line.split()[0])
                break

        # Find where convergence data starts.
        startline = None
        for index, line in enumerate(lines):
            if 'Starting optimization of cost function...' in line:
                startline = index
                break

        # Get parameters.
        ready = [False, False, False, False]
        for index, line in enumerate(lines[startline:]):
            if 'Energy goal' in line:
                ready[0] = True
                energygoal = float(line.split()[-1])
            elif 'Force goal' in line:
                ready[1] = True
                forcegoal = float(line.split()[-1])
            elif 'force coefficient' in line:
                ready[2] = True
                forcecoefficient = float(line.split()[-1])
            elif 'No force training.' in line:
                ready[1] = True
                ready[2] = True
                forcegoal = None
                forcecoefficient = None
            elif line.split()[0] == '0':
                ready[3] = True
                startline = startline + index
            if ready == [True, True, True, True]:
                break

        if forcegoal:
            E = energygoal**2 * no_images
            F = forcegoal**2 * no_images
            costfxngoal = E + forcecoefficient * F
        else:
            costfxngoal = energygoal**2 * no_images

        # Extract data.
        steps, es, fs, costfxns = [], [], [], []
        costfxnEs, costfxnFs = [], []
        for line in lines[startline:]:
            if 'Saving checkpoint' in line:
                continue
            elif 'convergence!' in line:
                continue
            elif 'unconverged!' in line:
                continue
            elif 'optimization completed successfully.' in line:
                break
            elif 'could not find parameters for the' in line:
                break
            if forcegoal:
                print(line)
                step, time, costfxn, e, f = line.split()
                fs.append(float(f))
            else:
                step, time, costfxn, e = line.split()
            steps.append(int(step))
            es.append(float(e))
            costfxns.append(costfxn)

            # Determine components of the cost function.
            if forcegoal:
                E = float(e)**2 * no_images
                F = float(f)**2 * no_images
                costfxnEs.append(E / float(costfxn))
                costfxnFs.append(forcecoefficient * F / float(costfxn))

        # Make plots.

        from matplotlib import pyplot

        fig = pyplot.figure(figsize=(6., 8.))
        # Margins, vertical gap, and top-to-bottom ratio of figure.
        lm, rm, bm, tm, vg, tb = 0.12, 0.05, 0.08, 0.03, 0.08, 4.
        bottomaxheight = (1. - bm - tm - vg) / (tb + 1.)

        ax = fig.add_axes((lm, bm + bottomaxheight + vg,
                           1. - lm - rm, tb * bottomaxheight))
        ax.semilogy(steps, es, 'b', lw=2, label='energy rmse')
        if forcegoal:
            ax.semilogy(steps, fs, 'g', lw=2, label='force rmse')
        ax.semilogy(steps, costfxns, color='0.5', lw=2, label='cost function')
        # Targets.
        ax.semilogy([steps[0], steps[-1]], [energygoal] * 2, color='b',
                    linestyle=':')
        if forcegoal:
            ax.semilogy([steps[0], steps[-1]], [forcegoal] * 2, color='g',
                        linestyle=':')
        ax.semilogy([steps[0], steps[-1]], [costfxngoal] * 2, color='0.5',
                    linestyle=':')
        ax.set_ylabel('error')
        ax.set_xlabel('BFGS step')
        ax.legend()

        if forcegoal:
            ax = fig.add_axes((lm, bm, 1. - lm - rm, bottomaxheight))
            ax.fill_between(x=np.array(steps), y1=costfxnEs, color='blue')
            ax.fill_between(x=np.array(steps), y1=costfxnEs,
                            y2=np.array(costfxnEs) + np.array(costfxnFs),
                            color='green')
            ax.set_ylabel('cost function component')
            ax.set_xlabel('BFGS step')
            ax.set_ylim(0, 1)

        self._fig = fig

    ###########################################################################

    def getfig(self):
        """
        Returns the figure object.
        
        """

        return self._fig

    ###########################################################################

    def savefig(self, plotfile='convergence.pdf'):
        """
        Saves the figure object to filename.

        """

        self._fig.savefig(plotfile)

###############################################################################


def plot_convergence(logfile, plotfile='convergence.pdf'):
    """
    Makes a plot of the convergence of the cost function and its energy
    and force components.

    :param logfile: Write function at which log data exists.
    :type log: Logger object
    :param plotfile: File for plots.
    :type plotfile: Object

    """

    plot = ConvergencePlot(logfile)
    plot.savefig(plotfile)

###############################################################################


def plot_parity(load,
                images,
                plot_forces=True,
                plotfile='parityplot.pdf',
                color='b.',):
    """
    Makes a parity plot of AMP energies and forces versus real energies and
    forces.

    :param load: Path for loading an existing parameters of AMP calculator.
    :type load: str
    :param images: List of ASE atoms objects with positions, symbols, energies,
                   and forces in ASE format. This is the training set of data.
                   This can also be the path to an ASE trajectory (.traj) or
                   database (.db) file. Energies can be obtained from any
                   reference, e.g. DFT calculations.
    :type images: list or str
    :param plot_forces: Determines whether or not forces should be plotted as
                        well.
    :type plot_forces: bool
    :param plotfile: File for plots.
    :type plotfile: Object
    :param color: Plot color.
    :type color: str

    """

    calc = AMP(load=load)

    if isinstance(images, str):
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = io.Trajectory(images, 'r')
        elif extension == '.db':
            images = io.read(images)

    # Make plots.

    from matplotlib import pyplot

    if plot_forces is None:

        fig = pyplot.figure(figsize=(5., 5.))

        # Plot I: Energy.
        ax = fig.add_subplot(111)
        for image in images:
            act_energy = image.get_potential_energy(apply_constraint=False)
            pred_energy = calc.get_potential_energy(image)
            ax.plot(act_energy, pred_energy, color)

        ax.set_xlabel('actual energy, eV')
        ax.set_ylabel('AMP energy, eV')
        ax.set_title('Energies')

    else:

        fig = pyplot.figure(figsize=(5., 10.))

        # Plot I: Energy.
        ax = fig.add_subplot(211)
        for image in images:
            act_energy = image.get_potential_energy(apply_constraint=False)
            pred_energy = calc.get_potential_energy(image)
            ax.plot(act_energy, pred_energy, color)

        ax.set_xlabel('actual energy, eV')
        ax.set_ylabel('AMP energy, eV')
        ax.set_title('Energies')

        #######################################################################

        # Plot II: Forces.
        ax = fig.add_subplot(212)
        for image in images:
            act_force = image.get_forces(apply_constraint=False)
            image.set_calculator(calc)
            amp_force = calc.get_forces(image)
            for atom in image:
                index = atom.index
                for k in range(3):
                    ax.plot(act_force[index][k], amp_force[index][k], color)

        ax.set_xlabel('actual force, eV/Ang')
        ax.set_ylabel('AMP force, eV/Ang')
        ax.set_title('Forces')

        #######################################################################

    fig.savefig(plotfile)

###############################################################################


def plot_error(load,
               images,
               plot_forces=True,
               plotfile='errorplot.pdf',
               color='b.',
               energy_rmse=None,
               force_rmse=None,):
    """
    Makes a plot of deviations in per atom energies and forces versus real
    energies and forces.

    :param load: Path for loading an existing parameters of AMP calculator.
    :type load: str
    :param images: List of ASE atoms objects with positions, symbols, energies,
                   and forces in ASE format. This is the training set of data.
                   This can also be the path to an ASE trajectory (.traj) or
                   database (.db) file. Energies can be obtained from any
                   reference, e.g. DFT calculations.
    :type images: list or str
    :param plot_forces: Determines whether or not forces should be plotted as
                        well.
    :type plot_forces: bool
    :param plotfile: File for plots.
    :type plotfile: Object
    :param color: Plot color.
    :type color: str
    :param energy_rmse: Energy per atom RMSE.
    :type energy_rmse: float
    :param force_rmse: Force RMSE.
    :type force_rmse: float

    """

    calc = AMP(load=load)

    if isinstance(images, str):
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = io.Trajectory(images, 'r')
        elif extension == '.db':
            images = io.read(images)

    # Make plots.

    from matplotlib import pyplot

    if plot_forces is None:

        fig = pyplot.figure(figsize=(5., 5.))

        # Plot I: Energy.
        ax = fig.add_subplot(111)
        for image in images:
            act_energy = image.get_potential_energy(apply_constraint=False)
            pred_energy = calc.get_potential_energy(image)
            energy_error = abs(pred_energy - act_energy) / len(image)
            act_energy_per_atom = act_energy / len(image)
            ax.plot(act_energy_per_atom, energy_error, color)

        if energy_rmse is not None:
            min_act_energy = \
                min([image.get_potential_energy(apply_constraint=False) /
                     len(image) for image in images])
            max_act_energy = \
                max([image.get_potential_energy(apply_constraint=False) /
                     len(image) for image in images])
            # draw horizontal line for rmse
            ax.plot([min_act_energy, max_act_energy],
                    [energy_rmse, energy_rmse],
                    'r-',
                    lw=2,)
            ax.text(max_act_energy,
                    energy_rmse,
                    'rmse = %6.5f' % energy_rmse,
                    ha='right',
                    va='bottom',
                    color='red')

        ax.set_xlabel('actual energy per atom, eV per atom')
        ax.set_ylabel('|actual energy - AMP energy| / number of atoms, '
                      'eV per atom')
        ax.set_title('Energies')

    else:

        fig = pyplot.figure(figsize=(5., 10.))

        # Plot I: Energy.
        ax = fig.add_subplot(211)
        for image in images:
            act_energy = image.get_potential_energy(apply_constraint=False)
            pred_energy = calc.get_potential_energy(image)
            energy_error = abs(pred_energy - act_energy) / len(image)
            act_energy_per_atom = act_energy / len(image)
            ax.plot(act_energy_per_atom, energy_error, color)

        if energy_rmse is not None:
            min_act_energy = \
                min([image.get_potential_energy(apply_constraint=False) /
                     len(image) for image in images])
            max_act_energy = \
                max([image.get_potential_energy(apply_constraint=False) /
                     len(image) for image in images])
            # draw horizontal line for rmse
            ax.plot([min_act_energy, max_act_energy],
                    [energy_rmse, energy_rmse],
                    'r-',
                    lw=2,)
            ax.text(max_act_energy,
                    energy_rmse,
                    'rmse = %6.5f' % energy_rmse,
                    ha='right',
                    va='bottom',
                    color='red')

        ax.set_xlabel('actual energy per atom, eV per atom')
        ax.set_ylabel('|actual energy - AMP energy| / number of atoms, '
                      'eV per atom')
        ax.set_title('Energies')

        #######################################################################

        # Plot II: Forces.
        ax = fig.add_subplot(212)
        for image in images:
            act_force = image.get_forces(apply_constraint=False)
            image.set_calculator(calc)
            amp_force = calc.get_forces(image)
            for atom in image:
                index = atom.index
                for k in range(3):
                    ax.plot(act_force[index][k],
                            abs(amp_force[index][k] - act_force[index][k]),
                            color)

        if force_rmse is not None:
            min_act_force = \
                min([min(np.array(
                    image.get_forces(apply_constraint=False)).ravel())
                    for image in images])
            max_act_force = \
                max([max(np.array(
                    image.get_forces(apply_constraint=False)).ravel())
                    for image in images])
            # draw horizontal line for rmse
            ax.plot([min_act_force, max_act_force],
                    [force_rmse, force_rmse],
                    'r-',
                    lw=2,)
            ax.text(max_act_force,
                    force_rmse,
                    'rmse = %5.4f' % force_rmse,
                    ha='right',
                    va='bottom',
                    color='red',)

        ax.set_xlabel('actual force, eV/Ang')
        ax.set_ylabel('|actual force - AMP force|, eV/Ang')
        ax.set_title('Forces')

        #######################################################################

    fig.savefig(plotfile)

###############################################################################


if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 2

    logfile = sys.argv[-1]

    plot_convergence(logfile=logfile)
