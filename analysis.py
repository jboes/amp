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
from ase.parallel import paropen
import json
from amp import AMP
from amp.utilities import hash_image
from matplotlib import rcParams
from matplotlib import pyplot
rcParams.update({'figure.autolayout': True})

###############################################################################


class ConvergencePlot:

    """
    Creates plot analyzing the convergence behavior.

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

        :param plotfile: Name or path to the plot file.
        :type plotfile: str
        """
        self._fig.savefig(plotfile)

###############################################################################


def plot_convergence(logfile, plotfile='convergence.pdf'):
    """
    Makes a plot of the convergence of the cost function and its energy
    and force components.

    :param logfile: Write function at which log data exists.
    :type log: Logger object
    :param plotfile: Name or path to the plot file.
    :type plotfile: str
    """
    plot = ConvergencePlot(logfile)
    plot.savefig(plotfile)

###############################################################################


def plot_parity(load,
                images,
                plot_forces=True,
                plotfile='parityplot.pdf',
                color='b.',
                overwrite=False):
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
    :param overwrite: If a plot or an script containing values found overwrite
                      it.
    :type overwrite: bool
    """
    base_filename = os.path.splitext(plotfile)[0]
    energyscript = os.path.join('energy-' + base_filename + '.json')

    if (not overwrite) and os.path.exists(plotfile):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile)
    if (not overwrite) and os.path.exists(energyscript):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % energyscript)

    if plot_forces is not None:
        forcescript = os.path.join('force-' + base_filename + '.json')
        if (not overwrite) and os.path.exists(forcescript):
            raise IOError('File exists: %s.\nIf you want to overwrite,'
                          ' set overwrite=True or manually delete.'
                          % forcescript)

    calc = AMP(load=load)

    if isinstance(images, str):
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = io.Trajectory(images, 'r')
        elif extension == '.db':
            images = io.read(images)

    # Images is converted to dictionary form; key is hash of image.
    dict_images = {}
    for image in images:
        key = hash_image(image)
        dict_images[key] = image
    images = dict_images.copy()
    del dict_images
    hashes = sorted(images.keys())

    energy_data = {}
    # Reading energy script
    try:
        fp = paropen(energyscript, 'rb')
        data = json.load(fp)
    except IOError:
        pass
    else:
        for hash in data.keys():
            energy_data[hash] = data[hash]

    # calculating energies for images if json is not found
    if len(energy_data.keys()) == 0:
        for hash in hashes:
            act_energy = \
                images[hash].get_potential_energy(apply_constraint=False)
            amp_energy = calc.get_potential_energy(images[hash])
            energy_data[hash] = [act_energy, amp_energy]
        # saving energy script
        try:
            json.dump(energy_data, energyscript)
            energyscript.flush()
            return
        except AttributeError:
            with paropen(energyscript, 'wb') as outfile:
                json.dump(energy_data, outfile)

    min_act_energy = min([energy_data[hash][0] for hash in hashes])
    max_act_energy = max([energy_data[hash][0] for hash in hashes])

    if plot_forces is None:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    # energy plot
    for hash in hashes:
        ax.plot(energy_data[hash][0], energy_data[hash][1], color)
    # draw line
    ax.plot([min_act_energy, max_act_energy],
            [min_act_energy, max_act_energy],
            'r-',
            lw=0.3,)
    ax.set_xlabel('actual energy, eV')
    ax.set_ylabel('AMP energy')
    ax.set_title('Energies')

    if plot_forces:

        force_data = {}
        # Reading force script
        try:
            fp = paropen(forcescript, 'rb')
            data = json.load(fp)
        except IOError:
            pass
        else:
            for hash in data.keys():
                force_data[hash] = {}
                for index in data[hash].keys():
                    force_data[hash][int(index)] = {}
                    for k in data[hash][index].keys():
                        force_data[hash][int(index)][int(k)] = \
                            data[hash][index][k]

        # calculating forces for images if json is not found
        if len(force_data.keys()) == 0:
            for hash in hashes:
                force_data[hash] = {}
                act_force = images[hash].get_forces(apply_constraint=False)
                images[hash].set_calculator(calc)
                amp_force = calc.get_forces(images[hash])
                for atom in images[hash]:
                    index = atom.index
                    force_data[hash][index] = {}
                    for k in range(3):
                        force_data[hash][index][k] = \
                            [act_force[index][k], amp_force[index][k]]

            # saving force script
            try:
                json.dump(force_data, forcescript)
                forcescript.flush()
                return
            except AttributeError:
                with paropen(forcescript, 'wb') as outfile:
                    json.dump(force_data, outfile)

        min_act_force = min([force_data[hash][atom.index][k][0]
                             for hash in hashes
                             for atom in images[hash]
                             for k in range(3)])
        max_act_force = max([force_data[hash][atom.index][k][0]
                             for hash in hashes
                             for atom in images[hash]
                             for k in range(3)])

        #######################################################################
        # force plot
        ax = fig.add_subplot(212)

        for hash in hashes:
            for atom in images[hash]:
                index = atom.index
                for k in range(3):
                    ax.plot(force_data[hash][index][k][0],
                            force_data[hash][index][k][1],
                            color)
        # draw line
        ax.plot([min_act_force, max_act_force],
                [min_act_force, max_act_force],
                'r-',
                lw=0.3,)

        ax.set_xlabel('actual force, eV/Ang')
        ax.set_ylabel('AMP force')
        ax.set_title('Forces')

        #######################################################################

    fig.savefig(plotfile)

###############################################################################


def plot_error(load,
               images,
               plot_forces=True,
               plotfile='errorplot.pdf',
               color='b.',
               overwrite=False):
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
    :param overwrite: If a plot or an script containing values found overwrite
                      it.
    :type overwrite: bool
    """
    base_filename = os.path.splitext(plotfile)[0]
    energyscript = os.path.join('energy-' + base_filename + '.json')

    if (not overwrite) and os.path.exists(plotfile):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile)
    if (not overwrite) and os.path.exists(energyscript):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % energyscript)

    if plot_forces is not None:
        forcescript = os.path.join('force-' + base_filename + '.json')
        if (not overwrite) and os.path.exists(forcescript):
            raise IOError('File exists: %s.\nIf you want to overwrite,'
                          ' set overwrite=True or manually delete.'
                          % forcescript)

    calc = AMP(load=load)

    if isinstance(images, str):
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = io.Trajectory(images, 'r')
        elif extension == '.db':
            images = io.read(images)

    # Images is converted to dictionary form; key is hash of image.
    dict_images = {}
    for image in images:
        key = hash_image(image)
        dict_images[key] = image
    images = dict_images.copy()
    del dict_images
    hashes = sorted(images.keys())

    energy_data = {}
    # Reading energy script
    try:
        fp = paropen(energyscript, 'rb')
        data = json.load(fp)
    except IOError:
        pass
    else:
        for hash in data.keys():
            energy_data[hash] = data[hash]

    # calculating errors for images if json is not found
    if len(energy_data.keys()) == 0:
        for hash in hashes:
            act_energy = \
                images[hash].get_potential_energy(apply_constraint=False)
            amp_energy = calc.get_potential_energy(images[hash])
            energy_error = abs(amp_energy - act_energy) / len(images[hash])
            act_energy_per_atom = act_energy / len(images[hash])
            energy_data[hash] = [act_energy_per_atom, energy_error]
        # saving energy script
        try:
            json.dump(energy_data, energyscript)
            energyscript.flush()
            return
        except AttributeError:
            with paropen(energyscript, 'wb') as outfile:
                json.dump(energy_data, outfile)

    # calculating energy per atom rmse
    energy_square_error = 0.
    for hash in hashes:
        energy_square_error += energy_data[hash][1] ** 2.

    energy_per_atom_rmse = np.sqrt(energy_square_error / len(hashes))

    min_act_energy = min([energy_data[hash][0] for hash in hashes])
    max_act_energy = max([energy_data[hash][0] for hash in hashes])

    if plot_forces is None:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    # energy plot
    for hash in hashes:
        ax.plot(energy_data[hash][0], energy_data[hash][1], color)
    # draw horizontal line for rmse
    ax.plot([min_act_energy, max_act_energy],
            [energy_per_atom_rmse, energy_per_atom_rmse], 'r-', lw=1,)
    ax.text(max_act_energy,
            energy_per_atom_rmse,
            'rmse = %6.5f' % energy_per_atom_rmse,
            ha='right',
            va='bottom',
            color='red')
    ax.set_xlabel('actual energy per atom, eV per atom')
    ax.set_ylabel('|actual energy - AMP energy| / number of atoms')
    ax.set_title('Energies')

    if plot_forces:

        force_data = {}
        # Reading force script
        try:
            fp = paropen(forcescript, 'rb')
            data = json.load(fp)
        except IOError:
            pass
        else:
            for hash in data.keys():
                force_data[hash] = {}
                for index in data[hash].keys():
                    force_data[hash][int(index)] = {}
                    for k in data[hash][index].keys():
                        force_data[hash][int(index)][int(k)] = \
                            data[hash][index][k]

        # calculating errors for images if json is not found
        if len(force_data.keys()) == 0:
            for hash in hashes:
                force_data[hash] = {}
                act_force = images[hash].get_forces(apply_constraint=False)
                images[hash].set_calculator(calc)
                amp_force = calc.get_forces(images[hash])
                for atom in images[hash]:
                    index = atom.index
                    force_data[hash][index] = {}
                    for k in range(3):
                        force_data[hash][index][k] = \
                            [act_force[index][k],
                             abs(amp_force[index][k] - act_force[index][k])]

            # saving force script
            try:
                json.dump(force_data, forcescript)
                forcescript.flush()
                return
            except AttributeError:
                with paropen(forcescript, 'wb') as outfile:
                    json.dump(force_data, outfile)

        # calculating force rmse
        force_square_error = 0.
        for hash in hashes:
            atoms = images[hash]
            for atom in atoms:
                index = atom.index
                for k in range(3):
                    force_square_error += \
                        ((1.0 / 3.0) * force_data[hash][index][k][1] ** 2.) / \
                        len(atoms)

        force_rmse = np.sqrt(force_square_error / len(hashes))

        min_act_force = min([force_data[hash][atom.index][k][0]
                             for hash in hashes
                             for atom in images[hash]
                             for k in range(3)])
        max_act_force = max([force_data[hash][atom.index][k][0]
                             for hash in hashes
                             for atom in images[hash]
                             for k in range(3)])

        #######################################################################
        # force plot
        ax = fig.add_subplot(212)

        for hash in hashes:
            for atom in images[hash]:
                index = atom.index
                for k in range(3):
                    ax.plot(force_data[hash][index][k][0],
                            force_data[hash][index][k][1],
                            color)
        # draw horizontal line for rmse
        ax.plot([min_act_force, max_act_force],
                [force_rmse, force_rmse],
                'r-',
                lw=1,)
        ax.text(max_act_force,
                force_rmse,
                'rmse = %5.4f' % force_rmse,
                ha='right',
                va='bottom',
                color='red',)

        ax.set_xlabel('actual force, eV/Ang')
        ax.set_ylabel('|actual force - AMP force|')
        ax.set_title('Forces')

        #######################################################################

    fig.savefig(plotfile)

###############################################################################


if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 2

    logfile = sys.argv[-1]

    plot_convergence(logfile=logfile)
