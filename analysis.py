import numpy as np
import os
from ase import io
from ase.parallel import paropen
import json
from matplotlib import rcParams
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
rcParams.update({'figure.autolayout': True})

###############################################################################


def read_trainlog(logfile):
    """
    Reads the log file from the training process, returning the relevant
    parameters.

    :param logfile: Name or path to the log file.
    :type logfile: str
    """
    data = {}

    with open(logfile, 'r') as f:
        lines = f.read().splitlines()

    # Get number of images.
    for line in lines:
        if 'unique images after hashing.' in line:
            no_images = float(line.split()[0])
            break
    data['no_images'] = no_images

    # Find where simulated annealing starts.
    annealingstartline = None
    for index, line in enumerate(lines):
        if 'Simulated annealing started.' in line:
            annealingstartline = index
            data['annealing'] = {}
            break
    annealingexitline = None
    for index, line in enumerate(lines):
        if 'Simulated annealing exited.' in line:
            annealingexitline = index
            break
    if annealingstartline is not None:
        # Extract data.
        steps, temps, costfxns, acceptratios = [], [], [], []
        for line in lines[annealingstartline + 3: annealingexitline]:
            step, temp, costfxn, _, _, _, _, accept = line.split()
            acceptratio = int(accept.split('(')[-1].split(')')[0])
            steps.append(int(step))
            temps.append(float(temp))
            costfxns.append(float(costfxn))
            acceptratios.append(float(acceptratio))
        data['annealing']['steps'] = steps
        data['annealing']['temps'] = temps
        data['annealing']['costfxns'] = costfxns
        data['annealing']['acceptratios'] = acceptratios

    # Find where convergence data starts.
    startline = None
    for index, line in enumerate(lines):
        if 'Starting optimization of cost function...' in line:
            startline = index
            data['convergence'] = {}
            d = data['convergence']
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
            d['energygoal'] = energygoal
            d['forcegoal'] = forcegoal
            d['forcecoefficient'] = forcecoefficient
            break

    if forcegoal:
        E = energygoal**2 * no_images
        F = forcegoal**2 * no_images
        costfxngoal = E + forcecoefficient * F
    else:
        costfxngoal = energygoal**2 * no_images
    d['costfxngoal'] = costfxngoal

    # Extract data.
    steps, es, fs, costfxns = [], [], [], []
    costfxnEs, costfxnFs = [], []
    index = startline
    while index < len(lines):
        line = lines[index]
        if 'Saving checkpoint' in line:
            index += 1
            continue
        elif 'convergence!' in line:
            index += 1
            continue
        elif 'unconverged!' in line:
            index += 1
            continue
        elif 'optimization completed successfully.' in line:
            break
        elif 'could not find parameters for the' in line:
            break
        elif 'Introducing new hidden-layer nodes...' in line:
            # Find where optimization resumes.
            print('!!!')
            nindex = index + 1
            found = False
            while not found:
                line = lines[nindex]
                print(line)
                print(line.split()[0])
                if line.split()[0] == '0':
                    print('found it')
                    index = nindex
                    found = True
                nindex += 1
            continue
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
        index += 1
    d['steps'] = steps
    d['es'] = es
    d['fs'] = fs
    d['costfxns'] = costfxns
    d['costfxnEs'] = costfxnEs
    d['costfxnFs'] = costfxnFs

    return data

###############################################################################


def plot_convergence(logfile, plotfile='convergence.pdf', returnfig=False):
    """
    Makes a plot of the convergence of the cost function and its energy
    and force components.

    :param logfile: Name or path to the log file.
    :type logfile: str
    :param plotfile: Name or path to the plot file. If "None" no output
                     written.
    :type plotfile: str
    :param returnfig: Whether to return a reference to the figure.
    :type returnfig: boolean
    """

    data = read_trainlog(logfile)

    if 'annealing' in data:
        # Make plots
        d = data['annealing']
        fig0 = pyplot.figure(figsize=(8.5, 11))
        ax = fig0.add_subplot(311)
        ax.set_title('Simulated annealing: trace of temperature')
        ax.plot(d['temps'], '-')
        ax.set_xlabel('step')
        ax.set_ylabel('temperature')

        ax = fig0.add_subplot(312)
        ax.set_title('trace of loss function')
        ax.plot(d['costfxns'], '-')
        ax.set_yscale('log')
        ax.set_xlabel('step')
        ax.set_ylabel('loss function')

        ax = fig0.add_subplot(313)
        ax.set_title('trace of acceptance rate')
        ax.plot(d['acceptratios'], '-')
        ax.set_xlabel('step')
        ax.set_ylabel('acceptance rate')

    # Find if multiple runs contained in data set.
    d = data['convergence']
    steps = range(len(d['steps']))
    breaks = []
    for index, step in enumerate(d['steps'][1:]):
        if step < d['steps'][index]:
            breaks.append(index)

    # Make plots.
    fig = pyplot.figure(figsize=(6., 8.))
    # Margins, vertical gap, and top-to-bottom ratio of figure.
    lm, rm, bm, tm, vg, tb = 0.12, 0.05, 0.08, 0.03, 0.08, 4.
    bottomaxheight = (1. - bm - tm - vg) / (tb + 1.)

    ax = fig.add_axes((lm, bm + bottomaxheight + vg,
                       1. - lm - rm, tb * bottomaxheight))
    ax.semilogy(steps, d['es'], 'b', lw=2, label='energy rmse')
    if d['forcegoal']:
        ax.semilogy(steps, d['fs'], 'g', lw=2, label='force rmse')
    ax.semilogy(steps, d['costfxns'], color='0.5', lw=2,
                label='loss function')
    # Targets.
    ax.semilogy([steps[0], steps[-1]], [d['energygoal']] * 2,
                color='b', linestyle=':')
    if d['forcegoal']:
        ax.semilogy([steps[0], steps[-1]], [d['forcegoal']] * 2,
                    color='g', linestyle=':')
    ax.semilogy([steps[0], steps[-1]], [d['costfxngoal']] * 2,
                color='0.5', linestyle=':')
    ax.set_ylabel('error')
    ax.set_xlabel('BFGS step')
    ax.legend(loc='best')
    if len(breaks) > 0:
        ylim = ax.get_ylim()
        for b in breaks:
            ax.plot([b] * 2, ylim, '--k')

    if d['forcegoal']:
        ax = fig.add_axes((lm, bm, 1. - lm - rm, bottomaxheight))
        ax.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                        color='blue')
        ax.fill_between(x=np.array(steps), y1=d['costfxnEs'],
                        y2=np.array(d['costfxnEs']) +
                        np.array(d['costfxnFs']),
                        color='green')
        ax.set_ylabel('loss function component')
        ax.set_xlabel('BFGS step')
        ax.set_ylim(0, 1)

    if plotfile is not None:
        with PdfPages(plotfile) as pdf:
            if 'annealing' in data:
                pdf.savefig(fig0)
            pdf.savefig(fig)

    if returnfig:
        figs = []
        if 'annealing' in data:
            figs.append(fig0)
        figs.append(fig)
        return figs

    # Close the figures (pyplot is sloppy).
    if 'annealing' in data:
        pyplot.close(fig0)
    pyplot.close(fig)

###############################################################################


def plot_parity(load,
                images,
                plot_forces=True,
                plotfile='parityplot.pdf',
                color='b.',
                overwrite=False):
    """
    Makes a parity plot of Amp energies and forces versus real energies and
    forces.

    :param load: Path for loading an existing parameters of Amp calculator.
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

    if plot_forces is not None:
        forcescript = os.path.join('force-' + base_filename + '.json')

    from amp import Amp
    from amp.utilities import hash_image
    from matplotlib import rc
    # activate latex text rendering
    rc('text', usetex=True)

    calc = Amp(load=load)

    if isinstance(images, str):
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = io.Trajectory(images, 'r')
        elif extension == '.db':
            images = io.read(images)

    # Images is converted to dictionary form; key is hash of image.
    dict_images = {}
    for image in images:
        hash = hash_image(image)
        dict_images[hash] = image
    images = dict_images.copy()
    del dict_images
    hashs = sorted(images.keys())
    no_of_images = len(hashs)

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
        count = 0
        while count < no_of_images:
            hash = hashs[count]
            atoms = images[hash]
            act_energy = atoms.get_potential_energy(apply_constraint=False)
            amp_energy = calc.get_potential_energy(atoms)
            energy_data[hash] = [act_energy, amp_energy]
            count += 1
        # saving energy script
        try:
            json.dump(energy_data, energyscript)
            energyscript.flush()
            return
        except AttributeError:
            with paropen(energyscript, 'wb') as outfile:
                json.dump(energy_data, outfile)
    del hash

    min_act_energy = min([energy_data[hash][0] for hash in hashs])
    max_act_energy = max([energy_data[hash][0] for hash in hashs])

    if plot_forces is None:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    # energy plot
    count = 0
    while count < no_of_images:
        hash = hashs[count]
        ax.plot(energy_data[hash][0], energy_data[hash][1], color)
        count += 1
    # draw line
    ax.plot([min_act_energy, max_act_energy],
            [min_act_energy, max_act_energy],
            'r-',
            lw=0.3,)
    ax.set_xlabel(r"\textit{ab initio} energy, eV")
    ax.set_ylabel(r"\textit{Amp} energy, eV")
    ax.set_title(r"Energies")

    if plot_forces:

        force_data = {}
        # Reading force script
        try:
            fp = paropen(forcescript, 'rb')
            data = json.load(fp)
        except IOError:
            pass
        else:
            hashs = data.keys()
            no_of_images = len(hashs)
            count0 = 0
            while count0 < no_of_images:
                hash = hashs[count0]
                force_data[hash] = {}
                indices = data[hash].keys()
                len_of_indices = len(indices)
                count1 = 0
                while count1 < len_of_indices:
                    index = indices[count1]
                    force_data[hash][int(index)] = {}
                    ks = data[hash][index].keys()
                    len_of_ks = len(ks)
                    count2 = 0
                    while count2 < len_of_ks:
                        k = ks[count2]
                        force_data[hash][int(index)][int(k)] = \
                            data[hash][index][k]
                        count2 += 1
                    count1 += 1
                count0 += 1

        # calculating forces for images if json is not found
        if len(force_data.keys()) == 0:
            count = 0
            while count < no_of_images:
                hash = hashs[count]
                atoms = images[hash]
                no_of_atoms = len(atoms)
                force_data[hash] = {}
                act_force = atoms.get_forces(apply_constraint=False)
                atoms.set_calculator(calc)
                amp_force = calc.get_forces(atoms)
                index = 0
                while index < no_of_atoms:
                    force_data[hash][index] = {}
                    k = 0
                    while k < 3:
                        force_data[hash][index][k] = \
                            [act_force[index][k], amp_force[index][k]]
                        k += 1
                    index += 1
                count += 1
            del hash, k, index

            # saving force script
            try:
                json.dump(force_data, forcescript)
                forcescript.flush()
                return
            except AttributeError:
                with paropen(forcescript, 'wb') as outfile:
                    json.dump(force_data, outfile)

        min_act_force = min([force_data[hash][index][k][0]
                             for hash in hashs
                             for index in range(len(images[hash]))
                             for k in range(3)])
        max_act_force = max([force_data[hash][index][k][0]
                             for hash in hashs
                             for index in range(len(images[hash]))
                             for k in range(3)])

        ##############################################################
        # force plot
        ax = fig.add_subplot(212)

        count = 0
        while count < no_of_images:
            hash = hashs[count]
            atoms = images[hash]
            no_of_atoms = len(atoms)
            index = 0
            while index < no_of_atoms:
                k = 0
                while k < 3:
                    ax.plot(force_data[hash][index][k][0],
                            force_data[hash][index][k][1],
                            color)
                    k += 1
                index += 1
            count += 1
        # draw line
        ax.plot([min_act_force, max_act_force],
                [min_act_force, max_act_force],
                'r-',
                lw=0.3,)

        ax.set_xlabel(r"\textit{ab initio} force, eV/\AA")
        ax.set_ylabel(r"\textit{Amp} force, eV/\AA")
        ax.set_title(r"Forces")

        ##############################################################

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

    :param load: Path for loading an existing parameters of Amp calculator.
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

    if plot_forces is not None:
        forcescript = os.path.join('force-' + base_filename + '.json')

    from amp import Amp
    from amp.utilities import hash_image
    from matplotlib import rc
    # activate latex text rendering
    rc('text', usetex=True)

    calc = Amp(load=load)

    if isinstance(images, str):
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = io.Trajectory(images, 'r')
        elif extension == '.db':
            images = io.read(images)

    # Images is converted to dictionary form; key is hash of image.
    dict_images = {}
    for image in images:
        hash = hash_image(image)
        dict_images[hash] = image
    images = dict_images.copy()
    del dict_images
    hashs = sorted(images.keys())
    no_of_images = len(hashs)

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
        count = 0
        while count < no_of_images:
            hash = hashs[count]
            atoms = images[hash]
            no_of_atoms = len(atoms)
            act_energy = atoms.get_potential_energy(apply_constraint=False)
            amp_energy = calc.get_potential_energy(atoms)
            energy_error = abs(amp_energy - act_energy) / no_of_atoms
            act_energy_per_atom = act_energy / no_of_atoms
            energy_data[hash] = [act_energy_per_atom, energy_error]
            count += 1
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
    count = 0
    while count < no_of_images:
        hash = hashs[count]
        energy_square_error += energy_data[hash][1] ** 2.
        count += 1
    del hash

    energy_per_atom_rmse = np.sqrt(energy_square_error / no_of_images)

    min_act_energy = min([energy_data[hash][0] for hash in hashs])
    max_act_energy = max([energy_data[hash][0] for hash in hashs])

    if plot_forces is None:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    # energy plot
    count = 0
    while count < no_of_images:
        hash = hashs[count]
        ax.plot(energy_data[hash][0], energy_data[hash][1], color)
        count += 1
    # draw horizontal line for rmse
    ax.plot([min_act_energy, max_act_energy],
            [energy_per_atom_rmse, energy_per_atom_rmse],
            color='black', linestyle='dashed', lw=1,)
    ax.text(max_act_energy,
            energy_per_atom_rmse,
            'energy rmse = %6.5f' % energy_per_atom_rmse,
            ha='right',
            va='bottom',
            color='black')

    ax.set_xlabel(r"\textit{ab initio} energy (eV) per atom")
    ax.set_ylabel(r"$|$\textit{ab initio} energy - \textit{Amp} energy$|$ / number of atoms")
    ax.set_title("Energies")

    if plot_forces:

        force_data = {}
        # Reading force script
        try:
            fp = paropen(forcescript, 'rb')
            data = json.load(fp)
        except IOError:
            pass
        else:
            hashs = data.keys()
            no_of_images = len(hashs)
            count0 = 0
            while count0 < no_of_images:
                hash = hashs[count0]
                force_data[hash] = {}
                indices = data[hash].keys()
                len_of_indices = len(indices)
                count1 = 0
                while count1 < len_of_indices:
                    index = indices[count1]
                    force_data[hash][int(index)] = {}
                    ks = data[hash][index].keys()
                    len_of_ks = len(ks)
                    count2 = 0
                    while count2 < len_of_ks:
                        k = ks[count2]
                        force_data[hash][int(index)][int(k)] = \
                            data[hash][index][k]
                        count2 += 1
                    count1 += 1
                count0 += 1

        # calculating errors for images if json is not found
        if len(force_data.keys()) == 0:
            count = 0
            while count < no_of_images:
                hash = hashs[count]
                atoms = images[hash]
                no_of_atoms = len(atoms)
                force_data[hash] = {}
                act_force = atoms.get_forces(apply_constraint=False)
                atoms.set_calculator(calc)
                amp_force = calc.get_forces(atoms)
                index = 0
                while index < no_of_atoms:
                    force_data[hash][index] = {}
                    k = 0
                    while k < 3:
                        force_data[hash][index][k] = \
                            [act_force[index][k],
                             abs(amp_force[index][k] - act_force[index][k])]
                        k += 1
                    index += 1
                count += 1

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
        count = 0
        while count < no_of_images:
            hash = hashs[count]
            atoms = images[hash]
            no_of_atoms = len(atoms)
            index = 0
            while index < no_of_atoms:
                k = 0
                while k < 3:
                    force_square_error += \
                        ((1.0 / 3.0) * force_data[hash][index][k][1] ** 2.) / \
                        no_of_atoms
                    k += 1
                index += 1
            count += 1
        del hash, index, k

        force_rmse = np.sqrt(force_square_error / no_of_images)

        min_act_force = min([force_data[hash][index][k][0]
                             for hash in hashs
                             for index in range(len(images[hash]))
                             for k in range(3)])
        max_act_force = max([force_data[hash][index][k][0]
                             for hash in hashs
                             for index in range(len(images[hash]))
                             for k in range(3)])

        ##############################################################
        # force plot
        ax = fig.add_subplot(212)

        count = 0
        while count < no_of_images:
            hash = hashs[count]
            atoms = images[hash]
            no_of_atoms = len(atoms)
            index = 0
            while index < no_of_atoms:
                k = 0
                while k < 3:
                    ax.plot(force_data[hash][index][k][0],
                            force_data[hash][index][k][1],
                            color)
                    k += 1
                index += 1
            count += 1
        # draw horizontal line for rmse
        ax.plot([min_act_force, max_act_force],
                [force_rmse, force_rmse],
                color='black',
                linestyle='dashed',
                lw=1,)
        ax.text(max_act_force,
                force_rmse,
                'force rmse = %5.4f' % force_rmse,
                ha='right',
                va='bottom',
                color='black',)

        ax.set_xlabel(r"\textit{ab initio} force, eV/\AA")
        ax.set_ylabel(r"$|$\textit{ab initio} force - \textit{Amp} force$|$")
        ax.set_title(r"Forces")

        ##############################################################

    fig.savefig(plotfile)

###############################################################################


if __name__ == '__main__':
    import sys
    assert len(sys.argv) == 2

    logfile = sys.argv[-1]

    plot_convergence(logfile=logfile)
