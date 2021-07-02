from argparse import (ArgumentParser, _ArgumentGroup, Namespace,
                      ArgumentDefaultsHelpFormatter, Action)
import json
import os
import pathlib
from typing import (Any, Collection, Dict, List,
                    Sequence, Tuple, Union, Optional)
import warnings

import numpy as np
from pint import UndefinedUnitError
import seekpath

from euphonic import (Crystal, DebyeWaller, ForceConstants, QpointFrequencies,
                      QpointPhononModes, Spectrum1D, Spectrum1DCollection,
                      Quantity, ureg)
import euphonic.util
Unit = ureg.Unit


def force_constants_from_file(filename: Union[str, os.PathLike]
                              ) -> ForceConstants:
    """
    Load force constants data from file

    Parameters
    ----------
    filename
        Data file

    Returns
    -------
    ForceConstants
    """
    path = pathlib.Path(filename)
    if path.suffix in ['.hdf5', '.yaml']:
        phonopy_kwargs = {}
        phonopy_kwargs['path'] = path.parent
        if (path.parent / 'BORN').is_file():
            phonopy_kwargs['born_name'] = 'BORN'
        # Set summary_name and fc_name depending on input file
        if path.suffix == '.hdf5':
            if (path.parent / 'phonopy.yaml').is_file():
                phonopy_kwargs['summary_name'] = 'phonopy.yaml'
                phonopy_kwargs['fc_name'] = path.name
            else:
                raise ValueError("Phonopy force_constants.hdf5 file "
                                 "must be accompanied by phonopy.yaml")
        elif path.suffix == '.yaml':
            phonopy_kwargs['summary_name'] = path.name
            # Assume this is a (renamed?) phonopy.yaml file
            if (path.parent / 'force_constants.hdf5').is_file():
                phonopy_kwargs['fc_name'] = 'force_constants.hdf5'
            else:
                phonopy_kwargs['fc_name'] = 'FORCE_CONSTANTS'
        return ForceConstants.from_phonopy(**phonopy_kwargs)
    elif path.suffix in ('.castep_bin', '.check'):
        return ForceConstants.from_castep(filename)
    elif path.suffix == '.json':
        return ForceConstants.from_json_file(filename)
    else:
        raise ValueError("File not recognised. Filename should be "
                         "*.yaml or force_constants.hdf5 (phonopy), "
                         "*.castep_bin or *.check "
                         "(castep) or *.json (JSON from Euphonic).")


def modes_from_file(filename: Union[str, os.PathLike]
                    ) -> QpointPhononModes:
    """
    Load phonon mode data from file

    Parameters
    ----------
    filename
        Data file

    Returns
    -------
    QpointPhononmodes
    """
    path = pathlib.Path(filename)
    if path.suffix == '.phonon':
        return QpointPhononModes.from_castep(path)
    elif path.suffix == '.json':
        return QpointPhononModes.from_json_file(path)
    elif path.suffix in ('.yaml', '.hdf5'):
        return QpointPhononModes.from_phonopy(path=path.parent,
                                              phonon_name=path.name)
    else:
        raise ValueError("File not recognised. Should have extension "
                         ".yaml or .hdf5 (phonopy), "
                         ".phonon (castep) or .json (JSON from Euphonic).")


def _load_json(filename: Union[str, os.PathLike]
               ) -> Union[QpointPhononModes, ForceConstants]:
    with open(filename, 'r') as f:
        data = json.load(f)

    if 'force_constants' in data:
        return force_constants_from_file(filename)
    elif 'eigenvectors' in data:
        return modes_from_file(filename)
    else:
        raise ValueError("Could not identify Euphonic data in JSON file.")


def load_data_from_file(filename: Union[str, os.PathLike]
                        ) -> Union[QpointPhononModes, ForceConstants]:
    """
    Load phonon mode or force constants data from file

    Parameters
    ----------
    filename : str
        The file with a path

    Returns
    -------
    QpointPhononmodes or ForceConstants
    """
    qpoint_phonon_modes_suffixes = ('.phonon')
    force_constants_suffixes = ('.castep_bin', '.check')

    path = pathlib.Path(filename)
    if path.suffix in qpoint_phonon_modes_suffixes:
        return modes_from_file(path)
    elif path.suffix in force_constants_suffixes:
        return force_constants_from_file(path)
    elif path.suffix == '.json':
        return _load_json(path)
    elif path.suffix in ('.hdf5', '.yaml'):
        try:
            return modes_from_file(path)
        except KeyError:
            return force_constants_from_file(path)
    else:
        raise ValueError(
            "File format was not recognised. Force constant data for import "
            f"should have extension from {force_constants_suffixes},"
            " phonon mode data for import should have extension "
            f"'{qpoint_phonon_modes_suffixes[0]}',"
            " data from phonpy should have extension .yaml or .hdf5,"
            " data from Euphonic should have extension '.json'.")


def get_args(parser: ArgumentParser, params: List[str] = None):
    """
    Get the arguments from the parser. params should only be none when
    running from command line.

    Parameters
    ----------
    parser : ArgumentParser
        The parser to get the arguments from
    params : List[str], optional
        The parameters to get arguments from, if None,
         then parse_args gets them from command line args

    Returns
    -------
    Arguments object for use e.g. args.unit
    """
    if params is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(params)
    return args


def matplotlib_save_or_show(save_filename: str = None):
    """
    Save or show the current matplotlib plot.
    Show if save_filename is not None which by default it is.

    Parameters
    ----------
    save_filename : str, optional
        The file to save the plot in
    """
    import matplotlib.pyplot as plt
    if save_filename is not None:
        plt.savefig(save_filename)
        print(f'Saved plot to {os.path.realpath(save_filename)}')
    else:
        plt.show()


def _get_q_distance(length_unit_string: str, q_distance: float) -> Quantity:
    """
    Parse user arguments to obtain reciprocal-length spacing Quantity
    """
    try:
        length_units = ureg(length_unit_string)
    except UndefinedUnitError:
        raise ValueError("Length unit not known. Euphonic uses Pint for units."
                         " Try 'angstrom' or 'bohr'. Metric prefixes "
                         "are also allowed, e.g 'nm'.")
    recip_length_units = 1 / length_units
    return q_distance * recip_length_units


def _get_energy_bins(
        modes: QpointPhononModes,
        n_ebins: int, emin: Optional[float] = None,
        emax: Optional[float] = None,
        headroom: float = 1.05) -> Quantity:
    """
    Gets recommeded energy bins, in same units as modes.frequencies.
    emin and emax are assumed to be in the same units as
    modes.frequencies, if not provided the min/max values of
    modes.frequencies are used to find the bin limits
    """
    if emin is None:
        # Subtract small amount from min frequency - otherwise due to unit
        # conversions binning of this frequency can vary with different
        # architectures/lib versions, making it difficult to test
        emin_room = 1e-5*ureg('meV').to(modes.frequencies.units).magnitude
        emin = min(np.min(modes.frequencies.magnitude - emin_room), 0.)
    if emax is None:
        emax = np.max(modes.frequencies.magnitude) * headroom
    if emin >= emax:
        raise ValueError("Maximum energy should be greater than minimum. "
                         "Check --e-min and --e-max arguments.")
    ebins = np.linspace(emin, emax, n_ebins) * modes.frequencies.units
    return ebins


def _get_tick_labels(bandpath: dict) -> List[Tuple[int, str]]:
    """Convert x-axis labels from seekpath format to euphonic format

    i.e.::

        ['L', '', '', 'X', '', 'GAMMA']   -->

        [(0, 'L'), (3, 'X'), (5, '$\\Gamma$')]
    """

    label_indices = np.where(bandpath["explicit_kpoints_labels"])[0]
    labels = [bandpath["explicit_kpoints_labels"][i] for i in label_indices]

    for i, label in enumerate(labels):
        if label == 'GAMMA':
            labels[i] = r'$\Gamma$'

    return list(zip(label_indices, labels))


def _get_break_points(bandpath: dict) -> List[int]:
    """Get information about band path labels and break points

    Parameters
    ----------
    bandpath
        Bandpath dictionary from Seekpath

    Returns
    -------
    list[int]

        Indices at which the spectrum should be split into subplots

    """
    # Find break points between continuous spectra: wherever there are two
    # adjacent labels
    labels = np.array(bandpath["explicit_kpoints_labels"])

    special_point_bools = np.fromiter(
        map(bool, labels), dtype=bool)

    # [T F F T T F T] -> [F F T T F T] AND [T F F T T F] = [F F F T F F] -> 3,
    adjacent_non_empty_labels = np.logical_and(special_point_bools[:-1],
                                               special_point_bools[1:])

    adjacent_different_labels = (labels[:-1] != labels[1:])

    break_points = np.where(np.logical_and(adjacent_non_empty_labels,
                                           adjacent_different_labels))[0]
    return (break_points + 1).tolist()


def _insert_gamma(bandpath: dict) -> None:
    """Modify seekpath.get_explicit_k_path() results; duplicate Gamma

    This enables LO-TO splitting to be included
    """
    import numpy as np
    gamma_indices = np.where(np.array(bandpath['explicit_kpoints_labels'][1:-1]
                                      ) == 'GAMMA')[0] + 1

    rel_kpts = bandpath['explicit_kpoints_rel'].tolist()
    labels = bandpath['explicit_kpoints_labels']
    for i in reversed(gamma_indices.tolist()):
        rel_kpts.insert(i, [0., 0., 0.])
        labels.insert(i, 'GAMMA')

    bandpath['explicit_kpoints_rel'] = np.array(rel_kpts)
    bandpath['explicit_kpoints_labels'] = labels

    # These unused properties have been invalidated: safer
    # to leave None than incorrect values
    bandpath['explicit_kpoints_abs'] = None
    bandpath['explicit_kpoints_linearcoord'] = None
    bandpath['explicit_segments'] = None


XTickLabels = List[Tuple[int, str]]
SplitArgs = Dict[str, Any]


def _bands_from_force_constants(data: ForceConstants,
                                q_distance: Quantity,
                                insert_gamma: bool = True,
                                frequencies_only: bool = False,
                                **calc_modes_kwargs
                                ) -> Tuple[Union[QpointPhononModes,
                                                 QpointFrequencies],
                                           XTickLabels, SplitArgs]:
    structure = data.crystal.to_spglib_cell()
    bandpath = seekpath.get_explicit_k_path(
        structure,
        reference_distance=q_distance.to('1 / angstrom').magnitude)

    if insert_gamma:
        _insert_gamma(bandpath)

    x_tick_labels = _get_tick_labels(bandpath)
    split_args = {'indices': _get_break_points(bandpath)}

    print(
        "Computing phonon modes: {n_modes} modes across {n_qpts} q-points"
        .format(n_modes=(data.crystal.n_atoms * 3),
                n_qpts=len(bandpath["explicit_kpoints_rel"])))
    qpts = bandpath["explicit_kpoints_rel"]

    if frequencies_only:
        modes = data.calculate_qpoint_frequencies(qpts,
                                                  reduce_qpts=False,
                                                  **calc_modes_kwargs)
    else:
        modes = data.calculate_qpoint_phonon_modes(qpts,
                                                   reduce_qpts=False,
                                                   **calc_modes_kwargs)
    return modes, x_tick_labels, split_args


def _grid_spec_from_args(crystal: Crystal,
                           grid: Optional[Sequence[int]] = None,
                           grid_spacing: Quantity = 0.1 * ureg('1/angstrom')
                           ) -> Sequence[int]:
    """Get Monkorst-Pack mesh divisions from user arguments"""
    if grid:
        grid_spec = grid
    else:
        grid_spec = crystal.get_mp_grid_spec(spacing=grid_spacing)
    return grid_spec


def _get_debye_waller(temperature: Quantity,
                      fc: ForceConstants,
                      grid: Optional[Sequence[int]] = None,
                      grid_spacing: Quantity = 0.1 * ureg('1/angstrom'),
                      **calc_modes_kwargs
                      ) -> DebyeWaller:
    """Generate Debye-Waller data from force constants and grid specification
    """
    mp_grid_spec = _grid_spec_from_args(fc.crystal, grid=grid,
                                        grid_spacing=grid_spacing)
    print("Calculating Debye-Waller factor on {} q-point grid"
          .format(' x '.join(map(str, mp_grid_spec))))
    dw_phonons = fc.calculate_qpoint_phonon_modes(
        euphonic.util.mp_grid(mp_grid_spec),
        **calc_modes_kwargs)
    return dw_phonons.calculate_debye_waller(temperature)


def _get_pdos_weighting(cl_arg_weighting: str) -> str:
    """
    Convert CL --weighting to weighting for calculate_pdos
    e.g. --weighting coherent-dos to weighting=coherent
    """
    if cl_arg_weighting == 'dos':
        pdos_weighting = None
    else:
        idx = cl_arg_weighting.rfind('-')
        if idx == -1:
            raise ValueError('Unexpected weighting {cl_arg_weighting}')
        pdos_weighting = cl_arg_weighting[:idx]
    return pdos_weighting


def _arrange_pdos_groups(pdos: Spectrum1DCollection,
                         cl_arg_pdos: Sequence[str]
                         ) -> Union[Spectrum1D, Spectrum1DCollection]:
    """
    Convert PDOS returned by calculate_pdos to PDOS/DOS
    wanted as CL output according to --pdos
    """
    dos = pdos.sum()
    if cl_arg_pdos is not None:
        # Only label total DOS if there are other lines on the plot
        dos.metadata['label'] = 'Total'
        pdos = pdos.group_by('species')
        for line_metadata in pdos.metadata['line_data']:
            line_metadata['label'] = line_metadata['species']
        if len(cl_arg_pdos) > 0:
            pdos = pdos.select(species=cl_arg_pdos)
            dos = pdos
        else:
            dos = Spectrum1DCollection.from_spectra([dos] + [*pdos])
    return dos


def _calc_modes_kwargs(args: Namespace) -> Dict[str, Any]:
    """Collect arguments that can be passed to calculate_qpoint_phonon_modes()
    """
    return dict(asr=args.asr, dipole_parameter=args.dipole_parameter,
                use_c=args.use_c, n_threads=args.n_threads)


def _get_cli_parser(features: Collection[str] = {}
                    ) -> Tuple[ArgumentParser,
                               Dict[str, _ArgumentGroup]]:
    """Instantiate an ArgumentParser with appropriate argument groups

    Args:
        sections:
            collection (e.g. set, list) of str for known argument groups.
            Known keys: read-fc, read-modes, ins-weighting, pdos-weighting,
            powder, mp-grid, plotting, ebins, adaptive-broadening, q-e,
            map, btol, dipole-parameter-optimisation

    Returns:
        Parser and a dict of parser subsections. This allows their help strings
        to be customised and specialised options to be added.

    """
    def deprecation_text(deprecated_arg: str, new_arg: str):
        return (f'--{deprecated_arg} is deprecated, '
                f'please use --{new_arg} instead')
    def deprecated_arg(recommended_arg: str):
        class DeprecatedArgAction(Action):
            def __call__(self, parser, args, values, option_string=None):
                # Need to filter to raise warnings from CL tools
                # Warnings not in __main__ are ignored by default
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'default', category=DeprecationWarning,
                        module=__name__)
                    warnings.warn(
                        deprecation_text(self.dest, recommended_arg),
                        DeprecationWarning)
                    setattr(args, recommended_arg, values)
        return DeprecatedArgAction

    _pdos_choices = ('coherent-dos', 'incoherent-dos',
                     'coherent-plus-incoherent-dos')
    _ins_choices = ('coherent',)

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    # Set up groups; these are only displayed if used, so simplest to
    # instantiate them all now and guarantee consistent order/presence
    # regardless of control logic
    section_defs = [('file', 'File I/O arguments'),
                    ('q', 'q-point sampling arguments'),
                    ('energy', 'energy/frequency arguments'),
                    ('interpolation',
                     'Force constants interpolation arguments'),
                    ('property', 'Property-calculation arguments'),
                    ('plotting', 'Plotting arguments'),
                    ('performance', 'Performance-related arguments')
                    ]

    sections = {label: parser.add_argument_group(doc)
                for label, doc in section_defs}

    if {'read-fc', 'read-modes'}.intersection(features):
        if {'read-fc', 'read-modes'}.issubset(features):
            filename_doc = (
                'Phonon data file. This should contain force constants or '
                'phonon mode data. Force constants formats: .yaml, '
                'force_constants.hdf5 (Phonopy); .castep_bin , .check '
                '(Castep); .json (Euphonic). Phonon mode data formats: '
                '{band,qpoints,mesh}.{hdf5,yaml} (Phonopy); '
                '.phonon (Castep); .json (Euphonic)')
        elif 'read-fc' in features:
            filename_doc = (
                'Phonon data file. This should contain force constants data. '
                'Accepted formats: .yaml, force_constants.hdf5 (Phonopy); '
                '.castep_bin, .check (Castep); .json (Euphonic).')
        else:
            raise ValueError('No band-data-only tools have been defined.')
        sections['file'].add_argument('filename', type=str, help=filename_doc)

    if 'read-fc' in features:
        sections['interpolation'].add_argument(
            '--asr', type=str, nargs='?', default=None, const='reciprocal',
            choices=('reciprocal', 'realspace'),
            help=('Apply an acoustic-sum-rule (ASR) correction to the '
                  'data: "realspace" applies the correction to the force '
                  'constant matrix in real space. "reciprocal" applies '
                  'the correction to the dynamical matrix at each q-point.'))
        if not 'dipole-parameter-optimisation' in features:
            sections['interpolation'].add_argument(
                '--dipole-parameter', type=float, default=1.0,
                dest='dipole_parameter',
                help=('Set the cutoff in real/reciprocal space for the dipole '
                      'Ewald sum; higher values use more reciprocal terms. If '
                      'tuned correctly this can result in performance '
                      'improvements. See euphonic-optimise-dipole-parameter '
                      'program for help on choosing a good DIPOLE_PARAMETER.'))

        use_c = sections['performance'].add_mutually_exclusive_group()
        use_c.add_argument(
            '--use-c', action='store_true', dest='use_c', default=None,
            help=('Force use of compiled C extension when computing '
                  'phonon frequencies/eigenvectors (or raise error).'))
        use_c.add_argument(
            '--disable-c', action='store_false', dest='use_c',
            help=('Do not attempt to use compiled C extension when computing '
                  'phonon frequencies/eigenvectors.'))
        sections['performance'].add_argument(
            '--n-threads', type=int, default=None, dest='n_threads',
            help=('Number of parallel processes for computing phonon modes. '
                  '(Only applies when using C extension.)'))

    pdos_desc = ('coherent neutron-weighted DOS, incoherent '
                 'neutron-weighted DOS or total (coherent + incoherent) '
                 'neutron-weighted DOS')
    ins_desc = ('coherent inelastic neutron scattering')
    if 'pdos-weighting' in features:
        # Currently do not support multiple PDOS for 2D plots
        if not 'q-e' in features:
            sections['property'].add_argument(
                '--pdos', type=str, action='store', nargs='*',
                help=('Plot PDOS. With --pdos, per-species PDOS will be plotted '
                      'alongside total DOS. A subset of species can also be '
                      'selected by adding more arguments e.g. --pdos Si O'))
        else:
            sections['property'].add_argument(
                '--pdos', type=str, action='store', nargs=1,
                help=('Plot PDOS for a specific species e.g. --pdos Si'))

    if {'pdos-weighting', 'ins-weighting'}.intersection(
            features):
        # We can plot plain DOS with all CL tools
        _weighting_choices = ('dos',)
        desc = 'DOS'
        if 'pdos-weighting' in features:
            _weighting_choices += _pdos_choices
            desc += (', coherent neutron-weighted DOS, incoherent '
                     'neutron-weighted DOS or total (coherent + incoherent) '
                     'neutron-weighted DOS')
        # We may have both 'pdos' and 'ins' so use separate if
        if 'ins-weighting' in features:
            _weighting_choices += _ins_choices
            desc += ', coherent inelastic neutron scattering'
            # --weights was deprecated before dos-weighting was added so
            # keep in this section
            sections['property'].add_argument(
                '--weights', default='dos', choices=_weighting_choices,
                action=deprecated_arg('weighting'),
                help=deprecation_text('weights', 'weighting'))
            sections['property'].add_argument(
                '--weighting', '-w', default='dos', choices=_weighting_choices,
                help=(f'Spectral weighting to plot: {desc}'))
        else:
            sections['property'].add_argument(
                '--weighting', '-w', default='dos', choices=_weighting_choices,
                help=f'Type of DOS to plot: {desc}')

    if 'ins-weighting' in features:
        sections['property'].add_argument(
            '--temperature', type=float, default=None,
            help=('Temperature in K; enable Debye-Waller factor calculation. '
                  '(Only applicable when --weighting=coherent).'))

    # 'ins-weighting' implies 'mp-grid' as well because mesh is needed for DW
    # factor
    if {'ins-weighting', 'mp-grid'}.intersection(features):
        grid_spec = sections['q'].add_mutually_exclusive_group()

        grid_spec.add_argument(
            '--grid', type=int, nargs=3, default=None,
            help=('Defines a Monkhorst-Pack grid.'))
        grid_spec.add_argument(
            '--grid-spacing', type=float, default=0.1, dest='grid_spacing',
            help=('q-point spacing of Monkhorst-Pack grid.'))

    if 'powder' in features:
        _sampling_choices = {'golden', 'sphere-projected-grid',
                             'spherical-polar-grid',
                             'spherical-polar-improved',
                             'random-sphere'}
        npts_group = sections['q'].add_mutually_exclusive_group()
        npts_group.add_argument('--npts', '-n', type=int, default=1000,
                                help=('Number of samples at each |q| sphere.'))
        npts_group.add_argument(
            '--npts-density', type=int, default=None, dest='npts_density',
            help=('NPTS specified as the number of points at surface of '
                  '1/LENGTH_UNIT-radius sphere; otherwise scaled to equivalent'
                  ' area density at sphere surface.'))

        sections['q'].add_argument(
            '--npts-min', type=int, default=100, dest='npts_min',
            help=('Minimum number of samples per sphere. This ensures adequate'
                  ' sampling at small q when using --npts-density.'))
        sections['q'].add_argument(
            '--npts-max', type=int, default=10000, dest='npts_max',
            help=('Maximum number of samples per sphere. This avoids '
                  'diminishing returns at large q when using --npts-density.'))
        sections['q'].add_argument(
            '--sampling', type=str, default='golden',
            choices=_sampling_choices,
            help=('Spherical sampling scheme; "golden" is generally '
                  'recommended uniform quasirandom sampling.'))
        sections['q'].add_argument(
            '--jitter', action='store_true',
            help=('Apply additional jitter to sample positions in angular '
                  'direction. Recommended for sampling methods other than '
                  '"golden" and "random-sphere".'))

    if 'plotting' in features:
        section = sections['plotting']
        section.add_argument(
            '-s', '--save-to', dest='save_to', default=None,
            help='Save resulting plot to a file with this name')
        section.add_argument('--title', type=str, default='',
                             help='Plot title')
        section.add_argument('--x-label', type=str, default=None,
                             dest='x_label', help='Plot x-axis label')
        section.add_argument('--y-label', type=str, default=None,
                             dest='y_label', help='Plot y-axis label')

    if {'ebins', 'q-e'}.intersection(features):
        section = sections['energy']
        section.add_argument('--e-min', type=float, default=None, dest='e_min',
                             help='Energy range minimum in ENERGY_UNIT')
        section.add_argument('--e-max', type=float, default=None, dest='e_max',
                             help='Energy range maximum in ENERGY_UNIT')
        section.add_argument('--energy-unit', '-u', dest='energy_unit',
                             type=str, default='meV', help='Energy units')

    if {'ebins', 'adaptive-broadening'}.intersection(features):
        if 'ebins' in features:
            section = sections['energy']
            section.add_argument('--ebins', type=int, default=200,
                                 help='Number of energy bins')
            if 'adaptive-broadening' in features:
                section.add_argument(
                    '--adaptive', action='store_true',
                    help=('Use adaptive broadening on the energy axis to '
                          'broaden based on phonon mode widths, rather than '
                          'using fixed width broadening'))
                eb_help = (
                    'If using fixed width broadening, the FWHM of broadening '
                    'on energy axis in ENERGY_UNIT (no broadening if '
                    'unspecified). If using adaptive broadening, this is a '
                    'scale factor multiplying the mode widths (no scale '
                    'factor applied if unspecified).')
            else:
                eb_help = ('The FWHM of broadening on energy axis in '
                           'ENERGY_UNIT (no broadening if unspecified).')
            section.add_argument(
                '--energy-broadening', '--eb', type=float, default=None,
                dest='energy_broadening',
                help=eb_help)
            section.add_argument(
                '--shape', type=str, nargs='?', default='gauss',
                choices=('gauss', 'lorentz'),
                help='The broadening shape')
        else:
            ValueError('"adaptive-broadening" cannot be applied without '
                       '"ebins"')

    if {'q-e', 'mp-grid'}.intersection(features):
        sections['q'].add_argument(
            '--length-unit', type=str, default='angstrom', dest='length_unit',
            help=('Length units; these will be inverted to obtain '
                  'units of distance between q-points (e.g. "bohr"'
                  ' for bohr^-1).'))

    if 'q-e' in features:
        sections['q'].add_argument(
            '--q-spacing', type=float, dest='q_spacing', default=0.025,
            help=('Target distance between q-point samples in 1/LENGTH_UNIT'))

    if {'q-e', 'map'}.issubset(features):
        sections['q'].add_argument(
            '--q-broadening', '--qb', type=float, default=None,
            dest='q_broadening',
            help=('FWHM of broadening on q axis in 1/LENGTH_UNIT '
                  '(no broadening if unspecified).'))

    if 'map' in features:
        sections['plotting'].add_argument(
            '--v-min', type=float, default=None, dest='v_min',
            help='Minimum of data range for colormap.')
        sections['plotting'].add_argument(
            '--v-max', type=float, default=None, dest='v_max',
            help='Maximum of data range for colormap.')
        sections['plotting'].add_argument(
            '--cmap', type=str, default='viridis', help='Matplotlib colormap')

    if 'btol' in features:
        sections['q'].add_argument(
            '--btol', default=10.0, type=float,
            help=('Distance threshold used for automatically splitting '
                  'discontinuous segments of reciprocal space onto separate '
                  'subplots. This is specified as a multiple of the median '
                  'distance between q-points.'))

    if 'dipole-parameter-optimisation' in features:
        parser.add_argument(
            '-n',
            default=500,
            type=int,
            help=('The number of times to loop over q-points. A higher '
                  'value will get a more reliable timing, but will take '
                  'longer')
        )
        parser.add_argument(
            '--dipole-parameter-min', '--min',
            default=0.25,
            type=float,
            help='The minimum value of dipole_parameter to test'
        )
        parser.add_argument(
            '--dipole-parameter-max', '--max',
            default=1.5,
            type=float,
            help='The maximum value of dipole_parameter to test'
        )
        parser.add_argument(
            '--dipole-parameter-step', '--step',
            default=0.25,
            type=float,
            help='The difference between each dipole_parameter to test'
        )

    return parser, sections
