from typing import List

from euphonic import (ureg, ForceConstants, QpointPhononModes,
                      Spectrum1DCollection)
from euphonic.util import mp_grid, mode_gradients_to_widths
from euphonic.plot import plot_1d
from .utils import (load_data_from_file, get_args, matplotlib_save_or_show,
                    _calc_modes_kwargs,
                    _get_cli_parser, _get_energy_bins,
                    _grid_spec_from_args, _get_pdos_weighting,
                    _arrange_pdos_groups)


def main(params: List[str] = None):
    parser = get_parser()
    args = get_args(parser, params)

    data = load_data_from_file(args.filename)
    mode_widths = None
    if isinstance(data, ForceConstants):

        recip_length_unit = ureg(f'1 / {args.length_unit}')
        grid_spec = _grid_spec_from_args(data.crystal, grid=args.grid,
                                         grid_spacing=(args.grid_spacing
                                                       * recip_length_unit))

        print("Force Constants data was loaded. Calculating phonon modes "
              "on {} q-point grid...".format(
                  ' x '.join([str(x) for x in grid_spec])))
        if args.adaptive:
            if args.shape != 'gauss':
                raise ValueError('Currently only Gaussian shape is supported '
                                 'with adaptive broadening')
            cmkwargs = _calc_modes_kwargs(args)
            cmkwargs['return_mode_gradients'] = True
            modes, mode_grads = data.calculate_qpoint_phonon_modes(
                mp_grid(grid_spec), **cmkwargs)
            mode_widths = mode_gradients_to_widths(mode_grads,
                                                   modes.crystal.cell_vectors)
            if args.energy_broadening:
                mode_widths *= args.energy_broadening
        else:
            modes = data.calculate_qpoint_phonon_modes(
                mp_grid(grid_spec), **_calc_modes_kwargs(args))

    elif isinstance(data, QpointPhononModes):
        if args.adaptive:
            raise ValueError('Cannot calculate mode gradients without force '
                             'constants data. Do not use --adaptive if using '
                             'precalculated phonon modes')
        print("Phonon band data was loaded.")
        modes = data
    modes.frequencies_unit = args.energy_unit
    ebins = _get_energy_bins(
        modes, args.ebins + 1, emin=args.e_min, emax=args.e_max)
    if args.weighting == 'dos' and args.pdos is None:
        dos = modes.calculate_dos(ebins, mode_widths=mode_widths)
    else:
        pdos = modes.calculate_pdos(
            ebins, mode_widths=mode_widths,
            weighting=_get_pdos_weighting(args.weighting))
        dos = _arrange_pdos_groups(pdos, args.pdos)

    if args.energy_broadening and not args.adaptive:
        dos = dos.broaden(args.energy_broadening*ebins.units, shape=args.shape)

    if args.x_label is None:
        x_label = f"Energy / {dos.x_data.units:~P}"
    else:
        x_label = args.x_label
    if args.y_label is None:
        y_label = ""
    else:
        y_label = args.y_label

    fig = plot_1d(dos, title=args.title, x_label=x_label, y_label=y_label,
                  y_min=0, lw=1.0)
    matplotlib_save_or_show(save_filename=args.save_to)


def get_parser():
    parser, _ = _get_cli_parser(features={'read-fc', 'read-modes', 'mp-grid',
                                          'plotting', 'ebins',
                                          'adaptive-broadening',
                                          'pdos-weighting'})
    parser.description = (
        'Plots a DOS from the file provided. If a force '
        'constants file is provided, a DOS is generated on the Monkhorst-Pack '
        'grid specified by the grid (or grid-spacing) argument.')

    return parser
