from __future__ import annotations

from contextlib import suppress
from copy import copy
import os
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

# Required for mocking
with suppress(ModuleNotFoundError):
    import matplotlib.pyplot  # noqa: ICN001

from tests_and_analysis.test.utils import get_data_path

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3D
    from typing_extensions import NotRequired


class LabelData(TypedDict):
    title: str | None
    x_ticklabels: list[list[str]]
    x_label: list[str]
    y_label: list[str]
    z_label: NotRequired[list[str]]


class LabelDataWithLines(LabelData):
    xy_data: list[list[list[float] | float] | float]


class ImageData(TypedDict):
    cmap: str
    extent: list[float]
    size: list[int]
    data_1: list[float]
    data_2: list[float]


class LabelDataWithImage(LabelData, ImageData):
    pass


def args_to_key(cl_args: list[str]) -> str:
    """
    From CL tool arguments, return the key that should be used to store
    its testing output
    """
    cl_args = copy(cl_args)
    if os.path.isfile(cl_args[0]):
        cl_args[0] = ' '.join([os.path.split(os.path.dirname(cl_args[0]))[1],
                               os.path.split(cl_args[0])[1]])
    return ' '.join(cl_args)


def get_script_test_data_path(*subpaths: str) -> str:
    """
    Returns
    -------
    str
        The data folder for scripts testing data
    """
    return get_data_path('script_data', *subpaths)


def get_plot_line_data(fig: Figure | None = None) -> LabelDataWithLines:
    if fig is None:
        fig = matplotlib.pyplot.gcf()

    data = cast('LabelDataWithLines', get_fig_label_data(fig))
    data['xy_data'] = []
    for ax in fig.axes:
        if '3D' in type(ax).__name__:
            data['xy_data'].append([np.array(
                cast('Line3D', line).get_data_3d()).tolist()
                                    for line in ax.lines])
        else:
            data['xy_data'].append([cast('np.ndarray',
                                         line.get_xydata()).T.tolist()
                                    for line in ax.lines])
    return data


def get_all_figs() -> list[Figure]:
    fignums = matplotlib.pyplot.get_fignums()
    return [matplotlib.pyplot.figure(fignum) for fignum in fignums]


def get_all_plot_line_data(figs: list[Figure]) -> list[LabelDataWithLines]:
    return [get_plot_line_data(fig) for fig in figs]

def get_fig_label_data(fig) -> LabelData:
    from mpl_toolkits.mplot3d import Axes3D

    label_data: LabelData = {'x_ticklabels': [],
                  'x_label': [],
                  'y_label': [],
                  'title': fig._suptitle.get_text() \
                           if fig._suptitle is not None else None}

    # Get axis/tick labels from all axes, collect only non-empty values
    # to avoid breaking tests if the way we set up axes changes
    for ax in fig.axes:

        if xlabel := ax.get_xlabel():
            label_data['x_label'].append(xlabel)
        if ylabel := ax.get_ylabel():
            label_data['y_label'].append(ylabel)

        if '3D' in type(ax).__name__:
            label_data.setdefault('z_label', [])
            if zlabel := ax.get_zlabel():
                label_data['z_label'].append(zlabel)

        # Collect tick labels from visible axes only,
        # we don't care about invisible axis tick labels
        if isinstance(ax, Axes3D) or ax.get_frame_on():
            xticklabels = [lab.get_text() for lab in ax.get_xticklabels()]
            label_data['x_ticklabels'].append(xticklabels)

    return label_data


def get_current_plot_offsets() -> list[list[float]]:
    """
    Get the positions (offsets) from an active matplotlib scatter plot

    This should work for both 2D and 3D scatter plots; in the 3D case these
    offsets are based on 2D projections.

    Returns
    -------
    List[List[float]]
        Scatter plot offsets
    """
    return matplotlib.pyplot.gca().collections[0].get_offsets().data.tolist()


def get_current_plot_image_data() -> LabelDataWithImage:
    fig = matplotlib.pyplot.gcf()

    for ax in fig.axes:
        if len(ax.get_images()) == 1:
            break
    else:
        msg = 'Could not find axes with a single image'
        raise ValueError(msg)

    data = get_fig_label_data(fig)
    data = cast('LabelDataWithImage', data)
    data.update(get_ax_image_data(ax))

    return data


def get_ax_image_data(ax: Axes) -> ImageData:
    im = ax.get_images()[0]
    # Convert negative zero to positive zero
    im_data = cast('np.ma.MaskedArray', im.get_array())

    assert im_data is not None, 'No data available.'

    data_slice_1 = im_data[:, (im_data.shape[1] // 2)].flatten()
    data_slice_2 = im_data[im_data.shape[0] // 2, :].flatten()

    return {
        'cmap': im.cmap.name,
        'extent': [float(x) for x in im.get_extent()],
        'size': [int(x) for x in im.get_size()],
        'data_1': list(map(float, data_slice_1.filled(np.nan))),
        'data_2': list(map(float, data_slice_2.filled(np.nan))),
    }
