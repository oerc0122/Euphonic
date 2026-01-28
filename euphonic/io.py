from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import numpy as np

from euphonic.ureg import Quantity, ureg
from euphonic.version import __version__

T = TypeVar('T')
D = TypeVar('D', bound=Mapping[str, Any])

class Storable(Protocol):
    @classmethod
    def from_dict(cls, d: D) -> T: ...

    def to_dict(self) -> D: ...


S = TypeVar('S', bound=Storable)


def _to_json_dict(dictionary: dict[str, Any]) -> dict[str, Any]:
    """
    Convert all keys in an output dictionary to a JSON serialisable
    format
    """
    for key, val in dictionary.items():
        if isinstance(val, np.ndarray):
            if val.dtype == np.complex128:
                val = val.view(np.float64,                   # noqa: PLW2901
                               ).reshape([*val.shape, 2])
            dictionary[key] = val.tolist()
        elif isinstance(val, dict):
            if key != 'metadata':
                dictionary[key] = _to_json_dict(val)
    return dictionary


def _from_json_dict(dictionary: dict[str, Any],
                    type_dict: dict[str, type[Any]] | None = None,
) -> dict[str, Any]:
    """
    For a dictionary read from a JSON file, convert all list key values
    to that specified in type_dict. If not specified in type_dict, just
    uses the default conversion provided by np.array(list)
    """
    type_dict = {} if type_dict is None else type_dict

    for key, val in dictionary.items():
        if isinstance(val, list):
            if key in type_dict and type_dict[key] is tuple:
                dictionary[key] = [tuple(x) for x in val]
            elif key in type_dict and type_dict[key] == np.complex128:
                dictionary[key] = np.array(
                    val, dtype=np.float64).view(np.complex128).squeeze(axis=-1)
            else:
                dictionary[key] = np.array(val)
        elif isinstance(val, dict) and key != 'metadata':
            dictionary[key] = _from_json_dict(
                val, type_dict)
    return dictionary


def _obj_to_dict(obj: T, attrs: Sequence[str]) -> dict[str, Any]:
    dout = {}
    for attr in attrs:
        val = getattr(obj, attr)
        # Don't write None values to dict
        if val is None:
            continue
        # Don't write empty metadata to dict
        if attr == 'metadata' and not val:
            continue

        if isinstance(val, np.ndarray):
            dout[attr] = np.copy(val)
        elif isinstance(val, list):
            dout[attr] = val.copy()
        elif isinstance(val, Quantity):
            if isinstance(val.magnitude, np.ndarray):
                dout[attr] = np.copy(val.magnitude)
            else:
                dout[attr] = val.magnitude
            dout[attr + '_unit'] = str(val.units)
        elif hasattr(val, 'to_dict'):
            dout[attr] = val.to_dict()
        else:
            dout[attr] = val
    return dout


def _process_dict(dictionary: D,
                  quantities: Sequence[str] = (),
                  optional: Sequence[str] = ()) -> D:
    """
    Process an input dictionary for creating objects. Convert keys in
    'quantities' to Quantity objects, and if any 'optional' keys are
    missing, set them to None
    """

    # Need to assert dictionary is mutable.
    dictionary_ = cast('dict[str, Any]', copy.deepcopy(dictionary))

    for okey in optional:
        dictionary_.setdefault(okey, None)

    for qkey in quantities:
        val = dictionary_.pop(qkey)
        if val is not None:
            val = val*ureg(dictionary_.pop(qkey + '_unit'))
        dictionary_[qkey] = val
    return cast('D', dictionary_)


def _obj_to_json_file(obj: Storable, filename: Path | str) -> None:
    """
    Generic function for writing to a JSON file from a Euphonic object
    """
    dout = _to_json_dict(obj.to_dict())
    dout['__euphonic_class__'] = obj.__class__.__name__
    dout['__euphonic_version__'] = __version__
    with open(filename, 'w') as f:
        json.dump(dout, f, indent=4, sort_keys=True)
    print(f'Written to {Path(f.name).resolve()}')


def _obj_from_json_file(cls: type[S], filename: Path | str,
                        type_dict: dict[str, type[Any]]|None = None) -> S:
    """
    Generic function for reading from a JSON file to a Euphonic object
    """
    type_dict = {} if type_dict is None else type_dict

    with open(filename) as f:
        obj_dict = json.loads(f.read())
    obj_dict = _from_json_dict(obj_dict, type_dict)
    return cls.from_dict(obj_dict)
