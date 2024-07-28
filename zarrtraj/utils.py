import re
import pathlib
import fsspec
from dataclasses import dataclass
import zarr
from typing import Union, Dict
import numpy as np
from kerchunk.hdf import SingleHdf5ToZarr

ZARRTRAJ_NETWORK_PROTOCOLS = ["s3", "http", "https", "adl", "abfs", "az", "gcs"]
ZARRTRAJ_EXPERIMENTAL_PROTOCOLS = ["adl", "abfs", "az", "gcs"]


class H5MDElement:
    """Convenience class for representing elements in an H5MD
    file.
    """

    def __init__(self, group):

        if "value" not in group:
            raise ValueError(
                f"H5MD element {group.name} must have a value array"
            )

        self._value = group["value"]

        self._is_time_independent = False
        self._is_fixed = False
        self._has_time = False

        if "step" not in group:
            self._is_time_independent = True

            if "time" in group:
                raise ValueError(
                    f"{group.name} was determined to be time-independent since "
                    "it doesn't contain a step dataset. Therefore, it cannot "
                    "contain a time dataset"
                )

        else:
            self._step = group["step"]
            self._is_fixed = self.step.shape == ()

            if self.is_fixed():
                self._converted_step = None
                self._step_offset = self._step.attrs.get("offset", 0)

            if "time" in group:
                self._has_time = True
                self._time = group["time"]

                if self.is_fixed():
                    if self._time.shape != ():
                        raise ValueError(
                            f"Fixed step element {group.name} must have fixed time"
                        )
                    self._converted_time = None
                    self._time_offset = self._time.attrs.get("offset", 0)

                    if (
                        self._time_offset != self._step_offset
                        or self._time[()] != self._step[()]
                    ):
                        raise ValueError(
                            "Fixed time and step datasets must have the same step length "
                            f"and offset for element {group.name}"
                        )
                else:
                    if self._time.shape != self._step.shape:
                        raise ValueError(
                            f"Time and step datasets must have the same shape for element {group.name}"
                        )

    def is_fixed(self):
        return self._is_fixed

    def is_time_independent(self):
        return self._is_time_independent

    @property
    def has_time(self):
        return self._has_time

    @property
    def step(self) -> Union[zarr.array, np.ndarray]:
        """Return the step dataset as an np.array if the element
        is fixed, otherwise return the step array as a zarr array
        """
        if self.is_time_independent():
            raise ValueError("Element is time-independent")
        if self._is_fixed:
            if self._converted_step is None:
                self._converted_step = fixed_to_explicit(
                    len(self.value), self._step[()], self._step_offset
                )
            return self._converted_step
        return self._step

    @property
    def time(self) -> Union[zarr.array, np.ndarray]:
        """Return the step dataset as an np.array if the element
        is fixed, otherwise return the step array as a zarr array
        """
        if not self.has_time:
            raise ValueError("Element does not have a time dataset")
        if self.is_time_independent():
            raise ValueError("Element is time-independent")
        if self._is_fixed:
            if self._converted_time is None:
                self._converted_time = fixed_to_explicit(
                    len(self.value), self._time[()], self._time_offset
                )
            return self._converted_time
        return self._time

    @property
    def value(self):
        return self._value

    @property
    def timeunit(self):
        if not self.has_time:
            raise ValueError("Element does not have a time dataset")
        if self.is_time_independent():
            raise ValueError("Element is time-independent")
        return self._time.attrs.get("unit", None)

    @property
    def valueunit(self):
        return self._value.attrs.get("unit", None)


def get_protocol(url: str) -> str:
    parts = re.split(r"(\:\:|\://)", url, maxsplit=1)
    if len(parts) > 1:
        return parts[0]
    return "file"


def get_extension(url: str) -> str:
    return pathlib.Path(url).suffix


def get_h5_zarr_mapping(
    url: str,
    protocol: str,
    so: dict,
) -> str:

    with fsspec.open(url, **so) as inf:
        h5chunks = SingleHdf5ToZarr(inf, url, inline_threshold=100)
        fo = h5chunks.translate(preserve_linked_dsets=True)

    if protocol in ZARRTRAJ_NETWORK_PROTOCOLS:
        fs = fsspec.filesystem(
            "reference",
            fo=fo,
            remote_protocol=protocol,
            remote_options=so,
            skip_instance_cache=True,
        )
    elif protocol == "file":
        fs = fsspec.filesystem(
            "reference",
            fo=fo,
            skip_instance_cache=True,
        )
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    return fs.get_mapper("")


def get_mapping_for(filename, protocol, ext, so):
    if ext == ".zarr" or ext == ".zarrmd":
        fs = fsspec.filesystem(protocol, **so)
        mapping = fs.get_mapper(filename)
    elif ext == ".h5md" or ext == ".h5":
        mapping = get_h5_zarr_mapping(filename, protocol, so)
    else:
        raise ValueError(f"Cannot create Zarr Group from file type {ext}")
    return mapping


def create_steplist(steparrays: list):
    return np.unique(np.concatenate(steparrays))


def create_stepmap(elements: Dict[str, H5MDElement]):
    stepmap = {}

    for elem, h5mdelement in elements.items():
        if h5mdelement.is_time_independent():
            continue

        stepmap[elem] = {}
        for i, step in enumerate(h5mdelement.step):
            stepmap[elem][step] = i

    return stepmap


def get_explicit_time(file, h5mdelement):
    """
    Convert a fixed time to explicit time array
    """
    if h5mdelement.is_fixed():
        l = file[h5mdelement.value].shape[0]
        stop = h5mdelement.time_offset + (l * file[h5mdelement.time])
        return file[h5mdelement.value][:]
    else:
        return file[h5mdelement.time]


def fixed_to_explicit(length, step, offset):
    """
    Convert a fixed step to explicit step array
    """
    stop = offset + (length * step)
    return np.arange(offset, stop, step)
