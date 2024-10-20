"""

Example: Loading a .h5md file from disk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a simulation from a ``.h5md`` trajectory file, pass a
topology file and a path to the ``.zarrmd`` file to a
:class:`~MDAnalysis.core.universe.Universe`::

    import zarrtraj
    import MDAnalysis as mda
    u = mda.Universe("topology.tpr", "trajectory.h5md")

The reader can also read ``.zarrmd`` files from disk.

``zarrmd`` files are H5MD-formatted files stored in the Zarr format.
To learn more, see the `H5MD documentation <https://nongnu.org/h5md/>`_,
the `Zarr documentation <https://zarr.readthedocs.io/en/stable/>`_,
and the :ref:`zarrmd format page <zarrmd>`.

Example: Reading from cloud services
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Zarrtraj currently supports reading from ``.h5md`` and ``.zarrmd`` files stored in
AWS S3 buckets and, experimentally, Google Cloud buckets, Azure Blob storage,
and Azure DataLakes.

AWS S3
------

To read from AWS S3, pass the S3 url path to the file as the trajectory
argument::

    import zarrtraj
    import MDAnalysis as mda
    import os

    # Using environmental variables is a convenient way
    # to manage AWS credentials
    os.environ["AWS_PROFILE"] = "sample_profile"
    os.environ["AWS_REGION"] = "us-west-1"

    u = mda.Universe("topology.tpr", "s3://sample-bucket/trajectory.h5md")

AWS provides a VSCode extension to manage AWS authentication profiles
`here <https://aws.amazon.com/visualstudiocode/>`_.

Google Cloud Storage
--------------------

First, follow `these instructions <https://cloud.google.com/docs/authentication/provide-credentials-adc>`_ 
to setup Application Default Credentials on the machine you're streaming the trajectory to.
Then, after ensuring your GCS bucket exists and you've logged in using the gcloud CLI with a user that has read access to the bucket, 
you can read from the GCS bucket as follows::

    import zarrtraj
    import MDAnalysis as mda

    u = mda.Universe("topology.tpr", "gcs://sample-bucket/trajectory.h5md")

Azure Blob Storage and Data Lakes
---------------------------------

After configuring your storage account and container, the easiest way to authenticate is to use your storage accounts'
connection string which can be found in the Azure Portal::

    import zarrtraj
    import MDAnalysis as mda

    # For production use, make sure to store your connection string in a secure location
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = <str>

    u = mda.Universe("topology.tpr", "az://sample-container/trajectory.h5md")

For more information on authenticating with Azure, see the `adlfs documentation <https://github.com/fsspec/adlfs>`_.

.. warning::

    Zarrtraj is not optimized for reading trajectories 
    in the cloud with random-access patterns. Iterate 
    sequentially for best performance.

Example: Writing directly to cloud storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, writing directly to cloud storage is only supported for the ``zarrmd`` format.
If you want to write directly to a cloud storage in the H5MD format, please raise an issue
on the `zarrtraj GitHub <https://github.com/Becksteinlab/zarrtraj>`_.

All datasets in the file will be written using the same chunking strategy: 
`~12MB per chunk <https://d1.awsstatic.com/whitepapers/AmazonS3BestPractices.pdf>`_,
regardless of the data type, number of atoms, or number of frames in the trajectory. The only
exceptions to this are when a single frame of the trajectory is larger than 12MB, in which case
the chunk size will be 1 frame, or when the dataset is smaller than 12MB, in which case the
dataset will be written in a single chunk.


.. code-block:: python

    import zarrtraj
    import MDAnalysis as mda
    from MDAnalysisTests.datafiles import PSF, DCD
    import os

    os.environ["AWS_PROFILE"] = "sample_profile"
    os.environ["AWS_REGION"] = "us-west-1"

    u = mda.Universe(PSF, DCD)
    with mda.Writer("s3://sample-bucket/trajectory.zarrmd", 
                    n_atoms=u.atoms.n_atoms) as w:
        for ts in u.trajectory:
            w.write(u.atoms)

For Google Cloud Storage, change the URL protocol to ``gcs`` and the authenticate with the gcloud CLI.

For Azure Blob Storage or Data Lakes, change the URL protocol to ``abfs``/``adl``/``az`` and authenticate with
your storage account's connection string.

For details on authenticating with different cloud services, see their respective trajectory reading sections above.

Classes
^^^^^^^

.. autoclass:: ZARRH5MDReader
   :members:
   :inherited-members:
.. autoclass:: ZARRMDWriter
   :members:
   :inherited-members:
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates import base, core
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.due import due, Doi
from MDAnalysis.lib.util import store_init_arguments
from enum import Enum
from .utils import *
from .cache import FrameCache
import collections
import numbers
import logging
import warnings


try:
    import zarr
    import numcodecs
except ImportError:
    HAS_ZARR = False

    # Allow building documentation even if zarr is not installed
    import types

    class MockZarrFile:
        pass

    zarr = types.ModuleType("zarr")
    zarr.File = MockZarrFile

else:
    HAS_ZARR = True


logger = logging.getLogger(__name__)


class ZARRH5MDReader(base.ReaderBase):
    format = ["H5MD", "H5", "ZARR", "ZARRMD"]
    # units is defined as instance-level variable and set from the
    # H5MD file in __init__() below

    # This dictionary is used to translate H5MD units to MDAnalysis units.
    # (https://nongnu.org/h5md/modules/units.html)
    _unit_translation = {
        "time": {
            "ps": "ps",
            "fs": "fs",
            "ns": "ns",
            "second": "s",
            "sec": "s",
            "s": "s",
            "AKMA": "AKMA",
        },
        "length": {
            "Angstrom": "Angstrom",
            "angstrom": "Angstrom",
            "A": "Angstrom",
            "nm": "nm",
            "pm": "pm",
            "fm": "fm",
        },
        "velocity": {
            "Angstrom ps-1": "Angstrom/ps",
            "A ps-1": "Angstrom/ps",
            "Angstrom fs-1": "Angstrom/fs",
            "A fs-1": "Angstrom/fs",
            "Angstrom AKMA-1": "Angstrom/AKMA",
            "A AKMA-1": "Angstrom/AKMA",
            "nm ps-1": "nm/ps",
            "nm ns-1": "nm/ns",
            "pm ps-1": "pm/ps",
            "m s-1": "m/s",
        },
        "force": {
            "kJ mol-1 Angstrom-1": "kJ/(mol*Angstrom)",
            "kJ mol-1 nm-1": "kJ/(mol*nm)",
            "Newton": "Newton",
            "N": "N",
            "J m-1": "J/m",
            "kcal mol-1 Angstrom-1": "kcal/(mol*Angstrom)",
            "kcal mol-1 A-1": "kcal/(mol*Angstrom)",
        },
    }

    @due.dcite(
        Doi("10.25080/majora-1b6fd038-005"),
        description="MDAnalysis trajectory reader/writer of the H5MD format",
        path=__name__,
    )
    @due.dcite(
        Doi("10.1016/j.cpc.2014.01.018"),
        description="Specifications of the H5MD standard",
        path=__name__,
        version="1.1",
    )
    @due.dcite(
        Doi("10.25080/Majora-629e541a-00e"),
        description="MDAnalysis 2016",
        path=__name__,
    )
    @due.dcite(
        Doi("10.1002/jcc.21787"), description="MDAnalysis 2011", path=__name__
    )
    @due.dcite(
        Doi("10.5281/zenodo.3773449"), description="Zarr", path=__name__
    )
    @store_init_arguments
    def __init__(
        self,
        filename,
        storage_options=None,
        convert_units=True,
        group=None,
        cache_size=(100 * 1024**2),
        **kwargs,
    ):
        """Reads ``.h5md`` and ``.zarrmd`` trajectory files using Zarr.

        .. note::

            Though the H5MD standard states that time datasets are optional,
            MDAnalysis requires that all time-dependent data have a time
            dataset in order to guarantee that all constructed
            :class:`~MDAnalysis.coordinates.timestep.Timestep` objects have
            a time associated with them.

        Parameters
        ----------
        filename : str
            trajectory filename or URL
        convert_units : bool (optional)
            convert units to MDAnalysis units
        storage_options : dict (optional)
            options to pass to the storage backend via ``fsspec``
        group : str (optional)
            group in 'particles' to read from. Not required if only one group
            is present in 'particles'
        **kwargs : dict
            General reader arguments.

        Raises
        ------
        RuntimeError
            when ``Zarr`` is not installed
        ValueError
            when a unit is not recognized by MDAnalysis
        ValueError
            when the length units of position and box/edges in the trajectory
            group do not match
        ValueError
            when the time units of all time-dependent datasets do not match
        ValueError
            when ``convert_units=True`` but the H5MD file contains no units
        ValueError
            when an unsupported URL protocol is provided
        ValueError
            when dimension of unitcell is not 3
        ValueError
            when the simulation box is not 'periodic' or 'none'
        ValueError
            when a position, velocity, force, or simulation group box is
            time-independent
        ValueError
            when the H5MD trajectory groups' 'box/edges' and 'position' do not
            share the same step and time datasets
        NoDataError
            when the H5MD file does not contain an 'h5md` group
        NoDataError
            when the H5MD file does not contain a 'box` group
        NoDataError
            when the simulation box is 'periodic' but the H5MD trajectory
            group does not contain a 'box/edges' group
        NoDataError
            when the H5MD file contains multiple groups in 'particles' and
            the ``group`` kwarg is not provided
        NoDataError
            when a time-dependent dataset does not have a time dataset
        NoDataError
            when the H5MD file has no 'position', 'velocity', or
            'force' group

        """
        # Set to none so close() can be called
        self._file = None
        self._cache = None

        if not HAS_ZARR:
            raise RuntimeError("Please install zarr")
        super(ZARRH5MDReader, self).__init__(filename, **kwargs)
        self.filename = filename
        self.convert_units = convert_units
        self._cache_size = cache_size
        so = storage_options if storage_options is not None else {}

        self._determine_protocol()

        self._mapping = get_mapping_for(
            filename, self._protocol, get_extension(self.filename), so
        )

        self._group = group
        self._open_trajectory()

        # For cases where the user wants only a subset of the elements,
        # a new steplist and dicts can be created

        # Don't include observables in steplist
        self._global_steparray = create_steplist(
            [
                h5mdelement.step
                for name, h5mdelement in self._elements.items()
                if not h5mdelement.is_time_independent()
                and name in ("position", "velocity", "force", "box/edges")
            ]
        )
        self._n_frames = len(self._global_steparray)

        self._stepmaps = create_stepmap(
            self._elements,
        )

        # Gets some info about what settings the datasets were created with
        # from first available group
        for name in ("position", "velocity", "force"):
            if name in self._elements:
                dset = self._elements[name].value
                self.n_atoms = dset.shape[1]
                self.compressor = dset.compressor
                self.chunks = dset.chunks
                self.filters = dset.filters
                break
        else:
            raise NoDataError(
                "Provide at least a position, velocity "
                "or force group in the h5md file."
            )

        self.ts = self._Timestep(
            self.n_atoms,
            positions=("position" in self._elements),
            velocities=("velocity" in self._elements),
            forces=("force" in self._elements),
            **self._ts_kwargs,
        )

        # Set all observables to None initially
        for elem in self._elements:
            if elem not in ("position", "velocity", "force", "box/edges"):
                self.ts.data[elem] = None

        self.units = {
            "time": None,
            "length": None,
            "velocity": None,
            "force": None,
        }
        self._set_translated_units()  # fills units dictionary
        self._check_units()  # raises error if units missing

        self._cache = self._cache_type(
            self._file,
            self._cache_size,
            self.ts,
            self.chunks[0],
            self._elements,
            self._global_steparray,
            self._stepmaps,
        )

        self._read_next_timestep()

    def _set_translated_units(self):
        """converts units from H5MD to MDAnalysis notation
        and fills units dictionary"""

        self.units = collections.defaultdict(type(None))

        # Datasets are self-describing, so units must be provided for each
        # additionally, all time units must be the same and box/edges
        # and positions need to share the same length unit

        # Length
        l_units = []

        for elem in ("box/edges", "position"):
            if elem in self._elements:
                try:
                    len_unit = self._unit_translation["length"][
                        self._elements[elem].valueunit
                    ]
                except KeyError:
                    raise ValueError(
                        f"length unit '{self._elements[elem].valueunit}' "
                        "is not recognized by ZarrH5MDReader. "
                    )
                l_units.append(len_unit)

        if all(l_unit == l_units[0] for l_unit in l_units):
            self.units["length"] = l_units[0]
        else:
            raise ValueError(
                "Length units of position and box/edges do not match"
            )

        if "velocity" in self._elements:
            try:
                self.units["velocity"] = self._unit_translation["velocity"][
                    self._elements["velocity"].valueunit
                ]
            except KeyError:
                raise ValueError(
                    f"velocity unit '{self._elements['velocity'].valueunit}' "
                    "is not recognized by ZarrH5MDReader. "
                )

        if "force" in self._elements:
            try:
                self.units["force"] = self._unit_translation["force"][
                    self._elements["force"].valueunit
                ]
            except KeyError:
                raise ValueError(
                    f"force unit '{self._elements['force'].valueunit}' "
                    "is not recognized by ZarrH5MDReader. "
                )

        t_units = []
        for elem in self._elements:
            if not self._elements[elem].is_time_independent():
                try:
                    time_unit = self._unit_translation["time"][
                        self._elements[elem].timeunit
                    ]
                except KeyError:
                    raise ValueError(
                        f"time unit '{self._elements[elem].timeunit}' "
                        "is not recognized by ZarrH5MDReader. "
                    )
                t_units.append(time_unit)

        if all(t_unit == t_units[0] for t_unit in t_units):
            self.units["time"] = t_units[0]
        else:
            raise ValueError(
                "Time units do not match across all time-dependent datasets."
            )

    def _check_units(self):
        """Raises error if no units are provided from H5MD file
        and convert_units=True"""

        if not self.convert_units:
            return

        errmsg = "H5MD file must have readable units if ``convert_units`` is"
        " set to ``True``. MDAnalysis sets ``convert_units=True`` by default."
        " Set ``convert_units=False`` to load Universe without units."

        _group_unit_dict = {
            "time": "time",
            "position": "length",
            "velocity": "velocity",
            "force": "force",
        }

        for group, unit in _group_unit_dict.items():
            if unit == "time":
                if self.units["time"] is None:
                    raise ValueError(errmsg)

            else:
                if group in self._elements:
                    if self.units[unit] is None:
                        raise ValueError(errmsg)

    def _determine_protocol(self):
        """Prepares the correct method for managing the file
        given the protocol"""
        self._protocol = get_protocol(self.filename)
        if self._protocol in ZARRTRAJ_NETWORK_PROTOCOLS:
            self._cache_type = ZarrLRUCache
        elif self._protocol == "file":
            self._cache_type = ZarrNoCache
        else:
            raise ValueError(
                f"Unsupported protocol '{self._protocol}' for Zarrtraj."
            )

        if self._protocol in ZARRTRAJ_EXPERIMENTAL_PROTOCOLS:
            warnings.warn(
                f"Zarrtraj is using the experimental protocol '{self._protocol}' "
                "which may lead to unexpected behavior. Please report any issues "
                "on the Zarrtraj GitHub."
            )

    def _open_trajectory(self):
        """Opens the trajectory file using zarr library,
        sets the group if not already set, and
        fills the elements dictionary."""
        self._frame = -1
        # special case: using builtin LRU cache
        # normally the cache object handles cache construction
        if self._cache_type == ZarrLRUCache:
            cache = zarr.storage.LRUStoreCache(
                self._mapping, max_size=self._cache_size
            )
            self._file = zarr.open_group(store=cache, mode="r")
        else:
            self._file = zarr.open_group(self._mapping, mode="r")

        if "h5md" not in self._file:
            raise NoDataError("H5MD file must contain an 'h5md' group")

        if self._group is None:
            traj_keys = list(self._file["particles"].group_keys())
            if len(traj_keys) == 1:
                self._group = traj_keys[0]
            else:
                raise NoDataError(
                    "If `group` kwarg not provided, H5MD file must "
                    "contain exactly one group in 'particles'"
                )

        particle_group = self._file["particles"][self._group]
        self._elements = dict()

        if "observables" in self._file:
            for obsv in self._file["observables"]:
                if "value" in self._file["observables"][obsv]:
                    self._elements[obsv] = H5MDElement(
                        self._file["observables"][obsv]
                    )
            if self._group in self._file["observables"]:
                for obsv in self._file["observables"][self._group]:
                    if "value" in self._file["observables"][self._group][obsv]:
                        self._elements[f"{self._group}/{obsv}"] = H5MDElement(
                            self._file["observables"][self._group][obsv]
                        )

        if "box" not in particle_group:
            raise NoDataError("H5MD file must contain a 'box' group")
        if particle_group["box"].attrs["dimension"] != 3:
            raise ValueError(
                "MDAnalysis only supports 3-dimensional simulation boxes"
            )

        particle_group_elems = ["position", "velocity", "force", "box/edges"]
        if particle_group["box"].attrs["boundary"] == ["periodic"] * 3:
            if "box/edges" not in particle_group:
                raise NoDataError(
                    "H5MD file must contain a 'box/edges' group if "
                    "simulation box is 'periodic'"
                )
        elif particle_group["box"].attrs["boundary"] != ["none"] * 3:
            raise ValueError(
                "MDAnalysis only supports H5MD simulation boxes for which "
                "all dimensions are 'periodic' or all dimensions are 'none"
            )

        for elem in particle_group_elems:
            if elem in particle_group:
                elem_gr = particle_group[elem]
                h5md_elem = H5MDElement(elem_gr)
                if h5md_elem.is_time_independent():
                    raise ValueError(
                        "MDAnalysis does not support time-independent "
                        "positions, velocities, forces, or simulation boxes"
                    )
                self._elements[elem] = h5md_elem

        for elem in self._elements:
            if not self._elements[elem].is_time_independent():
                if not self._elements[elem].has_time:
                    raise NoDataError(
                        "MDAnalysis requires that time data is available "
                        "for every time-dependent dataset. "
                        f"Element '{elem}' does not have a time dataset"
                    )

        if "box/edges" in self._elements and "position" in self._elements:
            if not np.array_equal(
                self._elements["box/edges"].step[:],
                self._elements["position"].step[:],
            ):
                # We already checked that the time datasets correspond
                # to the step datasets in the H5MDElement class
                # so only verify step here
                raise ValueError(
                    "Position step and time must be hard links to box/edges "
                    "step and time under the H5MD standard"
                )

    def _read_next_timestep(self):
        """read next frame in trajectory"""

        return self._read_frame(self._frame + 1)

    def _read_frame(self, frame):
        """reads data from h5md-formatted file and copies to current timestep"""
        if frame < 0 or frame >= self.n_frames:
            raise IOError("Frame index out of range")

        self._frame = self._cache.load_frame(frame)

        if self.convert_units:
            self._convert_units()

        return self.ts

    def _convert_units(self):
        """converts time, position, velocity, and force values if they
        are not given in MDAnalysis standard units
        """

        self.ts.time = self.convert_time_from_native(self.ts.time)

        if "box/edges" in self._elements and self.ts.dimensions is not None:
            self.convert_pos_from_native(self.ts.dimensions[:3])

        if self.ts.has_positions:
            self.convert_pos_from_native(self.ts.positions)

        if self.ts.has_velocities:
            self.convert_velocities_from_native(self.ts.velocities)

        if self.ts.has_forces:
            self.convert_forces_from_native(self.ts.forces)

    def close(self):
        """close reader"""
        if self._cache is not None:
            self._cache.cleanup()
        if self._file is not None:
            self._file.store.close()

    def _reopen(self):
        self.close()
        self._open_trajectory()

    def Writer(self, filename, n_atoms=None, **kwargs):
        if n_atoms is None:
            n_atoms = self.n_atoms
        kwargs.setdefault("compressor", self.compressor)
        kwargs.setdefault("filters", self.filters)
        kwargs.setdefault("positions", ("position" in self._elements))
        kwargs.setdefault("velocities", ("velocity" in self._elements))
        kwargs.setdefault("forces", ("force" in self._elements))
        return ZARRMDWriter(filename, n_atoms, **kwargs)

    @property
    def n_frames(self):
        """number of frames in trajectory"""
        return self._n_frames

    @staticmethod
    def _format_hint(thing):
        """Can this Reader read *thing*"""
        # When should the reader be used by default?

    @staticmethod
    def parse_n_atoms(filename, group=None, so=None):
        so = so if so is not None else {}
        protocol = get_protocol(filename)
        ext = get_extension(filename)
        mapping = get_mapping_for(filename, protocol, ext, so)
        file = zarr.open_group(mapping, mode="r")

        if group is None:
            traj_keys = list(file["particles"].group_keys())
            if len(traj_keys) == 1:
                group = traj_keys[0]
            else:
                raise NoDataError(
                    "Could not construct a minimal topology from the H5MD "
                    "trajectory file, as `group` kwarg was not provided, "
                    "and H5MD file did not contain exactly one group in 'particles'"
                )

        for dset in ("position", "velocity", "force"):
            if dset in file["particles"][group]:
                return file["particles"][group][dset]["value"].shape[1]
        raise NoDataError(
            "Could not construct minimal topology from the "
            "H5MD trajectory file, as it did not contain a "
            "'position', 'velocity', or 'force' group. "
            "You must include a topology file."
        )

    def copy(self):
        """Return independent copy of this Reader.

        New Reader will have its own file handle and can seek/iterate
        independently of the original.

        Will also copy the current state of the Timestep held in the original
        Reader.

        Note
        ----
        ZARRH5MDReader overrides this method to copy
        the copied reader's timestep to the cache's timestep

        .. versionchanged:: 2.2.0
           Arguments used to construct the reader are correctly captured and
           passed to the creation of the new class. Previously the only
           ``n_atoms`` was passed to class copies, leading to a class created
           with default parameters which may differ from the original class.
        """

        new = self.__class__(**self._kwargs)

        if self.transformations:
            new.add_transformations(*self.transformations)
        # seek the new reader to the same frame we started with
        new[self.ts.frame]
        # then copy over the current Timestep in case it has
        # been modified since initial load
        new.ts = self.ts.copy()
        new._cache._timestep = new.ts
        for auxname, auxread in self._auxs.items():
            new.add_auxiliary(auxname, auxread.copy())
        return new


class H5MDElementBuffer:
    def __init__(
        self,
        shape,
        dtype,
        n_frames,
        elem_grp,
        val_unit=None,
        t_unit=None,
        compressor="default",
        precision=None,
    ):
        """We don't actually know that this element will be written
        at every frame, but, n_frames is a max length for the buffer
        to limit memory usage

        Parameters
        ----------
        shape : tuple
            shape of the dataset for one frame. i.e. (n_atoms, 3)
        """
        # indices in the actual zarr dataset
        # to get buffer indices, do i.e. _val_idx % _val_frames_per_chunk
        self._val_idx = 0
        self._t_idx = 0
        self._n_frames = n_frames if n_frames is not None else np.inf

        val_filter = None
        time_filter = None
        if precision is not None:
            val_filter = [numcodecs.quantize.Quantize(precision, dtype)]
            time_filter = [numcodecs.quantize.Quantize(precision, np.float32)]
        bytes_per_frame = (
            np.prod(shape, dtype=np.int32) * np.dtype(dtype).itemsize
        )
        # Cloud IO works best with 8-16 MB chunks
        # Use 12MB to get a reasonable number of frames per chunk
        # if a single frame is >12MB, just select 1 FPC
        self._val_frames_per_chunk = min(
            max(1, (12582912 // bytes_per_frame)), self._n_frames
        )
        # If the dataset is smaller than 12MB, just write it all at once
        self._val_chunks = tuple((self._val_frames_per_chunk, *shape))
        self._val_buf = np.empty(self._val_chunks, dtype=dtype)

        elem_grp.empty(
            "value",
            shape=self._val_chunks,
            chunks=self._val_chunks,
            dtype=dtype,
            compressor=compressor,
            filters=val_filter,
        )
        self._val = elem_grp["value"]
        if val_unit is not None:
            self._val.attrs["unit"] = val_unit

        # Step and time both use 4 byte dtypes
        self._t_frames_per_chunk = min((12582912 // 4), self._n_frames)
        self._t_chunks = (self._t_frames_per_chunk,)

        self._t_buf = np.empty(self._t_chunks, dtype=np.float32)
        elem_grp.empty(
            "time",
            shape=self._t_chunks,
            chunks=self._t_chunks,
            dtype=np.float32,
            compressor=compressor,
            filters=time_filter,
        )
        self._t = elem_grp["time"]
        if t_unit is not None:
            self._t.attrs["unit"] = t_unit

        self._s_buf = np.empty(self._t_chunks, dtype=np.int32)
        elem_grp.empty(
            "step",
            shape=self._t_chunks,
            chunks=self._t_chunks,
            dtype=np.int32,
            compressor=compressor,
        )
        self._s = elem_grp["step"]

    def write(
        self,
        data,
        step,
        time,
    ):
        # flush buffer and extend zarr dset if reached end of chunk
        # this will never be called if n_frames is less than the chunk size
        if (
            self._val_idx != 0
            and self._val_idx % self._val_frames_per_chunk == 0
        ):
            self._val[self._val_idx - self._val_frames_per_chunk :] = (
                self._val_buf[:]
            )
            # extend the dataset by one chunk
            self._val.resize(
                self._val.shape[0] + self._val_frames_per_chunk,
                *self._val_chunks[1:],
            )

        if self._t_idx != 0 and self._t_idx % self._t_frames_per_chunk == 0:
            self._t[self._t_idx - self._t_frames_per_chunk :] = self._t_buf[:]
            self._s[self._t_idx - self._t_frames_per_chunk :] = self._s_buf[:]
            self._t.resize(self._t.shape[0] + self._t_frames_per_chunk)
            self._s.resize(self._s.shape[0] + self._t_frames_per_chunk)

        self._val_buf[self._val_idx % self._val_frames_per_chunk] = data
        self._val_idx += 1
        self._t_buf[self._t_idx % self._t_frames_per_chunk] = time
        self._s_buf[self._t_idx % self._t_frames_per_chunk] = step
        self._t_idx += 1

    def flush(self):
        """Write everything remaining in the buffers to the zarr datasets
        and shink the zarr datasets to the correct size.
        """
        num_v_frames = self._val_idx % self._val_frames_per_chunk
        if num_v_frames == 0:
            num_v_frames = self._val_frames_per_chunk

        self._val[self._val_idx - num_v_frames : self._val_idx] = (
            self._val_buf[:num_v_frames]
        )
        self._val.resize(self._val_idx, *self._val_chunks[1:])

        num_t_frames = self._t_idx % self._t_frames_per_chunk
        if num_t_frames == 0:
            num_t_frames = self._t_frames_per_chunk

        self._t[self._t_idx - num_t_frames : self._t_idx] = self._t_buf[
            :num_t_frames
        ]

        self._t.resize(self._t_idx)

        self._s[self._t_idx - num_t_frames : self._t_idx] = self._s_buf[
            :num_t_frames
        ]
        self._s.resize(self._t_idx)


class ZARRMDWriter(base.WriterBase):
    """
    Writer for the H5MD format using Zarr.

    Parameters
    ----------
    filename : str
        filename or URL to write to
    n_atoms : int
        number of atoms in the system
    n_frames : int (optional)
        number of frames to be written in the output trajectory. If not
        provided, the trajectory will allocate more memory than necessary
        which may slow down trajectory write speed.
    compressor : numcodecs.Codec (optional)
        compressor to use for the Zarr datasets. Will be applied to all datasets
    precision : int (optional)
        applies the numcodecs.Quantize filter to Zarr datasets.
        Will be applied to all floating point datasets
    storage_options : dict (optional)
        options to pass to the storage backend via ``fsspec``
    convert_units : bool (optional)
        Convert units from MDAnalysis to desired units [``True``]
    positions : bool (optional)
        Write positions into the trajectory [``True``]
    velocities : bool (optional)
        Write velocities into the trajectory [``True``]
    forces : bool (optional)
        Write forces into the trajectory [``True``]
    timeunit : str (optional)
        Option to convert values in the 'time' dataset to a custom unit,
        must be recognizable by MDAnalysis
    lengthunit : str (optional)
        Option to convert values in the 'position/value' dataset to a
        custom unit, must be recognizable by MDAnalysis
    velocityunit : str (optional)
        Option to convert values in the 'velocity/value' dataset to a
        custom unit, must be recognizable by MDAnalysis
    forceunit : str (optional)
        Option to convert values in the 'force/value' dataset to a
        custom unit, must be recognizable by MDAnalysis
    author : str (optional)
        Name of the author of the file
    author_email : str (optional)
        Email of the author of the file
    creator : str (optional)
        Software that wrote the file [``MDAnalysis``]
    creator_version : str (optional)
        Version of software that wrote the file
        [:attr:`MDAnalysis.__version__`]

    Raises
    ------
    RuntimeError
        when ``Zarr`` is not installed
    ValueError
        when ``n_atoms`` is 0
    ValueError
        when ``n_frames`` is provided and not positive
    ValueError
        when ``precision`` is less than 0
    ValueError
        when 'positions`, 'velocities', and 'forces' are all set to ``False``
    ValueError
        when a unit is not recognized by MDAnalysis
    TypeError
        when the input object is not a :class:`Universe` or
        :class:`AtomGroup`
    ValueError
        when any of the optional `timeunit`, `lengthunit`,
        `velocityunit`, or `forceunit` keyword arguments are
        not recognized by MDAnalysis
    ValueError
        when ``convert_units`` is set to ``True`` but the trajectory
        being written has no units
    NoDataError
        when a timestep being written contains positions but no dimensions
        or vice versa
    """

    format = "ZARRMD"
    multiframe = True
    #: These variables are not written from :attr:`Timestep.data`
    #: dictionary to the observables group in the H5MD file
    data_blacklist = ["step", "time", "dt"]

    #: currently written version of the file format
    H5MD_VERSION = (1, 1)

    # This dictionary is used to translate MDAnalysis units to H5MD units.
    # (https://nongnu.org/h5md/modules/units.html)
    _unit_translation_dict = {
        "time": {
            "ps": "ps",
            "fs": "fs",
            "ns": "ns",
            "second": "s",
            "sec": "s",
            "s": "s",
            "AKMA": "AKMA",
        },
        "length": {
            "Angstrom": "Angstrom",
            "angstrom": "Angstrom",
            "A": "Angstrom",
            "nm": "nm",
            "pm": "pm",
            "fm": "fm",
        },
        "velocity": {
            "Angstrom/ps": "Angstrom ps-1",
            "A/ps": "Angstrom ps-1",
            "Angstrom/fs": "Angstrom fs-1",
            "A/fs": "Angstrom fs-1",
            "Angstrom/AKMA": "Angstrom AKMA-1",
            "A/AKMA": "Angstrom AKMA-1",
            "nm/ps": "nm ps-1",
            "nm/ns": "nm ns-1",
            "pm/ps": "pm ps-1",
            "m/s": "m s-1",
        },
        "force": {
            "kJ/(mol*Angstrom)": "kJ mol-1 Angstrom-1",
            "kJ/(mol*nm)": "kJ mol-1 nm-1",
            "Newton": "Newton",
            "N": "N",
            "J/m": "J m-1",
            "kcal/(mol*Angstrom)": "kcal mol-1 Angstrom-1",
            "kcal/(mol*A)": "kcal mol-1 Angstrom-1",
        },
    }

    @due.dcite(
        Doi("10.25080/majora-1b6fd038-005"),
        description="MDAnalysis trajectory reader/writer of the H5MD" "format",
        path=__name__,
    )
    @due.dcite(
        Doi("10.1016/j.cpc.2014.01.018"),
        description="Specifications of the H5MD standard",
        path=__name__,
        version="1.1",
    )
    @due.dcite(
        Doi("10.25080/Majora-629e541a-00e"),
        description="MDAnalysis 2016",
        path=__name__,
    )
    @due.dcite(
        Doi("10.1002/jcc.21787"), description="MDAnalysis 2011", path=__name__
    )
    def __init__(
        self,
        filename,
        n_atoms,
        n_frames=None,
        compressor="default",
        precision=None,
        storage_options=None,
        convert_units=True,
        positions=True,
        velocities=True,
        forces=True,
        timeunit=None,
        lengthunit=None,
        velocityunit=None,
        forceunit=None,
        author="N/A",
        author_email=None,
        creator="MDAnalysis",
        creator_version=mda.__version__,
        **kwargs,
    ):
        self._file = None
        self._elements = dict()

        self.filename = filename
        self.compressor = compressor
        if precision is not None and precision < 0:
            raise ValueError("Precision must be greater than or equal to 0")
        self.prec = precision

        if not HAS_ZARR:
            raise RuntimeError("Please install zarr")
        if n_atoms == 0:
            raise ValueError("H5MDWriter: no atoms in output trajectory")

        self.n_atoms = n_atoms
        if n_frames is not None and n_frames <= 0:
            raise ValueError(
                "H5MDWriter: Please provide a positive value for 'n_frames' kwarg"
            )
        self.n_frames = n_frames
        self.storage_options = storage_options

        self.convert_units = convert_units

        # The writer defaults to writing all data from the parent Timestep if
        # it exists. If these are True, the writer will check each
        # Timestep.has_*  value and fill the self._has dictionary accordingly
        # in _initialize_hdf5_datasets()
        self._write_positions = positions
        self._write_velocities = velocities
        self._write_forces = forces
        if not any(
            [
                self._write_positions,
                self._write_velocities,
                self._write_velocities,
            ]
        ):
            raise ValueError(
                "At least one of positions, velocities, or "
                "forces must be set to ``True``."
            )

        self._new_units = {
            "time": timeunit,
            "length": lengthunit,
            "velocity": velocityunit,
            "force": forceunit,
        }

        # Pull out various keywords to store metadata in 'h5md' group
        self.author = author
        self.author_email = author_email
        self.creator = creator
        self.creator_version = creator_version

        protocol = get_protocol(filename)
        if protocol not in ZARRTRAJ_NETWORK_PROTOCOLS and protocol != "file":
            raise ValueError(
                f"Unsupported protocol '{protocol}' for Zarrtraj."
            )
        if protocol in ZARRTRAJ_EXPERIMENTAL_PROTOCOLS:
            warnings.warn(
                f"Zarrtraj is using the experimental protocol '{protocol}' "
                "which may lead to unexpected behavior. Please report any issues "
                "on the Zarrtraj GitHub."
            )

    def _write_next_frame(self, ag):
        """Write information associated with ``ag`` at current frame
        into trajectory

        Parameters
        ----------
        ag : AtomGroup or Universe

        """
        try:
            # Atomgroup?
            ts = ag.ts
        except AttributeError:
            try:
                # Universe?
                ts = ag.trajectory.ts
            except AttributeError:
                errmsg = "Input obj is neither an AtomGroup or Universe"
                raise TypeError(errmsg) from None

        if ts.n_atoms != self.n_atoms:
            raise IOError(
                "H5MDWriter: Timestep does not have"
                " the correct number of atoms"
            )

        # This should only be called once when first timestep is read.
        if self._file is None:
            self._determine_units(ag)
            self._open_file()
            self._initialize_zarrmd_file(ts)

        return self._write_next_timestep(ts)

    def _determine_units(self, ag):
        """determine which units the file will be written with

        By default, it fills the :attr:`self.units` dictionary by copying the
        units dictionary of the parent file. Because H5MD files have no
        standard unit restrictions, users may pass a kwarg in ``(timeunit,
        lengthunit, velocityunit, forceunit)`` to the writer so long as
        MDAnalysis has a conversion factor for it (:exc:`ValueError` raised if
        it does not). These custom unit arguments must be in
        MDAnalysis notation. If custom units are supplied from the user,
        :attr`self.units[unit]` is replaced with the corresponding
        `unit` argument.
        """

        self.units = ag.universe.trajectory.units.copy()

        # set user input units
        for key, value in self._new_units.items():
            if value is not None:
                if value not in self._unit_translation_dict[key]:
                    raise ValueError(
                        f"{value} is not a unit recognized by"
                        " MDAnalysis. Allowed units are:"
                        f" {self._unit_translation_dict.keys()}"
                        " For more information on units, see"
                        " `MDAnalysis units`_."
                    )
                else:
                    self.units[key] = self._new_units[key]

        if self.convert_units:
            # check if all units are None
            if not any(self.units.values()):
                raise ValueError(
                    "The trajectory has no units, but "
                    "`convert_units` is set to ``True`` by "
                    "default in MDAnalysis. To write the file "
                    "with no units, set ``convert_units=False``."
                )

    def _open_file(self):
        storage_options = (
            dict() if self.storage_options is None else self.storage_options
        )
        self._file = zarr.open_group(
            self.filename, storage_options=storage_options, mode="w"
        )

        # fill in H5MD metadata from kwargs
        self._file.require_group("h5md")
        self._file["h5md"].attrs["version"] = self.H5MD_VERSION
        self._file["h5md"].require_group("author")
        self._file["h5md/author"].attrs["name"] = self.author
        if self.author_email is not None:
            self._file["h5md/author"].attrs["email"] = self.author_email
        self._file["h5md"].require_group("creator")
        self._file["h5md/creator"].attrs["name"] = self.creator
        self._file["h5md/creator"].attrs["version"] = self.creator_version

        self._curr_time = None
        self._curr_step = None
        # for keeping track of num frames written
        self._counter = 0

    def _initialize_zarrmd_file(self, ts):
        # initialize trajectory group
        self._file.require_group("particles").require_group("trajectory")
        self._traj = self._file["particles/trajectory"]
        self.data_keys = [
            key for key in ts.data.keys() if key not in self.data_blacklist
        ]

        # box group is required for every group in 'particles'
        self._traj.require_group("box")
        self._traj["box"].attrs["dimension"] = 3

        # assume boundary condition is "none" until first
        # frame with dimensions is encountered in _write_next_timestep()
        self._traj["box"].attrs["boundary"] = 3 * ["none"]

        # assume we won't encounter observables, require group when we do
        self._obsv = None

    def _allocate_buffers(self, ts):
        """Allocates buffers for timestep data that wasn't already allocated"""
        t_unit = self._unit_translation_dict["time"][self.units["time"]]

        if (
            ts.dimensions is not None
            and np.all(ts.dimensions > 0)
            and "box/edges" not in self._elements
        ):
            length_unit = (
                self._unit_translation_dict["length"][self.units["length"]]
                if self.units["length"] is not None
                else None
            )
            self._traj["box"].attrs["boundary"] = 3 * ["periodic"]
            self._traj["box"].require_group("edges")
            self._elements["box/edges"] = H5MDElementBuffer(
                ts.triclinic_dimensions.shape,
                ts.triclinic_dimensions.dtype,
                self.n_frames,
                self._traj["box/edges"],
                val_unit=length_unit,
                t_unit=t_unit,
                compressor=self.compressor,
                precision=self.prec,
            )

        if (
            self._write_positions
            and ts.has_positions
            and "position" not in self._elements
        ):
            length_unit = (
                self._unit_translation_dict["length"][self.units["length"]]
                if self.units["length"] is not None
                else None
            )
            self._traj.require_group("position")
            self._elements["position"] = H5MDElementBuffer(
                ts.positions.shape,
                ts.positions.dtype,
                self.n_frames,
                self._traj["position"],
                val_unit=length_unit,
                t_unit=t_unit,
                compressor=self.compressor,
                precision=self.prec,
            )

        if (
            self._write_velocities
            and ts.has_velocities
            and "velocity" not in self._elements
        ):
            vel_unit = (
                self._unit_translation_dict["velocity"][self.units["velocity"]]
                if self.units["velocity"] is not None
                else None
            )
            self._traj.require_group("velocity")
            self._elements["velocity"] = H5MDElementBuffer(
                ts.velocities.shape,
                ts.velocities.dtype,
                self.n_frames,
                self._traj["velocity"],
                val_unit=vel_unit,
                t_unit=t_unit,
                compressor=self.compressor,
                precision=self.prec,
            )

        if (
            self._write_forces
            and ts.has_forces
            and "force" not in self._elements
        ):
            force_unit = (
                self._unit_translation_dict["force"][self.units["force"]]
                if self.units["force"] is not None
                else None
            )
            self._traj.require_group("force")
            self._elements["force"] = H5MDElementBuffer(
                ts.forces.shape,
                ts.forces.dtype,
                self.n_frames,
                self._traj["force"],
                val_unit=force_unit,
                t_unit=t_unit,
                compressor=self.compressor,
                precision=self.prec,
            )

        for obsv, value in ts.data.items():
            if (
                value is not None
                and obsv not in self.data_blacklist
                and obsv not in self._elements
            ):
                if self._obsv is None:
                    self._file.require_group("observables").require_group(
                        "trajectory"
                    )
                    self._obsv = self._file["observables/trajectory"]
                self._obsv.require_group(obsv)
                self._elements[obsv] = H5MDElementBuffer(
                    ts.data[obsv].shape,
                    ts.data[obsv].dtype,
                    self.n_frames,
                    self._obsv[obsv],
                    t_unit=self.units["time"],
                    compressor=self.compressor,
                    precision=self.prec,
                )

    def _write_next_timestep(self, ts):
        """Write coordinates and unitcell information to H5MD file.

        Do not call this method directly; instead use
        :meth:`write` because some essential setup is done
        there before writing the first frame.

        The first dimension of each dataset is extended by +1 and
        then the data is written to the new slot.

        Note
        ----
        Writing H5MD-formatted files with fancy trajectory slicing where the Timestep
        does not increase monotonically such as ``u.trajectory[[2,1,0]]``
        or ``u.trajectory[[0,1,2,0,1,2]]`` raises a :exc:`ValueError` as this
        violates the rules of the step dataset in the H5MD standard.

        """
        self._allocate_buffers(ts)

        prev_time = self._curr_time
        self._curr_time = (
            ts.time
            if self.units["time"] is None
            else self.convert_time_to_native(ts.time)
        )
        if prev_time is not None and self._curr_time < prev_time:
            raise ValueError(
                "The H5MD standard dictates that the time values "
                + "must increase monotonically"
            )
        prev_step = self._curr_step
        self._curr_step = ts.data["step"] if "step" in ts.data else ts.frame
        if prev_step is not None and self._curr_step < prev_step:
            raise ValueError(
                "The H5MD standard dictates that the step values "
                + "must increase monotonically"
            )

        if ts.dimensions is not None and "box/edges" in self._elements:

            box = (
                ts.triclinic_dimensions
                if self.units["length"] is None
                else self.convert_pos_to_native(ts.triclinic_dimensions)
            )
            self._elements["box/edges"].write(
                box, self._curr_step, self._curr_time
            )

            if not ts.has_positions and "position" in self._elements:
                raise NoDataError(
                    "H5MD requires that positions and box dimensions are "
                    "sampled at the same rate. No positions found in Timestep."
                )

        if ts.has_positions and "position" in self._elements:
            pos = (
                ts.positions
                if self.units["length"] is None
                else self.convert_pos_to_native(ts.positions)
            )
            self._elements["position"].write(
                pos, self._curr_step, self._curr_time
            )

            if ts.dimensions is None and "box/edges" in self._elements:
                raise NoDataError(
                    "H5MD requires that positions and box dimensions are "
                    "sampled at the same rate. No dimensions found in Timestep."
                )

        if ts.has_velocities and "velocity" in self._elements:
            vel = (
                ts.velocities
                if self.units["velocity"] is None
                else self.convert_velocities_to_native(ts.velocities)
            )
            self._elements["velocity"].write(
                vel, self._curr_step, self._curr_time
            )

        if ts.has_forces and "force" in self._elements:
            force = (
                ts.forces
                if self.units["force"] is None
                else self.convert_forces_to_native(ts.forces)
            )
            self._elements["force"].write(
                force, self._curr_step, self._curr_time
            )

        for obsv, value in ts.data.items():
            if (
                value is not None
                and obsv not in self.data_blacklist
                and obsv in self._elements
            ):
                self._elements[obsv].write(
                    value, self._curr_step, self._curr_time
                )

        self._counter += 1

    def close(self):
        # Prevent close from throwing errors if __init__ hasn't been called yet
        if hasattr(self, "_elements") and self._elements:
            for elembuffer in self._elements.values():
                elembuffer.flush()
                # To ensure idempotency:
                self._elements = dict()
        if hasattr(self, "_counter"):
            if self.n_frames is not None:
                # issue a warning if counter is less or more than n_frames
                if self._counter < self.n_frames:
                    warnings.warn(
                        f"H5MDWriter: `n_frames` kwarg set to {self.n_frames} but "
                        f"only {self._counter} frame(s) were written to the trajectory.",
                        RuntimeWarning,
                    )
                if self._counter >= self.n_frames:
                    warnings.warn(
                        f"H5MDWriter: `n_frames` kwarg set to {self.n_frames} but "
                        f"{self._counter} frame(s) were written to the trajectory.",
                        RuntimeWarning,
                    )
            del self._counter
        if hasattr(self, "_file") and self._file is not None:
            self._file.store.close()
            self._file = None


class ZarrNoCache(FrameCache):
    """No caching for Zarr-interfaced reads
    Used for reading H5MD formatted files from disk"""

    def __init__(
        self,
        open_file,
        cache_size,
        timestep,
        frames_per_chunk,
        elements,
        global_steparray,
        stepmaps,
    ):
        super().__init__(open_file, cache_size, timestep, frames_per_chunk)
        self._elements = elements
        self._global_steparray = global_steparray
        self._stepmaps = stepmaps

    def update_desired_dsets(
        self,
        elements: Dict[str, H5MDElement],
        global_steparray: np.ndarray,
        stepmaps: Dict[int, Dict[int, int]],
    ):
        self._elements = elements
        self._global_steparray = global_steparray
        self._stepmaps = stepmaps

    def load_frame(self, frame):
        """Reader responsible for raising StopIteration when no more frames"""
        self._load_timestep_frame(frame)
        return frame

    def _load_timestep_frame(self, frame):
        # Reader must handle unit conversions
        # reader must ensure time value present (not -1)
        step = self._global_steparray[frame]

        self._timestep.frame = frame
        self._timestep.data["step"] = step

        # Assume all time values from the same integration step
        # are exactly the same
        curr_time = None

        if "box/edges" in self._elements:
            if step in self._stepmaps["box/edges"]:
                edges_index = self._stepmaps["box/edges"][step]
                edges = self._elements["box/edges"].value[edges_index]
                if edges.shape == (3,):
                    self._timestep.dimensions = [*edges, 90, 90, 90]
                else:
                    self._timestep.dimensions = core.triclinic_box(*edges)
                if curr_time is None and self._elements["box/edges"].has_time:
                    curr_time = self._elements["box/edges"].time[edges_index]
            else:
                self._timestep.dimensions = None

        if "position" in self._elements:
            if step in self._stepmaps["position"]:
                self._timestep.has_positions = True
                pos_index = self._stepmaps["position"][step]
                self._timestep.positions = self._elements["position"].value[
                    pos_index
                ]
                if curr_time is None and self._elements["position"].has_time:
                    curr_time = self._elements["position"].time[pos_index]
            else:
                self._timestep.has_positions = False

        if "velocity" in self._elements:
            if step in self._stepmaps["velocity"]:
                self._timestep.has_velocities = True
                vel_index = self._stepmaps["velocity"][step]
                self._timestep.velocities = self._elements["velocity"].value[
                    vel_index
                ]
                if curr_time is None and self._elements["velocity"].has_time:
                    curr_time = self._elements["velocity"].time[vel_index]
            else:
                self._timestep.has_velocities = False

        if "force" in self._elements:
            if step in self._stepmaps["force"]:
                self._timestep.has_forces = True
                force_index = self._stepmaps["force"][step]
                self._timestep.forces = self._elements["force"].value[
                    force_index
                ]
                if curr_time is None and self._elements["force"].has_time:
                    curr_time = self._elements["force"].time[force_index]
            else:
                self._timestep.has_forces = False

        exclude = {"position", "velocity", "force", "box/edges"}

        for elem, h5mdelement in self._elements.items():
            if elem not in exclude:
                if not h5mdelement.is_time_independent():
                    if step in self._stepmaps[elem]:
                        obsv_index = self._stepmaps[elem][step]
                        self._timestep.data[elem] = h5mdelement.value[
                            obsv_index
                        ]
                        if curr_time is None and self._elements[elem].has_time:
                            curr_time = self._elements[elem].time[obsv_index]
                    elif elem in self._timestep.data:
                        self._timestep.data[elem] = None
                else:
                    # must be time independent
                    self._timestep.data[elem] = h5mdelement.value[:]

        self._timestep.time = curr_time

    def cleanup(self):
        pass


class ZarrLRUCache(ZarrNoCache):
    """Clone of ZarrNoCache to allow differentiation since
    ZarrLRUCache is a special case where the reader handles the cache"""
