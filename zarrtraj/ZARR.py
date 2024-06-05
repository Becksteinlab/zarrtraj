import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates import base, core
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.due import due, Doi
from MDAnalysis.lib.util import store_init_arguments
import dask.array as da
from enum import Enum
from .utils import *
from .cache import FrameCache, AsyncFrameCache
import collections
import numbers
import logging


try:
    import zarr
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
    format = ["H5MD", "H5", "ZARR"]
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

    @store_init_arguments
    def __init__(
        self,
        filename,
        storage_options=None,
        convert_units=True,
        group=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filename : str or :class:`h5py.File`
            trajectory filename or open h5py file
        convert_units : bool (optional)
            convert units to MDAnalysis units
        **kwargs : dict
            General reader arguments.

        Raises
        ------
        RuntimeError
            when `H5PY`_ is not installed
        RuntimeError
            when a unit is not recognized by MDAnalysis
        ValueError
            when ``n_atoms`` changes values between timesteps
        ValueError
            when ``convert_units=True`` but the H5MD file contains no units
        ValueError
            when dimension of unitcell is not 3
        ValueError
            when an MPI communicator object is passed to the reader
            but ``driver != 'mpio'``
        NoDataError
            when the H5MD file has no 'position', 'velocity', or
            'force' group

        """
        if not HAS_ZARR:
            raise RuntimeError("Please install zarr")
        super(ZARRH5MDReader, self).__init__(filename, **kwargs)
        self.filename = filename
        self.convert_units = convert_units
        self._group = group
        self._so = storage_options if storage_options is not None else {}

        self._frame_seq = None

        self._determine_protocol()
        self._open_trajectory()
        self._validate_file()

        # Gets some info about what settings the datasets were created with
        # from first available group
        for name in self._has:
            dset = self._particle_group[f"{name}/value"]
            self.n_atoms = dset.shape[1]
            self.compressor = dset.compressor
            self.chunks = dset.chunks
            self.filters = dset.filters
            break
        else:
            raise NoDataError(
                "Provide at least a position, velocity"
                " or force group in the h5md file."
            )

        self.ts = self._Timestep(
            self.n_atoms,
            positions=self.has_positions,
            velocities=self.has_velocities,
            forces=self.has_forces,
            **self._ts_kwargs,
        )

        self.units = {
            "time": None,
            "length": None,
            "velocity": None,
            "force": None,
        }
        self._set_translated_units()  # fills units dictionary

        # For testing, set the cache size to 100mb
        self._cache_size = 100 * 1024**2

        self._cache = self._cache_type(
            self._file, self._cache_size, self.ts, self.chunks[0], self._group
        )

        self._cache.get_first_frame()

    def _set_translated_units(self):
        """converts units from H5MD to MDAnalysis notation
        and fills units dictionary"""

        # need this dictionary to associate 'position': 'length'
        _group_unit_dict = {
            "time": "time",
            "position": "length",
            "velocity": "velocity",
            "force": "force",
        }

        for group, unit in _group_unit_dict.items():
            self._translate_h5md_units(group, unit)
            self._check_units(group, unit)

    def _translate_h5md_units(self, group, unit):
        """stores the translated unit string into the units dictionary"""

        errmsg = "{} unit '{}' is not recognized by H5MDReader. Please raise"
        " an issue in https://github.com/MDAnalysis/mdanalysis/issues"

        # doing time unit separately because time has to fish for
        # first available parent group - either position, velocity, or force
        if unit == "time":
            for name in self._has:
                if "unit" in self._particle_group[name]["time"].attrs:
                    try:
                        self.units["time"] = self._unit_translation["time"][
                            self._particle_group[name]["time"].attrs["unit"]
                        ]
                        break
                    except KeyError:
                        raise RuntimeError(
                            errmsg.format(
                                unit,
                                self._particle_group[name]["time"].attrs[
                                    "unit"
                                ],
                            )
                        ) from None

        else:
            if group in self._has:
                if "unit" in self._particle_group[group]["value"].attrs:
                    try:
                        self.units[unit] = self._unit_translation[unit][
                            self._particle_group[group]["value"].attrs["unit"]
                        ]
                    except KeyError:
                        raise RuntimeError(
                            errmsg.format(
                                unit,
                                self._particle_group[group]["value"].attrs[
                                    "unit"
                                ],
                            )
                        ) from None

            # if position group is not provided, can still get 'length' unit
            # from unitcell box
            if ("position" not in self._has) and (
                "edges" in self._particle_group["box"]
            ):
                if "unit" in self._particle_group["box/edges/value"].attrs:
                    try:
                        self.units["length"] = self._unit_translation["length"][
                            self._particle_group["box/edges/value"].attrs[
                                "unit"
                            ]
                        ]
                    except KeyError:
                        raise RuntimeError(
                            errmsg.format(
                                unit,
                                self._particle_group["box/edges/value"].attrs[
                                    "unit"
                                ],
                            )
                        ) from None

    def _check_units(self, group, unit):
        """Raises error if no units are provided from H5MD file
        and convert_units=True"""

        if not self.convert_units:
            return

        errmsg = "H5MD file must have readable units if ``convert_units`` is"
        " set to ``True``. MDAnalysis sets ``convert_units=True`` by default."
        " Set ``convert_units=False`` to load Universe without units."

        if unit == "time":
            if self.units["time"] is None:
                raise ValueError(errmsg)

        else:
            if self._has[group]:
                if self.units[unit] is None:
                    raise ValueError(errmsg)

    def _determine_protocol(self):
        """Determines the correct method for opening the file
        given the protocol and file extension"""

        self._mapping = None

        self._protocol = get_protocol(self.filename)
        if self._protocol == "s3":
            self._cache_type = ZarrLRUCache
            # NOTE: Import correct packages
        elif self._protocol == "file":
            self._cache_type = ZarrNoCache
        else:
            raise ValueError(
                f"Unsupported protocol '{self._protocol}' for H5MD file."
            )

        ext = get_extension(self.filename)
        if ext == ".zarr":
            self._mapping = zarr.storage.FSStore(
                self.filename, mode="r", **self._so
            )
        elif ext == ".h5md" or self._ext == ".h5":
            self._mapping = get_h5_zarr_mapping(
                self.filename, self._protocol, self._so
            )

    def _open_trajectory(self):
        """Opens the trajectory file using zarr library"""
        self._frame = -1

        self._file = zarr.open_group(self._mapping, mode="r")

    def _validate_file(self):
        """Validates the layout against H5MD standard as
        much as possible before handing off the open file
        to the cache object
        """
        if self._group is None:
            if len(self._file["particles"]) == 1:
                self._group = list(self._file["particles"])[0]
            else:
                raise ValueError(
                    "If `group` kwarg not provided, H5MD file must "
                    "contain exactly one group in 'particles'"
                )

        self._particle_group = self._file["particles"][self._group]

        self._has = {
            name
            for name in self._particle_group
            if name in ("position", "velocity", "force")
        }

        # _elements is a dictionary of
        # {h5md element name : dataset path in _file}
        self._elements = {}
        for elem in ("position", "velocity", "force"):
            if elem in self._particle_group:
                self._elements[elem] = self._particle_group[elem].name
        if (
            "box" in self._particle_group
            and "edges" in self._particle_group["box"]
        ):
            self._elements["edges"] = self._particle_group["box/edges"].name
        if "observables" in self._file:
            for obsv in self._file["observables"]:
                self._elements[obsv] = self._file["observables"][obsv].name
        elif "observables" in self._particle_group:
            for obsv in self._particle_group["observables"]:
                self._elements[obsv] = self._particle_group["observables"][
                    obsv
                ].name

        if self._particle_group["box"].attrs["dimension"] != 3:
            raise ValueError(
                "MDAnalysis only supports 3-dimensional" " simulation boxes"
            )

    def _read_next_timestep(self):
        """read next frame in trajectory"""
        if self._frame_seq is None:
            self._frame_seq = collections.deque(range(self.n_frames))
            print(self._frame_seq)
        return self._read_frame(self._frame + 1)

    def _read_frame(self, frame):
        """reads data from h5md file and copies to current timestep"""
        try:
            for name, value in self._has.items():
                if value:
                    _ = self._particle_group[name]["step"][frame]
                    break
            else:
                raise NoDataError(
                    "Provide at least a position, velocity"
                    " or force group in the h5md file."
                )
        except (ValueError, IndexError):
            raise IOError from None

        self._frame = frame
        ts = self.ts
        particle_group = self._particle_group
        ts.frame = frame

        # fills data dictionary from 'observables' group
        # Note: dt is not read into data as it is not decided whether
        # Timestep should have a dt attribute (see Issue #2825)
        self._copy_to_data()

        # Sets frame box dimensions
        # Note: H5MD files must contain 'box' group in each 'particles' group
        if "edges" in particle_group["box"]:
            edges = particle_group["box/edges/value"][frame, :]
            # A D-dimensional vector or a D × D matrix, depending on the
            # geometry of the box, of Float or Integer type. If edges is a
            # vector, it specifies the space diagonal of a cuboid-shaped box.
            # If edges is a matrix, the box is of triclinic shape with the edge
            # vectors given by the rows of the matrix.
            if edges.shape == (3,):
                ts.dimensions = [*edges, 90, 90, 90]
            else:
                ts.dimensions = core.triclinic_box(*edges)
        else:
            ts.dimensions = None

        # set the timestep positions, velocities, and forces with
        # current frame dataset
        if self._has["position"]:
            self._read_dataset_into_ts("position", ts.positions)
        if self._has["velocity"]:
            self._read_dataset_into_ts("velocity", ts.velocities)
        if self._has["force"]:
            self._read_dataset_into_ts("force", ts.forces)

        if self.convert_units:
            self._convert_units()

        return ts

    def _copy_to_data(self):
        """assigns values to keys in data dictionary"""

        if "observables" in self._file:
            for key in self._file["observables"].keys():
                self.ts.data[key] = self._file["observables"][key]["value"][
                    self._frame
                ]

        # pulls 'time' and 'step' out of first available parent group
        for name, value in self._has.items():
            if value:
                if "time" in self._particle_group[name]:
                    self.ts.time = self._particle_group[name]["time"][
                        self._frame
                    ]
                    break
        for name, value in self._has.items():
            if value:
                if "step" in self._particle_group[name]:
                    self.ts.data["step"] = self._particle_group[name]["step"][
                        self._frame
                    ]
                    break

    def _read_dataset_into_ts(self, dataset, attribute):
        """reads position, velocity, or force dataset array at current frame
        into corresponding ts attribute"""

        n_atoms_now = self._particle_group[f"{dataset}/value"][
            self._frame
        ].shape[0]
        if n_atoms_now != self.n_atoms:
            raise ValueError(
                f"Frame {self._frame} of the {dataset} dataset"
                f" has {n_atoms_now} atoms but the initial frame"
                " of either the postion, velocity, or force"
                f" dataset had {self.n_atoms} atoms."
                " MDAnalysis is unable to deal"
                " with variable topology!"
            )

        self._particle_group[f"{dataset}/value"].read_direct(
            attribute, source_sel=np.s_[self._frame, :]
        )

    def _convert_units(self):
        """converts time, position, velocity, and force values if they
        are not given in MDAnalysis standard units

        See https://userguide.mdanalysis.org/1.0.0/units.html
        """

        self.ts.time = self.convert_time_from_native(self.ts.time)

        if (
            "edges" in self._particle_group["box"]
            and self.ts.dimensions is not None
        ):
            self.convert_pos_from_native(self.ts.dimensions[:3])

        if self._has["position"]:
            self.convert_pos_from_native(self.ts.positions)

        if self._has["velocity"]:
            self.convert_velocities_from_native(self.ts.velocities)

        if self._has["force"]:
            self.convert_forces_from_native(self.ts.forces)

    def close(self):
        """close reader"""
        self._file.close()

    def _reopen(self):
        """reopen trajectory

        Note
        ----

        If the `driver` and `comm` arguments were used to open the
        hdf5 file (specifically, ``driver="mpio"``) then this method
        does *not* close and open the file like most readers because
        the information about the MPI communicator would be lost; instead
        it rewinds the trajectory back to the first timstep.

        """
        self.close()
        self._open_trajectory()

    def Writer(self, filename, n_atoms=None, **kwargs):
        """Return writer for trajectory format

        Note
        ----
        The chunk shape of the input file will not be copied to the output
        file, as :class:`H5MDWriter` uses a chunk shape of ``(1, n_atoms, 3)``
        by default. To use a custom chunk shape, you must specify the
        `chunks` argument. If you would like to copy an existing chunk
        format from a dataset (positions, velocities, or forces), do
        the following::

            chunks = u.trajectory._particle_group['position/value'].chunks

        Note that the writer will set the same layout for all particle groups.

        See Also
        --------
        :class:`H5MDWriter`  Output class for the H5MD format


        .. versionadded:: 2.0.0

        """
        if n_atoms is None:
            n_atoms = self.n_atoms
        kwargs.setdefault("driver", self._driver)
        kwargs.setdefault("compression", self.compression)
        kwargs.setdefault("compression_opts", self.compression_opts)
        kwargs.setdefault("positions", self.has_positions)
        kwargs.setdefault("velocities", self.has_velocities)
        kwargs.setdefault("forces", self.has_forces)
        return H5MDWriter(filename, n_atoms, **kwargs)

    def __getitem__(self, frame):
        """Return the Timestep corresponding to *frame*.

        If `frame` is a integer then the corresponding frame is
        returned. Negative numbers are counted from the end.

        If frame is a :class:`slice` then an iterator is returned that
        allows iteration over that part of the trajectory.

        Note
        ----
        *frame* is a 0-based frame index.

        Note
        ----
        ZarrtrajReader overrides this to get
        access to the the sequence of frames
        the user wants. If self._frame_seq is None
        by the time the first read is called, we assume
        the full full trajectory is being accessed
        in a for loop
        """
        if isinstance(frame, numbers.Integral):
            frame = self._apply_limits(frame)
            if self._frame_seq is None:
                self._frame_seq = collections.deque([frame])
                print(self._frame_seq)
            return self._read_frame_with_aux(frame)
        elif isinstance(frame, (list, np.ndarray)):
            if len(frame) != 0 and isinstance(frame[0], (bool, np.bool_)):
                # Avoid having list of bools
                frame = np.asarray(frame, dtype=bool)
                # Convert bool array to int array
                frame = np.arange(len(self))[frame]
            if isinstance(frame, np.ndarray):
                frame = frame.tolist()
            if self._frame_seq is None:
                self._frame_seq = collections.deque(frame)
                print(self._frame_seq)
            return base.FrameIteratorIndices(self, frame)
        elif isinstance(frame, slice):
            start, stop, step = self.check_slice_indices(
                frame.start, frame.stop, frame.step
            )
            if self._frame_seq is None:
                self._frame_seq = collections.deque(range(start, stop, step))
                print(self._frame_seq)
            if start == 0 and stop == len(self) and step == 1:
                return base.FrameIteratorAll(self)
            else:
                return base.FrameIteratorSliced(self, frame)
        else:
            raise TypeError(
                "Trajectories must be an indexed using an integer,"
                " slice or list of indices"
            )

    @property
    def n_frames(self):
        """number of frames in trajectory"""
        for name, value in self._has.items():
            if value:
                return self._particle_group[name]["value"].shape[0]

    @staticmethod
    def _format_hint(thing):
        """Can this Reader read *thing*"""
        # nb, filename strings can still get passed through if
        # format='H5MD' is used
        return HAS_H5PY and isinstance(thing, h5py.File)

    @staticmethod
    def parse_n_atoms(filename, group=None, so=None):
        mapping = None
        so = so if so is not None else {}

        protocol = get_protocol(filename)
        ext = get_extension(filename)

        if ext == ".zarr":
            mapping = zarr.storage.FSStore(filename, mode="r", **so)
        elif ext == ".h5md" or ext == ".h5":
            mapping = get_h5_zarr_mapping(filename, protocol, so)
        file = zarr.open_group(mapping, mode="r")
        if group is None:
            if len(file["particles"]) == 1:
                group = list(file["particles"])[0]
            else:
                raise ValueError(
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


class H5MDWriter(base.WriterBase):  #
    """Writer for `H5MD`_ format (version 1.1).

    H5MD trajectories are automatically recognised by the
    file extension ".h5md".

    All data from the input :class:`~MDAnalysis.coordinates.timestep.Timestep` is
    written by default. For detailed information on how :class:`H5MDWriter`
    handles units, compression, and chunking, see the Notes section below.

    Note
    ----
    Parellel writing with the use of a MPI communicator and the ``'mpio'``
    HDF5 driver is currently not supported.

    Note
    ----
    :exc:`NoDataError` is raised if no positions, velocities, or forces are
    found in the input trajectory. While the H5MD standard allows for this
    case, :class:`H5MDReader` cannot currently read files without at least
    one of these three groups.

    Note
    ----
    Writing H5MD files with fancy trajectory slicing where the Timestep
    does not increase monotonically such as ``u.trajectory[[2,1,0]]``
    or ``u.trajectory[[0,1,2,0,1,2]]`` raises a :exc:`ValueError` as this
    violates the rules of the step dataset in the H5MD standard.

    Parameters
    ----------
    filename : str or :class:`h5py.File`
        trajectory filename or open h5py file
    n_atoms : int
        number of atoms in trajectory
    n_frames : int (optional)
        number of frames to be written in trajectory
    driver : str (optional)
        H5PY file driver used to open H5MD file. See `H5PY drivers`_ for
        list of available drivers.
    convert_units : bool (optional)
        Convert units from MDAnalysis to desired units
    chunks : tuple (optional)
        Custom chunk layout to be applied to the position,
        velocity, and force datasets. By default, these datasets
        are chunked in ``(1, n_atoms, 3)`` blocks
    compression : str or int (optional)
        HDF5 dataset compression setting to be applied
        to position, velocity, and force datasets. Allowed
        settings are 'gzip', 'szip', 'lzf'. If an integer
        in range(10), this indicates gzip compression level.
        Otherwise, an integer indicates the number of a
        dynamically loaded compression filter.
    compression_opts : int or tup (optional)
        Compression settings.  This is an integer for gzip, 2-tuple for
        szip, etc. If specifying a dynamically loaded compression filter
        number, this must be a tuple of values. For gzip, 1 indicates
        the lowest level of compression and 9 indicates maximum compression.
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
        when `H5PY`_ is not installed
    ValueError
        when `n_atoms` is 0
    ValueError
        when ``chunks=False`` but the user did not specify `n_frames`
    ValueError
        when `positions`, `velocities`, and `forces` are all
        set to ``False``
    TypeError
        when the input object is not a :class:`Universe` or
        :class:`AtomGroup`
    IOError
        when `n_atoms` of the :class:`Universe` or :class:`AtomGroup`
        being written does not match `n_atoms` passed as an argument
        to the writer
    ValueError
        when any of the optional `timeunit`, `lengthunit`,
        `velocityunit`, or `forceunit` keyword arguments are
        not recognized by MDAnalysis

    Notes
    -----

    By default, the writer will write all available data (positions,
    velocities, and forces) if detected in the input
    :class:`~MDAnalysis.coordinates.timestep.Timestep`. In addition, the settings
    for `compression` and `compression_opts` will be read from
    the first available group of positions, velocities, or forces and used as
    the default value. To write a file without any one of these datsets,
    set `positions`, `velocities`, or `forces` to ``False``.

    .. rubric:: Units

    The H5MD format is very flexible with regards to units, as there is no
    standard defined unit for the format. For this reason, :class:`H5MDWriter`
    does not enforce any units. The units of the written trajectory can be set
    explicitly with the keyword arguments `lengthunit`, `velocityunit`,
    and `forceunit`. If units are not explicitly specified, they are set to
    the native units of the trajectory that is the source of the coordinates.
    For example, if one converts a DCD trajectory, then positions are written
    in ångstrom and time in AKMA units. A GROMACS XTC will be written in nm and
    ps. The units are stored in the metadata of the H5MD file so when
    MDAnalysis loads the H5MD trajectory, the units will be automatically
    set correctly.

    .. rubric:: Compression

    HDF5 natively supports various compression modes. To write the trajectory
    with compressed datasets, set ``compression='gzip'``, ``compression='lzf'``
    , etc. See `H5PY compression options`_ for all supported modes of
    compression. An additional argument, `compression_opts`, can be used to
    fine tune the level of compression. For example, for GZIP compression,
    `compression_opts` can be set to 1 for minimum compression and 9 for
    maximum compression.

    .. rubric:: HDF5 Chunking

    HDF5 datasets can be *chunked*, meaning the dataset can be split into equal
    sized pieces and stored in different, noncontiguous places on disk.
    If HDF5 tries to read an element from a chunked dataset, the *entire*
    dataset must be read, therefore an ill-thought-out chunking scheme can
    drastically effect file I/O performance. In the case of all MDAnalysis
    writers, in general, the number of frames being written is not known
    apriori by the writer, therefore the HDF5 must be extendable. However, the
    allocation of diskspace is defined when the dataset is created, therefore
    extendable HDF5 datasets *must* be chunked so as to allow dynamic storage
    on disk of any incoming data to the writer. In such cases where chunking
    isn't explicity defined by the user, H5PY automatically selects a chunk
    shape via an algorithm that attempts to make mostly square chunks between
    1 KiB - 1 MiB, however this can lead to suboptimal I/O performance.
    :class:`H5MDWriter` uses a default chunk shape of ``(1, n_atoms, 3)`` so
    as to mimic the typical access pattern of a trajectory by MDAnalysis. In
    our tests ([Jakupovic2021]_), this chunk shape led to a speedup on the
    order of 10x versus H5PY's auto-chunked shape. Users can set a custom
    chunk shape with the `chunks` argument. Additionaly, the datasets in a
    file can be written with a contiguous layout by setting ``chunks=False``,
    however this must be accompanied by setting `n_frames` equal to the
    number of frames being written, as HDF5 must know how much space to
    allocate on disk when creating the dataset.

    .. _`H5PY compression options`: https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
    .. _`H5PY drivers`: https://docs.h5py.org/en/stable/high/file.html#file-drivers


    .. versionadded:: 2.0.0

    """

    format = "H5MD"
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
    def __init__(
        self,
        filename,
        n_atoms,
        n_frames=None,
        driver=None,
        convert_units=True,
        chunks=None,
        compression=None,
        compression_opts=None,
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

        if not HAS_H5PY:
            raise RuntimeError("H5MDWriter: Please install h5py")
        self.filename = filename
        if n_atoms == 0:
            raise ValueError("H5MDWriter: no atoms in output trajectory")
        self._driver = driver
        if self._driver == "mpio":
            raise ValueError(
                "H5MDWriter: parallel writing with MPI I/O "
                "is not currently supported."
            )
        self.n_atoms = n_atoms
        self.n_frames = n_frames
        self.chunks = (1, n_atoms, 3) if chunks is None else chunks
        if self.chunks is False and self.n_frames is None:
            raise ValueError(
                "H5MDWriter must know how many frames will be "
                "written if ``chunks=False``."
            )
        self.contiguous = self.chunks is False and self.n_frames is not None
        self.compression = compression
        self.compression_opts = compression_opts
        self.convert_units = convert_units
        self.h5md_file = None

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
        if self.h5md_file is None:
            self._determine_units(ag)
            self._open_file()
            self._initialize_hdf5_datasets(ts)

        return self._write_next_timestep(ts)

    def _determine_units(self, ag):
        """determine which units the file will be written with

        By default, it fills the :attr:`self.units` dictionary by copying the
        units dictionary of the parent file. Because H5MD files have no
        standard unit restrictions, users may pass a kwarg in ``(timeunit,
        lengthunit, velocityunit, forceunit)`` to the writer so long as
        MDAnalysis has a conversion factor for it (:exc:`ValueError` raised if
        it does not). These custom unit arguments must be in
        `MDAnalysis notation`_. If custom units are supplied from the user,
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
        """Opens file with `H5PY`_ library and fills in metadata from kwargs.

        :attr:`self.h5md_file` becomes file handle that links to root level.

        """

        self.h5md_file = h5py.File(
            name=self.filename, mode="w", driver=self._driver
        )

        # fill in H5MD metadata from kwargs
        self.h5md_file.require_group("h5md")
        self.h5md_file["h5md"].attrs["version"] = np.array(self.H5MD_VERSION)
        self.h5md_file["h5md"].require_group("author")
        self.h5md_file["h5md/author"].attrs["name"] = self.author
        if self.author_email is not None:
            self.h5md_file["h5md/author"].attrs["email"] = self.author_email
        self.h5md_file["h5md"].require_group("creator")
        self.h5md_file["h5md/creator"].attrs["name"] = self.creator
        self.h5md_file["h5md/creator"].attrs["version"] = self.creator_version

    def _initialize_hdf5_datasets(self, ts):
        """initializes all datasets that will be written to by
        :meth:`_write_next_timestep`

        Note
        ----
        :exc:`NoDataError` is raised if no positions, velocities, or forces are
        found in the input trajectory. While the H5MD standard allows for this
        case, :class:`H5MDReader` cannot currently read files without at least
        one of these three groups. A future change to both the reader and
        writer will allow this case.


        """

        # for keeping track of where to write in the dataset
        self._counter = 0

        # ask the parent file if it has positions, velocities, and forces
        # if prompted by the writer with the self._write_* attributes
        self._has = {
            group: (
                getattr(ts, f"has_{attr}")
                if getattr(self, f"_write_{attr}")
                else False
            )
            for group, attr in zip(
                ("position", "velocity", "force"),
                ("positions", "velocities", "forces"),
            )
        }

        # initialize trajectory group
        self.h5md_file.require_group("particles").require_group("trajectory")
        self._traj = self.h5md_file["particles/trajectory"]
        self.data_keys = [
            key for key in ts.data.keys() if key not in self.data_blacklist
        ]
        if self.data_keys:
            self._obsv = self.h5md_file.require_group("observables")

        # box group is required for every group in 'particles'
        self._traj.require_group("box")
        self._traj["box"].attrs["dimension"] = 3
        if ts.dimensions is not None and np.all(ts.dimensions > 0):
            self._traj["box"].attrs["boundary"] = 3 * ["periodic"]
            self._traj["box"].require_group("edges")
            self._edges = self._traj.require_dataset(
                "box/edges/value",
                shape=(0, 3, 3),
                maxshape=(None, 3, 3),
                dtype=np.float32,
            )
            self._step = self._traj.require_dataset(
                "box/edges/step", shape=(0,), maxshape=(None,), dtype=np.int32
            )
            self._time = self._traj.require_dataset(
                "box/edges/time", shape=(0,), maxshape=(None,), dtype=np.float32
            )
            self._set_attr_unit(self._edges, "length")
            self._set_attr_unit(self._time, "time")
        else:
            # if no box, boundary attr must be "none" according to H5MD
            self._traj["box"].attrs["boundary"] = 3 * ["none"]
            self._create_step_and_time_datasets()

        if self.has_positions:
            self._create_trajectory_dataset("position")
            self._pos = self._traj["position/value"]
            self._set_attr_unit(self._pos, "length")
        if self.has_velocities:
            self._create_trajectory_dataset("velocity")
            self._vel = self._traj["velocity/value"]
            self._set_attr_unit(self._vel, "velocity")
        if self.has_forces:
            self._create_trajectory_dataset("force")
            self._force = self._traj["force/value"]
            self._set_attr_unit(self._force, "force")

        # intialize observable datasets from ts.data dictionary that
        # are NOT in self.data_blacklist
        if self.data_keys:
            for key in self.data_keys:
                self._create_observables_dataset(key, ts.data[key])

    def _create_step_and_time_datasets(self):
        """helper function to initialize a dataset for step and time

        Hunts down first available location to create the step and time
        datasets. This should only be called if the trajectory has no
        dimension, otherwise the 'box/edges' group creates step and time
        datasets since 'box' is the only required group in 'particles'.

        :attr:`self._step` and :attr`self._time` serve as links to the created
        datasets that other datasets can also point to for their step and time.
        This serves two purposes:
            1. Avoid redundant writing of multiple datasets that share the
               same step and time data.
            2. In HDF5, each chunked dataset has a cache (default 1 MiB),
               so only 1 read is required to access step and time data
               for all datasets that share the same step and time.

        """

        for group, value in self._has.items():
            if value:
                self._step = self._traj.require_dataset(
                    f"{group}/step",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int32,
                )
                self._time = self._traj.require_dataset(
                    f"{group}/time",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float32,
                )
                self._set_attr_unit(self._time, "time")
                break

    def _create_trajectory_dataset(self, group):
        """helper function to initialize a dataset for
        position, velocity, and force"""

        if self.n_frames is None:
            shape = (0, self.n_atoms, 3)
            maxshape = (None, self.n_atoms, 3)
        else:
            shape = (self.n_frames, self.n_atoms, 3)
            maxshape = None

        chunks = None if self.contiguous else self.chunks

        self._traj.require_group(group)
        self._traj.require_dataset(
            f"{group}/value",
            shape=shape,
            maxshape=maxshape,
            dtype=np.float32,
            chunks=chunks,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        if "step" not in self._traj[group]:
            self._traj[f"{group}/step"] = self._step
        if "time" not in self._traj[group]:
            self._traj[f"{group}/time"] = self._time

    def _create_observables_dataset(self, group, data):
        """helper function to initialize a dataset for each observable"""

        self._obsv.require_group(group)
        # guarantee ints and floats have a shape ()
        data = np.asarray(data)
        self._obsv.require_dataset(
            f"{group}/value",
            shape=(0,) + data.shape,
            maxshape=(None,) + data.shape,
            dtype=data.dtype,
        )
        if "step" not in self._obsv[group]:
            self._obsv[f"{group}/step"] = self._step
        if "time" not in self._obsv[group]:
            self._obsv[f"{group}/time"] = self._time

    def _set_attr_unit(self, dset, unit):
        """helper function to set a 'unit' attribute for an HDF5 dataset"""

        if self.units[unit] is None:
            return

        dset.attrs["unit"] = self._unit_translation_dict[unit][self.units[unit]]

    def _write_next_timestep(self, ts):
        """Write coordinates and unitcell information to H5MD file.

        Do not call this method directly; instead use
        :meth:`write` because some essential setup is done
        there before writing the first frame.

        The first dimension of each dataset is extended by +1 and
        then the data is written to the new slot.

        Note
        ----
        Writing H5MD files with fancy trajectory slicing where the Timestep
        does not increase monotonically such as ``u.trajectory[[2,1,0]]``
        or ``u.trajectory[[0,1,2,0,1,2]]`` raises a :exc:`ValueError` as this
        violates the rules of the step dataset in the H5MD standard.

        """

        i = self._counter

        # H5MD step refers to the integration step at which the data were
        # sampled, therefore ts.data['step'] is the most appropriate value
        # to use. However, step is also necessary in H5MD to allow
        # temporal matching of the data, so ts.frame is used as an alternative
        self._step.resize(self._step.shape[0] + 1, axis=0)
        try:
            self._step[i] = ts.data["step"]
        except KeyError:
            self._step[i] = ts.frame
        if len(self._step) > 1 and self._step[i] < self._step[i - 1]:
            raise ValueError(
                "The H5MD standard dictates that the step "
                "dataset must increase monotonically in value."
            )

        # the dataset.resize() method should work with any chunk shape
        self._time.resize(self._time.shape[0] + 1, axis=0)
        self._time[i] = ts.time

        if "edges" in self._traj["box"]:
            self._edges.resize(self._edges.shape[0] + 1, axis=0)
            self._edges.write_direct(
                ts.triclinic_dimensions, dest_sel=np.s_[i, :]
            )
        # These datasets are not resized if n_frames was provided as an
        # argument, as they were initialized with their full size.
        if self.has_positions:
            if self.n_frames is None:
                self._pos.resize(self._pos.shape[0] + 1, axis=0)
            self._pos.write_direct(ts.positions, dest_sel=np.s_[i, :])
        if self.has_velocities:
            if self.n_frames is None:
                self._vel.resize(self._vel.shape[0] + 1, axis=0)
            self._vel.write_direct(ts.velocities, dest_sel=np.s_[i, :])
        if self.has_forces:
            if self.n_frames is None:
                self._force.resize(self._force.shape[0] + 1, axis=0)
            self._force.write_direct(ts.forces, dest_sel=np.s_[i, :])

        if self.data_keys:
            for key in self.data_keys:
                obs = self._obsv[f"{key}/value"]
                obs.resize(obs.shape[0] + 1, axis=0)
                obs[i] = ts.data[key]

        if self.convert_units:
            self._convert_dataset_with_units(i)

        self._counter += 1

    def _convert_dataset_with_units(self, i):
        """convert values in the dataset arrays with self.units dictionary"""

        # Note: simply doing convert_pos_to_native(self._pos[-1]) does not
        # actually change the values in the dataset, so assignment required
        if self.units["time"] is not None:
            self._time[i] = self.convert_time_to_native(self._time[i])
        if self.units["length"] is not None:
            if self._has["position"]:
                self._pos[i] = self.convert_pos_to_native(self._pos[i])
            if "edges" in self._traj["box"]:
                self._edges[i] = self.convert_pos_to_native(self._edges[i])
        if self._has["velocity"]:
            if self.units["velocity"] is not None:
                self._vel[i] = self.convert_velocities_to_native(self._vel[i])
        if self._has["force"]:
            if self.units["force"] is not None:
                self._force[i] = self.convert_forces_to_native(self._force[i])

    @property
    def has_positions(self):
        """``True`` if writer is writing positions from Timestep."""
        return self._has["position"]

    @property
    def has_velocities(self):
        """``True`` if writer is writing velocities from Timestep."""
        return self._has["velocity"]

    @property
    def has_forces(self):
        """``True`` if writer is writing forces from Timestep."""
        return self._has["force"]


class ZarrNoCache(FrameCache):
    """No caching for Zarr-interfaced reads
    Used for reading H5MD formatted files from disk"""

    def __init__(
        self, open_file, cache_size, timestep, frames_per_chunk, group
    ):
        super().__init__(open_file, cache_size, timestep, frames_per_chunk)
        self._group = group
        self._particle_group = self._file["particles"][group]

    def update_desired_dsets(self, dsets: dict):
        self._dsets = dsets
        self._step_maps = {dset: {} for dset in self._dsets.keys()}
        # A step map is a map from an H5MDElements' step values
        # i.e. element["step"][:]
        # to the index at which you can find data for that step
        # i.e to check if there is position data at step x and then grab it,
        # you would do
        # if step in self._step_maps["position"]:
        #   step_map = self._step_maps["position"]
        #   pos = position["value"][step_map[step]]
        # "all" is a special value that means all steps
        self._construct_step_time_lists()

    def _construct_step_time_lists(self):
        """Constructs a list of 1. unique steps and 2. unique times
        from the H5MD file

        reader[frame] will return a timestep filled with
        data from all elements that contain data at self._step_list[frame]"""
        all_elem_steps = []
        all_elem_times = []
        for dset, zarrpath in self._dsets.items():
            if (
                "step" in self._file[zarrpath]
                and "time" in self._file[zarrpath]
            ):
                step = self._file[zarrpath]["step"][:]
                all_elem_steps.append(step)
                time = self._file[zarrpath]["time"][:]
                all_elem_times.append(time)
                self._step_maps[dset] = {
                    step_val: i for i, step_val in enumerate(step)
                }
        self._step_list = np.unique(np.concatenate(all_elem_steps))
        self._time_list = np.unique(np.concatenate(all_elem_times))

    def get_first_frame(self):
        # Determine which datasets have values for this timestep
        self._timestep
        # Read all data into the timestep


class ZarrLRUCache(FrameCache):
    pass
