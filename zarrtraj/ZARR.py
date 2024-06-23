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
        # Set to none so close() can be called
        self._file = None
        self._cache = None
        # Read first timestep
        self._frame_seq = collections.deque([0])
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

        # Though the user may not want all elements and therefore
        # won't want a steplist constructed from all elements,
        # this will cover the majority of cases.
        # For cases where the user wants only a subset of the elements,
        # a new steplist and dicts can be created

        self._global_steparray = create_steplist(
            [h5mdelement.step for h5mdelement in self._elements.values()]
        )

        self._stepmap = create_stepmap(
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
            self._stepmap,
        )
        self._cache.update_frame_seq(self._frame_seq)
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
                    raise RuntimeError(
                        f"length unit '{self._elements[elem].valueunit}' "
                        "is not recognized by ZarrH5MDReader. "
                    )
                l_units.append(len_unit)

        if all(l_unit == l_units[0] for l_unit in l_units):
            self.units["length"] = l_units[0]
        else:
            raise RuntimeError(
                "Length units of position and box/edges do not match"
            )

        if "velocity" in self._elements:
            try:
                self.units["velocity"] = self._unit_translation["velocity"][
                    self._elements["velocity"].valueunit
                ]
            except KeyError:
                raise RuntimeError(
                    f"velocity unit '{self._elements['velocity'].valueunit}' "
                    "is not recognized by ZarrH5MDReader. "
                )

        if "force" in self._elements:
            try:
                self.units["force"] = self._unit_translation["force"][
                    self._elements["force"].valueunit
                ]
            except KeyError:
                raise RuntimeError(
                    f"force unit '{self._elements['force'].valueunit}' "
                    "is not recognized by ZarrH5MDReader. "
                )

        t_units = []
        for elem in self._elements:
            if self._elements[elem].has_time:
                try:
                    time_unit = self._unit_translation["time"][
                        self._elements[elem].timeunit
                    ]
                except KeyError:
                    raise RuntimeError(
                        f"time unit '{self._elements[elem].timeunit}' "
                        "is not recognized by ZarrH5MDReader. "
                    )
                t_units.append(time_unit)

        if all(t_unit == t_units[0] for t_unit in t_units):
            self.units["time"] = t_units[0]

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
        if self._protocol == "s3":
            self._cache_type = ZarrLRUCache
            # NOTE: Import correct packages
        elif self._protocol == "file":
            self._cache_type = ZarrNoCache
        else:
            raise ValueError(
                f"Unsupported protocol '{self._protocol}' for H5MD file."
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

        if self._group is None:
            if len(self._file["particles"]) == 1:
                self._group = list(self._file["particles"])[0]
            else:
                raise ValueError(
                    "If `group` kwarg not provided, H5MD file must "
                    "contain exactly one group in 'particles'"
                )

        particle_group = self._file["particles"][self._group]
        self._elements = dict()

        if "observables" in self._file:
            for obsv in self._file["observables"]:
                if "value" in obsv:
                    self._elements[obsv.basename] = H5MDElement(obsv)
            if self._group in self._file["observables"]:
                for obsv in self._file["observables"][self._group]:
                    if "value" in obsv:
                        self._elements[f"{self._group}/{obsv.name}"] = (
                            H5MDElement(obsv)
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

    def _read_next_timestep(self):
        """read next frame in trajectory"""
        # if len(self._frame_seq) == 0 and self._frame == -1:
        #    self._frame_seq = collections.deque(range(0, self.n_frames))
        # if self._frame == -1:
        #    # If loading first frame or rewinding, inform cache
        #    # we want first frame
        #    # If iterating normally, inform cache of sequence as well
        #    self._cache.update_frame_seq(self._frame_seq)

        return self._read_frame(self._frame + 1)

    def _read_frame(self, frame):
        """reads data from h5md file and copies to current timestep"""
        # frame seq update case 1: read called from iterator-like context
        if not self._frame_seq:
            self._frame_seq = None
            self._cache.update_frame_seq(self._frame_seq)
            raise StopIteration

        self._frame = self._cache.load_frame()

        if self.ts.time == -1.0:
            raise ValueError(
                "MDAnalysis requires that time data is available "
                "for each integration step in the trajectory"
            )

        if self.convert_units:
            self._convert_units()

        # frame seq update case 1: read called from __getitem__-like context
        if len(self._frame_seq) == 0:
            self._frame_seq = None
            self._cache.update_frame_seq(self._frame_seq)

        return self.ts

    def _convert_units(self):
        """converts time, position, velocity, and force values if they
        are not given in MDAnalysis standard units

        See https://userguide.mdanalysis.org/1.0.0/units.html
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
        self._frame_seq = None
        if self._cache is not None:
            self._cache.cleanup()
        if self._file is not None:
            self._file.store.close()

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
        kwargs.setdefault("compressor", self.compressor)
        kwargs.setdefault("filters", self.filters)
        kwargs.setdefault("chunks", self.chunks)
        kwargs.setdefault("positions", ("position" in self._elements))
        kwargs.setdefault("velocities", ("velocity" in self._elements))
        kwargs.setdefault("forces", ("force" in self._elements))
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
        ZarrtrajReader overrides this method to get
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
                self._cache.update_frame_seq(self._frame_seq)
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
                self._cache.update_frame_seq(self._frame_seq)
            return base.FrameIteratorIndices(self, frame)
        elif isinstance(frame, slice):
            start, stop, step = self.check_slice_indices(
                frame.start, frame.stop, frame.step
            )
            if self._frame_seq is None:
                self._frame_seq = collections.deque(range(start, stop, step))
                self._cache.update_frame_seq(self._frame_seq)
            if start == 0 and stop == len(self) and step == 1:
                return base.FrameIteratorAll(self)
            else:
                return base.FrameIteratorSliced(self, frame)
        else:
            raise TypeError(
                "Trajectories must be an indexed using an integer,"
                " slice or list of indices"
            )

    def __iter__(self):
        """Iterate over all frames in the trajectory"""
        self._reopen()
        self._frame_seq = collections.deque(range(0, self.n_frames))
        self._cache.update_frame_seq(self._frame_seq)
        return self

    def next(self):
        if self._frame_seq is None and self._frame + 1 < self.n_frames:
            self._frame_seq = collections.deque([self._frame + 1])
            self._cache.update_frame_seq(self._frame_seq)
        elif self._frame_seq is None:
            self.rewind()
            raise StopIteration from None
        try:
            ts = self._read_next_timestep()
        except (EOFError, IOError):
            self.rewind()
            raise StopIteration from None
        else:
            for auxname, reader in self._auxs.items():
                ts = self._auxs[auxname].update_ts(ts)

            ts = self._apply_transformations(ts)

        return ts

    def iter_as_aux(self, auxname):
        """Iterate over the trajectory with an auxiliary reader"""
        aux = self._check_for_aux(auxname)
        self._reopen()
        self._frame_seq = collections.deque(range(0, self.n_frames))
        self._cache.update_frame_seq(self._frame_seq)
        aux._restart()
        while True:
            try:
                yield self.next_as_aux(auxname)
            except StopIteration:
                return

    def copy(self):
        """Return independent copy of this Reader.

        New Reader will have its own file handle and can seek/iterate
        independently of the original.

        Will also copy the current state of the Timestep held in the original
        Reader.


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

    @property
    def n_frames(self):
        """number of frames in trajectory"""
        for elem, h5mdelement in self._elements.items():
            if elem in ("position", "velocity", "force"):
                return h5mdelement.value.shape[0]

    @staticmethod
    def _format_hint(thing):
        """Can this Reader read *thing*"""
        # nb, filename strings can still get passed through if
        # format='H5MD' is used
        pass

    @staticmethod
    def parse_n_atoms(filename, group=None, so=None):
        so = so if so is not None else {}
        protocol = get_protocol(filename)
        ext = get_extension(filename)
        mapping = get_mapping_for(filename, protocol, ext, so)
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


class H5MDWriter(base.WriterBase):
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
        self,
        open_file,
        cache_size,
        timestep,
        frames_per_chunk,
        elements,
        global_steparray,
        stepmap,
    ):
        super().__init__(open_file, cache_size, timestep, frames_per_chunk)
        self._elements = elements
        self._global_steparray = global_steparray
        self._stepmap = stepmap

    def update_desired_dsets(
        self,
        elements: Dict[str, H5MDElement],
        global_steparray: np.ndarray,
        stepmap: Dict[int, Dict[int, int]],
    ):
        self._elements = elements
        self._global_steparray = global_steparray
        self._stepmap = stepmap

    def load_frame(self):
        """Reader responsbile for raising StopIteration when no more frames"""
        frame = self._frame_seq.popleft()
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
            if step in self._stepmap["box/edges"]:
                edges_index = self._stepmap["box/edges"][step]
                edges = self._elements["box/edges"].value[edges_index]
                if edges.shape == (3,):
                    self._timestep.dimensions = [*edges, 90, 90, 90]
                else:
                    self._timestep.dimensions = core.triclinic_box(*edges)
                if curr_time is None and self._elements["box/edges"].has_time():
                    curr_time = self._elements["box/edges"].time[edges_index]
            else:
                self._timestep.dimensions = None

        if "position" in self._elements:
            if step in self._stepmap["position"]:
                self._timestep.has_positions = True
                pos_index = self._stepmap["position"][step]
                self._timestep.positions = self._elements["position"].value[
                    pos_index
                ]
                if curr_time is None and self._elements["position"].has_time():
                    curr_time = self._elements["position"].time[pos_index]
            else:
                self._timestep.has_positions = False

        if "velocity" in self._elements:
            if step in self._stepmap["velocity"]:
                self._timestep.has_velocities = True
                vel_index = self._stepmap["velocity"][step]
                self._timestep.velocities = self._elements["velocity"].value[
                    vel_index
                ]
                if curr_time is None and self._elements["velocity"].has_time():
                    curr_time = self._elements["velocity"].time[vel_index]
            else:
                self._timestep.has_velocities = False

        if "force" in self._elements:
            if step in self._stepmap["force"]:
                self._timestep.has_forces = True
                force_index = self._stepmap["force"][step]
                self._timestep.forces = self._elements["force"].value[
                    force_index
                ]
                if curr_time is None and self._elements["force"].has_time():
                    curr_time = self._elements["force"].time[force_index]
            else:
                self._timestep.has_forces = False

        if curr_time is not None:
            self._timestep.time = curr_time
        else:
            self._timestep.time = -1.0

        exclude = {"position", "velocity", "force", "box/edges"}

        for elem, h5mdelement in self._elements.items():
            if elem not in exclude:
                if step in self._stepmap[elem]:
                    obsv_index = self._stepmap[elem][step]
                    self._timestep.data[elem] = h5mdelement.value[obsv_index]
                else:
                    # must be time independent
                    self._timestep.data[elem] = h5mdelement.value[()]

    def cleanup(self):
        pass

class ZarrLRUCache(ZarrNoCache):
    """Clone of ZarrNoCache to allow differentiation since 
    ZarrLRUCache is a special case where the reader handles the cache"""
    pass