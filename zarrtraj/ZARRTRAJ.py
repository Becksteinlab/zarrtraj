"""

Example: Loading a .zarrtraj file from disk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a ZarrTraj simulation from a .zarrtraj trajectory file, pass a
topology file and a `zarr.Group` object to a
:class:`~MDAnalysis.core.universe.Universe`::

    import zarrtraj
    import zarr
    import MDAnalysis as mda
    u = mda.Universe("topology.tpr", zarr.open_group("trajectory.zarrtraj",
                                                     mode="r"))

Example: Reading from cloud services
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Zarrtraj currently supports AWS, Google Cloud, and Azure Block Storage backed
Zarr groups.

To read from AWS S3, wrap your Zarr group in a Least-Recently-Used cache to
reduce I/O::

    import zarrtraj
    import s3fs
    import zarr
    import MDAnalysis as mda

    # S3FS exposes a filesystem-like interface to S3 buckets,
    # which zarr uses to interact with the trajectory
    s3 = s3fs.S3FileSystem(
        # Anon must be false to allow authentication
        anon=False,
        # Authentication profiles are setup in ./aws/credentials
        profile='sample_profile',
        client_kwargs=dict(
            region_name='us-east-1''
            )
    )

    cloud_store = s3fs.S3Map(
        root='<bucket-name>/trajectory.zarrtraj',
        s3=s3,
        check=False
    )

    # max_size is cache size in bytes
    cache = LRUStoreCache(cloud_store, max_size=2**25)
    zgroup = zarr.group(store=cache)
    u = mda.Universe("topology.tpr", zgroup)

AWS provides a VSCode extension to manage AWS authentication profiles
`here <https://aws.amazon.com/visualstudiocode/>`_.

.. warning::

    Because of this local cache model, random-access trajectory reading for
    cloud-backed Zarr Groups is not currently supported.

Example: Writing to cloud services
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ZarrTrajWriter can write to AWS, Google Cloud, and Azure Block Storage
in addition to local storage. To write to a cloud-backed Zarr group, pass
the `n_frames` and `format` kwargs in addition to other MDAnalysis
writer arguments::

    import s3fs
    import zarrtraj
    import MDAnalysis as mda

    s3 = s3fs.S3FileSystem(
        anon=False,
        profile='sample_profile',
        client_kwargs=dict(
            region_name='us-east-1''
            )
    )

    cloud_store = s3fs.S3Map(
        root='<bucket-name>/trajectory.zarrtraj',
        s3=s3,
        check=False
    )

    zgroup = zarr.open_group(store=cloud_store)
    with mda.Writer(zgroup, u.trajectory.n_atoms,
                    n_frames=u.trajectory.n_frames,
                    format='ZARRTRAJ') as w:
        for ts in u.trajectory:
            w.write(u.atoms)

Classes
^^^^^^^

.. autoclass:: ZarrTrajReader
   :members:
   :inherited-members:

.. autoclass:: ZarrTrajWriter
   :members:
   :inherited-members:
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates import base, core
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.due import due, Doi
from MDAnalysis.lib.util import store_init_arguments
import dask.array as da
from enum import Enum
from .CACHE import FrameCache, AsyncFrameCache
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

#: Default writing buffersize is 10 MB
ZARRTRAJ_DEFAULT_BUFSIZE = 10485760
#: Default frames per data chunk is 10
ZARRTRAJ_DEFAULT_FPC = 10
#: Currently implemented version of the file format
ZARRTRAJ_VERSION = "0.1.0"


class ZarrTrajBoundaryConditions(Enum):
    ZARRTRAJ_NONE = 0
    ZARRTRAJ_PERIODIC = 1


class ZarrTrajReader(base.ReaderBase):
    """Reader for the `Zarrtraj` format version 0.1.0

    For more information on the format, see the :ref:`zarrtraj_spec`.
    """

    format = ["ZARRTRAJ", "ZARR"]

    @store_init_arguments
    def __init__(
        self, filename, storage_options=None, cache_size=2**28, **kwargs
    ):
        """
        Parameters
        ----------
        filename : :class:`zarr.Group`
            Fsspec-style path to the Zarr group
        storage_options : dict (optional)
            Additional options to pass to the storage backend
        **kwargs : dict
            General reader arguments.

        Raises
        ------
        RuntimeError
            when ``zarr`` is not installed
        PermissionError
            when the Zarr group is not readable
        RuntimeError
            when an incorrect unit is provided or a unit is missing
        ValueError
            when ``n_atoms`` changes values between timesteps
        NoDataError
            when the Zarrtraj file has a boundary condition of 'periodic'
            but no 'dimensions' array is provided
        NoDataError
            when the Zarrtraj file has no 'position', 'velocity', or
            'force' group
        RuntimeError
            when the Zarrtraj file version is incompatibile with the reader
        ValueError
            when an observables dataset is not sampled at the same rate as
            the position, velocity, and force datasets
        """
        # The reader is responsible for
        # 1. Verifying the file (version, units, array shapes and sizes, etc)
        # 2. Opening and closing the file

        # Ensure close() doesn't fail if the cache wasn't allocated
        self._cache = None

        # NOTE: this class likely needs a timestep still. check this
        if not HAS_ZARR:
            raise RuntimeError("ZarrTrajReader: Please install zarr")
        super(ZarrTrajReader, self).__init__(filename, **kwargs)
        self.storage_options = storage_options
        self._file = None
        # Before opening the zarr group, check filename's protocol
        # to see if we need to import the correct filesystem
        self._protocol_import()
        self._open_group()
        if not self._file:
            raise PermissionError(
                "ZarrTrajReader: The Zarr group is not readable"
            )
        if self._file.attrs["version"] != ZARRTRAJ_VERSION:
            raise RuntimeError(
                "ZarrTrajReader: Zarrtraj file version "
                + f"{self._file.attrs['version']} "
                + "is not compatible with reader version "
                + f"{ZARRTRAJ_VERSION}"
            )
        self._particle_group = self._file["particles"]
        self._step_array = self._particle_group["step"]
        self._time_array = self._particle_group["time"]
        # IO CALL
        self._boundary = (
            ZarrTrajBoundaryConditions.ZARRTRAJ_NONE
            if self._particle_group["box"].attrs["boundary"] == "none"
            else ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC
        )

        if (
            self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC
            and "dimensions" not in self._particle_group["box"]
        ):
            raise NoDataError(
                "ZarrTrajReader: Triclinic box dimension array must "
                + "be provided if `boundary` is set to 'periodic'"
            )

        # Keep this so other objects can call has_positions, etc
        self._has = set(
            name
            for name in self._particle_group
            if name in ("positions", "velocities", "forces")
        )

        # Get n_atoms + numcodecs compressor & filter objects from
        # first available dataset so we can pass this in reader.Writer
        for name in self._has:
            dset = self._particle_group[name]
            self.n_atoms = dset.shape[1]
            self.compressor = dset.compressor
            self.chunks = dset.chunks
            # NOTE: add filters
            break
        else:
            raise NoDataError(
                "ZarrTrajReader: Provide at least a position, velocity "
                + "or force array in the ZarrTraj file."
            )

        # Keep this so other objects can call reader.n_frames
        for name in self._has:
            self._n_frames = self._particle_group[name].shape[0]
            break

        
        self.units = {
            "time": "ps",
            "length": "nm",
            "velocity": "nm/ps",
            "force": "kJ/(mol*nm)",
        }

        self._obsv = set()
        if "observables" in self._particle_group:
            self._obsv = set(self._particle_group["observables"])

        self._verify_correct_units()

        # Timestep shared between reader and cache
        self.ts = self._Timestep(
            self.n_atoms,
            positions=self.has_positions,
            velocities=self.has_velocities,
            forces=self.has_forces,
            **self._ts_kwargs,
        )

        self._frame_seq = None
        self._cache = ZarrTrajAsyncFrameCache(
            self._file,
            cache_size,
            self.ts,
            self._chunks[0],
            self.has_positions
            self._obsv,
        )
        self._first_read = False

    def _protocol_import(self):
        """Import the correct filesystem for the protocol"""
        if "s3://" in self.filename:
            try:
                import s3fs
            except ImportError:
                raise ImportError(
                    "ZarrTrajReader: Reading from AWS S3 requires installing s3fs"
                )
        elif "gcs://" in self.filename or "gs://" in self.filename:
            try:
                import gcsfs
            except ImportError:
                raise ImportError(
                    "ZarrTrajReader: Reading from Google Cloud requires installing gcsfs"
                )
        elif "az://" in self.filename:
            try:
                import adlfs
            except ImportError:
                raise ImportError(
                    "ZarrTrajReader: Reading from  Azure Blog Storage requires installing adlfs"
                )

    def _open_group(self):
        """Open the Zarr group for reading"""
        storage_options = (
            dict() if self.storage_options is None else self.storage_options
        )
        store = zarr.storage.FSStore(
            url=self.filename, mode="r", **storage_options
        )
        self._file = zarr.open_group(store=store, mode="r")

    def _verify_correct_units(self):
        self._unit_group = self._particle_group["units"]
        if (
            "length" not in self._unit_group.attrs
            or self._unit_group.attrs["length"] != "nm"
        ):
            raise RuntimeError(
                "ZarrTrajReader: Zarrtraj file with positions must contain "
                + "'nm' length unit"
            )
        if (
            "velocity" not in self._unit_group.attrs
            or self._unit_group.attrs["velocity"] != "nm/ps"
        ):
            raise RuntimeError(
                "ZarrTrajReader: Zarrtraj file must contain "
                + "'nm/ps' velocity unit"
            )
        if (
            "force" not in self._unit_group.attrs
            or self._unit_group.attrs["force"] != "kJ/(mol*nm)"
        ):
            raise RuntimeError(
                "ZarrTrajReader: Zarrtraj file with forces must contain "
                + "'kJ/(mol*nm)' force unit"
            )
        if self._unit_group.attrs["time"] != "ps":
            raise RuntimeError(
                "ZarrTrajReader: Zarrtraj file must contain 'ps' for time unit"
            )

    def _read_next_timestep(self):
        """Read next frame in trajectory"""
        if self._frame_seq is None:
            self._frame_seq = collections.deque(range(self._n_frames))
        print(self._frame_seq)
        # Frame sequence is already determined in the frame_seq queue
        return self._read_frame(-1)

    def _read_frame(self, frame):
        """Reads data from zarrtraj file and copies to current timestep"""
        # If this is the first read in the sequence, we need to
        # give the cache the sequence
        if self._first_read:
            self._cache.update_frame_seq(self._frame_seq)
            self._first_read = False

        # Stop  iteration when we've read all frames in the frame queue
        if not self._frame_seq:
            raise IOError from None

        # Get the next frame from the cache
        ts = self._cache.get_frame()

        # Reader is responsible for converting units
        # since this isn't related to io
        self._convert_units(ts)

        return ts

    def _convert_units(self, ts):
        """Converts position, velocity, and force values to
        MDAnalysis units. Time does not need to be converted"""

        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            # Only convert [:3] since last 3 elements are angle
            self.convert_pos_from_native(ts.dimensions[:3])

        if self.has_positions:
            self.convert_pos_from_native(ts.positions)

        if self.has_velocities:
            self.convert_velocities_from_native(ts.velocities)

        if self.has_forces:
            self.convert_forces_from_native(ts.forces)

    @staticmethod
    def parse_n_atoms(filename, storage_options=None):
        storage_options = dict() if storage_options is None else storage_options
        store = zarr.storage.FSStore(url=filename, mode="r", **storage_options)
        file = zarr.open_group(store=store, mode="r")

        for group in file["particles"]:
            if group in ("positions", "velocities", "forces"):
                n_atoms = file[f"particles/{group}"].shape[1]
                return n_atoms

        raise NoDataError(
            "Could not construct minimal topology from the "
            + "Zarrtraj trajectory file, as it did not contain "
            + "a 'position', 'velocity', or 'force' group. "
            + "You must include a topology file."
        )

    def close(self):
        """Close reader"""
        if self._cache is not None:
            self._cache.cleanup()
        self._frame_seq = None
        if self._file is not None:
            self._file.store.close()

    def _reopen(self):
        """reopen trajectory"""
        self.close()
        self._open_group()

    @property
    def n_frames(self):
        """number of frames in trajectory"""
        return self._n_frames

    def Writer(self, filename, n_atoms=None, **kwargs):
        """Return writer for trajectory format"""
        if n_atoms is None:
            n_atoms = self.n_atoms
        kwargs.setdefault("n_frames", self.n_frames)
        kwargs.setdefault("format", "ZARRTRAJ")
        kwargs.setdefault("compressor", self.compressor)
        kwargs.setdefault("chunks", self.chunks)
        # NOTE: add filters
        kwargs.setdefault("positions", self.has_positions)
        kwargs.setdefault("velocities", self.has_velocities)
        kwargs.setdefault("forces", self.has_forces)
        kwargs.setdefault("storage_options", self.storage_options)
        return ZarrTrajWriter(filename, n_atoms, **kwargs)

    @property
    def has_positions(self):
        """``True`` if 'position' group is in trajectory."""
        return "positions" in self._has

    @property
    def has_velocities(self):
        """``True`` if 'velocity' group is in trajectory."""
        return "velocities" in self._has

    @property
    def has_forces(self):
        """``True`` if 'force' group is in trajectory."""
        return "forces" in self._has

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_particle_group"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._particle_group = self._file["particles"]
        self[self.ts.frame]

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
            if self._frame_seq is not None:
                self._frame_seq = collections.deque([frame])
            return self._read_frame_with_aux(frame)
        elif isinstance(frame, (list, np.ndarray)):
            if len(frame) != 0 and isinstance(frame[0], (bool, np.bool_)):
                # Avoid having list of bools
                frame = np.asarray(frame, dtype=bool)
                # Convert bool array to int array
                frame = np.arange(len(self))[frame]
            if isinstance(frame, np.ndarray):
                frame = frame.tolist()
            if self._frame_seq is not None:
                self._frame_seq = collections.deque(frame)
            return base.FrameIteratorIndices(self, frame)
        elif isinstance(frame, slice):
            start, stop, step = self.check_slice_indices(
                frame.start, frame.stop, frame.step
            )
            if self._frame_seq is not None:
                self._frame_seq = collections.deque(range(start, stop, step))
            if start == 0 and stop == len(self) and step == 1:
                return base.FrameIteratorAll(self)
            else:
                return base.FrameIteratorSliced(self, frame)
        else:
            raise TypeError(
                "Trajectories must be an indexed using an integer,"
                " slice or list of indices"
            )


class ZarrTrajWriter(base.WriterBase):
    """Writer for `Zarrtraj` format version 0.1.0.

    All data from the input
    :class:`~MDAnalysis.coordinates.timestep.Timestep` is
    written by default.

    Parameters
    ----------
    filename : :class:`zarr.Group`
       fsspec-style path to the Zarr group
    n_atoms : int
        number of atoms in trajectory
    n_frames : int (required for cloud and buffered writing)
        number of frames to be written in trajectory
    chunks : tuple (optional)
        Custom chunk layout to be applied to the position,
        velocity, and force datasets. By default, these datasets
        are chunked in ``(10, n_atoms, 3)`` blocks
    max_memory : int (optional)
        Maximum memory buffer size in bytes for writing to a
        a cloud-backed Zarr group or when ``force_buffered=True``.
        By default, this is set to 10 MB.
    compressor : str or int (optional)
        ``numcodecs`` compressor object to be applied
        to position, velocity, force, and observables datasets.
    filters : list (optional)
        list of ``numcodecs`` filter objects to be applied to
        to position, velocity, force, and observables datasets.
    storage_options : dict (optional)
        Additional options to pass to the storage backend
    positions : bool (optional)
        Write positions into the trajectory [``True``]
    velocities : bool (optional)
        Write velocities into the trajectory [``True``]
    forces : bool (optional)
        Write forces into the trajectory [``True``]
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
        when `zarr` is not installed
    PermissionError
        when the Zarr group is not writeable
    ImportError
        when the correct package for handling a cloud-backed
        Zar group is not installed
    RuntimeError
        when the Zarr group is cloud-backed or `force_buffer=True`
        and `n_frames` is not provided
    ValueError
        when `n_atoms` is 0
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
        when the trajectory is missing units
    ValueError
        when `max_memory` is not large enough to fit at least one
        chunk of the trajectory data
    ValueError
        when the step or time dataset does not increase monotonically
    ValueError
        when an observables dataset is not sampled at the same rate as
        positions, velocities, and forces
    """

    format = "ZARRTRAJ"
    multiframe = True

    #: These variables are not written from :attr:`Timestep.data`
    #: dictionary to the observables group in the Zarrtraj file
    data_blacklist = ["step", "time", "dt"]

    def __init__(
        self,
        filename,
        n_atoms,
        n_frames=None,
        chunks=None,
        positions=True,
        velocities=True,
        forces=True,
        compressor=None,
        filters=None,
        storage_options=None,
        max_memory=None,
        force_buffered=False,
        author_email=None,
        author="N/A",
        creator="MDAnalysis",
        creator_version=mda.__version__,
        **kwargs,
    ):

        if not HAS_ZARR:
            raise RuntimeError("ZarrTrajWriter: Please install zarr")
        self.filename = filename
        self.storage_options = storage_options
        self._open_group()
        # Verify group is open for writing
        if not self._file.store.is_writeable():
            raise PermissionError(
                "ZarrTrajWriter: The Zarr group is not writeable"
            )
        if n_atoms == 0:
            raise ValueError("ZarrTrajWriter: no atoms in output trajectory")

        self.n_atoms = n_atoms
        self.n_frames = n_frames
        self.force_buffered = force_buffered

        # Fill in Zarrtraj metadata from kwargs
        # IO CALL
        self._file.attrs["version"] = ZARRTRAJ_VERSION
        self._file.require_group("metadata")
        self._file["metadata"].attrs["author"] = author
        if author_email is not None:
            self._file["metadata"].attrs["author_email"] = author_email
        self._file["metadata"].attrs["creator"] = creator
        if creator == "MDAnalysis":
            self._file["metadata"].attrs["creator_version"] = creator_version

        self._determine_if_buffered_storage()
        if self._is_buffered_store:
            # Ensure n_frames exists
            if n_frames is None:
                raise RuntimeError(
                    "ZarrTrajWriter: Buffered writing requires "
                    + "'n_frames' kwarg"
                )
            self.max_memory = (
                ZARRTRAJ_DEFAULT_BUFSIZE if max_memory is None else max_memory
            )

        self.chunks = (
            (ZARRTRAJ_DEFAULT_FPC, self.n_atoms, 3)
            if chunks is None
            else chunks
        )
        self.filters = filters if filters is not None else []
        self.compressor = (
            compressor
            if compressor is not None
            else zarr.storage.default_compressor
        )
        # The writer defaults to writing all data from the parent Timestep if
        # it exists. If these are True, the writer will check each
        # Timestep.has_*  value and fill the self._has dictionary accordingly
        # in _initialize_hdf5_datasets()
        self._write = set()
        if positions:
            self._write.add("positions")
        if velocities:
            self._write.add("velocities")
        if forces:
            self._write.add("forces")

        self.units = {
            "time": "ps",
            "length": "nm",
            "velocity": "nm/ps",
            "force": "kJ/(mol*nm)",
        }

        if not self._write:
            raise ValueError(
                "ZarrTrajWriter: At least one of positions, velocities, or "
                "forces must be set to ``True``."
            )

        self._initial_write = True

    def _open_group(self):
        """Open the Zarr group for writing"""
        self._file = zarr.open_group(
            self.filename, storage_options=self.storage_options, mode="a"
        )

    def _determine_if_buffered_storage(self):
        # Check if we are working with a cloud storage type
        store = self._file.store
        if isinstance(store, zarr.storage.FSStore):
            if "s3" in store.fs.protocol:
                # Verify s3fs is installed
                # NOTE: Verify this is necessary
                try:
                    import s3fs
                except ImportError:
                    raise Exception(
                        "ZarrTrajWriter: Writing to AWS S3 requires installing "
                        + +"s3fs"
                    )
                self._is_buffered_store = True
            elif "gcs" in store.fs.protocol:
                # Verify gcsfs is installed
                try:
                    import gcsfs
                except ImportError:
                    raise Exception(
                        "ZarrTrajWriter: Writing to Google Cloud Storage "
                        + +"requires installing gcsfs"
                    )
                self._is_buffered_store = True
        elif isinstance(store, zarr.storage.ABSStore):
            self._is_buffered_store = True
        elif self.force_buffered:
            self._is_buffered_store = True
        else:
            self._is_buffered_store = False

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
                errmsg = "ZarrTrajWriter: Input obj is neither an AtomGroup or Universe"
                raise TypeError(errmsg) from None

        if ts.n_atoms != self.n_atoms:
            raise IOError(
                "ZarrTrajWriter: Timestep does not have"
                " the correct number of atoms"
            )

        # This will only be called once when first timestep is read.
        if self._initial_write:
            self._determine_has(ts)
            self._determine_units(ag)
            if self._is_buffered_store:
                self._check_max_memory(ts)
                self._initialize_zarr_datasets(ts)
                self._initialize_memory_buffers()
            else:
                self._initialize_zarr_datasets(ts)
            self._initial_write = False

        if self._is_buffered_store:
            return self._write_next_buffered_timestep(ts)
        else:
            return self._write_next_timestep(ts)

    def _determine_units(self, ag):
        """Verifies the trajectory contains all
        necessary units so conversion to zarrtraj units can happen.
        If the trajectory being written does not contain a unit but that
        unit isn't being written, ZarrTrajWriter will still write the unit.
        Eg. The trajectory doesn't contain force units but force isn't
        being written"""

        from_units = ag.universe.trajectory.units.copy()

        if self.has_positions and not from_units["length"]:
            raise ValueError(
                "ZarrTrajWriter: The trajectory is missing length units."
            )
        if self.has_velocities and not from_units["velocity"]:
            raise ValueError(
                "ZarrTrajWriter: The trajectory is missing velocity units."
            )
        if self.has_forces and not from_units["force"]:
            raise ValueError(
                "ZarrTrajWriter: The trajectory is missing force units."
            )
        if not from_units["time"]:
            raise ValueError(
                "ZarrTrajWriter: The trajectory is missing time units."
            )

    def _determine_has(self, ts):
        """ask the parent file if it has positions, velocities, and forces,
        dimensions, and observables"""
        self._has = set()
        if "positions" in self._write and ts.has_positions:
            self._has.add("positions")
        if "velocities" in self._write and ts.has_velocities:
            self._has.add("velocities")
        if "forces" in self._write and ts.has_forces:
            self._has.add("forces")
        self._boundary = (
            ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC
            if ts.dimensions is not None
            else ZarrTrajBoundaryConditions.ZARRTRAJ_NONE
        )
        # include observable datasets from ts.data dictionary that
        # are NOT in self.data_blacklist
        self.data_keys = [
            key for key in ts.data.keys() if key not in self.data_blacklist
        ]

    def _check_max_memory(self, ts):
        """
        Determines if at least one chunk of size ``chunks`` fits in the
        ``max_memory``sized buffer. If not, the writer will fail without
        allocating space for trajectory data on the cloud.
        """
        # For each ts element in step, time, dimensions, positions, velocities,
        # forces, and observables, add ts[0].size * ts.itemsize * self.chunks[0]
        # to mem_per_chunk
        has = []
        mem_per_chunk = 0
        try:
            has.append(np.asarray(ts.data["step"]))
        except KeyError:
            has.append(np.asarray(ts.frame))
        has.append(np.asarray(ts.time))
        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            has.append(ts.triclinic_dimensions)
        if self.has_positions:
            has.append(ts.positions)
        if self.has_velocities:
            has.append(ts.velocities)
        if self.has_forces:
            has.append(ts.forces)
        for key in self.data_keys:
            data = np.asarray(ts.data[key])
            has.append(data)
        for dataset in has:
            mem_per_chunk += dataset.size * dataset.itemsize * self.chunks[0]

        if mem_per_chunk > self.max_memory:
            raise ValueError(
                "ZarrTrajWriter: `max_memory` kwarg "
                + f"must be at least {mem_per_chunk} for "
                + f"chunking pattern of {self.chunks}"
            )
        else:
            self.n_buffer_frames = self.chunks[0] * (
                self.max_memory // mem_per_chunk
            )

    def _initialize_zarr_datasets(self, ts):
        """initializes all datasets that will be written to by
        :meth:`_write_next_timestep`. Datasets must be sampled at the same
        rate in version 0.1.0 of zarrtraj

        Note
        ----
        :exc:`NoDataError` is raised if no positions, velocities, or forces are
        found in the input trajectory.
        """
        if self.n_frames is None:
            self._first_dim = 0
        else:
            self._first_dim = self.n_frames

        # for keeping track of where to write in the dataset
        self._counter = 0
        self._particle_group = self._file.require_group("particles")
        # NOTE: subselection init goes here when implemented

        # Initialize units group
        self._particle_group.require_group("units")
        self._particle_group["units"].attrs["time"] = self.units["time"]
        self._particle_group["units"].attrs["length"] = self.units["length"]
        self._particle_group["units"].attrs["velocity"] = self.units["velocity"]
        self._particle_group["units"].attrs["force"] = self.units["force"]

        self._particle_group.require_group("box")
        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            self._particle_group["box"].attrs["boundary"] = "periodic"
            self._particle_group["box"]["dimensions"] = zarr.empty(
                shape=(self._first_dim, 3, 3),
                dtype=np.float32,
                compressor=self.compressor,
                filters=self.filters,
            )
            self._dimensions = self._particle_group["box"]["dimensions"]
        else:
            # boundary attr must be "none"
            self._particle_group["box"].attrs["boundary"] = "none"

        self._particle_group["step"] = zarr.empty(
            shape=(self._first_dim,), dtype=np.int32
        )
        self._step = self._particle_group["step"]
        self._particle_group["time"] = zarr.empty(
            shape=(self._first_dim,), dtype=np.int32
        )
        self._time = self._particle_group["time"]

        if self.has_positions:
            self._create_trajectory_dataset("positions")
            self._pos = self._particle_group["positions"]
        if self.has_velocities:
            self._create_trajectory_dataset("velocities")
            self._vel = self._particle_group["velocities"]
        if self.has_forces:
            self._create_trajectory_dataset("forces")
            self._force = self._particle_group["forces"]
        if self.data_keys:
            self._obsv = self._particle_group.require_group("observables")
            for key in self.data_keys:
                data = np.asarray(ts.data[key])
                self._create_observables_dataset(key, data)

    def _create_observables_dataset(self, group, data):
        """helper function to initialize a dataset for each observable"""
        self._obsv[group] = zarr.empty(
            shape=(self._first_dim,) + data.shape, dtype=data.dtype
        )

    def _create_trajectory_dataset(self, group):
        """helper function to initialize a dataset for
        position, velocity, and force"""
        self._particle_group[group] = zarr.empty(
            shape=(self._first_dim, self.n_atoms, 3),
            dtype=np.float32,
            chunks=self.chunks,
            filters=self.filters,
            compressor=self.compressor,
        )

    def _initialize_memory_buffers(self):
        self._time_buffer = np.empty((self.n_buffer_frames,), dtype=np.float32)
        self._step_buffer = np.empty((self.n_buffer_frames,), dtype=np.int32)
        self._dimensions_buffer = np.empty(
            (self.n_buffer_frames, 3, 3), dtype=np.float32
        )
        if self.has_positions:
            self._pos_buffer = np.empty(
                (self.n_buffer_frames, self.n_atoms, 3), dtype=np.float32
            )
        if self.has_velocities:
            self._force_buffer = np.empty(
                (self.n_buffer_frames, self.n_atoms, 3), dtype=np.float32
            )
        if self.has_forces:
            self._vel_buffer = np.empty(
                (self.n_buffer_frames, self.n_atoms, 3), dtype=np.float32
            )
        if self.data_keys:
            self._obsv_buffer = {}
            for key in self.data_keys:
                self._obsv_buffer[key] = np.empty(
                    (self.n_buffer_frames,) + self._obsv[key].shape[1:],
                    dtype=self._obsv[key].dtype,
                )
        # Reduce cloud I/O by storing previous step & time val for error checking
        self._prev_step = None
        self._prev_time = None

    def _write_next_buffered_timestep(self, ts):
        """Write the next timestep to a cloud or buffered zarr group.
        Will only actually perform write if buffer is full"""
        i = self._counter
        buffer_index = i % self.n_buffer_frames
        # Add the current timestep information to the buffer
        try:
            curr_step = ts.data["step"]
        except KeyError:
            curr_step = ts.frame
        self._step_buffer[buffer_index] = curr_step
        if self._prev_step is not None and curr_step < self._prev_step:
            raise ValueError(
                "ZarrTrajWriter: The Zarrtraj standard dictates that the step "
                "dataset must increase monotonically in value."
            )
        self._prev_step = curr_step

        curr_time = self.convert_time_to_native(ts.time, inplace=False)
        self._time_buffer[buffer_index] = curr_time
        if self._prev_time is not None and curr_time < self._prev_time:
            raise ValueError(
                "ZarrTrajWriter: The Zarrtraj standard dictates that the time "
                "dataset must increase monotonically in value."
            )
        self._prev_time = curr_time

        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            self._dimensions_buffer[buffer_index] = self.convert_pos_to_native(
                ts.triclinic_dimensions, inplace=False
            )

        if self.has_positions:
            self._pos_buffer[buffer_index] = self.convert_pos_to_native(
                ts.positions, inplace=False
            )

        if self.has_velocities:
            self._vel_buffer[buffer_index] = self.convert_velocities_to_native(
                ts.velocities, inplace=False
            )

        if self.has_forces:
            self._force_buffer[buffer_index] = self.convert_forces_to_native(
                ts.forces, inplace=False
            )

        for key in self.data_keys:
            try:
                data = np.asarray(ts.data[key])
                self._obsv_buffer[key][buffer_index] = data
            except IndexError:
                raise ValueError(
                    "ZarrTrajWriter: Observables data must be sampled at the same rate as the position, velocity, and force data."
                )

        # If buffer is full or last write call, write buffers to cloud
        if ((i + 1) % self.n_buffer_frames == 0) or (i == self.n_frames - 1):
            da.from_array(self._step_buffer[: buffer_index + 1]).to_zarr(
                self._step, region=(slice(i - buffer_index, i + 1),)
            )
            da.from_array(self._time_buffer[: buffer_index + 1]).to_zarr(
                self._time, region=(slice(i - buffer_index, i + 1),)
            )
            if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
                da.from_array(
                    self._dimensions_buffer[: buffer_index + 1]
                ).to_zarr(
                    self._dimensions, region=(slice(i - buffer_index, i + 1),)
                )
            if self.has_positions:
                da.from_array(self._pos_buffer[: buffer_index + 1]).to_zarr(
                    self._pos, region=(slice(i - buffer_index, i + 1),)
                )
            if self.has_velocities:
                da.from_array(self._vel_buffer[: buffer_index + 1]).to_zarr(
                    self._vel, region=(slice(i - buffer_index, i + 1),)
                )
            if self.has_forces:
                da.from_array(self._force_buffer[: buffer_index + 1]).to_zarr(
                    self._force, region=(slice(i - buffer_index, i + 1),)
                )
            for key in self.data_keys:
                da.from_array(
                    self._obsv_buffer[key][: buffer_index + 1]
                ).to_zarr(
                    self._obsv[key], region=(slice(i - buffer_index, i + 1),)
                )
        self._counter += 1

    def _write_next_timestep(self, ts):
        """Write coordinates and unitcell information to Zarr group.

        Do not call this method directly; instead use
        :meth:`write` because some essential setup is done
        there before writing the first frame.

        The first dimension of each dataset is extended by +1 and
        then the data is written to the new slot.

        """
        i = self._counter

        # Resize all datasets if needed
        # These datasets are not resized if n_frames was provided as an
        # argument, as they were initialized with their full size.
        if self.n_frames is None:
            self._step.resize((self._step.shape[0] + 1,))
            self._time.resize((self._time.shape[0] + 1,))
            if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
                self._dimensions.resize(
                    (self._dimensions.shape[0] + 1,)
                    + self._dimensions.shape[1:]
                )
            if self.has_positions:
                self._pos.resize(
                    (self._pos.shape[0] + 1,) + self._pos.shape[1:]
                )
            if self.has_velocities:
                self._vel.resize(
                    (self._vel.shape[0] + 1,) + self._vel.shape[1:]
                )
            if self.has_forces:
                self._force.resize(
                    (self._force.shape[0] + 1,) + self._force.shape[1:]
                )
            for key in self.data_keys:
                self._obsv[key].resize(
                    (self._obsv[key].shape[0] + 1,) + self._obsv[key].shape[1:]
                )

        # Zarrtraj step refers to the integration step at which the data were
        # sampled, therefore ts.data['step'] is the most appropriate value
        # to use. However, step is also necessary in Zarrtraj to allow
        # temporal matching of the data, so ts.frame is used as an alternative
        try:
            self._step[i] = ts.data["step"]
        except KeyError:
            self._step[i] = ts.frame
        if len(self._step) > 1 and self._step[i] < self._step[i - 1]:
            raise ValueError(
                "ZarrTrajWriter: The Zarrtraj standard dictates that the step "
                "dataset must increase monotonically in value."
            )
        self._time[i] = self.convert_time_to_native(ts.time, inplace=False)
        if len(self._time) > 1 and self._time[i] < self._time[i - 1]:
            raise ValueError(
                "ZarrTrajWriter: The Zarrtraj standard dictates that the time "
                "dataset must increase monotonically in value."
            )
        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            self._dimensions[i] = self.convert_pos_to_native(
                ts.triclinic_dimensions, inplace=False
            )
        if self.has_positions:
            self._pos[i] = self.convert_pos_to_native(
                ts.positions, inplace=False
            )
        if self.has_velocities:
            self._vel[i] = self.convert_velocities_to_native(
                ts.velocities, inplace=False
            )
        if self.has_forces:
            self._force[i] = self.convert_forces_to_native(
                ts.forces, inplace=False
            )
        for key in self.data_keys:
            try:
                data = np.asarray(ts.data[key])
                self._obsv[key][i] = data
            except IndexError:
                raise ValueError(
                    "ZarrTrajWriter: Observables data must be sampled at the same rate as the position, velocity, and force data."
                )

        self._counter += 1

    @property
    def has_positions(self):
        """``True`` if writer is writing positions from Timestep."""
        return "positions" in self._has

    @property
    def has_velocities(self):
        """``True`` if writer is writing velocities from Timestep."""
        return "velocities" in self._has

    @property
    def has_forces(self):
        """``True`` if writer is writing forces from Timestep."""
        return "forces" in self._has


class ZarrTrajAsyncFrameCache(AsyncFrameCache):
    # https://www.geeksforgeeks.org/optimal-page-replacement-algorithm/
    def cleanup(self):
        pass

    def _load_timestep(self, frame):
        ts = self.ts
        particle_group = self._particle_group
        ts.frame = frame

        self.ts.time = self._time_array[self._frame]
        self.ts.data["step"] = self._step_array[self._frame]

        # Handle observables
        if "observables" in particle_group:
            try:
                for key in particle_group["observables"].keys():
                    self.ts.data[key] = self._particle_group["observables"][
                        key
                    ][frame]
            except IndexError:
                raise ValueError(
                    "ZarrTrajReader: Observables data must be sampled at the same "
                    + "rate as the position, velocity, and force data."
                )

        # Sets frame box dimensions
        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            edges = particle_group["box/dimensions"][frame, :]
            ts.dimensions = core.triclinic_box(*edges)
        else:
            ts.dimensions = None

        # set the timestep positions, velocities, and forces with
        # current frame dataset
        if self.has_positions:
            self._read_dataset_into_ts("positions", ts.positions)
        if self.has_velocities:
            self._read_dataset_into_ts("velocities", ts.velocities)
        if self.has_forces:
            self._read_dataset_into_ts("forces", ts.forces)

    def _read_dataset_into_ts(self, dataset, attribute):
        """Reads position, velocity, or force dataset array at current frame
        into corresponding ts attribute"""

        n_atoms_now = self._particle_group[dataset][self._frame].shape[0]
        if n_atoms_now != self.n_atoms:
            raise ValueError(
                f"ZarrTrajReader: Frame {self._frame} of the {dataset} dataset"
                f" has {n_atoms_now} atoms but the initial frame"
                " of either the postion, velocity, or force"
                f" dataset had {self.n_atoms} atoms."
                " MDAnalysis is unable to deal"
                " with variable topology!"
            )

        attribute[:] = self._particle_group[dataset][self._frame]
