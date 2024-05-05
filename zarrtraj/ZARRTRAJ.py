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

#: Default writing buffersize is 10 MB
ZARRTRAJ_DEFAULT_BUFSIZE = 10485760
#: Default frames per data chunk is 10
ZARRTRAJ_DEFAULT_FPC = 10
#: Currently implemented version of the file format
ZARRTRAJ_VERSION = '0.1.0'


class ZarrTrajBoundaryConditions(Enum):
    ZARRTRAJ_NONE = 0
    ZARRTRAJ_PERIODIC = 1


class ZarrTrajReader(base.ReaderBase):
    """Reader for the `Zarrtraj` format version 0.1.0

    For more information on the format, see the :ref:`zarrtraj_spec`.
    """

    format = 'ZARRTRAJ'

    @store_init_arguments
    def __init__(self, filename,
                 **kwargs):
        """
        Parameters
        ----------
        filename : :class:`zarr.Group`
            Open, readable zarrtraj file
        convert_units : bool (optional)
            convert units to MDAnalysis units
        **kwargs : dict
            General reader arguments.

        Raises
        ------
        RuntimeError
            when `zarr` is not installed
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
        """
        if not HAS_ZARR:
            raise RuntimeError("Please install zarr")
        super(ZarrTrajReader, self).__init__(filename, **kwargs)
        # ReaderBase calls self.filename = str(filename), which we want to undo
        self.filename = filename
        if not self.filename:
            raise PermissionError("The Zarr group is not readable")
        if self.filename.attrs["version"] != ZARRTRAJ_VERSION:
            raise RuntimeError("Zarrtraj file version " +
                               f"{self.filename.attrs['version']} " +
                               "is not compatible with reader version " +
                               f"{ZARRTRAJ_VERSION}")

        self._frame = -1
        self._file = self.filename
        self._particle_group = self._file['particles']
        self._step_array = self._particle_group['step']
        self._time_array = self._particle_group['time']
        # IO CALL
        self._boundary = (ZarrTrajBoundaryConditions.ZARRTRAJ_NONE if
                          self._particle_group["box"].attrs[
                              "boundary"] == "none"
                          else ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC)

        if (self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC and
                'dimensions' not in self._particle_group['box']):
            raise NoDataError("Triclinic box dimension array must " +
                              "be provided if `boundary` is set to 'periodic'")

        # IO CALL
        self._has = set(name for name in self._particle_group if
                        name in ('positions', 'velocities', 'forces'))

        # Get n_atoms + numcodecs compressor & filter objects from
        # first available dataset
        # IO CALLS
        for name in self._has:
            dset = self._particle_group[name]
            self.n_atoms = dset.shape[1]
            self.compressor = dset.compressor
            # NOTE: add filters
            break
        else:
            raise NoDataError("Provide at least a position, velocity " +
                              "or force array in the ZarrTraj file.")

        for name in self._has:
            self._n_frames = self._particle_group[name].shape[0]
            break

        self.ts = self._Timestep(self.n_atoms,
                                 positions=self.has_positions,
                                 velocities=self.has_velocities,
                                 forces=self.has_forces,
                                 **self._ts_kwargs)

        self._verify_correct_units()
        self.units = {'time': 'ps',
                      'length': 'nm',
                      'velocity': 'nm/ps',
                      'force': 'kJ/(mol*nm)'}
        self._read_next_timestep()

    def _verify_correct_units(self):
        self._unit_group = self._particle_group['units']
        if ('length' not in self._unit_group.attrs or 
            self._unit_group.attrs['length'] != "nm"):
            raise RuntimeError("Zarrtraj file with positions must contain " +
                               "'nm' length unit")
        if ('velocity' not in self._unit_group.attrs or
            self._unit_group.attrs['velocity'] != "nm/ps"):
            raise RuntimeError("Zarrtraj file must contain " +
                               "'nm/ps' velocity unit")
        if ('force' not in self._unit_group.attrs or
            self._unit_group.attrs['force'] != "kJ/(mol*nm)"):
            raise RuntimeError("Zarrtraj file with forces must contain " +
                               "'kJ/(mol*nm)' force unit")
        if (self._unit_group.attrs['time'] != "ps"):
            raise RuntimeError("Zarrtraj file must contain 'ps' for time unit")

    def _read_next_timestep(self):
        """Read next frame in trajectory"""
        return self._read_frame(self._frame + 1)

    def _read_frame(self, frame):
        """Reads data from zarrtraj file and copies to current timestep"""
        if frame >= self._n_frames:
            raise IOError from None

        self._frame = frame
        ts = self.ts
        particle_group = self._particle_group
        ts.frame = frame

        # NOTE: handle observables
        # Fills data dictionary from 'observables' group
        # Note: dt is not read into data as it is not decided whether
        # Timestep should have a dt attribute (see Issue #2825)
        self.ts.time = self._time_array[self._frame]
        self.ts.data['step'] = self._step_array[self._frame]

        # Sets frame box dimensions
        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            edges = particle_group["box/dimensions"][frame, :]
            ts.dimensions = core.triclinic_box(*edges)
        else:
            ts.dimensions = None

        # set the timestep positions, velocities, and forces with
        # current frame dataset
        if self.has_positions:
            self._read_dataset_into_ts('positions', ts.positions)
        if self.has_velocities:
            self._read_dataset_into_ts('velocities', ts.velocities)
        if self.has_forces:
            self._read_dataset_into_ts('forces', ts.forces)

        self._convert_units()

        return ts

    def _read_dataset_into_ts(self, dataset, attribute):
        """Reads position, velocity, or force dataset array at current frame
        into corresponding ts attribute"""

        n_atoms_now = self._particle_group[dataset][
                                           self._frame].shape[0]
        if n_atoms_now != self.n_atoms:
            raise ValueError(f"Frame {self._frame} of the {dataset} dataset"
                             f" has {n_atoms_now} atoms but the initial frame"
                             " of either the postion, velocity, or force"
                             f" dataset had {self.n_atoms} atoms."
                             " MDAnalysis is unable to deal"
                             " with variable topology!")

        attribute[:] = self._particle_group[dataset][self._frame, :]

    def _convert_units(self):
        """Converts position, velocity, and force values to
        MDAnalysis units. Time does not need to be converted"""

        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            # Only convert [:3] since last 3 elements are angle
            self.convert_pos_from_native(self.ts.dimensions[:3])

        if self.has_positions:
            self.convert_pos_from_native(self.ts.positions)

        if self.has_velocities:
            self.convert_velocities_from_native(self.ts.velocities)

        if self.has_forces:
            self.convert_forces_from_native(self.ts.forces)

    @staticmethod
    def _format_hint(thing):
        """Can this Reader read *thing*"""
        if not HAS_ZARR or not isinstance(thing, zarr.Group):
            return False
        else:
            return True

    @staticmethod
    def parse_n_atoms(filename, storage_options=None):
        for group in filename['particles']:
            if group in ('positions', 'velocities', 'forces'):
                n_atoms = filename[f'particles/{group}'].shape[1]
                return n_atoms

        raise NoDataError("Could not construct minimal topology from the " +
                          "Zarrtraj trajectory file, as it did not contain " +
                          "a 'position', 'velocity', or 'force' group. " +
                          "You must include a topology file.")

    def close(self):
        """Close reader"""
        self._file.store.close()

    def _reopen(self):
        """reopen trajectory"""
        self.close()
        self._frame = -1

    @property
    def n_frames(self):
        """number of frames in trajectory"""
        return self._n_frames

    def Writer(self, filename, n_atoms=None, **kwargs):
        """Return writer for trajectory format
        """
        if n_atoms is None:
            n_atoms = self.n_atoms
        kwargs.setdefault('format', "ZARRTRAJ")
        kwargs.setdefault('compressor', self.compressor)
        # NOTE: add filters
        kwargs.setdefault('positions', self.has_positions)
        kwargs.setdefault('velocities', self.has_velocities)
        kwargs.setdefault('forces', self.has_forces)
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
        del state['_particle_group']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._particle_group = self._file['particles']
        self[self.ts.frame]


class ZarrTrajWriter(base.WriterBase):
    """Writer for `Zarrtraj` format version 0.1.0.

    All data from the input
    :class:`~MDAnalysis.coordinates.timestep.Timestep` is
    written by default.

    Parameters
    ----------
    filename : :class:`zarr.Group`
        Open, readable zarrtraj file
    n_atoms : int
        number of atoms in trajectory
    n_frames : int (required for cloud and buffered writing)
        number of frames to be written in trajectory
    chunks : tuple (optional)
        Custom chunk layout to be applied to the position,
        velocity, and force datasets. By default, these datasets
        are chunked in ``{10, n_atoms, 3}`` blocks
    max_memory : int (optional)
        Maximum memory buffer size in bytes for writing to a
        a cloud-backed Zarr group or when `force_bufferd=True`.
        By default, this is set to 10 MB.
    compressor : str or int (optional)
        `numcodecs` compressor object to be applied
        to position, velocity, force, and observables datasets.
    filters : list (optional)
        list of `numcodecs` filter objects to be applied to
        to position, velocity, force, and observables datasets.
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
        """

    format = 'ZARRTRAJ'
    multiframe = True

    #: These variables are not written from :attr:`Timestep.data`
    #: dictionary to the observables group in the Zarrtraj file
    data_blacklist = ['step', 'time', 'dt']

    def __init__(self, filename, n_atoms, n_frames=None,
                 chunks=None,
                 positions=True, velocities=True,
                 forces=True, compressor=None,
                 filters=None, max_memory=None,
                 force_buffered=False,
                 author_email=None, author='N/A',
                 creator='MDAnalysis', creator_version=mda.__version__,
                 **kwargs):

        if not HAS_ZARR:
            raise RuntimeError("ZarrTrajWriter: Please install zarr")
        # Verify group is open for writing
        if not filename.store.is_writeable():
            raise PermissionError("The Zarr group is not writeable")
        if n_atoms == 0:
            raise ValueError("ZarrTrajWriter: no atoms in output trajectory")
        self.filename = filename
        self._file = filename
        self.n_atoms = n_atoms
        self.n_frames = n_frames
        self.force_buffered = force_buffered

        # Fill in Zarrtraj metadata from kwargs
        # IO CALL
        self._file.attrs['version'] = ZARRTRAJ_VERSION
        self._file.require_group('metadata')
        self._file['metadata'].attrs['author'] = author
        if author_email is not None:
            self._file['metadata'].attrs['author_email'] = author_email
        self._file['metadata'].attrs['creator'] = creator
        if creator == 'MDAnalysis':
            self._file['metadata'].attrs['creator_version'] = creator_version

        self._determine_if_buffered_storage()
        if self._is_buffered_store:
            # Ensure n_frames exists
            if n_frames is None:
                raise RuntimeError("ZarrTrajWriter: Buffered writing requires " +
                                "'n_frames' kwarg")
            self.max_memory = (ZARRTRAJ_DEFAULT_BUFSIZE if max_memory is None
                               else max_memory)

        self.chunks = ((ZARRTRAJ_DEFAULT_FPC, self.n_atoms, 3)
                       if chunks is None else chunks)
        self.filters = filters if filters is not None else []
        self.compressor = (compressor if compressor is not None else 
                           zarr.storage.default_compressor)
        # The writer defaults to writing all data from the parent Timestep if
        # it exists. If these are True, the writer will check each
        # Timestep.has_*  value and fill the self._has dictionary accordingly
        # in _initialize_hdf5_datasets()
        self._write = set()
        if positions:
            self._write.add('positions')
        if velocities:
            self._write.add('velocities')
        if forces:
            self._write.add('forces')

        self.units = {'time': 'ps',
                      'length': 'nm',
                      'velocity': 'nm/ps',
                      'force': 'kJ/(mol*nm)'}

        if not self._write:
            raise ValueError("At least one of positions, velocities, or "
                             "forces must be set to ``True``.")

        self._initial_write = True

    def _determine_if_buffered_storage(self):
        # Check if we are working with a cloud storage type
        store = self._file.store
        if isinstance(store, zarr.storage.FSStore):
            if 's3' in store.fs.protocol:
                # Verify s3fs is installed
                # NOTE: Verify this is necessary
                try:
                    import s3fs
                except ImportError:
                    raise Exception("Writing to AWS S3 requires installing " +
                                    + "s3fs")
                self._is_buffered_store = True
            elif 'gcs' in store.fs.protocol:
                # Verify gcsfs is installed
                try:
                    import gcsfs
                except ImportError:
                    raise Exception("Writing to Google Cloud Storage " +
                                    + "requires installing gcsfs")
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
                errmsg = "Input obj is neither an AtomGroup or Universe"
                raise TypeError(errmsg) from None

        if ts.n_atoms != self.n_atoms:
            raise IOError("ZarrTrajWriter: Timestep does not have"
                          " the correct number of atoms")

        # This will only be called once when first timestep is read.
        if self._initial_write:
            self._determine_has(ts)
            self._determine_units(ag)
            if self._is_buffered_store:
                self._check_max_memory()
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
            raise ValueError("The trajectory is missing length units.")
        if self.has_velocities and not from_units["velocity"]:
            raise ValueError("The trajectory is missing velocity units.")
        if self.has_forces and not from_units["force"]:
            raise ValueError("The trajectory is missing force units.")
        if not from_units["time"]:
            raise ValueError("The trajectory is missing time units.")

    def _determine_has(self, ts):
        # ask the parent file if it has positions, velocities, and forces,
        # and dimensions
        self._has = set()
        if "positions" in self._write and ts.has_positions:
            self._has.add("positions")
        if "velocities" in self._write and ts.has_velocities:
            self._has.add("velocities")
        if "forces" in self._write and ts.has_forces:
            self._has.add("forces")
        self._boundary = (ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC
                          if ts.dimensions is not None
                          else ZarrTrajBoundaryConditions.ZARRTRAJ_NONE)

    def _check_max_memory(self):
        """
        Determines if at least one chunk of size ``chunks`` fits in the
        ``max_memory``sized buffer. If not, the writer will fail without
        allocating space for trajectory data on the cloud.
        """

        float32_size = np.dtype(np.float32).itemsize
        int32_size = np.dtype(np.int32).itemsize
        mem_per_chunk = 0

        # Write edges, step, and time in the same pattern as
        # velocity, force, and position, though it is not
        # strictly necessary for simplicity
        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            mem_per_chunk += float32_size * self.chunks[0] * 9
        # Step
        mem_per_chunk += int32_size * self.chunks[0]
        # Time
        mem_per_chunk += float32_size * self.chunks[0]

        if self.has_positions:
            mem_per_chunk += float32_size * self.chunks[0] * self.n_atoms * 3

        if self.has_forces:
            mem_per_chunk += float32_size * self.chunks[0] * self.n_atoms * 3

        if self.has_velocities:
            mem_per_chunk += float32_size * self.chunks[0] * self.n_atoms * 3

        if mem_per_chunk > self.max_memory:
            raise ValueError("`max_memory` kwarg " +
                             f"must be at least {mem_per_chunk} for " +
                             f"chunking pattern of {self.chunks}")
        else:
            self.n_buffer_frames = self.chunks[0] * (self.max_memory //
                                                     mem_per_chunk)

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
        self._particle_group = self._file.require_group('particles')
        # NOTE: subselection init goes here when implemented

        # Initialize units group
        self._particle_group.require_group('units')
        self._particle_group["units"].attrs['time'] = self.units['time']
        self._particle_group["units"].attrs['length'] = self.units['length']
        self._particle_group["units"].attrs['velocity'] = self.units['velocity']
        self._particle_group["units"].attrs['force'] = self.units['force']

        self._particle_group.require_group('box')
        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            self._particle_group['box'].attrs['boundary'] = 'periodic'
            self._particle_group['box']['dimensions'] = zarr.empty(
                shape=(self._first_dim, 3, 3),
                dtype=np.float32,
                compressor=self.compressor,
                filters=self.filters
            )
            self._dimensions = self._particle_group['box']['dimensions']
        else:
            # boundary attr must be "none"
            self._particle_group['box'].attrs['boundary'] = 'none'

        self._particle_group['step'] = (zarr.empty(shape=(self._first_dim,),
                                                   dtype=np.int32))
        self._step = self._particle_group['step']
        self._particle_group['time'] = (zarr.empty(shape=(self._first_dim,),
                                                   dtype=np.int32))
        self._time = self._particle_group['time']

        if self.has_positions:
            self._create_trajectory_dataset('positions')
            self._pos = self._particle_group['positions']
        if self.has_velocities:
            self._create_trajectory_dataset('velocities')
            self._vel = self._particle_group['velocities']
        if self.has_forces:
            self._create_trajectory_dataset('forces')
            self._force = self._particle_group['forces']

        # intialize observable datasets from ts.data dictionary that
        # are NOT in self.data_blacklist
        self.data_keys = [
            key for key in ts.data.keys() if key not in self.data_blacklist]
        if self.data_keys:
            self._obsv = self._particle_group.require_group('observables')
        if self.data_keys:
            for key in self.data_keys:
                self._create_observables_dataset(key, ts.data[key])

    def _create_observables_dataset(self, group, data):
        """helper function to initialize a dataset for each observable"""

        self._obsv.require_group(group)
        # guarantee ints and floats have a shape ()
        data = np.asarray(data)
        self._obsv[group] = zarr.empty(shape=(self._first_dim,) + data.shape,
                                       dtype=data.dtype)

    def _create_trajectory_dataset(self, group):
        """helper function to initialize a dataset for
        position, velocity, and force"""
        self._particle_group[group] = zarr.empty(shape=(self._first_dim,
                                                        self.n_atoms, 3),
                                                 dtype=np.float32,
                                                 chunks=self.chunks,
                                                 filters=self.filters,
                                                 compressor=self.compressor)

    def _initialize_memory_buffers(self):
        self._time_buffer = np.zeros((self.n_buffer_frames,), dtype=np.float32)
        self._step_buffer = np.zeros((self.n_buffer_frames,), dtype=np.int32)
        self._dimensions_buffer = np.zeros((self.n_buffer_frames, 3, 3),
                                           dtype=np.float32)
        if self.has_positions:
            self._pos_buffer = np.zeros((self.n_buffer_frames, self.n_atoms,
                                         3), dtype=np.float32)
        if self.has_velocities:
            self._force_buffer = np.zeros((self.n_buffer_frames, self.n_atoms,
                                           3), dtype=np.float32)
        if self.has_forces:
            self._vel_buffer = np.zeros((self.n_buffer_frames, self.n_atoms,
                                         3), dtype=np.float32)
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
            curr_step = ts.data['step']
        except KeyError:
            curr_step = ts.frame
        self._step_buffer[buffer_index] = curr_step
        if self._prev_step is not None and curr_step < self._prev_step:
            raise ValueError("The Zarrtraj standard dictates that the step "
                             "dataset must increase monotonically in value.")
        self._prev_step = curr_step

        curr_time = (self.convert_time_to_native(ts.time, inplace=False))
        self._time_buffer[buffer_index] = curr_time
        if self._prev_time is not None and curr_time < self._prev_time:
            raise ValueError("The Zarrtraj standard dictates that the time "
                             "dataset must increase monotonically in value.")
        self._prev_time = curr_time

        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            self._dimensions_buffer[buffer_index, :] = (
                self.convert_pos_to_native(ts.triclinic_dimensions,
                                           inplace=False))

        if self.has_positions:
            self._pos_buffer[buffer_index, :] = self.convert_pos_to_native(
                ts.positions, inplace=False)

        if self.has_velocities:
            self._vel_buffer[buffer_index, :] = (
                self.convert_velocities_to_native(ts.velocities, inplace=False))

        if self.has_forces:
            self._force_buffer[buffer_index, :] = (
                self.convert_forces_to_native(ts.forces, inplace=False))

        # If buffer is full or last write call, write buffers to cloud
        if (((i + 1) % self.n_buffer_frames == 0) or
                (i == self.n_frames - 1)):
            da.from_array(self._step_buffer[:buffer_index + 1]).to_zarr(self._step, region=(slice(i - buffer_index, i + 1),))
            da.from_array(self._time_buffer[:buffer_index + 1]).to_zarr(self._time, region=(slice(i - buffer_index, i + 1),))
            if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
                da.from_array(self._dimensions_buffer[:buffer_index + 1]).to_zarr(self._dimensions, region=(slice(i - buffer_index, i + 1),))
            if self.has_positions:
                da.from_array(self._pos_buffer[:buffer_index + 1]).to_zarr(self._pos, region=(slice(i - buffer_index, i + 1),))
            if self.has_velocities:
                da.from_array(self._vel_buffer[:buffer_index + 1]).to_zarr(self._vel, region=(slice(i - buffer_index, i + 1),))
            if self.has_forces:
                da.from_array(self._force_buffer[:buffer_index + 1]).to_zarr(self._force, region=(slice(i - buffer_index, i + 1),))

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
                self._dimensions.resize((self._dimensions.shape[0] + 1,) +
                                        self._dimensions.shape[1:])
            if self.has_positions:
                self._pos.resize((self._pos.shape[0] + 1,) +
                                 self._pos.shape[1:])
            if self.has_velocities:
                self._vel.resize((self._vel.shape[0] + 1,) +
                                 self._vel.shape[1:])
            if self.has_forces:
                self._force.resize((self._force.shape[0] + 1,) +
                                   self._force.shape[1:])

        # Zarrtraj step refers to the integration step at which the data were
        # sampled, therefore ts.data['step'] is the most appropriate value
        # to use. However, step is also necessary in Zarrtraj to allow
        # temporal matching of the data, so ts.frame is used as an alternative
        try:
            self._step[i] = ts.data['step']
        except KeyError:
            self._step[i] = ts.frame
        if len(self._step) > 1 and self._step[i] < self._step[i-1]:
            raise ValueError("The Zarrtraj standard dictates that the step "
                             "dataset must increase monotonically in value.")
        self._time[i] = self.convert_time_to_native(ts.time, inplace=False)
        if len(self._time) > 1 and self._time[i] < self._time[i-1]:
            raise ValueError("The Zarrtraj standard dictates that the time "
                             "dataset must increase monotonically in value.")
        if self._boundary == ZarrTrajBoundaryConditions.ZARRTRAJ_PERIODIC:
            self._dimensions[i, :] = self.convert_pos_to_native(
                ts.triclinic_dimensions, inplace=False)
        if self.has_positions:
            self._pos[i, :] = self.convert_pos_to_native(
                ts.positions, inplace=False)
        if self.has_velocities:
            self._vel[i, :] = self.convert_velocities_to_native(
                ts.velocities, inplace=False)
        if self.has_forces:
            self._force[i, :] = self.convert_forces_to_native(
                ts.forces, inplace=False)
        # NOTE: Fix me. add observables
        # if self.convert_units:
        #    self._convert_dataset_with_units(i)

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