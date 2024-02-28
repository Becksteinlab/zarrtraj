"""

Example: Loading a .zarrtraj file from disk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a ZarrTraj simulation from a .zarrtraj trajectory file, pass a topology file
and a `zarr.Group`_ object to :class:`~MDAnalysis.core.universe.Universe`::

    import zarrtraj
    import MDAnalysis as mda
    u = mda.Universe("topology.tpr", zarr.open_group("trajectory.zarrtraj", mode="r"))

Example: Reading from cloud services
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Zarrtraj currently supports AWS, Google Cloud, and Azure Block Storage backed Zarr groups.

To read from AWS S3, wrap your Zarr group in a Least-Recently-Used cache to reduce I/O::

    import s3fs
    import zarrtraj
    import MDAnalysis as mda
    key = os.getenv('AWS_KEY')
    secret = os.getenv('AWS_SECRET_KEY')
    s3 = s3fs.S3FileSystem(key=key, secret=secret)
    store = s3fs.S3Map(root='<bucket-name>/trajectory.zarrtraj', s3=s3, check=False)
    cache = LRUStoreCache(store, max_size=2**25) # max_size is cache size in bytes
    root = zarr.group(store=cache)
    u = mda.Universe("topology.tpr", root)

Because of this local cache model, random-access trajectory reading for cloud-backed Zarr Groups
is not currently supported.

Example: Writing to cloud services
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ZarrTrajWriter can write to AWS, Google Cloud, and Azure Block Storage as well. 
The writer must be passed the `n_frames`_ and `format`_ kwargs in addition to 
other writer arguments to function.

To write to a `zarr.Group`_ from a trajectory loaded in MDAnalysis, do::

    import s3fs
    import zarrtraj
    import MDAnalysis as mda
    key = os.getenv('AWS_KEY')
    secret = os.getenv('AWS_SECRET_KEY')
    s3 = s3fs.S3FileSystem(key=key, secret=secret)
    store = s3fs.S3Map(root='<bucket-name>/trajectory.zarrtraj', s3=s3, check=False)
    root = zarr.open_group(store=store)
    with mda.Writer(root, u.trajectory.n_atoms, n_frames=u.trajectory.n_frames, format='ZARRTRAJ') as w:
        for ts in u.trajectory:
            w.write(u.atoms)

Classes
-------

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
import time # NOTE: REMOVE after test


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


class ZarrTrajReader(base.ReaderBase):
    format = 'ZARRTRAJ'

    # This dictionary is used to translate from Zarrtraj units to MDAnalysis units
    _unit_translation = {
        'time': {
            'ps': 'ps',
            'fs': 'fs',
            'ns': 'ns',
            'second': 's',
            'sec': 's',
            's': 's',
            'AKMA': 'AKMA',
        },
        'length': {
            'Angstrom': 'Angstrom',
            'angstrom': 'Angstrom',
            'A': 'Angstrom',
            'nm': 'nm',
            'pm': 'pm',
            'fm': 'fm',
        },
        'velocity': {
            'Angstrom ps-1': 'Angstrom/ps',
            'A ps-1': 'Angstrom/ps',
            'Angstrom fs-1': 'Angstrom/fs',
            'A fs-1': 'Angstrom/fs',
            'Angstrom AKMA-1': 'Angstrom/AKMA',
            'A AKMA-1': 'Angstrom/AKMA',
            'nm ps-1': 'nm/ps',
            'nm ns-1': 'nm/ns',
            'pm ps-1': 'pm/ps',
            'm s-1': 'm/s'
        },
        'force':  {
            'kJ mol-1 Angstrom-1': 'kJ/(mol*Angstrom)',
            'kJ mol-1 nm-1': 'kJ/(mol*nm)',
            'Newton': 'Newton',
            'N': 'N',
            'J m-1': 'J/m',
            'kcal mol-1 Angstrom-1': 'kcal/(mol*Angstrom)',
            'kcal mol-1 A-1': 'kcal/(mol*Angstrom)'
        }
    }

    @store_init_arguments
    def __init__(self, filename,
                 convert_units=True,
                 **kwargs):
        
        if not HAS_ZARR:
            raise RuntimeError("Please install zarr")
        super(ZarrTrajReader, self).__init__(filename, **kwargs)
        self.filename = filename
        self.convert_units = convert_units

        self.open_trajectory()
        
        # _has dictionary used for checking whether zarrtraj file has
        # 'position', 'velocity', or 'force' groups in the file
        self._has = {name: name in self._particle_group for
                     name in ('position', 'velocity', 'force')}
        
        self._has_edges = 'edges' in self._particle_group['box']
        
        # Gets some info about what settings the datasets were created with
        # from first available group
        for name, value in self._has.items():
            if value:
                dset = self._particle_group[f'{name}/value']
                self.n_atoms = dset.shape[1]
                self.compressor = dset.compressor
                break
        else:
            raise NoDataError("Provide at least a position, velocity"
                              " or force group in the h5md file.")
        
        self.ts = self._Timestep(self.n_atoms,
                                 positions=self.has_positions,
                                 velocities=self.has_velocities,
                                 forces=self.has_forces,
                                 **self._ts_kwargs)
        
        self.units = {'time': None,
                      'length': None,
                      'velocity': None,
                      'force': None}
        self._set_translated_units()  # fills units dictionary
        self._read_next_timestep()

    def _set_translated_units(self):
        """converts units from ZARRTRAJ to MDAnalysis notation
        and fills units dictionary"""

        # need this dictionary to associate 'position': 'length'
        _group_unit_dict = {'time': 'time',
                            'position': 'length',
                            'velocity': 'velocity',
                            'force': 'force'
                            }

        for group, unit in _group_unit_dict.items():
            self._translate_zarrtraj_units(unit)
            self._check_units(group, unit)

    def _translate_zarrtraj_units(self, unit):
        """stores the translated unit string into the units dictionary"""

        errmsg = "{} unit '{}' is not recognized by ZarrTrajReader."

        try:
            self.units[unit] = self._unit_translation[unit][
                self._file['particles']['units'].attrs[unit]]
        except KeyError:
            raise RuntimeError(errmsg.format(
                                unit, self._file['particles'][
                                    'units'].attrs['time'])
                               ) from None
        
    def _check_units(self, group, unit):
        """Raises error if no units are provided from Zarrtraj file
        and convert_units=True"""

        if not self.convert_units:
            return

        errmsg = "Zarrtraj file must have readable units if ``convert_units`` is"
        " set to ``True``. MDAnalysis sets ``convert_units=True`` by default."
        " Set ``convert_units=False`` to load Universe without units."

        if unit == 'time':
            if self.units['time'] is None:
                raise ValueError(errmsg)

        else:
            if self._has[group]:
                if self.units[unit] is None:
                    raise ValueError(errmsg)

    @staticmethod
    def _format_hint(thing):
        """Can this Reader read *thing*"""
        # Check if the object is already a zarr.Group
        # If it isn't, try opening it as a group and if it excepts, return False
        if not HAS_ZARR or not isinstance(thing, zarr.Group):
            return False
        else:
            return True

    def open_trajectory(self):
        """opens the trajectory file using zarr library"""
        if not self.filename.store.is_readable():
            raise PermissionError("The Zarr group is not readable")

        self._frame = -1
        self._file = self.filename
        # pulls first key out of 'particles'
        # allows for arbitrary name of group1 in 'particles'
        self._particle_group = self._file['particles'][
            list(self._file['particles'])[0]]

    @staticmethod
    def parse_n_atoms(filename, storage_options=None):
        for group in filename['particles/trajectory']:
            if group in ('position', 'velocity', 'force'):
                n_atoms = filename[f'particles/trajectory/{group}/value'].shape[1]
                return n_atoms

        raise NoDataError("Could not construct minimal topology from the "
                        "Zarrtraj trajectory file, as it did not contain a "
                        "'position', 'velocity', or 'force' group. "
                        "You must include a topology file.")

    def close(self):
        """close reader"""
        self._file.store.close()
    
    def _reopen(self):
        """reopen trajectory"""
        self.close()
        self.open_trajectory()

    @property
    def n_frames(self):
        """number of frames in trajectory"""
        for name, value in self._has.items():
            if value:
                return self._particle_group[name]['value'].shape[0]
            
    def _read_frame(self, frame):
        """reads data from zarrtraj file and copies to current timestep"""
        try:
            for name, value in self._has.items():
                if value and 'step' in self._particle_group[name]:
                    _ = self._particle_group[name]['step'][frame]
                    break
            else:
                if self._has_edges and 'step' in self._particle_group['box']['edges']:
                    _ = self._particle_group['box']['edges']['step'][frame]
                else:
                    raise NoDataError("Provide at least a position, velocity"
                                    " or force group in the zarrtraj file.")
            
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
        # Note: Zarrtraj files must contain 'box' group in each 'particles' group
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
        if self._has['position']:
            self._read_dataset_into_ts('position', ts.positions)
        if self._has['velocity']:
            self._read_dataset_into_ts('velocity', ts.velocities)
        if self._has['force']:
            self._read_dataset_into_ts('force', ts.forces)

        if self.convert_units:
            self._convert_units()

        return ts
    
    def _copy_to_data(self):
        """assigns values to keys in data dictionary"""

        # pulls 'time' and 'step' out of first available parent group
        for name, value in self._has.items():
            if value:
                if 'time' in self._particle_group[name]:
                    self.ts.time = self._particle_group[name][
                        'time'][self._frame]
                    break
        for name, value in self._has.items():
            if value:
                if 'step' in self._particle_group[name]:
                    self.ts.data['step'] = self._particle_group[name][
                        'step'][self._frame]
                    break

    def _read_dataset_into_ts(self, dataset, attribute):
        """reads position, velocity, or force dataset array at current frame
        into corresponding ts attribute"""

        n_atoms_now = self._particle_group[f'{dataset}/value'][
                                           self._frame].shape[0]
        if n_atoms_now != self.n_atoms:
            raise ValueError(f"Frame {self._frame} of the {dataset} dataset"
                             f" has {n_atoms_now} atoms but the initial frame"
                             " of either the postion, velocity, or force"
                             f" dataset had {self.n_atoms} atoms."
                             " MDAnalysis is unable to deal"
                             " with variable topology!")
        
        self._particle_group[f'{dataset}/value'].get_basic_selection(
            selection=np.s_[self._frame, :],
            out=attribute
        )

    def _convert_units(self):
        """converts time, position, velocity, and force values if they
        are not given in MDAnalysis standard units

        See https://userguide.mdanalysis.org/stable/units.html
        """

        self.ts.time = self.convert_time_from_native(self.ts.time)

        if self._has_edges and self.ts.dimensions is not None:
            self.convert_pos_from_native(self.ts.dimensions[:3])

        if self._has['position']:
            self.convert_pos_from_native(self.ts.positions)

        if self._has['velocity']:
            self.convert_velocities_from_native(self.ts.velocities)

        if self._has['force']:
            self.convert_forces_from_native(self.ts.forces)

    def _read_next_timestep(self):
        """read next frame in trajectory"""
        return self._read_frame(self._frame + 1)

    def Writer(self, filename, n_atoms=None, **kwargs):
        """Return writer for trajectory format
        
        """
        if n_atoms is None:
            n_atoms = self.n_atoms
        kwargs.setdefault('format', "ZARRTRAJ") # NOTE: change main codebase to recognize zarr grps
        kwargs.setdefault('compressor', self.compressor)
        kwargs.setdefault('positions', self.has_positions)
        kwargs.setdefault('velocities', self.has_velocities)
        kwargs.setdefault('forces', self.has_forces)
        return ZarrTrajWriter(filename, n_atoms, **kwargs)

    @property
    def has_positions(self):
        """``True`` if 'position' group is in trajectory."""
        return self._has['position']

    @has_positions.setter
    def has_positions(self, value: bool):
        self._has['position'] = value

    @property
    def has_velocities(self):
        """``True`` if 'velocity' group is in trajectory."""
        return self._has['velocity']

    @has_velocities.setter
    def has_velocities(self, value: bool):
        self._has['velocity'] = value

    @property
    def has_forces(self):
        """``True`` if 'force' group is in trajectory."""
        return self._has['force']

    @has_forces.setter
    def has_forces(self, value: bool):
        self._has['force'] = value

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_particle_group']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._particle_group = self._file['particles'][
                               list(self._file['particles'])[0]]
        self[self.ts.frame]


class ZarrTrajWriter(base.WriterBase):
    format = 'ZARRTRAJ'
    multiframe = True

    #: currently written version of the file format
    ZARRTRAJ_VERSION = 1

    _unit_translation_dict = {
        'time': {
            'ps': 'ps',
            'fs': 'fs',
            'ns': 'ns',
            'second': 's',
            'sec': 's',
            's': 's',
            'AKMA': 'AKMA'},
        'length': {
            'Angstrom': 'Angstrom',
            'angstrom': 'Angstrom',
            'A': 'Angstrom',
            'nm': 'nm',
            'pm': 'pm',
            'fm': 'fm'},
        'velocity': {
            'Angstrom/ps': 'Angstrom ps-1',
            'A/ps': 'Angstrom ps-1',
            'Angstrom/fs': 'Angstrom fs-1',
            'A/fs': 'Angstrom fs-1',
            'Angstrom/AKMA': 'Angstrom AKMA-1',
            'A/AKMA': 'Angstrom AKMA-1',
            'nm/ps': 'nm ps-1',
            'nm/ns': 'nm ns-1',
            'pm/ps': 'pm ps-1',
            'm/s': 'm s-1'},
        'force':  {
            'kJ/(mol*Angstrom)': 'kJ mol-1 Angstrom-1',
            'kJ/(mol*nm)': 'kJ mol-1 nm-1',
            'Newton': 'Newton',
            'N': 'N',
            'J/m': 'J m-1',
            'kcal/(mol*Angstrom)': 'kcal mol-1 Angstrom-1',
            'kcal/(mol*A)': 'kcal mol-1 Angstrom-1'}}

    def __init__(self, filename, n_atoms, n_frames=None,
                 convert_units=True, chunks=None,
                 positions=True, velocities=True,
                 forces=True, timeunit=None, lengthunit=None,
                 velocityunit=None, forceunit=None, compressor=None,
                 filters=None, max_memory=None, **kwargs):
        
        if not HAS_ZARR:
            raise RuntimeError("ZarrTrajWriter: Please install zarr")
        if n_atoms == 0:
            raise ValueError("ZarrTrajWriter: no atoms in output trajectory")
        self.filename = filename
        self.n_atoms = n_atoms
        self.n_frames = n_frames
        self.zarr_group = None
        self._open_file()
        self._determine_if_cloud_storage()
        if self._is_cloud_storage:
            # Ensure n_frames exists
            if n_frames is None:
                raise TypeError("ZarrTrajWriter: Cloud writing requires " +
                                "'n_frames' kwarg")
            # Determine if chunk size works with memory size

            # If not chunk size, choose sensible default that 
            # fits in memory size
            # and takes up >1 mB of memory
            self.chunks = (100, n_atoms, 3)
            self.n_buffer_frames = self.chunks[0]
            # NOTE: if buffer ends up being in memory,
            # n_buffer frames will end up being calculated based on 
            # number of chunks that can fit in memory, not just chunks arg

        else:
            self.chunks = (1, n_atoms, 3) if chunks is None else chunks

        self.filters = filters
        self.compressor = compressor
        self.convert_units = convert_units
        
        # The writer defaults to writing all data from the parent Timestep if
        # it exists. If these are True, the writer will check each
        # Timestep.has_*  value and fill the self._has dictionary accordingly
        # in _initialize_hdf5_datasets()
        self._write_positions = positions
        self._write_velocities = velocities
        self._write_forces = forces
        if not any([self._write_positions,
                    self._write_velocities,
                    self._write_velocities]):
            raise ValueError("At least one of positions, velocities, or "
                             "forces must be set to ``True``.")

        self._new_units = {'time': timeunit,
                           'length': lengthunit,
                           'velocity': velocityunit,
                           'force': forceunit}
        self._initial_write = True

    def _determine_if_cloud_storage(self):
        # Check if we are working with a cloud storage type
        store = self.zarr_group.store
        if isinstance(store, zarr.storage.FSStore):
            if 's3' in store.fs.protocol:
                # Verify s3fs is installed
                # NOTE: Verify this is necessary
                try:
                    import s3fs
                except ImportError:
                    raise Exception("Writing to AWS S3 requires installing " +
                                    + "s3fs")
                self._is_cloud_storage = True
            elif 'gcs' in store.fs.protocol:
                # Verify gcsfs is installed
                try:
                    import gcsfs
                except ImportError:
                    raise Exception("Writing to Google Cloud Storage " +
                                    + "requires installing gcsfs")
                self._is_cloud_storage = True
        elif isinstance(store, zarr.storage.ABSStore):
            self._is_cloud_storage = True
        else:
            self._is_cloud_storage = False
        

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

        # This should only be called once when first timestep is read.
        if self._initial_write:
            self._determine_units(ag)
            self._initialize_zarr_datasets(ts)
            if self._is_cloud_storage:
                self._initialize_memory_buffers()
                self._write_next_timestep = self._write_next_cloud_timestep
            self._initial_write = False
        return self._write_next_timestep(ts)
    
    def _determine_units(self, ag):
        """determine which units the file will be written with

        By default, it fills the :attr:`self.units` dictionary by copying the
        units dictionary of the parent file. Because Zarrtraj files have no
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
                    raise ValueError(f"{value} is not a unit recognized by"
                                     " MDAnalysis. Allowed units are:"
                                     f" {self._unit_translation_dict.keys()}"
                                     " For more information on units, see"
                                     " `MDAnalysis units`_.")
                else:
                    self.units[key] = self._new_units[key]

        if self.convert_units:
            # check if all units are None
            if not any(self.units.values()):
                raise ValueError("The trajectory has no units, but "
                                 "`convert_units` is set to ``True`` by "
                                 "default in MDAnalysis. To write the file "
                                 "with no units, set ``convert_units=False``.")

    def _open_file(self):
        if not isinstance(self.filename, zarr.Group):
            raise TypeError("Expected a Zarr group object, but " +
                            "received an instance of type {}"
                            .format(type(self.filename).__name__))
        # Verify group is open for writing
        if not self.filename.store.is_writeable():
            raise PermissionError("The Zarr group is not writeable")
        self.zarr_group = self.filename

        # fill in Zarrtraj metadata from kwargs
        self.zarr_group.require_group('zarrtraj')
        self.zarr_group['zarrtraj'].attrs['version'] = self.ZARRTRAJ_VERSION


    def _initialize_zarr_datasets(self, ts):
        """initializes all datasets that will be written to by
        :meth:`_write_next_timestep`. Datasets must be sampled at the same
        rate in version 1.0 of zarrtraj

        Note
        ----
        :exc:`NoDataError` is raised if no positions, velocities, or forces are
        found in the input trajectory.


        """

        # for keeping track of where to write in the dataset
        self._counter = 0
        first_dim = self.n_frames if self.n_frames is not None else 0

        # ask the parent file if it has positions, velocities, and forces
        # if prompted by the writer with the self._write_* attributes
        self._has = {group: getattr(ts, f'has_{attr}')
                     if getattr(self, f'_write_{attr}')
                     else False for group, attr in zip(
                     ('position', 'velocity', 'force'),
                     ('positions', 'velocities', 'forces'))}
        # initialize trajectory group
        self.zarr_group.require_group('particles').require_group('trajectory')
        self._traj = self.zarr_group['particles/trajectory']
        # NOTE: subselection init goes here when implemented
        # box group is required for every group in 'particles'
        # Initialize units group
        self.zarr_group['particles'].require_group('units')
        self._unit_group = self.zarr_group['particles']['units']

        self._traj.require_group('box')
        if ts.dimensions is not None and np.all(ts.dimensions > 0):
            self._traj['box'].attrs['boundary'] = 3*['periodic']
            self._traj['box'].require_group('edges')
            self._edges = self._traj.require_dataset('box/edges/value',
                                                     shape=(first_dim, 3, 3),
                                                     dtype=np.float32)
            self._step = self._traj.require_dataset('box/edges/step',
                                                    shape=(first_dim,),
                                                    dtype=np.int32)
            self._time = self._traj.require_dataset('box/edges/time',
                                                    shape=(first_dim,),
                                                    dtype=np.float32)
            self._has_edges = True
            self._set_attr_unit('length')
            self._set_attr_unit('time')
        else:
            # if no box, boundary attr must be "none" 
            self._traj['box'].attrs['boundary'] = 3*['none']
            self._create_step_and_time_datasets()
            self._has_edges = False

        if self.has_positions:
            self._create_trajectory_dataset('position')
            self._pos = self._traj['position/value']
            self._set_attr_unit('length')
        if self.has_velocities:
            self._create_trajectory_dataset('velocity')
            self._vel = self._traj['velocity/value']
            self._set_attr_unit('velocity')
        if self.has_forces:
            self._create_trajectory_dataset('force')
            self._force = self._traj['force/value']
            self._set_attr_unit('force')

        # intialize observable datasets from ts.data dictionary that
        # are NOT in self.data_blacklist
        #if self.data_keys:
        #    for key in self.data_keys:
        #        self._create_observables_dataset(key, ts.data[key])

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
        first_dim = self.n_frames if self.n_frames is not None else 0
        for group, value in self._has.items():
            if value:
                self._step = self._traj.require_dataset(f'{group}/step',
                                                        shape=(first_dim,),
                                                        dtype=np.int32)
                self._time = self._traj.require_dataset(f'{group}/time',
                                                        shape=(first_dim,),
                                                        dtype=np.float32)
                self._set_attr_unit('time')
                break

    def _create_trajectory_dataset(self, group):
        """helper function to initialize a dataset for
        position, velocity, and force"""

        if self.n_frames is None:
            shape = (0, self.n_atoms, 3)
        else:
            shape = (self.n_frames, self.n_atoms, 3)

        self._traj.require_group(group)
        self._traj.require_dataset(f'{group}/value',
                                   shape=shape,
                                   dtype=np.float32,
                                   chunks=self.chunks,
                                   filters=self.filters,
                                   compressor=self.compressor)
        # Hard linking in zarr is not possible, so we only keep one step and time aray
        # if 'step' not in self._traj[group]:
        #     self._traj[f'{group}/step'] = self._step
        # if 'time' not in self._traj[group]:
        #     self._traj[f'{group}/time'] = self._time

    def _set_attr_unit(self, unit):
        """helper function to set a unit attribute for an Zarr dataset"""

        if self.units[unit] is None:
            return

        self._unit_group.attrs[unit] = self._unit_translation_dict[unit][self.units[unit]]

    def _initialize_memory_buffers(self):
        # NOTE: chunks may change for time, step, and edges if using
        # in memory buffer that can fit multiple chunks
        self._time_buffer = np.zeros((self.n_buffer_frames,), dtype=np.float32)
        self._step_buffer = np.zeros((self.n_buffer_frames,), dtype=np.int32)
        self._edges_buffer = np.zeros((self.n_buffer_frames, 3, 3), dtype=np.float32)
        if self.has_positions:
            self._pos_buffer = np.zeros((self.n_buffer_frames, self.n_atoms, 
                                         3), dtype=np.float32)
        if self.has_velocities:
            self._force_buffer = np.zeros((self.n_buffer_frames, self.n_atoms, 
                                         3), dtype=np.float32)
        if self.has_forces:
            self._vel_buffer = np.zeros((self.n_buffer_frames, self.n_atoms, 
                                         3), dtype=np.float32)
        # Reduce cloud I/O by storing previous step val for error checking
        self._prev_step = None

    def _write_next_cloud_timestep(self, ts):
        """
        This method performs two steps:

        1. First, it appends the next frame to 
        
        
        """
        
        i = self._counter
        buffer_index = i % self.n_buffer_frames
        # Add the current timestep information to the buffer
        try:
            curr_step = ts.data['step']
        except KeyError:
            curr_step = ts.frame
        self._step_buffer[buffer_index] = ts.frame
        if self._prev_step is not None and curr_step < self._prev_step:
            raise ValueError("The Zarrtraj standard dictates that the step "
                             "dataset must increase monotonically in value.")
        self._prev_step = curr_step
        start = time.time()

        if self.units['time'] is not None:
            self._time_buffer[buffer_index] = self.convert_time_to_native(ts.time)
        else:
            self._time_buffer[buffer_index] = ts.time
    
        if self._has_edges:
            if self.units['length'] is not None:
                self._edges_buffer[buffer_index, :] = self.convert_pos_to_native(ts.triclinic_dimensions)
            else:
                self._edges_buffer[buffer_index, :] = ts.triclinic_dimensions

        if self.has_positions:
            if self.units['length'] is not None:
                self._pos_buffer[buffer_index, :] = self.convert_pos_to_native(ts.positions)
            else:
                self._pos_buffer[buffer_index, :] = ts.positions

        if self.has_velocities:
            if self.units['velocity'] is not None:
                self._vel_buffer[buffer_index, :] = self.convert_velocities_to_native(ts.velocities)
            else:
                self._vel_buffer[buffer_index, :] = ts.velocities

        if self.has_forces:
            if self.units['force'] is not None:
                self._force_buffer[buffer_index, :] = self.convert_forces_to_native(ts.forces)
            else:
                self._force_buffer[buffer_index, :] = ts.forces
            
        stop = time.time()
        print(f"Writing to the buffer took {stop-start} seconds")
        # If buffer is full or last write call, write buffer to cloud
        if (((i + 1) % self.n_buffer_frames == 0) or
                (i == self.n_frames - 1)):
            start = time.time()
            da.from_array(self._step_buffer[:buffer_index + 1]).to_zarr(self._step, region=(slice(i - buffer_index, i + 1),), return_stored=True)
            da.from_array(self._time_buffer[:buffer_index + 1]).to_zarr(self._time, region=(slice(i - buffer_index, i + 1),))
            if self._has_edges:
                da.from_array(self._edges_buffer[:buffer_index + 1]).to_zarr(self._edges, region=(slice(i - buffer_index, i + 1),))
            if self.has_positions:
                da.from_array(self._pos_buffer[:buffer_index + 1]).to_zarr(self._pos, region=(slice(i - buffer_index, i + 1),))
            if self.has_velocities:
                da.from_array(self._vel_buffer[:buffer_index + 1]).to_zarr(self._vel, region=(slice(i - buffer_index, i + 1),))
            if self.has_forces:
                da.from_array(self._force_buffer[:buffer_index + 1]).to_zarr(self._force, region=(slice(i - buffer_index, i + 1),))
            stop = time.time()
            print(f"Flushing this buffer took {stop-start} seconds")

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

        # Zarrtraj step refers to the integration step at which the data were
        # sampled, therefore ts.data['step'] is the most appropriate value
        # to use. However, step is also necessary in Zarrtraj to allow
        # temporal matching of the data, so ts.frame is used as an alternative
        new_shape = (self._step.shape[0] + 1,) + self._step.shape[1:]
        print(self._step.shape)
        self._step.resize(new_shape)
        print(self._step.shape)
        print(self._step[:])
        try:
            self._step[i] = ts.data['step']
        except KeyError:
            self._step[i] = ts.frame
        if len(self._step) > 1 and self._step[i] < self._step[i-1]:
            raise ValueError("The Zarrtraj standard dictates that the step "
                             "dataset must increase monotonically in value.")

        # the dataset.resize() method should work with any chunk shape
        new_shape = (self._time.shape[0] + 1,) + self._time.shape[1:]
        self._time.resize(new_shape)
        self._time[i] = ts.time

        if 'edges' in self._traj['box']:
            new_shape = (self._edges.shape[0] + 1,) + self._edges.shape[1:]
            self._edges.resize(new_shape)
            self._edges[i, :] = ts.triclinic_dimensions
        # These datasets are not resized if n_frames was provided as an
        # argument, as they were initialized with their full size.
        if self.has_positions:
            if self.n_frames is None:
                new_shape = (self._pos.shape[0] + 1,) + self._pos.shape[1:]
                self._pos.resize(new_shape)
            self._pos[i, :] = ts.positions
        if self.has_velocities:
            if self.n_frames is None:
                new_shape = (self._vel.shape[0] + 1,) + self._vel.shape[1:]
                self._vel.resize(new_shape)
            self._vel[i, :] = ts.velocities
        if self.has_forces:
            if self.n_frames is None:
                new_shape = (self._force.shape[0] + 1,) + self._force.shape[1:]
                self._force.resize(new_shape)
            self._force[i, :] = ts.forces

        #if self.convert_units:
        #    self._convert_dataset_with_units(i)

        self._counter += 1

    @property
    def has_positions(self):
        """``True`` if writer is writing positions from Timestep."""
        return self._has['position']

    @property
    def has_velocities(self):
        """``True`` if writer is writing velocities from Timestep."""
        return self._has['velocity']

    @property
    def has_forces(self):
        """``True`` if writer is writing forces from Timestep."""
        return self._has['force']
