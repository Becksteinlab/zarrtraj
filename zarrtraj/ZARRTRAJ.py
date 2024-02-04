import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates import base, core
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.due import due, Doi
from MDAnalysis.lib.util import store_init_arguments


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
    r"""
        Notation:
            (name) is an Zarr group
            {name} is an Zarr  group with arbitrary name
            [variable] is an Zarr array
            <dtype> is Zarr array datatype
            +-- is an attribute of a group or Zarr array

            Zarr root
                \-- (particles)
                \-- (units)
                    +-- distance <str>
                    +-- velocity <str>
                    +-- force <str>
                    +-- time <str>
                    +-- angle <str>
                \-- {group1}
                    \-- (box)
                        \-- (edges)
                            \-- [step] <int>, gives frame
                            +-- boundary : <str>, boundary conditions of unit cell
                            \-- [value] <float>, gives box dimensions
                                +-- unit <str>
                    \-- (position)
                        \-- [step] <int>, gives frame
                        \-- [time] <float>, gives time
                        \-- [value] <float>, gives numpy array of positions
                                                with shape (frame, n_atoms, 3)
                    \-- (velocity)
                        \-- [step] <int>, gives frame
                        \-- [time] <float>, gives time
                        \-- [value] <float>, gives numpy array of velocities
                                                with shape (frame, n_atoms, 3)
                    \-- (force)
                        \-- [step] <int>, gives frame
                        \-- [time] <float>, gives time
                        \-- [value] <float>, gives numpy array of forces
                                                with shape (frame, n_atoms, 3)
    """

    format = 'ZARRTRAJ'

    @store_init_arguments
    def __init__(self, filename,
                 **kwargs):
        
        if not HAS_ZARR:
            raise RuntimeError("Please install zarr")
        super(ZarrTrajReader, self).__init__(filename, **kwargs)
        self.filename = filename
        # NOTE: Not yet implemented
        # self.convert_units = convert_units 

        self.open_trajectory()
        
        # _has dictionary used for checking whether zarrtraj file has
        # 'position', 'velocity', or 'force' groups in the file
        self._has = {name: name in self._particle_group for
                     name in ('position', 'velocity', 'force')} 
        
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
        # self._set_translated_units()  # fills units dictionary NOTE not yet implemented
        self._read_next_timestep() 

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
                if value:
                    _ = self._particle_group[name]['step'][frame]
                    break
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

        # NOTE: Not sure about unit conversions yet
        #if self.convert_units:
        #    self._convert_units()

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
    
    def _read_next_timestep(self):
        """read next frame in trajectory"""
        return self._read_frame(self._frame + 1)
    
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

    # raise NotImplementedError("There is currently no writer for Zarrtraj files")

    def __init__(self, filename, n_atoms, n_frames=None,
                 convert_units=True, chunks=None,
                 positions=True, velocities=True,
                 forces=True, timeunit=None, lengthunit=None,
                 velocityunit=None, compressor=None,
                 filters=None, **kwargs):
        
        if not HAS_ZARR:
            raise RuntimeError("ZarrTrajWriter: Please install zarr")
        
        self.filename = filename
        if n_atoms == 0:
            raise ValueError("ZarrTrajWriter: no atoms in output trajectory")
        self.n_atoms = n_atoms
        self.n_frames = n_frames
        # NOTE: Consider changing default shape for cloud trajectories
        self.chunks = (1, n_atoms, 3) if chunks is None else chunks
        if self.chunks is False and self.n_frames is None:
            raise ValueError("ZarrTrajWriter must know how many frames will " +
                             "be written if ``chunks=False``.")
        self.filters = filters
        self.compressor = compressor
        self.contiguous = self.chunks is False and self.n_frames is not None
        self.convert_units = convert_units
        self.zarr_group = None
        
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

        #self._new_units = {'time': timeunit,
        #                   'length': lengthunit,
        #                   'velocity': velocityunit,
        #                   'force': forceunit}

        # Pull out various keywords to store metadata in 'h5md' group
        #self.author = author
        #self.author_email = author_email
        #self.creator = creator
        #self.creator_version = creator_version

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
        if self.zarr_group is None:
            # NOTE: not yet implemented
            #self._determine_units(ag)
            self._open_file()
            self._initialize_zarr_datasets(ts)

        return self._write_next_timestep(ts)

    #def _determine_units(self, ag):
    #    # NOTE: Rewrite docstring
    #    """determine which units the file will be written with
#
    #    By default, it fills the :attr:`self.units` dictionary by copying the
    #    units dictionary of the parent file. Because H5MD files have no
    #    standard unit restrictions, users may pass a kwarg in ``(timeunit,
    #    lengthunit, velocityunit, forceunit)`` to the writer so long as
    #    MDAnalysis has a conversion factor for it (:exc:`ValueError` raised if
    #    it does not). These custom unit arguments must be in
    #    `MDAnalysis notation`_. If custom units are supplied from the user,
    #    :attr`self.units[unit]` is replaced with the corresponding
    #    `unit` argument.
#
    #    """
#
    #    self.units = ag.universe.trajectory.units.copy()
#
    #    # set user input units
    #    #for key, value in self._new_units.items():
    #    #    if value is not None:
    #    #        if value not in self._unit_translation_dict[key]:
    #    #            raise ValueError(f"{value} is not a unit recognized by"
    #    #                             " MDAnalysis. Allowed units are:"
    #    #                             f" {self._unit_translation_dict.keys()}"
    #    #                             " For more information on units, see"
    #    #                             " `MDAnalysis units`_.")
    #    #        else:
    #    #            self.units[key] = self._new_units[key]
#
    #    if self.convert_units:
    #        # check if all units are None
    #        if not any(self.units.values()):
    #            raise ValueError("The trajectory has no units, but "
    #                             "`convert_units` is set to ``True`` by "
    #                             "default in MDAnalysis. To write the file "
    #                             "with no units, set ``convert_units=False``.")

    def _open_file(self):
        # NOTE: verify type checking is not handled before
        # writer object created
        if not isinstance(self.filename, zarr.Group):
            raise TypeError("Expected a Zarr group object, but " +
                            "received an instance of type {}"
                            .format(type(self.filename).__name__))
        # Verify group is open for writing
        if not self.filename.store.is_writeable():
            raise PermissionError("The Zarr group is not writeable")

        self.zarr_group = self.filename
        # NOTE: verify if there is use case for a non-empty group
        #if len(self.filename != 0):
        #    raise ValueError("Expected an empty Zarr group")

        # NOTE: Decide if author metadata necessary
        # fill in H5MD metadata from kwargs
        #self.zarr_group['h5md'].attrs['version'] = np.array(self.H5MD_VERSION)
        #self.zarr_group.require_group('h5md')
        #self.zarr_group['h5md'].require_group('author')
        #self.zarr_group['h5md/author'].attrs['name'] = self.author
        #if self.author_email is not None:
        #    self.zarr_group['h5md/author'].attrs['email'] = self.author_email
        #self.zarr_group['h5md'].require_group('creator')
        #self.zarr_group['h5md/creator'].attrs['name'] = self.creator
        #self.zarr_group['h5md/creator'].attrs['version'] = self.creator_version

    def _initialize_zarr_datasets(self, ts):
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
        self._traj.require_group('box')
        if ts.dimensions is not None and np.all(ts.dimensions > 0):
            self._traj['box'].attrs['boundary'] = 3*['periodic']
            self._traj['box'].require_group('edges')
            self._edges = self._traj.require_dataset('box/edges/value',
                                                     shape=(0, 3, 3),
                                                     maxshape=(None, 3, 3),
                                                     dtype=np.float32)
            self._step = self._traj.require_dataset('box/edges/step',
                                                    shape=(0,),
                                                    maxshape=(None,),
                                                    dtype=np.int32)
            self._time = self._traj.require_dataset('box/edges/time',
                                                    shape=(0,),
                                                    maxshape=(None,),
                                                    dtype=np.float32)
            #self._set_attr_unit(self._edges, 'length')
            #self._set_attr_unit(self._time, 'time')
        else:
            # if no box, boundary attr must be "none" 
            self._traj['box'].attrs['boundary'] = 3*['none']
            self._create_step_and_time_datasets()

        if self.has_positions:
            self._create_trajectory_dataset('position')
            self._pos = self._traj['position/value']
            #self._set_attr_unit(self._pos, 'length')
        if self.has_velocities:
            self._create_trajectory_dataset('velocity')
            self._vel = self._traj['velocity/value']
            #self._set_attr_unit(self._vel, 'velocity')
        if self.has_forces:
            self._create_trajectory_dataset('force')
            self._force = self._traj['force/value']
            #self._set_attr_unit(self._force, 'force')

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

        for group, value in self._has.items():
            if value:
                self._step = self._traj.require_dataset(f'{group}/step',
                                                        shape=(0,),
                                                        maxshape=(None,),
                                                        dtype=np.int32)
                self._time = self._traj.require_dataset(f'{group}/time',
                                                        shape=(0,),
                                                        maxshape=(None,),
                                                        dtype=np.float32)
                #self._set_attr_unit(self._time, 'time')
                break

    def _create_trajectory_dataset(self, group):
        """helper function to initialize a dataset for
        position, velocity, and force"""

        if self.n_frames is None:
            shape = (0, self.n_atoms, 3)
        else:
            shape = (self.n_frames, self.n_atoms, 3)

        chunks = None if self.contiguous else self.chunks

        self._traj.require_group(group)
        self._traj.require_dataset(f'{group}/value',
                                   shape=shape,
                                   dtype=np.float32,
                                   chunks=chunks,
                                   filters=self.filters,
                                   compressor=self.compressor)
        if 'step' not in self._traj[group]:
            self._traj[f'{group}/step'] = self._step
        if 'time' not in self._traj[group]:
            self._traj[f'{group}/time'] = self._time

    def _write_next_timestep(self, ts):
        """Write coordinates and unitcell information to Zarr group.

        Do not call this method directly; instead use
        :meth:`write` because some essential setup is done
        there before writing the first frame.

        The first dimension of each dataset is extended by +1 and
        then the data is written to the new slot.

        """

        i = self._counter

        # H5MD step refers to the integration step at which the data were
        # sampled, therefore ts.data['step'] is the most appropriate value
        # to use. However, step is also necessary in H5MD to allow
        # temporal matching of the data, so ts.frame is used as an alternative
        new_shape = (self._step.shape[0] + 1,) + self._step.shape[1:]
        self._step.resize(new_shape)
        try:
            self._step[i] = ts.data['step']
        except(KeyError):
            self._step[i] = ts.frame
        if len(self._step) > 1 and self._step[i] < self._step[i-1]:
            raise ValueError("The H5MD standard dictates that the step "
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

    

