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

    format = 'ZARRTRAJ'

    @store_init_arguments
    def __init__(self, filename,
                 convert_units=True,
                 driver=None,
                 comm=None,
                 **kwargs):
        
        if not HAS_ZARR:
            raise RuntimeError("Please install h5py")
        super(ZarrTrajReader, self).__init__(filename, **kwargs)
        self.filename = filename
        self.convert_units = convert_units # NOTE: Not yet implemented

        self.open_trajectory()
        if self._particle_group['box'].attrs['dimension'] != 3:
            raise ValueError("MDAnalysis only supports 3-dimensional"
                             " simulation boxes")
        
        # _has dictionary used for checking whether zarrtraj file has
        # 'position', 'velocity', or 'force' groups in the file
        self._has = {name: name in self._particle_group for
                     name in ('position', 'velocity', 'force')} # NOTE: _has not yet implemented
        
        # Gets some info about what settings the datasets were created with
        # from first available group
        for name, value in self._has.items():
            if value:
                dset = self._particle_group[f'{name}/value']
                self.n_atoms = dset.shape[1]
                self.compression = dset.compression
                self.compression_opts = dset.compression_opts
                break
        else:
            raise NoDataError("Provide at least a position, velocity"
                              " or force group in the h5md file.")
        
        # NOTE: _Timestep not yet implemented
        self.ts = self._Timestep(self.n_atoms,
                                 positions=self.has_positions,
                                 velocities=self.has_velocities,
                                 forces=self.has_forces,
                                 **self._ts_kwargs)
        
        self.units = {'time': None,
                      'length': None,
                      'velocity': None,
                      'force': None}
        self._set_translated_units()  # fills units dictionary NOTE not yet implemented
        self._read_next_timestep() # NOTE: Not yet implemented

    @staticmethod
    def _format_hint(thing):
        """Can this Reader read *thing*"""
        # Check if the object is already a zarr.Group
        # If it isn't, try opening it as a group and if it excepts, return False
        if not HAS_ZARR:
            return False
      
        if isinstance(thing, zarr.Group):
            return True

        try:
            # Try opening the file with Zarr
            zarr.open(thing, mode='r')
            return True
        except Exception:
            # If an error occurs, it's likely not a Zarr file
            return False
        
    def open_trajectory(self):
        """opens the trajectory file using zarr library"""
        self._frame = -1
        if isinstance(self.filename, zarr.Group):
            self._file = self.filename
        else:
            self._file = zarr.open_group(self.filename,
                                         mode='r')
     
    @staticmethod
    def parse_n_atoms(filename):
        with zarr.open_group(filename, 'r') as f:
            for group in f['particles/trajectory']:
                if group in ('position', 'velocity', 'force'):
                    n_atoms = f[f'particles/trajectory/{group}/value'].shape[1]
                    return n_atoms

            raise NoDataError("Could not construct minimal topology from the "
                            "Zarrtraj trajectory file, as it did not contain a "
                            "'position', 'velocity', or 'force' group. "
                            "You must include a topology file.")
        
    

class ZarrTrajWriter(base.WriterBase):
    format = 'ZARRTRAJ'
    multiframe = True

    raise NotImplementedError("There is currently no writer for TNG files")

