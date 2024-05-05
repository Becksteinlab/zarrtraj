from zarrtraj import *

class TrajReaderDiskIteration(object):
    """Benchmarks for zarrtraj file striding."""
    # parameterize the input zarr group
    # these zarr groups should vary on
    # long vs short traj length, compression, filters, chunking
    # reads should be parameterized based on LRU cache- size + presence
    params = ([])
    param_names = ['traj_length', 'compressor', 'filters', 'chunks', 'cache']

    def setup(self):
        self.traj_file, self.traj_reader = [ZARRTRAJ, ZarrTrajReader]
        self.reader_object = self.traj_reader(self.traj_file)

    def time_strides(self):
        """Benchmark striding over full trajectory
        test files for each format.
        """
        for ts in self.reader_object:
            pass

class TrajReaderAWSIterations