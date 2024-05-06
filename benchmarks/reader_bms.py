from zarrtraj import *
# from asv_runner.benchmarks.mark import skip_for_params
from zarr.storage import DirectoryStore, LRUStoreCache

import os

BENCHMARK_DATA_DIR = os.getenv("BENCHMARK_DATA_DIR")

class TrajReaderDiskBenchmarks(object):
    """Benchmarks for zarrtraj file striding."""
    # parameterize the input zarr group
    # these zarr groups should vary on
    # compression, filter_precision, chunk_frames
    # reads should be parameterized based on LRU cache_size- size + presence
    # cache_size sizes are 1, 10, 50, 98 (all) frames
    params = ([0, 1, 9], ["all", 3], [1, 10, 50], [40136, 401360, 2006800, 3933328])
    param_names = ['compressor_level', 'filter_precision', 'chunk_frames', 'cache_size']

    def setup(self, compressor_level, filter_precision, chunk_frames, cache_size):
        store = DirectoryStore(f"{BENCHMARK_DATA_DIR}/short_{compressor_level}_{filter_precision}_{chunk_frames}.zarrtraj")
        lruc = LRUStoreCache(store, max_size=cache_size)
        self.traj_file = zarr.open_group(store=lruc, mode='r')
        self.reader_object = ZarrTrajReader(self.traj_file)

    def time_strides(self, compressor_level, filter_precision, chunk_frames, cache_size):
        """Benchmark striding over full trajectory"""
        for ts in self.reader_object:
            pass