import abc
import threading
from collections import deque


class FrameCache(abc.ABC):

    def __init__(
        self, open_file, cache_size, timestep, frames_per_chunk, *args, **kwargs
    ):
        self._file = open_file
        self._cache_size = cache_size
        self._timestep = timestep
        self._frames_per_chunk = frames_per_chunk

    def update_frame_seq(self, frame_seq):
        """Call this in the reader's _read_next_timestep
        method on the first read
        """
        self._frame_seq = frame_seq

    @abc.abstractmethod
    def update_desired_dsets(self, *args, **kwargs):
        """Call this in the reader on the first read
        to inform the cache of which datasets it
        should pull into the cache for each call to
        load_frame()

        The reader is reponsible for ensuring
        the requested datasets are actually
        present in the file before requesting them
        from the cache
        """

    @abc.abstractmethod
    def load_frame(self, *args, **kwargs):
        """Call this in the reader's
        _read_next_frame() method
        """

    @abc.abstractmethod
    def cleanup(self, *args, **kwargs):
        """Call this in the reader's close() method"""


# Not yet implemented
class AsyncFrameCache(FrameCache, threading.Thread):

    def __init__(self):
        super(FrameCache, self).__init__()
        self._frame_seq = deque([])
        self._stop_event = threading.Event()
        self._first_read = True
        self._mutex = threading.Lock()
        self._frame_available = threading.Condition(self._mutex)

    def load_first_frame(self):
        pass

    def load_frame(self):
        if self._first_read:
            self._first_read = False
            self.start()
        frame = self._reader_q.popleft()
        with self._frame_available:
            while not self._cache_contains(frame):
                self._frame_available.wait()

            self._load_timestep(frame)

    def run(self):
        while self._frame_seq and not self._stop_event:
            frame = self._frame_seq.pop(0)
            key = frame % self._frames_per_chunk

            if self._cache_contains(key):
                continue
            elif self._num_cache_frames < self._max_cache_frames:
                self._get_key(key)
            else:
                with self._mutex:
                    eviction_key = self._predict()
                    self._evict(eviction_key)
                    self._get_key(key)
                    self._frame_available.notify()

    def _stop(self):
        self._stop_event.set()

    def cleanup(self):
        self._stop()

    def _predict(frame_seq, cache, frame_seq_len, index):
        """
        1. Attempt to find a page that is
           never referenced in the future
        2. If not possible, return the page that is referenced
           furthest in the future

        Cache is a list of available chunks

        returns the key of the chunk to be replaced
        chunks have keys based on frame number % chunksize
        """
        res = -1
        farthest = index
        for i in range(len(cache)):
            j = 0
            for j in range(index, frame_seq_len):
                if cache[i] == frame_seq[j]:
                    if j > farthest:
                        farthest = j
                        res = i
                    break
            # If a page is never referenced in future, return it.
            if j == frame_seq_len:
                return i
        # If all of the frames were not in future, return any of them, we return 0. Otherwise we return res.
        return 0 if (res == -1) else res

    @abc.abstractmethod
    def _stop(self):
        pass

    @abc.abstractmethod
    def _cache_contains(self, frame):
        pass

    @abc.abstractmethod
    def _get_key(self, key):
        """Loads the chunk with the given key
        from the file into the cache"""

    @abc.abstractmethod
    def _evict(self, key):
        """Removes the chunk with the given key from
        the cache"""

    @abc.abstractmethod
    def _load_timestep(self, frame):
        """Loads the frame with the given frame number
        into the timestep object associated with the cache class
        """
