Benchmarks
==========

Speed benchmarks are available via AirSpeedVelocity
`here <https://becksteinlab.github.io/zarrtraj/>`_

Initial benchmarks were performed in the `Beckstein Lab <https://becksteinlab.physics.asu.edu/>`
on Spudda, which has:
- 2 Intel(R) Xeon(R) CPU E5-2620 0 @ 2.00GHz
- 12 total cores
- 32GB RAM

Local file speed tests were performed in the 1.31 TB SSD scratch space using RAID 0.

The following metrics were measured:
- ``ZARRH5MDDiskStrideTime``: Time to iterate through all timesteps in SSD-stored trajectory files
    using compressed & uncompressed zarrmd and h5md files.
- ``ZARRH5MDS3StrideTime``: Time to iterate through all timesteps in S3-stored trajectory files
    using compressed & uncompressed zarrmd and h5md files.
- ``H5MDReadersDiskStrideTime``: Time to iterate through all timesteps in an SSD-stored trajectory file 
    using compressed & uncompressed h5md files comparing the :class:`MDAnalysis.coordinates.H5MDReader` 
    and :class:`zarrtraj.ZARRH5MDReader` classes.
- ``H5MDFmtDiskRMSFTime``: Time to calculate the root mean square fluctuation (RMSF) of the trajectory 
    using compressed & uncompressed SSD-stored zarrmd files comparing the :class:`MDAnalysis.analysis.rms.RMSF`
    method and a ``dask`` parallelized version of the same method.
- ``H5MDFmtAWSRMSFTime``: Time to calculate the root mean square fluctuation (RMSF) of the trajectory 
    using compressed & uncompressed S3-stored zarrmd files comparing the :class:`MDAnalysis.analysis.rms.RMSF`
    method and a ``dask`` parallelized version of the same method.

For all benchmarks, the trajectory file used was the 
`YiiP trajectory <https://www.mdanalysis.org/MDAnalysisData/yiip_equilibrium.html>`_
aligned using the ``MDAnalysis`` :class:`MDAnalysis.analysis.align.AlignTraj` class
rewritten in the ``zarrmd`` and ``H5MD`` formats using the ``zarrtraj`` package.

Highlights:
- The dask parallelized RMSF calculation performed ~4x faster than the serial calculation via MDAnalysis
  on both local and S3-stored trajectory files. While this method is not yet implemented in ``zarrtraj``,
  it may be in a future version
- The ``ZARRH5MDReader`` class performed ~2-4x faster than the ``H5MDReader`` class on iterating through
  local trajectory files, though this may be because the files were written using a chunking strategy
  favorable to the ``ZARRH5MDReader`` class.
- For each trajectory file, iterating through its timesteps from S3 storage took about twice as logging
  as iterating through the same file from local SSD storage.
  
