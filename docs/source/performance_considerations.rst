Performance Considerations
--------------------------

For optimal performance in writing ``.zarrmd`` trajectories, we reccomend using 
Zarrtraj with the following settings:

Set ``n_frames`` to the number of frames being written
======================================================

By default, Zarrtraj will allocate ~12MB chunks at a time for the output file and deallocate the unused
memory by resizing the underlying Zarr dataset when the writer writes the last frame,
but providing the ``n_frames`` kwarg allows Zarrtraj to allocate only the memory that is needed. 
This will boost writing speed and reduce memory overhead when writing small trajectories.

Set ``precision=3``
===================

Under the hood, this kwarg is creating a ``numcodecs.quantize.Quantize`` filter to reduce
the precision of floating point data in the ``.zarrmd`` file to the number of digits specified. 
3 decimal places is the default precision for XTC and should be sufficient in the majority of cases.

Use ``compressor=numcodecs.Blosc(cname="zstd", clevel=9)``
==========================================================

From early prototyping, this compressor was found to provide the best compression
ratio for ``zarrmd`` trajectory data. While further benchmarking and experimentation is needed,
this setting in addition to ``precision=3`` provides the closest compression to 
XTC achieved thus far.


Example
=======

.. code-block:: python

    import numcodecs
    import zarrtraj
    import MDAnalysis as mda
    from MDAnalysisTests.datafiles import PSF, DCD

    u = mda.Universe(PSF, DCD)

    with mda.Writer(
        "test.zarrmd",
        n_atoms=u.trajectory.n_atoms,
        n_frames=u.trajectory.n_frames,
        precision=3,
        compressor=numcodecs.Blosc(cname="zstd", clevel=9),
    ) as W:
        for ts in u.trajectory:
            W.write(u)



