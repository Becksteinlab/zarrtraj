YiiP Protein Example
====================

To get started immediately with *Zarrtraj*, we have made the topology and trajectory of the 
`YiiP protein in a POPC membrane <https://www.mdanalysis.org/MDAnalysisData/yiip_equilibrium.html>`_
publicly available for streaming. The trajectory is stored in in the `zarrmd` format 
for optimal streaming performance. 

To access the trajectory, follow this example:

.. code-block:: python 

    import zarrtraj
    import MDAnalysis as mda
    import fsspec


    with fsspec.open("gcs://zarrtraj-test-data/YiiP_system.pdb", "r") as top:

        u = mda.Universe(
            top, "gcs://zarrtraj-test-data/yiip.zarrmd", topology_format="PDB"
        )

        for ts in u.trajectory:
            # Do something


While there is not yet an officially recommended way to access cloud-stored topologies, this
method of opening a Python `File`-like object from the topology URL in PDB format using *FSSpec*
works with MDAnalysis 2.7.0. Check back later for further development!