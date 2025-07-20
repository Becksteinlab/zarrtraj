YiiP Protein Example
====================

To get started immediately with *Zarrtraj*, we have made the topology and trajectory of the 
`YiiP protein in a POPC membrane <https://www.mdanalysis.org/MDAnalysisData/yiip_equilibrium.html>`_
publicly available for streaming (no read/write credentials are needed as in :ref:`the walkthrough <_walkthrough>`). 
The trajectory is stored in in the :ref:`ZarrMD format <_zarrmd>` for optimal streaming performance. 

To access the trajectory, follow this example:

.. code-block:: python 

    import zarrtraj
    import MDAnalysis as mda
    import fsspec

    with fsspec.open("gcs://zarrtraj-test-data/YiiP_system.pdb", "r") as top:

        u = mda.Universe(
            top, "gcs://zarrtraj-test-data/yiip.zarrmd", topology_format="PDB"
        )
        protein = u.select_atoms("protein")

        for ts in u.trajectory[::100]:
            print(f"{ts.frame}, {ts.time}, {protein.center_of_mass()}")

While there is not yet an officially recommended way to access cloud-stored topologies, this
method of opening a Python `File`-like object from the topology URL in PDB format using 
`FSSpec <https://filesystem-spec.readthedocs.io/en/latest/>`_
works with MDAnalysis 2.7.0. Check back later for further development!

To see an executable example of running an MDAnalysis 
`RMSD <https://docs.mdanalysis.org/1.1.1/documentation_pages/analysis/rms.html>` analysis on this 
trajectory in a jupyter notebook, see 
`the example notebook on Github <https://github.com/Becksteinlab/zarrtraj/blob/main/examples/rmsd_yiip.ipynb>`.