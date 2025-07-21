YiiP Protein Example
====================

To get started immediately with Zarrtraj, we have made the topology and trajectory of the 
`YiiP protein in a POPC membrane <https://www.mdanalysis.org/MDAnalysisData/yiip_equilibrium.html>`_
publicly available for streaming (no read/write credentials are needed as in :ref:`the walkthrough <walkthrough>`). 
The trajectory is stored in in the :ref:`ZarrMD format <zarrmd>` for optimal streaming performance. 

To access the trajectory, you can copy and paste this code into a python script:

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

In this example, we first import all necessary packages:
* :mod:`zarrtraj` for the functionality to read a trajectory from cloud-storage; it automatically hooks into :mod:`MDAnalysis` to make the trajectory available as part of a :class:`~MDAnalysis.core.universe.Universe`
* `fsspec <https://filesystem-spec.readthedocs.io>`_ to access a simple file in cloud storage with a file-like interface

We then create the basic MDAnalysis data structure, the :class:`~MDAnalysis.core.universe.Universe`, from the *topology file* "YiiP_system.pdb"
(which contains static information about the individual atoms (their names, types, and
organization as a biomolecule) and the data that change over time, namely the positions
of the atoms in the *trajectory* "yiip.zarrmd". Both topology and trajectory files can
be stored in the cloud (here in `Google Cloud Storage<https://cloud.google.com/storage>`_). 
The specific URI string for the *trajectory* tells MDAnalysis to use :mod:`zarrtraj`
to access the file.

We then use standard MDAnalysis functionality to first select a part of the system for
analysis, namely, the protein, and then *iterate* over the trajectory in steps of 100
frames. For each loaded frame, we print information about the frame number and recorded
time in the trajectory and perform a simple analysis task by calculating and printing
the center of mass (see :meth:`AtomGroup.center_of_mass <MDAnalysis.core.groups.AtomGroup.center_of_mass>`
of the protein.


While there is not yet an officially recommended way to access cloud-stored topologies, this
method of opening a Python `File`-like object from the topology URL in PDB format using 
`FSSpec <https://filesystem-spec.readthedocs.io/en/latest/>`_
works with MDAnalysis 2.7.0. Check back later for further development!

.. note:: 
   Whenever you want to read or write the :ref:`ZarrMD format <zarrmd>`, you need 
   to ``import zarrtraj``. You do not have to explicitly call any functions or classes
   inside the :mod:`zarrtraj` package because on import it automatically registers itself
   as a reader/writer with :mod:`MDAnalysis`. The ``import zarrtraj`` together with
   importing MDAnalysis (in any order) is sufficient for MDAnalysis to "know" how to
   work with trajectories stored in the cloud in zarr and h5md format.



.. SeeAlso::
   To see an executable example of running a full MDAnalysis
   :mod:`~MDAnalysis.analysis.rms.RMSD` analysis on this trajectory in a 
   Jupyter notebook, see the `rmsd_yiip.ipynb example notebook`_ on GitHub.


.. _`rmsd_yiip.ipynb example notebook`:
   https://github.com/Becksteinlab/zarrtraj/blob/main/examples/rmsd_yiip.ipynb