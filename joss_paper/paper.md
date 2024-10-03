---
title: 'Zarrtraj: A Python package for streaming molecular dynamics trajectories from cloud services'
tags:
  - streaming
  - molecular-dynamics
  - file-format
  - mdanalysis
  - zarr
authors:
  - name: Lawson Woods
    orcid: 0009-0003-0713-4167
    affiliation: [3, 4]
  - name: Hugo Macdermott-Opeskin
    orcid: 0000-0002-7393-7457
  - name: Edis Jakupovic 
    affiliation: [1, 2]
  - name: Yuxuan Zhuang
    orcid: 0000-0003-4390-8556
    affiliation: [5, 6]
  - name: Richard J Gowers
    orcid: 0000-0002-3241-1846
  - name: Oliver Beckstein
    orcid: 000-0003-1340-0831
    affiliation: [1, 2]
affiliations:
 - name: Department of Physics, Arizona State University, Tempe, Arizona, United States of America
   index: 1
 - name: Center for Biological Physics, Arizona State University, Tempe, AZ, United States of America
   index: 2
 - name: School of Computing and Augmented Intelligence, Arizona State University, Tempe, Arizona, United States of America
   index: 3
 - name: School of Molecular Sciences, Arizona State University, Tempe, Arizona, United States of America
   index: 4
 - name: Department of Computer Science, Stanford University, Stanford, CA 94305, USA.
   index: 5
 - name: Departments of Molecular and Cellular Physiology and Structural Biology, Stanford University School of Medicine, Stanford, CA 94305, USA.
   index: 6
date: 22 September 2024
bibliography: paper.bib
---

# Summary

Molecular dynamics (MD) simulations provide a microscope into the behavior of 
atomic-scale environments otherwise prohibitively difficult to observe, however,
the resulting trajectory data is too often siloed in a single institutions' 
HPC environment, rendering it unusable by the broader scientific community.
Zarrtraj enables these trajectories to be read directly from cloud storage providers
like AWS, Google Cloud, and Microsoft Azure into MDAnalysis, a popular Python 
package for analyzing trajectory data, providing a method to open up access to
trajectory data to anyone with an internet connection. Enabling cloud streaming
for MD trajectories empowers easier replication of published analysis results,
analyses of large, conglomerate datasets from different sources, and training
machine learning models without downloading and storing trajectory data.

# Statement of need

The computing power in HPC environments has increased to the point where
running simulation algorithms is often no longer the constraint in
obtaining scientific insights from molecular dynamics trajectory data. 
Instead, the ability to process, analyze and share large volumes of data provide 
new constraints on research in this field [@SharingMD:2019].

Other groups in the field recognize this same need for adherence to 
FAIR principles [@FAIR:2019] including 
MDsrv, a tool that can stream MD trajectories into a web browser for visual exploration [@MDsrv:2022], 
GCPRmd, a web service that builds on MDsrv to provide a predefined set of analysis results and simple 
geometric features for G-protein-coupled receptors [@GPCRmd:2019] [@GPCRome:2020], 
MDDB (Molecular Dynamics Data Bank), an EU-scale 
repository for biosimulation data [@MDDB:2024],
and MDverse, a prototype search engine 
for publicly-available GROMACS simulation data [@MDverse:2024]. 

While these efforts currently offer solutions for indexing,
searching, and vizualizing MD trajectory data, the problem of distributing trajectories 
in way that enables *NumPy*-like slicing and parallel reading for use in arbitrary analysis 
tasks remains.

Though exposing download links on the open internet offers a simple solution to this problem,
on-disk representations of molecular dynamics trajectories often range in size 
with large datasets up to TBs in scale [@ParallelAnalysis:2010] [@FoldingAtHome:2020],
so a solution which could prevent this 
duplication of storage and unnecessary download step would provide greater utility 
for the computational molecular sciences ecosystem.

Enter *Zarrtraj*, the first fully-functioning tool to our knowledge that allows 
streaming trajectories into analysis software using an established trajectory format.
*Zarrtraj* is implemented as an MDAnalysis [@MDAnalysis:2016] MDAKit [@MDAKits:2023] that
enables streaming MD trajectories in the popular HDF5-based H5MD format [@H5MD:2014]
from AWS S3, Google Cloud Buckets, and Azure Blob Storage & Data Lakes without ever downloading them.
This is possible thanks to the *Zarr* [@Zarr:2024] package which allows 
streaming array-like data from a variety of storage mediums and [Kerchunk](https://github.com/fsspec/kerchunk), 
which extends the capability of *Zarr* by allowing it to read HDF5 files.
Because it implements the standard MDAnalysis trajectory reader API,
*Zarrtraj* can leverage *Zarr*'s ability to read a slice of a file and 
to read a file in parallel, making it compatible with
analysis algorithms that use the "split-apply-combine" parallelization strategy [@SplitApplyCombine:2011].
In addition to the H5MD format, 
*Zarrtraj* can stream and write trajectories in the experimental ZarrMD
format, which ports the H5MD layout to the *Zarr* filetype.

This work builds on the existing MDAnalysis `H5MDReader`
[@H5MDReader:2021], and similarly uses *NumPy* [@NumPy:2020] as a common interface in-between MDAnalysis
and the file storage medium. *Zarrtraj* was inspired and made possible by similar efforts in the 
geosciences community to align data practices with FAIR principles [@PANGEO:2022].

With *Zarrtraj*, we envision research groups making their data publicly available 
via a cloud URL so that anyone can reuse their trajectories and reproduce their results.
Large databases, like MDDB and MDverse, can expose a URL associated with each 
trajectory in their databases so that users can make a query and immediately use the resulting
trajectories to run an analysis on the hits that match their search. Groups seeking to 
collect a large volume of trajectory data to train machine learning models [@MLMDMethods:2023] can make use
of our tool to efficiently and inexpensively obtain the data they need from these published 
URLs.

# Features and Benchmarks

Once imported, *Zarrtraj* allows passing trajectory URLs just like ordinary files:
```python
import zarrtraj
import MDAnalysis as mda

u = mda.Universe("topology.pdb", "s3://sample-bucket-name/trajectory.h5md")
```

Initial benchmarks show that *Zarrtraj* can iterate serially
through an AWS S3 cloud trajectory (load into memory one frame at a time)
at roughly 1/2 or 1/3 the speed it can iterate through the same trajectory from disk and roughly 
1/5 to 1/10 the speed it can iterate through the same trajectory on disk in XTC format (\autoref{fig:benchmark}).
However, it should be noted that this speed is influenced by network bandwidth and that
writing parallelized algorithms can offset this loss of speed as in \autoref{fig:RMSD}. 

![Benchmarks performed on a machine with 2 Intel Xeon 2.00GHz CPUs, 32GB of RAM, and an SSD configured with RAID 0. The trajectory used for benchmarking was the YiiP trajectory from MDAnalysisData [@YiiP:2019], a 9000-frame (90ns), 111,815 particle simulation of a membrane-protein system. The original 3.47GB XTC trajectory was converted into an uncompressed 11.3GB H5MD trajectory and an uncompressed 11.3GB ZarrMD trajectory using the MDAnalysis `H5MDWriter` and *Zarrtraj* `ZarrMD` writers, respectively. XTC trajectory read using the MDAnalysis `XTCReader` for comparison. \label{fig:benchmark}](benchmark.png)

![RMSD benchmarks performed on the same machine as \autoref{fig:benchmark}. YiiP trajectory aligned to first frame as reference using `MDAnalysis.analysis.align.AlignTraj` and converted to compressed, quantized H5MD (7.8GB) and ZarrMD (4.9GB) trajectories. RMSD performed using development branch of MDAnalysis (2.8.0dev) with "serial" and "dask" backends. See [this notebook](https://github.com/Becksteinlab/zarrtraj/blob/d4ab7710ec63813750d7224fe09bf5843e513570/joss_paper/figure_2.ipynb) for full benchmark codes. \label{fig:RMSD}](RMSD.png)

*Zarrtraj* is capable of making use of *Zarr*'s powerful compression and quantization when writing ZarrMD trajectories. 
The uncompressed MDAnalysisData YiiP trajectory in ZarrMD format is reduced from 11.3GB uncompressed 
to just 4.9GB after compression with the Zstandard algorithm [@Zstandard:2021] 
and quantization to 3 digits of precision. See [performance considerations](https://zarrtraj.readthedocs.io/en/latest/performance_considerations.html)
for more.


# Acknowledgements

We thank Dr. Jenna Swarthout Goddard for supporting the GSoC program at MDAnalysis and 
Martin Durant, author of Kerchunk, for helping refine and merge features in his upstream codebase 
necessary for this project. LW was a participant in the Google Summer of Code 2024 program.

# References