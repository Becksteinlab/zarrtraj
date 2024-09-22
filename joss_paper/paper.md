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
    affiliation: 1 
  - name: Hugo Macdermott-Opeskin
    orcid: 0000-0002-7393-7457
    affiliation: 1
  - name: Oliver Beckstein
    orcid: 000-0003-1340-0831
    affiliation: 1
  - name: Edis Jakupovic 
    affiliation: 1
  - name: Yuxuan Zhuang
    orcid: 0000-0003-4390-8556
    affiliations: 1
  - name: Richard J Gowers
    orcid: 0000-0002-3241-1846
    affiliations: 1
affiliations:
 - name: Placeholder
   index: 1
date: 22 September 2024
bibliography: paper.bib
---

# Summary

Molecular dynamics simulations provide a microscope into the behavior of 
atomic-scale environments otherwise prohibitively diffult to observe, however,
the resulting trajectory data is too often siloed in a single institutions' 
HPC environment, rendering it unusable by the broader scientific community.
Zarrtraj enables these trajectories to be read directly from cloud storage providers
like AWS, Google Cloud, and Microsoft Azure into MDAnalysis, a popular Python 
package for analyzing trajectory data, providing a method to open up access to
trajectory data to anyone with an internet connection.

# Statement of need

The computing power in HPC environments has increased to the point where
running simulation algorithms is often no longer the constraint in obtaining
molecular dynamics trajectory data for analysis. Instead, the speed of writing to disk and
the ability to share generated data provide new constraints on research in this field.
While exposing download links on the open internet offers one solution this problem,
molecular dynamics trajectories are often massive files which are slow to download and expensive
to store at scale, so a solution which could prevent this duplication of storage and uneccessary 
download step would be more ideal.

Enter `Zarrtraj`, an `MDAnalysis` [@MDAnalysis:2016] `MDAKit` [@MDAKits:2023] which enables 
streaming these trajectories from AWS S3, Google Cloud Buckets, and Azure Blob Storage and Data
Lakes without ever downloading them using the standard `MDAnalysis` trajectory reader API.
This is possible thanks to the `Zarr` [@Zarr:2024] package which allows streaming array-like
data from a variety of storage mediums and `Kerchunk`, which extends the capability of `Zarr`
by allowing it to read `HDF5` files in addition to `Zarr` files. Trajectory data can be streamed
in the `H5MD` format [@H5MD:2014], which builds on top of `HDF5`, and the experimental `ZarrMD` format,
which ports `H5MD` to the `Zarr` filetype. This work builds on the existing `MDAnalysis` `H5MDReader`
[@H5MDReader:2021], and similarly uses `NumPy` [@NumPy:2020] as a common interface in-between `MDAnalysis`
and the file storage medium.

<!-- 
# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements
Thank you to Google for supporting the Google Summer of Code program (GSoC) which provided
financial support for this project. Thank you to Dr. Hugo MacDermott-Opeskin and Dr. Yuxuan Zhuang 
for their mentorship and feedback and to Dr. Jenna Swarthout Goddard for supporting the GSoC program at MDAnalysis. 
Thank you to Dr. Oliver Beckstein and Edis Jakupovic for lending their expertise in H5MD and all things MDAnalysis. 
Finally, thanks to Martin Durant, author of Kerchunk, for helping refine and merge features in his upstream codebase 
necessary for this project.

# References