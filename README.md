zarrtraj
==============================
[//]: # (Badges)

| **Latest release** | [![Last release tag][badge_release]][url_latest_release] ![GitHub commits since latest release (by date) for a branch][badge_commits_since]  [![Documentation Status][badge_docs]][url_docs]|
| :----------------- | :------- |
| **Status**         | [![GH Actions Status][badge_actions]][url_actions] [![codecov][badge_codecov]][url_codecov] |
| **Community**      | [![License: MIT][badge_license]][url_license]  [![Powered by MDAnalysis][badge_mda]][url_mda]|

[badge_actions]: https://github.com/Becksteinlab/zarrtraj/actions/workflows/gh-ci.yaml/badge.svg
[badge_codecov]: https://codecov.io/gh/Becksteinlab/zarrtraj/branch/main/graph/badge.svg
[badge_commits_since]: https://img.shields.io/github/commits-since/Becksteinlab/zarrtraj/latest
[badge_docs]: https://readthedocs.org/projects/zarrtraj/badge/?version=latest
[badge_license]: https://img.shields.io/badge/License-MIT-blue.svg
[badge_mda]: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
[badge_release]: https://img.shields.io/github/release-pre/Becksteinlab/zarrtraj.svg
[url_actions]: https://github.com/Becksteinlab/zarrtraj/actions?query=branch%3Amain+workflow%3Agh-ci
[url_codecov]: https://codecov.io/gh/Becksteinlab/zarrtraj/branch/main
[url_docs]: https://zarrtraj.readthedocs.io/en/latest/?badge=latest
[url_latest_release]: https://github.com/Becksteinlab/zarrtraj/releases
[url_license]: https://opensource.org/license/mit
[url_mda]: https://www.mdanalysis.org

Zarrtraj is an MDAnalysis MDAKit that provides the ability to read and write H5MD-formatted trajectory data in MDAnalysis using Zarr. 
Zarrtraj can read trajectories locally and from AWS S3, Google Cloud Buckets, and Azure Blob Storage & DataLakes.
It can read both [H5MD-formatted files stored in hdf5](https://www.nongnu.org/h5md/h5md.html) (.h5md files) and [H5MD-formatted files stored 
in Zarr](https://zarrtraj.readthedocs.io/en/latest/zarrmd-file-spec/v0.2.0.html) (.zarrmd files).

Zarrtraj is installable via both pip and conda forge:
```bash
pip install zarrtraj
```
```bash
conda install -c conda-forge zarrtraj
```

For more information on installation and usage, see the [zarrtraj documentation](https://zarrtraj.readthedocs.io/en/latest/index.html)

Zarrtraj is bound by a [Code of Conduct](https://github.com/Becksteinlab/zarrtraj/blob/main/CODE_OF_CONDUCT.md).

### Copyright

The Zarrtraj source code is hosted at https://github.com/Becksteinlab/zarrtraj
and is available under the MIT License (see the file [LICENSE](https://github.com/Becksteinlab/zarrtraj/blob/main/LICENSE)).

Copyright (c) 2024, Lawson Woods


#### Acknowledgements
 
Project based on the 
[MDAnalysis Cookiecutter](https://github.com/MDAnalysis/cookiecutter-mda) version 0.1.
Please cite [MDAnalysis](https://github.com/MDAnalysis/mdanalysis#citation) when using Zarrtraj in published work.
