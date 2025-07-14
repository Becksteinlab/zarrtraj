# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
The rules for this file:
  * entries are sorted newest-first.
  * summarize sets of changes - don't reproduce every git log comment here.
  * don't ever delete anything.
  * keep the format consistent:
    * do not use tabs but use spaces for formatting
    * 79 char width
    * YYYY-MM-DD date format (following ISO 8601)
  * accompany each entry with github issue/PR number (Issue #xyz)
-->
## [0.3.2]

## Authors
- ljwoods2

### Fixed
- Pinned Zarr<3.0 to avoid issues importing LRUCache (Issue #81, PR #80)

### Changed
- Added tests for Python 3.13 (Issue #84, PR #83)

## [0.3.1]

## Authors
- ljwoods2

### Fixed
- Fixed ZARRH5MDWriter bug which caused writer to fail unexpectedly on
  writing scalar ts.data attributes

### Changed
- Relicense to MIT (Issue #74)

## [0.3.0] 2024-10-24

## Authors
- ljwoods2

## Added
- added CITATION.cff file (issue #69, PR #68)

## [0.2.1] 2024-07-28


### Authors
- ljwoods2

### Added
- Experimental support for Google Cloud Buckets and Azure Blob Storage & Data Lakes

### Fixed
- Fixed bug which caused writer to fail when optional `n_frames` kwarg not provided

### Changed


### Deprecated

## [0.2.0]


### Authors
- ljwoods2

### Added
- Rewrite of repository, zarrtraj supports reading H5MD-formatted files

### Fixed


### Changed


### Deprecated
- Proprietary zarrtraj format no longer supported

## [0.1.0]

### Authors
- ljwoods2

### Added
- implemented v0.1.0 zarrtraj spec

### Fixed


### Changed


### Deprecated



