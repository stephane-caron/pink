# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Check configuration limits against model
- Mock configuration type for unit testing
- Unit test the body task

### Changed

- Specify path when loading a model description
- Switch to the Apache 2.0 license
- ``build_jvrc_model`` is now ``build_from_urdf``

### Fixed

- Don't distribute robot models with the library

## [0.1.0] - 2022/03/17

### Added

- Body task
- Humanoid example

### Changed

- ConfiguredRobot(model, data) type is now Configuration(model, data, q)

### Fixed

- Add floating base joint when loading JVRC model

## [0.0.2] - 2022/03/16

### Added

- Base class for kinematic tasks
- Box minus operator
- ConfiguredRobot type in which we can query frame transforms and Jacobians
- Document joint limits API
- Use JVRC model for unit testing

### Fixed

- Documentation reads versions from the main module
- Remove unused mock imports

## [0.0.1] - 2022/02/06

Python package infrastructure.
