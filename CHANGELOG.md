# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Body task targets can be read directly from a robot configuration
- Example: double pendulum
- Example: Kinova Gen2 arm
- Example: loading a custom URDF description
- Example: visualization in MeshCat
- Example: visualization in yourdfpy
- Generalize configuration limits to any root joint
- Handle descriptions that have no velocity limit
- Handle general root joint in configuration limits
- Handle general root joint in posture task
- Posture task targets can be read directly from a configuration
- Simple rate limiter in ``pink.utils``

### Changed

- Raise an error when querying a body that doesn't exist
- Simplify Upkie wheeled biped example
- Transition from ``pink.models`` to ``robot_descriptions``
- Warn when the backend QP solver is not explicitly selected

## [0.4.0] - 2022/06/21

### Added

- Coveralls for continuous coverage testing
- Document differential inverse kinematics and task targets
- Single-task test on task target translations mapped to IK output translations

### Changed

- Argument to ``build_from_urdf`` functions is now the path to the URDF file
- Bumped status to beta
- Examples use the ``jvrc_description`` and ``upkie_description`` packages
- Use jvrc\_description and upkie\_description packages from PyPI
- Task is now an abstract base class

### Fixed

- Unit tests for robot models

## [0.3.0] - 2022/03/30

### Added

- Joint velocity limits
- Configuration limits

### Changed

- Bumped status to alpha
- Configuration limit check now has a tolerance argument

## [0.2.0] - 2022/03/29

### Added

- Check configuration limits against model
- Mock configuration type for unit testing
- Tangent member of a configuration
- Unit test the body task

### Changed

- Specify path when loading a model description
- Switch to the Apache 2.0 license
- ``build_jvrc_model`` is now ``build_from_urdf``

### Fixed

- Don't distribute robot models with the library
- IK unit test that used robot instead of configuration

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
