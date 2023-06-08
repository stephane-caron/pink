# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Example: UR5 arm
- Example: flying dual-arm with UR3

### Changed

- Drop support for Python 3.7

## [0.11.0] - 2023/05/01

### Added

- Example: Stretch RE1

### Changed

- Handle all frames, including non-body frames (thanks to @proyan)
- Rename ``BodyTask`` to ``FrameTask``

## [0.10.0] - 2023/03/30

### Added

- Base class ``Limit`` for configuration and velocity limits
- In-place integration of a velocity from a configuration
- Unit tests for configuration limits
- Unit tests for velocity limits

### Changed

- Add ``pink.limits`` submodule
- Configuration and velocity limits are now stacked rather than pre-reduced
- Move configuration limit to ``ConfigurationLimit`` class
- Move velocity limit to ``VelocityLimit`` class

## [0.9.0] - 2023/03/15

### Added

- Example: Draco 3 humanoid by @shbang91
- Joint-coupling task by @shbang91
- Linear holonomic task by @shbang91
- Unit tests for joint-coupling and linear holonomic tasks by @shbang91

### Changed

- Refactor ``apply_configuration`` into ``Configuration`` constructor
- Refactor ``assume_configuration`` into ``Configuration`` constructor
- Refactor task dynamics into ``compute_error`` and ``compute_jacobian``

## [0.8.0] - 2023/02/09

### Added

- BodyNotFound exception
- Document spatial and body minus between transforms
- Example: SigmaBan humanoid
- Forward keyword arguments of ``solve_ik`` to the backend QP solver

### Changed

- Return type of ``compute_velocity_limits`` now includes optionals
- VectorSpace type is now immutable

### Fixed

- Jacobian of the body task

## [0.7.0] - 2023/01/16

### Added

- Bounded subspace of tangent space, that is, restricted to bounded joints
- CI: test on both Ubuntu and macOS images
- Tangent space from Pinocchio model

### Changed

- Example: UR3 arm
- Expose ``pink.Task`` from top-level module
- Improve joint limit computations

### Fixed

- Empty inequalities when model has no bounded joint

## [0.6.0] - 2022/12/01

### Added

- Example: Omnidirectional three-wheeled robot

### Changed

- Example: reduce Upkie model in leg extension example
- Solver keyword argument is now mandatory

## [0.5.0] - 2022/09/26

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
- Handle unbounded velocity limits in QP formulation
- Posture task targets can be read directly from a configuration
- Simple rate limiter in ``pink.utils``

### Changed

- Raise an error when querying a body that doesn't exist
- Transition from ``pink.models`` to ``robot_descriptions``
- Update reference posture in Upkie wheeled biped example
- Warn when the backend QP solver is not explicitly selected

### Fixed

- Unbounded velocities when the backend solver is CVXOPT

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
