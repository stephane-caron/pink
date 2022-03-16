# Pink

[**Installation**](https://github.com/stephane-caron/pink#installation)
| [**Documentation**](https://scaron.info/doc/pink/)
| [**Usage**](https://github.com/stephane-caron/pink#usage)
| [**Example**](https://github.com/stephane-caron/pink#example)

[![build](https://img.shields.io/github/workflow/status/stephane-caron/pink/CI)](https://github.com/stephane-caron/pink/actions)
[![PyPI package](https://img.shields.io/pypi/v/pin-pink)](https://pypi.org/project/pin-pink/)
![Status](https://img.shields.io/pypi/status/pin-pink)

**P**ython **in**verse **k**inematics for your robot model based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Upcoming changes

Pink's API is **not stable**. Expect the following upcoming changes:

- **WIP:** Use ``ConfiguredRobot`` type with ``solve_ik``
- Formalize exceptions and configuration assertions
- Add body task
- v0.1: First version with functional body task and unit tests
- v0.2: Add posture task
- v0.3: Add joint angle limits
- v0.4: Reformulate task gains as time constants

Open questions:

- Expect kinematics to be already up-to-date in robot data?

## Installation

First, install [Pinocchio](https://github.com/stack-of-tasks/pinocchio), for instance by ``pip install pin``.

Then install Pink by:

```sh
pip install pin-pink
```

## Usage

*Under construction...*

## Example

*Under construction...*

## History

Pink implements the same task-based inverse kinematics as [pymanoid](https://github.com/stephane-caron/pymanoid), but it is much simpler to install and runs faster thanks to Pinocchio. Its internal math is summarized in [this note](https://scaron.info/robot-locomotion/inverse-kinematics.html). If you find yourself needing to read that in order to use the library, it means the API has abstraction leakage, please open an issue :-)
