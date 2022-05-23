# Pink

[**Installation**](https://github.com/tasts-robots/pink#installation)
| [**Documentation**](https://tasts-robots.org/doc/pink/)
| [**Example**](https://github.com/tasts-robots/pink#example)
| [**Contributing**](CONTRIBUTING.md)

[![Build](https://img.shields.io/github/workflow/status/tasts-robots/pink/CI)](https://github.com/tasts-robots/pink/actions)
[![Coverage](https://coveralls.io/repos/github/tasts-robots/pink/badge.svg?branch=master)](https://coveralls.io/github/tasts-robots/pink?branch=master)
[![PyPI version](https://img.shields.io/pypi/v/pin-pink)](https://pypi.org/project/pin-pink/)
![Status](https://img.shields.io/pypi/status/pin-pink)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**P**ython **in**verse **k**inematics for your robot model based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Usage

*Under construction...*

Pink implements the same task-based inverse kinematics as [pymanoid](https://github.com/stephane-caron/pymanoid), but it is much simpler to install and runs faster thanks to Pinocchio. Its internal math is summarized in [this note](https://scaron.info/robot-locomotion/inverse-kinematics.html). If you find yourself needing to read that in order to use the library, it means the API has abstraction leakage, please open an issue :-)

## Example

*Under construction...*

## Installation

Make sure [Pinocchio](https://github.com/stack-of-tasks/pinocchio) is
installed, for instance by ``pip install pin``. Then install Pink by:

```sh
pip install pin-pink
```

## 🏗️ Upcoming changes

Pink is under active development and its API is **not stable**. Expect the following upcoming changes:

- v0.4: Handle models without and with floating base
- Add a manipulator example
- Complete this README
- Reformulate task gains as time constants

All [contributions](CONTRIBUTING.md) are welcome :)
