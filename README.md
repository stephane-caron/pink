# Pink

[**Installation**](https://github.com/tasts-robots/pink#installation)
| [**Documentation**](https://tasts-robots.org/doc/pink/)
| [**Example**](https://github.com/tasts-robots/pink#example)
| [**Contributing**](CONTRIBUTING.md)

[![Build](https://img.shields.io/github/workflow/status/tasts-robots/pink/CI)](https://github.com/tasts-robots/pink/actions)
[![Coverage](https://coveralls.io/repos/github/tasts-robots/pink/badge.svg?branch=master)](https://coveralls.io/github/tasts-robots/pink?branch=master)
[![PyPI version](https://img.shields.io/pypi/v/pin-pink)](https://pypi.org/project/pin-pink/)
![Status](https://img.shields.io/pypi/status/pin-pink)

**P**ython **in**verse **k**inematics for your robot model based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## 🏗️ Upcoming changes

Pink is under active development and its API is **not stable**. Expect the following upcoming changes:

- Add posture task to Upkie example
- v0.4: Handle models without and with floating base
- Add a manipulator example
- Reformulate task gains as time constants

And more, all [contributions](CONTRIBUTING.md) are welcome :)

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
