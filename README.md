# Pink

[**Installation**](https://github.com/stephane-caron/pink#installation)
| [**Documentation**](https://scaron.info/doc/pink/)
| [**Example**](https://github.com/stephane-caron/pink#example)
| [**Contributing**](CONTRIBUTING.md)

[![build](https://img.shields.io/github/workflow/status/stephane-caron/pink/CI)](https://github.com/stephane-caron/pink/actions)
[![PyPI version](https://img.shields.io/pypi/v/pin-pink)](https://pypi.org/project/pin-pink/)
![Status](https://img.shields.io/pypi/status/pin-pink)

**P**ython **in**verse **k**inematics for your robot model based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## 🏗️ Upcoming changes

Pink is under active development and its API is **not stable**. Expect the following upcoming changes:

- Body task convergence unit test with several competing tasks
- v0.3: Add joint angle limits
- v0.4: Handle models without and with floating base
- v0.5: Reformulate task gains as time constants

And more: all [contributions](CONTRIBUTING.md) are welcome :)

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
