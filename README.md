# Pink

[**Installation**](https://github.com/tasts-robots/pink#installation)
| [**Documentation**](https://tasts-robots.org/doc/pink/)
| [**Example**](https://github.com/tasts-robots/pink#example)
| [**Contributing**](CONTRIBUTING.md)

[![Build](https://img.shields.io/github/workflow/status/tasts-robots/pink/CI)](https://github.com/tasts-robots/pink/actions)
[![Coverage](https://coveralls.io/repos/github/tasts-robots/pink/badge.svg?branch=master)](https://coveralls.io/github/tasts-robots/pink?branch=master)
[![PyPI version](https://img.shields.io/pypi/v/pin-pink)](https://pypi.org/project/pin-pink/)
![Status](https://img.shields.io/pypi/status/pin-pink)

**P**ython **in**verse **k**inematics for articulated robot models, based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Installation

```sh
pip install pin-pink
```

## Usage

Inverse kinematics in Pink is defined by [weighted tasks](https://scaron.info/robot-locomotion/inverse-kinematics.html). A task characterizes an objective to achieve, such as "put the foot frame at this location", by means of an objective function to be minimized, such as $\| {}^{world}p_{foot}^{target} - {}^{world}p_{foot} \|^2$. The robot is given multiple tasks to achieve, some of which may come into conflict. Conflicts are resolved by casting all objectives to the same unit, say ``[cost]``, and weighing all these normalized objectives relative to each other.

### Task costs

Here is the example of a biped robot that wants to control the position and orientation of of three of its frame, its base link, left and right contact frames. A fourth "posture" task, giving a desired angle for each joint, is added for regularization:

```python
from pink.tasks import BodyTask, PostureTask

tasks = {
    "base": BodyTask(
        "base",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    ),
    "left_contact": BodyTask(
        "left_contact",
        position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
        orientation_cost=0.0,  # [cost] / [rad]
    ),
    "right_contact": BodyTask(
        "right_contact",
        position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
        orientation_cost=0.0,  # [cost] / [rad]
    ),
    "posture": PostureTask(
        cost=1e-3,  # [cost] / [rad]
    ),
}
```

Position costs, which can be scalars or 3D vectors, specify how much each meter of position error "costs" in the overall normalized objective.

## Example

*Under construction...*

## Installation

```sh
pip install pin-pink
```

## 🏗️ Upcoming changes

Pink is under active development and its API is **not stable**. Expect the following upcoming changes:

- v0.5: Handle models without and with floating base
- Add a manipulator example
- Complete this README
- Reformulate task gains as time constants

All [contributions](CONTRIBUTING.md) are welcome :)

## History

Pink implements the same task-based inverse kinematics as [pymanoid](https://github.com/stephane-caron/pymanoid), but it is much simpler to install and runs faster thanks to Pinocchio. Its internal math is summarized in [this note](https://scaron.info/robot-locomotion/inverse-kinematics.html). If you find yourself needing to read that in order to use the library, it means the API has abstraction leakage, please open an issue :-)
