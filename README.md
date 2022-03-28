# Pink

[**Installation**](https://github.com/stephane-caron/pink#installation)
| [**Documentation**](https://scaron.info/doc/pink/)
| [**Usage**](https://github.com/stephane-caron/pink#usage)
| [**Example**](https://github.com/stephane-caron/pink#example)

[![build](https://img.shields.io/github/workflow/status/stephane-caron/pink/CI)](https://github.com/stephane-caron/pink/actions)
[![PyPI version](https://img.shields.io/pypi/v/pin-pink)](https://pypi.org/project/pin-pink/)
![Status](https://img.shields.io/pypi/status/pin-pink)

**P**ython **in**verse **k**inematics for your robot model based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## 🏗️ Upcoming changes

Pink is under active development and its API is **not stable**. Expect the following upcoming changes and more:

- Configuration vector copy when applying (currently a reference, can lead to inconsistencies)
- Body task convergence unit test with several competing tasks
- v0.2: Add posture task
- v0.3: Add joint angle limits
- v0.4: Reformulate task gains as time constants

## 👷 Contributing

There are many ways you can contribute to Pink, all of them welcome! Here are some ideas of increasing difficulty:

- Check out the [documentation](https://scaron.info/doc/pink/) and report mistakes or ask questions
- Try out the [examples](examples) and report any issue
- Suggests improvements to simplify the API
- Add your own [robot model](https://scaron.info/doc/pink/extras.html#module-pink.models) to the extras so that it can load by ``pink.models.build_from_urdf("/home/.../my_robot_description")``
- Write your own example
- Find a use case that is not covered and write a unit test for it
- Benchmark the performance of the current Configuration interface (``pin.computeJointJacobians`` + ``pin.getFrameJacobian``) compared to using ``pin.computeFrameJacobian``, e.g. depending on the number and locations of tasks
- Solve the question of numerical instability incurred by almost-unfeasible targets where LM damping does not kick in

If you are interested in helping out, open an issue so we can track progress. If you already have a PR, open it so we can review it :)

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
