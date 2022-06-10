# Pink

<img src="https://user-images.githubusercontent.com/1189580/172797197-9aa46561-cfaa-4046-bd60-f681d85b055d.png" align="right" height=140>

[![Build](https://img.shields.io/github/workflow/status/tasts-robots/pink/CI)](https://github.com/tasts-robots/pink/actions)
[![Coverage](https://coveralls.io/repos/github/tasts-robots/pink/badge.svg?branch=master)](https://coveralls.io/github/tasts-robots/pink?branch=master)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://tasts-robots.org/doc/pink/)
[![PyPI version](https://img.shields.io/pypi/v/pin-pink)](https://pypi.org/project/pin-pink/)
![Status](https://img.shields.io/pypi/status/pin-pink)

**P**ython **in**verse **k**inematics for articulated robot models, based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Installation

```console
pip install pin-pink
```

## Usage

Inverse kinematics in Pink is defined by [weighted tasks](https://scaron.info/robot-locomotion/inverse-kinematics.html). A task characterizes an objective to achieve, such as "put the foot frame at this location", by means of an objective function to be minimized, such as $\Vert {}^{world}p_{foot}^{target} - {}^{world}p_{foot} \Vert^2$. The robot is given multiple tasks to achieve, some of which may come into conflict. Conflicts are resolved by casting all objectives to the same unit, say ``[cost]``, and weighing all these normalized objectives relative to each other.

### Task costs

Here is the example of a biped robot that controls the position and orientation of its base, left-contact and right-contact frames. A fourth "posture" task, giving a desired angle for each joint, is added for regularization:

```python
from pink.tasks import BodyTask, PostureTask

tasks = {
    "base": BodyTask(
        "base",
        position_cost=1.0,              # [cost] / [m]
        orientation_cost=1.0,           # [cost] / [rad]
    ),
    "left_contact": BodyTask(
        "left_contact",
        position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
        orientation_cost=0.0,           # [cost] / [rad]
    ),
    "right_contact": BodyTask(
        "right_contact",
        position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
        orientation_cost=0.0,           # [cost] / [rad]
    ),
    "posture": PostureTask(
        cost=1e-3,                      # [cost] / [rad]
    ),
}
```

Orientation (similarly position) costs, which can be scalars or 3D vectors, specify how much each radian of angular error "costs" in the overall normalized objective. When using 3D vectors, components are weighted anisotropically along each axis of the body frame.

### Task targets

Aside from their costs, most tasks take a second set of parameters called *targets*, for example a target transform for a body task or a target configuration vector for a posture task. Targets are set by the `set_target` function:

```python
    tasks["posture"].set_target(
        [1.0, 0.0, 0.0, 0.0] +           # floating base quaternion
        [0.0, 0.0, 0.0] +                # floating base position
        [0.0, 0.2, 0.0, 0.0, -0.2, 0.0]  # joint angles
    )
```

Body tasks can be initialized, for example, from the robot's neutral configuration:

```python
import pink
import upkie_description

robot = pink.models.build_from_urdf(upkie_description.urdf_path)
configuration = pink.apply_configuration(robot, robot.q0)
for body, task in tasks.items():
    if type(task) is BodyTask:
        task.set_target(configuration.get_transform_body_to_world(body))
```

Once a task has its cost and (if applicable) target defined, it can be used to solve inverse kinematics.

### Differential inverse kinematics

Pink solves differential inverse kinematics, meaning it outputs a velocity that steers the robot model towards a solution configuration (rather than solving for that configuration itself). If we keep integrating that velocity, and task targets don't change in the meantime, we will converge to that configuration:

```python
dt = 6e-3  # [s]
for t in np.arange(0.0, 42.0, dt):
    velocity = solve_ik(configuration, tasks.values(), dt)  # includes joint limits
    q = configuration.integrate(velocity, dt)
    configuration = pink.apply_configuration(robot, q)
    time.sleep(dt)
```

If task targets are continuously updated there will be no stationary solution to converge to, but the model will keep on tracking each target at best. Note that [`solve_ik`](https://scaron.info/doc/pink/inverse-kinematics.html#pink.solve_ik.solve_ik) takes into account both joint position and velocity limits read from URDF.

## Examples

All steps above are illustrated in the following examples:

- [Humanoid standing waving:](examples/humanoid_standing_waving.py) a humanoid model that positions its pelvis and feet while waiving its right wrist.
- [Upkie crouching:](examples/upkie_crouching.py) a wheeled-biped model that lifts its feet up and down.

## 🏗️ Upcoming changes

Pink is in beta and its API is not fully stable. Expect the following upcoming changes:

- Handle models without and with floating base
- Add a manipulator example
- Reformulate task gains as time constants

All [contributions](CONTRIBUTING.md) are welcome :)

## History

Pink implements the same task-based inverse kinematics as [pymanoid](https://github.com/stephane-caron/pymanoid), but it is much simpler to install and runs faster thanks to Pinocchio. Its internal math is summarized on [this page](https://scaron.info/robot-locomotion/inverse-kinematics.html). If you find yourself needing to read that in order to use the library, it means the API has abstraction leakage, please open an issue :-)
