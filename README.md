# Pink

[![Build](https://img.shields.io/github/actions/workflow/status/stephane-caron/pink/ci.yml?branch=main)](https://github.com/stephane-caron/pink/actions)
[![Documentation](https://img.shields.io/github/actions/workflow/status/stephane-caron/pink/docs.yml?branch=main&label=docs)](https://stephane-caron.github.io/pink/)
[![Coverage](https://coveralls.io/repos/github/stephane-caron/pink/badge.svg?branch=main)](https://coveralls.io/github/stephane-caron/pink?branch=main)
[![Conda version](https://anaconda.org/conda-forge/pink/badges/version.svg)](https://anaconda.org/conda-forge/pink)
[![PyPI version](https://img.shields.io/pypi/v/pin-pink)](https://pypi.org/project/pin-pink/)

**P**ython **in**verse **k**inematics for articulated robot models, based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

![Banner for Pink v0.5.0](https://user-images.githubusercontent.com/1189580/192318997-ed7574c3-8238-451d-9548-a769d46ec03b.png)

## Installation

For best performance we recommended installing Pink from Conda:

```console
conda install -c conda-forge pink
```

You can also install it from PyPI:

```console
pip install pin-pink
```

## Usage

Pink solves differential inverse kinematics by [weighted tasks](https://scaron.info/robot-locomotion/inverse-kinematics.html). A task is defined by a *residual* function $e(q)$ of the robot configuration $q \in \mathcal{C}$ to be driven to zero. For instance, putting a foot position $p_{foot}(q)$ at a given target $p_{foot}^{\star}$ can be described by the position residual:

$$
e(q) = p_{foot}^{\star} - p_{foot}(q)
$$

In differential inverse kinematics, we compute a velocity $v \in \mathfrak{c}$ that satisfies the first-order differential equation:

$$
J_e(q) v = \dot{e}(q) = -\alpha e(q)
$$

where $J\_e(q) := \frac{\partial e}{\partial q}$ is the [task Jacobian](https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html). We can define multiple tasks, but some of them will come into conflict if they can't be all fully achieved at the same time. Conflicts are resolved by casting all objectives to a common unit, and weighing these normalized objectives relative to each other. We also include configuration and velocity limits, making our overall optimization problem a quadratic program:

$$
\begin{align}
\underset{v \in \mathfrak{c}}{\text{minimize}} \ & \sum_{\text{task } e} \Vert J_e(q) v + \alpha e(q) \Vert^2_{W_e} \\
\text{subject to} \ & v_{\text{min}}(q) \leq v \leq v_{\text{max}}(q)
\end{align}
$$

Pink provides an API to describe the problem as tasks with targets, and automatically build and solve the underlying quadratic program.

### Task costs

Here is the example of a biped robot that controls the position and orientation of its base, left and right contact frames. A fourth "posture" task, giving a preferred angle for each joint, is added for regularization:

```python
from pink.tasks import FrameTask, PostureTask

tasks = {
    "base": FrameTask(
        "base",
        position_cost=1.0,              # [cost] / [m]
        orientation_cost=1.0,           # [cost] / [rad]
    ),
    "left_contact": FrameTask(
        "left_contact",
        position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
        orientation_cost=0.0,           # [cost] / [rad]
    ),
    "right_contact": FrameTask(
        "right_contact",
        position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
        orientation_cost=0.0,           # [cost] / [rad]
    ),
    "posture": PostureTask(
        cost=1e-3,                      # [cost] / [rad]
    ),
}
```

Orientation (similarly position) costs can be scalars or 3D vectors. They specify how much each radian of angular error "costs" in the overall normalized objective. When using 3D vectors, components are weighted anisotropically along each axis of the body frame.

### Task targets

Aside from their costs, most tasks take a second set of parameters called *target*. For example, a frame task aims for a target transform, while a posture task aims for a target configuration vector. Targets are set by the `set_target` function:

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
from robot_descriptions.loaders.pinocchio import load_robot_description

robot = load_robot_description("upkie_description")
configuration = pink.Configuration(robot.model, robot.data, robot.q0)
for body, task in tasks.items():
    if type(task) is FrameTask:
        task.set_target(configuration.get_transform_frame_to_world(body))
```

A task can be added to the inverse kinematics once both its cost and target (if applicable) are defined.

### Differential inverse kinematics

Pink solves differential inverse kinematics, meaning it outputs a velocity that steers the robot towards achieving all tasks at best. If we keep integrating that velocity, and task targets don't change over time, we will converge to a stationary configuration:

```python
dt = 6e-3  # [s]
for t in np.arange(0.0, 42.0, dt):
    velocity = solve_ik(configuration, tasks.values(), dt, solver="quadprog")
    configuration.integrate_inplace(velocity, dt)
    time.sleep(dt)
```

If task targets are continuously updated, there will be no stationary solution to converge to, but the model will keep on tracking each target at best. Note that [`solve_ik`](https://stephane-caron.github.io/pink/inverse-kinematics.html#pink.solve_ik.solve_ik) will take care of both configuration and velocity limits read from the robot model.

## Examples

Illustrated examples showcase how Pink performs on various robot morphologies:

- Arm: [UR5](https://github.com/stephane-caron/pink/tree/main/examples#arm-ur5) and [UR5 with end-effector limits](https://github.com/stephane-caron/pink/tree/main/examples/barriers#arm-ur5)
- Dual arms: [Flying dual-arm UR3](https://github.com/stephane-caron/pink/tree/main/examples#flying-dual-arm-ur3)
- Dual arms: [Yumi with spherical self-collision avoidance](https://github.com/stephane-caron/pink/tree/main/examples/barriers#yumi-end-effector-self-collision-avoidance)
- Dual arms: [Iiwa with whole-body self-collision avoidance](https://github.com/stephane-caron/pink/tree/main/examples/barriers#iiwa-whole-body-collision-avoidance)
- Humanoid: [Draco 3](https://github.com/stephane-caron/pink/tree/main/examples#humanoid-draco-3)
- Mobile base: [Stretch R1](https://github.com/stephane-caron/pink/tree/main/examples#mobile-stretch)
- Quadruped: [Go2 squatting with floating-base limits](https://github.com/stephane-caron/pink/tree/main/examples/barriers#go2-squat)
- Wheeled biped: [Upkie rolling without slipping](https://github.com/stephane-caron/pink/tree/main/examples#wheeled-biped-upkie)

There are also more basic examples to get started:

- [Double pendulum](https://github.com/stephane-caron/pink/blob/main/examples/double_pendulum.py)
- [Loading a custom URDF](https://github.com/stephane-caron/pink/blob/main/examples/load_custom_urdf.py)
- [Visualization in MeshCat](https://github.com/stephane-caron/pink/blob/main/examples/visualize_in_meshcat.py)
- [Visualization in yourdfpy](https://github.com/stephane-caron/pink/blob/main/examples/visualize_in_yourdfpy.py)

Check out the [examples](https://github.com/stephane-caron/pink/tree/main/examples) directory for more.

## Frequently Asked Questions

- [Can I solve **global** inverse kinematics?](https://github.com/stephane-caron/pink/discussions/66#discussioncomment-8224315)
- [Can I make velocities smoother?](https://github.com/stephane-caron/pink/discussions/103)
- [My configuration gets stuck somewhere and does not solve the task, what is going on?](https://github.com/stephane-caron/pink/discussions/66#discussioncomment-8224315)

## Global inverse kinematics

Pink implements differential inverse kinematics, a first-order algorithm that converges to the closest optimum of its cost function. It is a **local** method that does not solve the more difficult problem of [global inverse kinematics](https://github.com/stephane-caron/pink/discussions/66). That is, it may converge to a global optimum, or to a local one stuck to some configuration limits. This behavior is illustrated in the [simple pendulum with configuration limit](https://github.com/stephane-caron/pink/blob/main/examples/simple_pendulum_configuration_limit.py) example.

## How can I help?

Install the library and use it! Report bugs in the [issue tracker](https://github.com/stephane-caron/pink/issues). If you are a developer with some robotics experience looking to hack on open source, check out the [contribution guidelines](CONTRIBUTING.md).

## Citation

If you use Pink in your scientific works, please cite it *e.g.* as follows:

```bibtex
@software{pink,
  title = {{Pink: Python inverse kinematics based on Pinocchio}},
  author = {Caron, Stéphane and De Mont-Marin, Yann and Budhiraja, Rohan and Bang, Seung Hyeon and Domrachev, Ivan and Nedelchev, Simeon, and @peterd-NV and Vaillant, Joris},
  license = {Apache-2.0},
  url = {https://github.com/stephane-caron/pink},
  version = {3.3.0},
  year = {2025}
}
```

Don't forget to add yourself to the BibTeX above and to `CITATION.cff` if you contribute to this repository.

## See also

Software:

- [mink](https://github.com/kevinzakka/mink): differential inverse kinematics in Python, based on the MuJoCo physics engine.
- [Jink.jl](https://github.com/adubredu/Jink.jl): Julia package for differential multi-task inverse kinematics.
- [PlaCo](https://github.com/rhoban/placo): C++ inverse kinematics based on Pinocchio.
- [pymanoid](https://github.com/stephane-caron/pymanoid): precursor to Pink based on OpenRAVE.
- [TSID](https://github.com/stack-of-tasks/tsid): C++ inverse kinematics based on Pinocchio.

Technical notes:

- [Inverse kinematics](https://scaron.info/robotics/inverse-kinematics.html): a general introduction to differential inverse kinematics.
- [Jacobian of a kinematic task and derivatives on manifolds](https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html).
- [Control Barrier Functions](https://simeon-ned.com/blog/2024/cbf/).
