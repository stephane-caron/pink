# Pink

<img src="https://user-images.githubusercontent.com/1189580/172797197-9aa46561-cfaa-4046-bd60-f681d85b055d.png" align="right" height=140>

[![Build](https://img.shields.io/github/actions/workflow/status/tasts-robots/pink/main.yml?branch=master)](https://github.com/tasts-robots/pink/actions)
[![Coverage](https://coveralls.io/repos/github/tasts-robots/pink/badge.svg?branch=master)](https://coveralls.io/github/tasts-robots/pink?branch=master)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?style=flat)](https://tasts-robots.org/doc/pink/)
[![PyPI version](https://img.shields.io/pypi/v/pin-pink)](https://pypi.org/project/pin-pink/)

**P**ython **in**verse **k**inematics for articulated robot models, based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Installation

```console
pip install pin-pink
```

On Raspberry Pi, you will need to install [from source](https://tasts-robots.org/doc/pink/installation.html#from-source).

## Usage

Pink solves inverse kinematics by [weighted tasks](https://scaron.info/robot-locomotion/inverse-kinematics.html). A task is defined by a *residual* function $e(q)$ of the robot configuration $q \in \mathcal{C}$ to be driven to zero. For instance, putting a foot position $p_{foot}(q)$ at a given target $p_{foot}^{\star}$ can be described by the position residual:

$$
e(q) = p_{foot}^{\star} - p_{foot}(q)
$$

In inverse kinematics, we compute a velocity $v \in \mathfrak{c}$ that satisfies the first-order differential equation:

$$
J_e(q) v = \dot{e}(q) = -\alpha e(q)
$$

where $J\_e(q) := \frac{\partial e}{\partial q}$ is the [task Jacobian](https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html). We can define multiple tasks, but some of them will come into conflict if they can't be all fully achieved at the same time. Conflicts are resolved by casting all objectives to a common unit, and weighing these normalized objectives relative to each other:

$$
\underset{v}{\text{minimize}} \sum_{\text{task } e} \Vert J_e(q) v + \alpha e(q) \Vert^2_{W_e}
$$

Finally, we include configuration and velocity limits, making our problem a quadratic program:

$$
\begin{align}
\underset{v}{\text{minimize}} & \sum_{\text{task } e} \Vert J_e(q) v + \alpha e(q) \Vert^2_{W_e} \\
\text{subject to}
    & v_{\text{min}}(q) \leq v \leq v_{\text{max}}(q)
\end{align}
$$

Pink provides an API to describe the problem as tasks with targets, and automatically build and solve the underlying quadratic program.

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
from robot_descriptions.loaders.pinocchio import load_robot_description

robot = load_robot_description("upkie_description")
configuration = pink.apply_configuration(robot, robot.q0)
for body, task in tasks.items():
    if type(task) is BodyTask:
        task.set_target(configuration.get_transform_body_to_world(body))
```

Once a task has its cost and (if applicable) target defined, it can be used to solve inverse kinematics.

### Differential inverse kinematics

Pink solves differential inverse kinematics, meaning it outputs a velocity that steers the robot model towards a configuration that achieves all tasks at best. If we keep integrating that velocity, and task targets don't change in the meantime, we will converge to that configuration:

```python
dt = 6e-3  # [s]
for t in np.arange(0.0, 42.0, dt):
    velocity = solve_ik(configuration, tasks.values(), dt, solver="quadprog")
    q = configuration.integrate(velocity, dt)
    configuration = pink.apply_configuration(robot, q)
    time.sleep(dt)
```

If task targets are continuously updated there will be no stationary solution to converge to, but the model will keep on tracking each target at best. Note that [`solve_ik`](https://scaron.info/doc/pink/inverse-kinematics.html#pink.solve_ik.solve_ik) takes into account both joint position and velocity limits read from the robot model.

## Examples

Pink works with all kinds of robot morphologies:

* Arm: [Kinova Gen2](https://github.com/tasts-robots/pink/blob/master/examples/arm_kinova_gen2.py)
* Arm: [UR3](https://github.com/tasts-robots/pink/blob/master/examples/arm_ur3.py)
* Humanoid: [JVRC-1](https://github.com/tasts-robots/pink/blob/master/examples/humanoid_jvrc.py)
* Humanoid: [SigmaBan](https://github.com/tasts-robots/pink/blob/master/examples/humanoid_sigmaban.py)
* Wheeled biped: [Upkie](https://github.com/tasts-robots/pink/blob/master/examples/wheeled_biped_upkie.py)
* Wheeled: [Omnidirectional robot](https://github.com/tasts-robots/pink/blob/master/examples/omnidirectional_wheeled_robot.py)

Basic examples are the best to get started:

* [Double pendulum](https://github.com/tasts-robots/pink/blob/master/examples/double_pendulum.py)
* [Loading a custom URDF](https://github.com/tasts-robots/pink/blob/master/examples/load_custom_urdf.py)
* [Visualization in MeshCat](https://github.com/tasts-robots/pink/blob/master/examples/visualize_in_meshcat.py)
* [Visualization in yourdfpy](https://github.com/tasts-robots/pink/blob/master/examples/visualize_in_yourdfpy.py)

Check out the [examples](https://github.com/tasts-robots/pink/tree/master/examples) folder for more.

## How can I help?

Install the library and use it! Report bugs in the [issue tracker](https://github.com/tasts-robots/pink/issues). If you are a developer with some robotics experience looking to hack on open source, check out the [contribution guidelines](CONTRIBUTING.md).

## See also

- [Jink.jl](https://github.com/adubredu/Jink.jl): Julia package for differential multi-task inverse kinematics.
- [pymanoid](https://github.com/stephane-caron/pymanoid): precursor to Pink based on OpenRAVE.
