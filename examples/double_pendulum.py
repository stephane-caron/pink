#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Raise the double pendulum up and down.
"""

from functools import partial
from os import path

import numpy as np
import pinocchio as pin
import yourdfpy

import pink
from pink import solve_ik
from pink.tasks import BodyTask, PostureTask
from pink.utils import RateLimiter

if __name__ == "__main__":
    urdf_path = path.join(path.dirname(__file__), "double_pendulum.urdf")
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )

    tasks = {
        "tip": BodyTask(
            "link3",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        ),
        "posture": PostureTask(
            cost=1e-3,  # [cost] / [rad]
        ),
    }

    configuration = pink.apply_configuration(robot, robot.q0)
    for task in tasks.values():
        task.set_target_from_configuration(configuration)

    animation_time = 0.0  # [s]
    visualizer_fps = 100  # [Hz]
    rate = RateLimiter(frequency=visualizer_fps)

    def callback(scene, robot, viz):
        global animation_time, configuration
        dt = rate.period

        # Update task targets
        t = animation_time
        T = tasks["tip"].transform_target_to_world
        T.translation[1] = 0.1 * np.sin(t)

        tip_task = tasks["tip"]
        J, e = tip_task.compute_task_dynamics(configuration)
        print(f"{J=}, {e=}")

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks.values(), dt, solver="osqp")
        print(velocity)
        q = configuration.integrate(velocity, dt)
        configuration = pink.apply_configuration(robot, q)

        # Display resulting configuration
        viz.update_cfg(configuration.q)

        # Regulate visualizer FPS
        animation_time += dt
        rate.sleep()

    viz = yourdfpy.URDF.load(urdf_path)
    viz.show(callback=partial(callback, robot=robot, viz=viz))
