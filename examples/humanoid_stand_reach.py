#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Stéphane Caron.
#
# This file is part of Pink.
#
# Pink is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Pink is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Pink. If not, see <http://www.gnu.org/licenses/>.

"""
Humanoid robot model standing on two feet and reaching with a hand.
"""

import pink
import pink.models

from pink.tasks import BodyTask
from pink import solve_ik

if __name__ == "__main__":
    robot = pink.models.build_jvrc_model()
    configured_robot = pink.configure_robot(robot, robot.q0)
    dt = 5e-3

    left_foot_task = BodyTask(
        "l_ankle", position_cost=1.0, orientation_cost=3.0
    )
    right_foot_task = BodyTask(
        "r_ankle", position_cost=1.0, orientation_cost=3.0
    )
    pelvis_task = BodyTask("PELVIS_S", position_cost=1.0, orientation_cost=3.0)
    tasks = [pelvis_task, left_foot_task, right_foot_task]

    left_foot_task.set_target(
        configured_robot.get_transform_body_to_world("l_ankle")
    )
    right_foot_task.set_target(
        configured_robot.get_transform_body_to_world("r_ankle")
    )
    pelvis_task.set_target(
        configured_robot.get_transform_body_to_world("PELVIS_S")
    )

    velocity = solve_ik(configured_robot, tasks, dt)
