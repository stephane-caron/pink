#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Test acceleration limit."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration, solve_ik
from pink.limits import AccelerationLimit
from pink.tasks import FrameTask


class TestAccelerationLimit(unittest.TestCase):
    """Test fixture for acceleration limit."""

    def setUp(self):
        """Set test fixture up."""
        robot = load_robot_description("ur3_description")
        model = robot.model
        a_max = np.ones(robot.model.nv)
        self.a_max = a_max
        self.data = robot.data
        self.q_ref = robot.q0
        self.limit = AccelerationLimit(model, a_max)
        self.model = model

    def test_dimensions(self):
        """Check dimensions of configuration limit projection."""
        nb = len(self.limit.indices)
        nv = self.model.nv
        self.assertEqual(self.limit.projection_matrix.shape, (nb, nv))

    def test_model_with_no_limit(self):
        """Check that unbounded models don't fail."""
        empty_model = pin.Model()
        empty_bounded = AccelerationLimit(empty_model, self.a_max)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(
            empty_bounded.compute_qp_inequalities(np.empty(0), 1e-3)
        )

    def test_limit_has_an_effect(self):
        """Check that the limit has an effect on a reaching task."""
        end_effector_task = FrameTask(
            "ee_link",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
        )
        configuration = Configuration(self.model, self.data, self.q_ref)
        end_effector_task.set_target_from_configuration(configuration)
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[1] = 0.3
        end_effector_target.translation[2] = 0.2
        configuration_limit = configuration.model.configuration_limit
        velocity_limit = configuration.model.velocity_limit
        tasks = [end_effector_task]
        dt = 5e-3
        solver_settings = {
            "eps_abs": 1e-7,
            "eps_rel": 0.0,
        }
        v_prev = solve_ik(
            configuration,
            tasks,
            dt,
            solver="scs",
            limits=[configuration_limit, velocity_limit],
            **solver_settings,
        )
        configuration.integrate_inplace(v_prev, dt)
        self.limit.set_last_integration(v_prev, dt)
        end_effector_target.translation[1] = 0.6
        v_with = solve_ik(
            configuration,
            tasks,
            dt,
            solver="scs",
            limits=[configuration_limit, velocity_limit, self.limit],
            **solver_settings,
        )
        v_without = solve_ik(
            configuration,
            tasks,
            dt,
            solver="scs",
            limits=[configuration_limit, velocity_limit],
            **solver_settings,
        )
        a_with = (v_with - v_prev) / dt
        a_without = (v_without - v_prev) / dt
        tolerance = 1e-3  # [rad] / [s]^2
        max_a_with = float(np.max(np.abs(a_with)))
        a_max = float(np.max(self.a_max))
        self.assertTrue(
            np.all(np.abs(a_with) < self.a_max + tolerance),
            f"max(a_with) = {max_a_with} > a_max = {a_max}",
        )
        self.assertFalse(np.all(np.abs(a_without) < self.a_max + tolerance))
