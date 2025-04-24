#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2024 Inria

"""Test fixture for the frame task."""

import unittest

import numpy as np
import pinocchio as pin
from qpsolvers import solve_qp
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.exceptions import TargetNotSet, TaskDefinitionError
from pink.tasks import FrameTask


class TestFrameTask(unittest.TestCase):
    """Test consistency of the frame task.

    Note:
        This fixture only tests the task itself. Integration tests with the IK
        are carried out in :class:`TestSolveIK`.
    """

    def setUp(self):
        """Prepare test fixture."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        frame_name = "ee_frame"
        joint_name = robot.model.names[-1]
        parent_joint = robot.model.getJointId(joint_name)
        parent_frame = robot.model.getFrameId(joint_name)
        placement = pin.SE3.Identity()
        robot.model.addFrame(
            pin.Frame(
                frame_name,
                parent_joint,
                parent_frame,
                placement,
                pin.FrameType.OP_FRAME,
            )
        )
        robot.data = pin.Data(robot.model)
        self.configuration = Configuration(robot.model, robot.data, robot.q0)

    def test_set_target_from_configuration(self):
        """Read target after setting it to a known value."""
        task = FrameTask("l_ankle", position_cost=1.0, orientation_cost=0.1)
        task.set_target_from_configuration(self.configuration)
        transform_ankle_to_world = (
            self.configuration.get_transform_frame_to_world("l_ankle")
        )
        self.assertTrue(
            np.allclose(
                transform_ankle_to_world, task.transform_target_to_world
            )
        )

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        earflap_task = FrameTask(
            "earflap", position_cost=1.0, orientation_cost=0.1
        )
        self.assertTrue("frame=" in repr(earflap_task))
        self.assertTrue("gain=" in repr(earflap_task))
        self.assertTrue("orientation_cost=" in repr(earflap_task))
        self.assertTrue("position_cost=" in repr(earflap_task))
        self.assertTrue("lm_damping=" in repr(earflap_task))

    def test_target_not_set(self):
        """Raise an exception when the target is not set."""
        task = FrameTask("l_ankle", position_cost=1.0, orientation_cost=0.1)
        with self.assertRaises(TargetNotSet):
            task.compute_error(self.configuration)
        with self.assertRaises(TargetNotSet):
            task.compute_jacobian(self.configuration)

    def test_target_set_properly(self):
        """Return target properly once it's set."""
        task = FrameTask("l_ankle", position_cost=1.0, orientation_cost=0.1)
        T = self.configuration.get_transform_frame_to_world("l_ankle")
        task.set_target(T)
        self.assertIsNotNone(task.transform_target_to_world)
        if task.transform_target_to_world is not None:  # help mypy
            self.assertTrue(
                np.allclose(
                    T.homogeneous,
                    task.transform_target_to_world.homogeneous,
                )
            )

    def test_target_is_a_copy(self):
        """Target is saved as a copy, not a reference to the original."""
        task = FrameTask("l_ankle", position_cost=1.0, orientation_cost=0.1)
        target = self.configuration.get_transform_frame_to_world("l_ankle")
        task.set_target(target)
        y = target.translation[1]
        target.translation[1] += 12.0
        if task.transform_target_to_world is not None:  # help mypy
            self.assertAlmostEqual(
                task.transform_target_to_world.translation[1], y
            )
            self.assertNotAlmostEqual(
                task.transform_target_to_world.translation[1],
                target.translation[1],
            )

    def test_zero_error_when_target_at_body(self):
        """Error is zero when the target and body are at the same location."""
        task = FrameTask("r_ankle", position_cost=1.0, orientation_cost=0.1)
        target = self.configuration.get_transform_frame_to_world("r_ankle")
        task.set_target(target)  # error == 0
        e = task.compute_error(self.configuration)
        J = task.compute_jacobian(self.configuration)
        J_ref = -self.configuration.get_frame_jacobian("r_ankle")  # at target
        self.assertLess(np.linalg.norm(e), 1e-10)
        self.assertTrue(np.allclose(J, J_ref))

    def test_unit_cost_qp_objective(self):
        """Unit cost means the QP objective is exactly (J^T J, -e^T J)."""
        task = FrameTask("r_wrist", position_cost=1.0, orientation_cost=0.1)
        transform_target_to_frame = pin.SE3(
            np.eye(3), np.array([0.0, 0.01, 0.0])
        )
        target = (
            self.configuration.get_transform_frame_to_world("r_wrist")
            * transform_target_to_frame
        )
        task.set_target(target)
        J = task.compute_jacobian(self.configuration)
        e = task.compute_error(self.configuration)
        task.set_position_cost(1.0)
        task.set_orientation_cost(1.0)
        task.lm_damping = 0.0
        H, c = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(e.T @ J, c))

    def test_zero_costs_same_as_disabling_lines(self):
        """Setting a position or orientation cost to zero.

        A zero yields the same QP objective as disabling the corresponding
        Jacobian and error coordinates.
        """
        task = FrameTask("l_wrist", position_cost=1.0, orientation_cost=0.1)
        transform_target_to_frame = pin.SE3(
            np.eye(3), np.array([0.1, 0.02, 0.01])
        )
        target = (
            self.configuration.get_transform_frame_to_world("l_wrist")
            * transform_target_to_frame
        )
        task.set_target(target)
        J = task.compute_jacobian(self.configuration)
        e = task.compute_error(self.configuration)
        qd = np.random.random(J.shape[1:])
        test_cases = {
            "position_only": (1.0, 0.0, slice(0, 3)),
            "orientation_only": (0.0, 1.0, slice(3, 6)),
            "position_0": ([1.0, 0.0, 0.0], 0.0, slice(0, 1)),
            "position_1": ([0.0, 1.0, 0.0], 0.0, slice(1, 2)),
            "position_2": ([0.0, 0.0, 1.0], 0.0, slice(2, 3)),
            "orientation_0": (0.0, [1.0, 0.0, 0.0], slice(3, 4)),
            "orientation_1": (0.0, [0.0, 1.0, 0.0], slice(4, 5)),
            "orientation_2": (0.0, [0.0, 0.0, 1.0], slice(5, 6)),
        }
        for case in test_cases.values():
            position_cost, orientation_cost, indexes = case
            task.set_position_cost(position_cost)
            task.set_orientation_cost(orientation_cost)
            task.lm_damping = 0.0
            H, c = task.compute_qp_objective(self.configuration)
            H_check = J[indexes].T @ J[indexes]
            c_check = e[indexes].T @ J[indexes]
            cost = qd.T @ H @ qd + c @ qd
            cost_check = qd.T @ H_check @ qd + c_check @ qd
            self.assertAlmostEqual(cost, cost_check)

    def test_lm_damping_has_no_effect_at_target(self):
        """Levenberg-Marquardt damping has no effect when the error is zero."""
        task = FrameTask("l_wrist", position_cost=1.0, orientation_cost=0.1)
        target = self.configuration.get_transform_frame_to_world("l_wrist")
        task.set_target(target)
        task.lm_damping = 1e-8
        H_1, c_1 = task.compute_qp_objective(self.configuration)
        task.lm_damping = 1e-4
        H_2, c_2 = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(H_1, H_2))
        self.assertTrue(np.allclose(c_1, c_2))

    def test_lm_damping_has_effect_under_error(self):
        """Levenberg-Marquardt damping is indeed a damping.

        That is, unless the task is fulfilled, it reduces velocities.
        """
        task = FrameTask("r_wrist", position_cost=1.0, orientation_cost=0.1)
        transform_target_to_frame = pin.SE3(
            np.eye(3), np.array([0.0, 2.0, 0.0])
        )
        target = (
            self.configuration.get_transform_frame_to_world("r_wrist")
            * transform_target_to_frame
        )
        task.set_target(target)
        task.lm_damping = 1e-8
        H_1, c_1 = task.compute_qp_objective(self.configuration)
        task.lm_damping = 1e-4
        H_2, c_2 = task.compute_qp_objective(self.configuration)
        qd_1 = solve_qp(H_1, c_1, solver="daqp")  # H_1 p.s.d. (LM damping)
        qd_2 = solve_qp(H_2, c_2, solver="daqp")  # idem for H_2
        self.assertGreater(np.linalg.norm(qd_2 - qd_1), 1e-6)

    def test_task_on_user_added_op_frame(self):
        """Error is zero when the target and body are at the same location."""
        task = FrameTask("ee_frame", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        target = self.configuration.get_transform_frame_to_world("ee_frame")
        task.set_target(target)  # error == 0
        e = task.compute_error(self.configuration)
        J = task.compute_jacobian(self.configuration)
        J_ref = -self.configuration.get_frame_jacobian("ee_frame")  # at target
        self.assertLess(np.linalg.norm(e), 1e-10)
        self.assertTrue(np.allclose(J, J_ref))

    def test_inconsistent_cost(self):
        """Exception when the cost is not a vector any more."""
        task = FrameTask("ee_frame", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        task.cost = 42.0
        with self.assertRaises(TaskDefinitionError):
            task.set_position_cost(1.0)
        with self.assertRaises(TaskDefinitionError):
            task.set_orientation_cost(1.0)
