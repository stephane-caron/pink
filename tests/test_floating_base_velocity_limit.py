#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for floating base velocity limits."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink.limits import FloatingBaseVelocityLimit
from pink.utils import get_joint_idx


class TestFloatingBaseVelocityLimitPlanar(unittest.TestCase):
    """Check velocity bounds for a planar floating base."""

    def setUp(self):
        """Load a robot with a planar root joint."""
        robot = load_robot_description(
            "stretch_description", root_joint=pin.JointModelPlanar()
        )
        self.robot = robot
        self.model = robot.model
        self.dt = 0.1  # [s]
        self.linear_max = np.array([0.4, 0.2, np.inf])  # [m] / [s]
        self.angular_max = np.array([np.inf, np.inf, 1.0])  # [rad] / [s]
        root_joint_id = self.model.getJointId("root_joint")
        base_frame = None
        for frame in self.model.frames:
            if frame.parentJoint == root_joint_id:
                base_frame = frame.name
                break
        if base_frame is None:
            self.fail("Planar robot has no frame attached to the root joint")
        self.limit = FloatingBaseVelocityLimit(
            model=self.model,
            base_frame=base_frame,
            max_linear_velocity=self.linear_max,
            max_angular_velocity=self.angular_max,
        )

    def test_qp_constraints_match_frame_jacobian(self):
        """Computed inequalities should mirror the base frame Jacobian."""
        result = self.limit.compute_qp_inequalities(self.robot.q0, self.dt)
        self.assertIsNotNone(result)
        G, h = result

        data = self.model.createData()
        pin.computeJointJacobians(self.model, data, self.robot.q0)
        pin.updateFramePlacements(self.model, data)
        frame_id = self.model.getFrameId(self.limit.base_frame)
        jacobian = pin.getFrameJacobian(
            self.model, data, frame_id, pin.ReferenceFrame.LOCAL
        )

        expected_rows = []
        expected_bounds = []
        for row_idx, max_velocity in enumerate(self.linear_max):
            if not np.isfinite(max_velocity):
                continue
            row = jacobian[row_idx, :]
            expected_rows.extend([row, -row])
            expected_bounds.extend([self.dt * max_velocity] * 2)
        for row_idx, max_velocity in enumerate(self.angular_max):
            if not np.isfinite(max_velocity):
                continue
            row = jacobian[3 + row_idx, :]
            expected_rows.extend([row, -row])
            expected_bounds.extend([self.dt * max_velocity] * 2)

        expected_G = np.vstack(expected_rows)
        expected_h = np.array(expected_bounds)

        np.testing.assert_allclose(G, expected_G)
        np.testing.assert_allclose(h, expected_h)

    def test_velocity_projection_indices(self):
        """Bounded tangent directions should come from the root joint."""
        result = self.limit.compute_qp_inequalities(self.robot.q0, self.dt)
        self.assertIsNotNone(result)
        G, _ = result
        _, idx_v = get_joint_idx(self.model, "root_joint")
        root_nv = self.model.joints[self.model.getJointId("root_joint")].nv
        root_columns = G[:, idx_v : idx_v + root_nv]
        non_root_columns = np.delete(
            G, np.s_[idx_v : idx_v + root_nv], axis=1
        )
        self.assertGreater(np.linalg.norm(root_columns), 0.0)
        self.assertTrue(np.allclose(non_root_columns, 0.0))


class TestFloatingBaseVelocityLimitFreeFlyer(unittest.TestCase):
    """Check velocity bounds for a free flyer floating base."""

    def setUp(self):
        """Load a robot with a free flyer root joint."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        self.robot = robot
        self.model = robot.model
        self.dt = 0.05  # [s]
        self.linear_max = np.array([0.3, 0.3, 0.2])  # [m] / [s]
        self.angular_max = np.array([1.0, 1.0, 1.5])  # [rad] / [s]
        root_joint_id = self.model.getJointId("root_joint")
        base_frame = None
        for frame in self.model.frames:
            if frame.parentJoint == root_joint_id:
                base_frame = frame.name
                break
        if base_frame is None:
            self.fail("Free-flyer robot has no frame attached to the root joint")
        self.limit = FloatingBaseVelocityLimit(
            model=self.model,
            base_frame=base_frame,
            max_linear_velocity=self.linear_max,
            max_angular_velocity=self.angular_max,
        )

    def test_expected_number_of_constraints(self):
        """Every finite bound contributes two inequality rows."""
        result = self.limit.compute_qp_inequalities(self.robot.q0, self.dt)
        self.assertIsNotNone(result)
        G, h = result
        finite_bounds = np.isfinite(self.linear_max).sum() + np.isfinite(
            self.angular_max
        ).sum()
        expected_rows = 2 * finite_bounds
        self.assertEqual(G.shape, (expected_rows, self.model.nv))
        self.assertEqual(h.shape[0], expected_rows)

    def test_velocity_within_limits_satisfies_constraints(self):
        """A displacement matching the limit should satisfy the inequalities."""
        result = self.limit.compute_qp_inequalities(self.robot.q0, self.dt)
        self.assertIsNotNone(result)
        G, h = result

        _, idx_v = get_joint_idx(self.model, "root_joint")
        dq = np.zeros(self.model.nv)
        dq[idx_v : idx_v + 6] = self.dt * np.hstack(
            [self.linear_max, self.angular_max]
        )
        # Add a tiny epsilon so we stay strictly inside the bound.
        dq[idx_v : idx_v + 6] *= 0.999
        self.assertTrue(np.all(G @ dq <= h + 1e-12))

    def test_missing_root_joint_returns_none(self):
        """Manipulator only robots should not produce floating-base limits."""
        model = pin.Model()
        limit = FloatingBaseVelocityLimit(
            model=model,
            base_frame="base_link",
            max_linear_velocity=self.linear_max,
            max_angular_velocity=self.angular_max,
        )
        self.assertIsNone(limit.compute_qp_inequalities(np.array([]), self.dt))
