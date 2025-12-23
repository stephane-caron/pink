#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for floating base velocity limits."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
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
        self.configuration = Configuration(self.model, robot.data, robot.q0)
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

    def test_constraints_clip_base_twist(self):
        """Constraints keep twists inside bounds."""
        result = self.limit.compute_qp_inequalities(self.configuration, self.dt)
        self.assertIsNotNone(result)
        G, h = result

        twist_bounds = np.hstack([self.linear_max, self.angular_max])
        _, idx_v = get_joint_idx(self.model, "root_joint")

        dq_inside = np.zeros(self.model.nv)
        dq_inside[idx_v : idx_v + 6] = self.dt * twist_bounds * 0.999
        self.assertTrue(np.all(G @ dq_inside <= h + 1e-12))

        dq_outside = dq_inside.copy()
        finite_axes = np.nonzero(np.isfinite(twist_bounds))[0]
        dq_outside[idx_v + finite_axes[0]] = (
            self.dt * twist_bounds[finite_axes[0]] * 1.01
        )
        self.assertTrue(np.any(G @ dq_outside > h + 1e-12))

    def test_velocity_projection_indices(self):
        """Bounded tangent directions should come from the root joint."""
        result = self.limit.compute_qp_inequalities(self.configuration, self.dt)
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
        self.configuration = Configuration(self.model, robot.data, robot.q0)
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
        result = self.limit.compute_qp_inequalities(self.configuration, self.dt)
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
        result = self.limit.compute_qp_inequalities(self.configuration, self.dt)
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

    def test_missing_root_joint_raises(self):
        """Manipulator only robots should fail fast when no floating base exists."""
        model = pin.Model()
        with self.assertRaises(ValueError):
            FloatingBaseVelocityLimit(
                model=model,
                base_frame="base_link",
                max_linear_velocity=self.linear_max,
                max_angular_velocity=self.angular_max,
            )

    def test_default_base_frame_detection(self):
        """Omitting base_frame should pick the frame attached to root_joint."""
        limit = FloatingBaseVelocityLimit(
            model=self.model,
            base_frame=None,
            max_linear_velocity=self.linear_max,
            max_angular_velocity=self.angular_max,
        )
        root_joint_id = self.model.getJointId("root_joint")
        self.assertEqual(
            self.model.frames[limit.frame_id].parentJoint, root_joint_id
        )
