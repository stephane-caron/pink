#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Tests that should pass for all barriers."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.barriers import PositionBarrier


class TestBarrier(unittest.TestCase):
    """Tests that should pass for all barriers."""

    def setUp(self):
        """Set up test fixtures."""

        self.robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        self.conf = Configuration(
            self.robot.model,
            self.robot.data,
            q=np.zeros(self.robot.nq),
        )
        self.dt = 1e-3  # [s]

    def test_diff_form_dimensions(self):
        """Velocity barrier dimension is the number of bounded joints."""
        for barrier in [
            PositionBarrier("left_hip", p_min=np.zeros(3), p_max=np.zeros(3))
        ]:
            H, c = barrier.compute_qp_objective(self.conf)
            G, h = barrier.compute_qp_inequalities(self.conf, self.dt)
            self.assertEqual(H.shape[0], self.robot.nv)
            self.assertEqual(H.shape[1], self.robot.nv)
            self.assertEqual(c.shape[0], self.robot.nv)
            self.assertEqual(G.shape[0], barrier.dim)
            self.assertEqual(G.shape[1], self.robot.nv)

    def test_barrier_value_dimension(self):
        """Test tha shape of value in all barriers is correct."""

        for barrier in [
            PositionBarrier("left_hip", p_min=np.zeros(3), p_max=np.zeros(3))
        ]:
            v = barrier.compute_barrier(self.conf)
            self.assertEqual(v.shape[0], barrier.dim)

    def test_barrier_jacobians_dimension(self):
        """Test that shapes of jacobians in all barriers are correct."""

        for barrier in [
            PositionBarrier("left_hip", p_min=np.zeros(3), p_max=np.zeros(3))
        ]:
            J = barrier.compute_jacobian(self.conf)
            self.assertEqual(J.shape[0], barrier.dim)
            self.assertEqual(J.shape[1], self.robot.nv)

    def test_barrier_without_penalty_weight(self):
        """Test that objective is zero if no penalty weight is provided."""
        for barrier in [
            PositionBarrier("left_hip", p_min=np.zeros(3), p_max=np.zeros(3))
        ]:
            H, c = barrier.compute_qp_objective(self.conf)
            self.assertTrue(np.allclose(H, 0))
            self.assertTrue(np.allclose(c, 0))

    def test_barrier_penalty_weight(self):
        """Test that objective is non-zeros, if penalty weight is provided."""
        for barrier in [
            PositionBarrier(
                "left_hip",
                p_min=np.zeros(3),
                p_max=np.zeros(3),
                safe_displacement_gain=1.0,
            )
        ]:
            H, c = barrier.compute_qp_objective(self.conf)
            self.assertFalse(np.allclose(H, 0))
            if np.any(barrier.compute_safe_displacement(self.conf) != 0):
                self.assertFalse(np.allclose(c, 0))

    def test_task_repr(self):
        """Test task string representation."""
        for limit in [
            PositionBarrier(
                "universe", safe_displacement_gain=0.0, p_min=np.zeros(3)
            )
        ]:
            self.assertTrue("gain=" in repr(limit))
            self.assertTrue("safe_displacement=" in repr(limit))
            self.assertTrue("safe_displacement_gain" in repr(limit))
            self.assertTrue("dim" in repr(limit))

    def test_cached(self):
        """Test that cached results are reused."""

        barrier = PositionBarrier(
            "left_hip", p_min=np.zeros(3), p_max=np.zeros(3)
        )
        barrier.compute_qp_objective(self.conf)
        # Check that cache is initially triggered
        self.assertIsNotNone(barrier._Barrier__q_cache)

        # Check that cache is resetted after update
        self.conf.update(self.conf.q + 1.0)
        barrier.compute_qp_inequalities(self.conf)
        self.assertTrue(np.allclose(barrier._Barrier__q_cache, self.conf.q))
