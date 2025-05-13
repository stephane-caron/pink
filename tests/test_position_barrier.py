#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Tests for position barrier limit."""

import unittest

import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.barriers import PositionBarrier
from pink.exceptions import NoPositionLimitProvided


class TestPositionBarrier(unittest.TestCase):
    """Testing position barrier limit."""

    def setUp(self):
        """Set test fixture up."""
        robot = load_robot_description("ur3_description")
        model = robot.model
        self.data = robot.data
        self.model = model
        self.robot = robot
        self.configuration = Configuration(
            self.model, self.data, np.zeros(self.model.nq)
        )

    def test_no_position_limit(self):
        """Raise an error when the robot body is not found."""
        with self.assertRaises(NoPositionLimitProvided):
            PositionBarrier("foo")

    def test_dimension(self):
        """Check dimensions of configuration limit projection."""
        self.assertEqual(PositionBarrier("ee_link", p_min=np.zeros(3)).dim, 3)
        self.assertEqual(PositionBarrier("ee_link", p_max=np.zeros(3)).dim, 3)
        self.assertEqual(
            PositionBarrier(
                "ee_link", p_min=np.zeros(3), p_max=np.zeros(3)
            ).dim,
            6,
        )

    def test_gains(self):
        """Check gains of configuration limit projection."""
        # One limit, scalar gain
        self.assertEqual(
            PositionBarrier("ee_link", p_min=np.zeros(3), gain=1).gain.shape,
            (3,),
        )
        # One limit, vector gain
        self.assertEqual(
            PositionBarrier(
                "ee_link", p_max=np.zeros(3), gain=np.array([1, 2, 3])
            ).gain.shape,
            (3,),
        )
        # Two limits, scalar gain
        self.assertEqual(
            PositionBarrier(
                "ee_link", p_min=np.zeros(3), p_max=np.zeros(3), gain=1
            ).gain.shape,
            (6,),
        )
        # Two limits, vector gain
        self.assertEqual(
            PositionBarrier(
                "ee_link",
                p_min=np.zeros(3),
                p_max=np.zeros(3),
                gain=np.array([1, 2, 3]),
            ).gain.shape,
            (6,),
        )

    def test_positive_when_in_safety_zone(self):
        """Check that the barrier is positive when in the safety zone."""

        barrier = PositionBarrier("ee_link", p_min=np.zeros(3))
        h = barrier.compute_barrier(self.configuration)
        self.assertTrue(np.all(h > 0))

    def test_negative_when_out_of_safety_zone(self):
        """Test that the barrier is negative when out of the safety zone."""

        for violated_idx in range(3):
            p_min = np.zeros(3)
            p_min[violated_idx] = 1.0
            barrier = PositionBarrier("ee_link", p_min=p_min)
            h = barrier.compute_barrier(self.configuration)
            self.assertTrue(np.any(h < 0))
