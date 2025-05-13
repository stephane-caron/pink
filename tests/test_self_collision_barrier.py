#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""Tests for position barrier limit."""

import os
import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.iiwa14_description import PACKAGE_PATH, REPOSITORY_PATH

from pink import Configuration
from pink.barriers import SelfCollisionBarrier
from pink.exceptions import InvalidCollisionPairs, NegativeMinimumDistance
from pink.utils import process_collision_pairs


class TestSelfCollisionBarrier(unittest.TestCase):
    """Testing position barrier limit."""

    def setUp(self):
        """Set test fixture up."""
        urdf_path = os.path.join(
            PACKAGE_PATH, "urdf", "iiwa14_spheres_collision.urdf"
        )
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, package_dirs=[os.path.dirname(REPOSITORY_PATH)]
        )

        self.non_colliding_objects_pair_id = 5
        self.q0 = np.zeros(self.robot.model.nq)
        self.robot.collision_data = process_collision_pairs(
            self.robot.model, self.robot.collision_model
        )
        self.configuration = Configuration(
            self.robot.model,
            self.robot.data,
            self.q0,
            collision_model=self.robot.collision_model,
            collision_data=self.robot.collision_data,
        )

    def test_invalid_distance(self):
        """Raise an error when the robot body is not found."""
        with self.assertRaises(NegativeMinimumDistance):
            SelfCollisionBarrier(n_collision_pairs=10, d_min=-1)

    def test_invalid_collision_pairs(self):
        """Raise an error when the robot body is not found."""
        with self.assertRaises(InvalidCollisionPairs):
            SelfCollisionBarrier(n_collision_pairs=-1, d_min=0.02)
        with self.assertRaises(InvalidCollisionPairs):
            SelfCollisionBarrier(
                n_collision_pairs=len(
                    self.robot.collision_model.collisionPairs
                )
                + 5,
                d_min=0.02,
            ).compute_barrier(self.configuration)

    def test_dimension(self):
        """Check dimensions of configuration limit projection."""
        # The dimension of the barrier depends on the number of collision pairs
        self.assertEqual(
            SelfCollisionBarrier(
                n_collision_pairs=10,
                d_min=0.02,
            ).dim,
            10,
        )
        self.assertEqual(
            SelfCollisionBarrier(
                n_collision_pairs=len(
                    self.robot.collision_model.collisionPairs
                ),
                d_min=0.02,
            ).dim,
            len(self.robot.collision_model.collisionPairs),
        )

    def test_gains(self):
        """Check gains of configuration limit projection."""
        # One limit, scalar gain
        self.assertEqual(
            SelfCollisionBarrier(
                n_collision_pairs=10,
                d_min=0.02,
            ).gain.shape,
            (10,),
        )

    def test_jacobians(self):
        """Test that shapes of jacobians in all barriers are correct."""

        barrier = SelfCollisionBarrier(
            n_collision_pairs=len(self.robot.collision_model.collisionPairs),
            d_min=0.02,
        )
        J = barrier.compute_jacobian(self.configuration)
        self.assertEqual(J.ndim, 2)
        self.assertEqual(
            J.shape[0], len(self.robot.collision_model.collisionPairs)
        )

    def test_positive_when_in_safety_zone(self):
        """Check that the barrier is positive when in the safety zone."""

        barrier = SelfCollisionBarrier(
            n_collision_pairs=len(self.robot.collision_model.collisionPairs),
            d_min=0.02,
        )
        h = barrier.compute_barrier(self.configuration)
        self.assertTrue(np.all(h[self.non_colliding_objects_pair_id] > 0))

    def test_negative_when_out_of_safety_zone(self):
        """Test that the barrier is negative when out of the safety zone."""
        barrier = SelfCollisionBarrier(
            n_collision_pairs=len(self.robot.collision_model.collisionPairs),
            d_min=0.02,
        )
        # See https://github.com/stephane-caron/pink/pull/129 for an
        # illustration of this test configuration
        q_test = np.zeros(self.robot.model.nq)
        q_test[1] = 2.0
        q_test[3] = -2.5
        configuration = Configuration(
            self.robot.model,
            self.robot.data,
            q_test,
            collision_model=self.robot.collision_model,
            collision_data=self.robot.collision_data,
        )
        h = barrier.compute_barrier(configuration)
        self.assertTrue(np.min(h) < 0)  # there is a collision

    def test_closest_collision_pairs(self):
        """Test that the closest collision pairs are considered if number of
        collision pairs is less than actual number of collision pairs"""
        barrier = SelfCollisionBarrier(
            n_collision_pairs=5,
            d_min=0.02,
        )
        h = barrier.compute_barrier(self.configuration)
        distances = np.array(
            [
                self.configuration.collision_data.distanceResults[
                    k
                ].min_distance
                - 0.02
                for k in range(
                    len(self.configuration.collision_model.collisionPairs)
                )
            ]
        )
        for h_i in h:
            self.assertTrue(np.sum(distances < h_i) < barrier.dim)
