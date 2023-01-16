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
Test submodels.
"""

import unittest

import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink.bounded_tangent import BoundedTangent
from pink.utils import VectorSpace


class TestBoundedTangent(unittest.TestCase):

    """
    Test fixture for bounded tangent subspace.
    """

    def setUp(self):
        """
        Set test fixture up.
        """
        model = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        ).model
        self.bounded_tangent = BoundedTangent(model)
        self.model = model
        self.tangent = VectorSpace(model.nv)

    def test_tangent(self):
        """
        Check dimensions of regular tangent space.
        """
        nv = self.model.nv
        self.assertEqual(self.tangent.eye.shape, (nv, nv))
        self.assertEqual(self.tangent.ones.shape, (nv,))
        self.assertEqual(self.tangent.zeros.shape, (nv,))

    def test_bounded_tangent(self):
        """
        Check dimensions in bounded tangent space.
        """
        for joint in self.bounded_tangent.joints:
            self.assertGreaterEqual(joint.idx_q, 0)
            self.assertGreaterEqual(joint.idx_v, 0)
        nb = len(self.bounded_tangent.joints)  # those are only revolute joints
        nv = self.model.nv
        self.assertEqual(self.bounded_tangent.dim, nb)
        self.assertEqual(self.bounded_tangent.nv, nv)
        self.assertEqual(
            self.bounded_tangent.projection_matrix.shape, (nb, nv)
        )

    def test_unbounded_tangent(self):
        """
        Check that unbounded models don't fail.
        """
        empty_model = pin.Model()
        empty_bounded = BoundedTangent(empty_model)
        self.assertEqual(empty_bounded.dim, 0)
