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

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink.submodels import Subspace, add_submodels


class TestSubmodels(unittest.TestCase):
    def setUp(self):
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        self.model = robot.model
        add_submodels(self.model)

    def test_tangent(self):
        nv = self.model.nv
        tangent = self.model.tangent
        self.assertEqual(tangent.eye.shape, (nv, nv))
        self.assertEqual(tangent.ones.shape, (nv,))
        self.assertEqual(tangent.zeros.shape, (nv,))

    def test_bounded(self):
        bounded = self.model.bounded
        for joint in bounded.joints:
            self.assertGreaterEqual(joint.idx_q, 0)
            self.assertGreaterEqual(joint.idx_v, 0)
        nb = len(bounded.joints)  # those are only revolute joints

        nv = self.model.nv
        self.assertEqual(bounded.tangent.dim, nb)
        self.assertEqual(bounded.tangent.input_dim, nv)
        self.assertEqual(bounded.tangent.projection_matrix.shape, (nb, nv))

        nq = self.model.nq
        self.assertEqual(bounded.configuration.dim, nb)
        self.assertEqual(bounded.configuration.input_dim, nq)
        self.assertEqual(
            bounded.configuration.projection_matrix.shape, (nb, nq)
        )

    def test_subspace(self):
        """
        Check Subspace class.
        """
        subspace = Subspace(2, [1])
        P = subspace.projection_matrix
        self.assertLess((P - np.array([0.0, 1.0])).max(), 1e-10)

    def test_unbounded(self):
        """
        Check that unbounded models don't fail.
        """
        empty_model = pin.Model()
        add_submodels(empty_model)
