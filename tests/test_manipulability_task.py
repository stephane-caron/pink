#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test fixture for the manipulability task."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.tasks import ManipulabilityTask


class TestManipulabilityTask(unittest.TestCase):
    """Test consistency of the manipulability task.

    Note:
        This fixture only tests the task itself. Integration tests with the IK
        are carried out in :class:`TestSolveIK`.
    """

    def setUp(self):
        """Prepare test fixture."""
        robot = load_robot_description("ur3_description", root_joint=None)
        self.configuration = Configuration(robot.model, robot.data, robot.q0)
        # Use the end-effector frame for manipulability computation
        self.frame_name = "ee_link"

    def test_task_repr(self):
        """String representation reports the task parameters."""
        task = ManipulabilityTask(self.frame_name, cost=1.0)
        self.assertTrue("frame=" in repr(task))
        self.assertTrue("cost=" in repr(task))
        self.assertTrue("gain=" in repr(task))
        self.assertTrue("lm_damping=" in repr(task))
        self.assertTrue("manipulability_rate=" in repr(task))

    def test_compute_manipulability(self):
        """Manipulability should be a non-negative scalar."""
        task = ManipulabilityTask(self.frame_name, cost=1.0)
        m = task.compute_manipulability(self.configuration)
        self.assertIsInstance(m, float)
        self.assertGreaterEqual(m, 0.0)

    def test_compute_jacobian_shape(self):
        """Jacobian should have shape (1, nv)."""
        task = ManipulabilityTask(self.frame_name, cost=1.0)
        J = task.compute_jacobian(self.configuration)
        nv = self.configuration.model.nv
        self.assertEqual(J.shape, (1, nv))

    def test_compute_error_shape(self):
        """Error should have shape (1,)."""
        task = ManipulabilityTask(self.frame_name, cost=1.0)
        e = task.compute_error(self.configuration)
        self.assertEqual(e.shape, (1,))

    def test_compute_error_value(self):
        """Error should be the negative of the manipulability rate."""
        manipulability_rate = 0.5
        task = ManipulabilityTask(
            self.frame_name, cost=1.0, manipulability_rate=manipulability_rate
        )
        e = task.compute_error(self.configuration)
        self.assertAlmostEqual(e[0], -manipulability_rate)

    def test_compute_kinematic_hessian_shape(self):
        """Kinematic Hessian should have shape (nv, 6, nv)."""
        task = ManipulabilityTask(self.frame_name, cost=1.0)
        H = task.compute_kinematic_hessian(self.configuration)
        nv = self.configuration.model.nv
        self.assertEqual(H.shape, (nv, 6, nv))

    def test_qp_objective_shapes(self):
        """QP objective matrices should have correct shapes."""
        task = ManipulabilityTask(self.frame_name, cost=1.0)
        H, c = task.compute_qp_objective(self.configuration)
        nv = self.configuration.model.nv
        self.assertEqual(H.shape, (nv, nv))
        self.assertEqual(c.shape, (nv,))

    def test_unit_cost_qp_objective(self):
        """Unit cost means the QP objective is exactly (J^T J, -e^T J)."""
        task = ManipulabilityTask(self.frame_name, cost=1.0, lm_damping=0.0)
        J = task.compute_jacobian(self.configuration)
        e = task.compute_error(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(e.T @ J, c))

    def test_zero_cost_disables_task(self):
        """The task has no effect when its cost is zero."""
        task = ManipulabilityTask(self.frame_name, cost=0.0)
        J = task.compute_jacobian(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        qd = np.random.random(J.shape[1])
        cost = qd.T @ H @ qd + c @ qd
        self.assertAlmostEqual(cost, 0.0)

    def test_invalid_reference_frame_raises(self):
        """Invalid reference frame should raise ValueError."""
        with self.assertRaises(ValueError):
            ManipulabilityTask(
                self.frame_name,
                cost=1.0,
                reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

    def test_mask_position(self):
        """Position mask selects only linear velocity components."""
        task = ManipulabilityTask(self.frame_name, cost=1.0, mask="position")
        self.assertTrue(
            np.array_equal(task.mask, np.array([1, 1, 1, 0, 0, 0]))
        )
        m = task.compute_manipulability(self.configuration)
        self.assertGreaterEqual(m, 0.0)

    def test_mask_orientation(self):
        """Orientation mask selects only angular velocity components."""
        task = ManipulabilityTask(
            self.frame_name, cost=1.0, mask="orientation"
        )
        self.assertTrue(
            np.array_equal(task.mask, np.array([0, 0, 0, 1, 1, 1]))
        )
        m = task.compute_manipulability(self.configuration)
        self.assertGreaterEqual(m, 0.0)

    def test_mask_planar_xy(self):
        """Planar XY mask selects only x and y linear velocity components."""
        task = ManipulabilityTask(self.frame_name, cost=1.0, mask="planar_xy")
        self.assertTrue(
            np.array_equal(task.mask, np.array([1, 1, 0, 0, 0, 0]))
        )
        m = task.compute_manipulability(self.configuration)
        self.assertGreaterEqual(m, 0.0)

    def test_custom_mask(self):
        """Custom binary mask is applied correctly."""
        custom_mask = np.array([1, 0, 1, 0, 1, 0])
        task = ManipulabilityTask(self.frame_name, cost=1.0, mask=custom_mask)
        self.assertTrue(np.array_equal(task.mask, custom_mask))
        m = task.compute_manipulability(self.configuration)
        self.assertGreaterEqual(m, 0.0)

    def test_invalid_mask_string_raises(self):
        """Invalid mask string should raise ValueError."""
        with self.assertRaises(ValueError):
            ManipulabilityTask(self.frame_name, cost=1.0, mask="invalid_mask")

    def test_invalid_mask_shape_raises(self):
        """Custom mask with wrong shape should raise ValueError."""
        with self.assertRaises(ValueError):
            ManipulabilityTask(
                self.frame_name, cost=1.0, mask=np.array([1, 1, 1])
            )

    def test_invalid_mask_values_raises(self):
        """Custom mask with non-binary values should raise ValueError."""
        with self.assertRaises(ValueError):
            ManipulabilityTask(
                self.frame_name, cost=1.0, mask=np.array([1, 2, 1, 0, 0, 0])
            )

    def test_different_manipulability_rates(self):
        """Different manipulability rates produce different errors."""
        task1 = ManipulabilityTask(
            self.frame_name, cost=1.0, manipulability_rate=0.1
        )
        task2 = ManipulabilityTask(
            self.frame_name, cost=1.0, manipulability_rate=0.5
        )
        e1 = task1.compute_error(self.configuration)
        e2 = task2.compute_error(self.configuration)
        self.assertNotAlmostEqual(e1[0], e2[0])

    def test_lm_damping_effect(self):
        """LM damping adds regularization to QP objective."""
        task_no_damp = ManipulabilityTask(
            self.frame_name, cost=1.0, lm_damping=0.0
        )
        task_with_damp = ManipulabilityTask(
            self.frame_name, cost=1.0, lm_damping=1e-3
        )
        H1, _ = task_no_damp.compute_qp_objective(self.configuration)
        H2, _ = task_with_damp.compute_qp_objective(self.configuration)
        self.assertGreater(np.trace(H2), np.trace(H1))

    def test_gain_effect(self):
        """Task gain affects the QP objective linear term."""
        task_full = ManipulabilityTask(
            self.frame_name, cost=1.0, gain=1.0, manipulability_rate=0.1
        )
        task_half = ManipulabilityTask(
            self.frame_name, cost=1.0, gain=0.5, manipulability_rate=0.1
        )
        _, c1 = task_full.compute_qp_objective(self.configuration)
        _, c2 = task_half.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(c2, 0.5 * c1))

    def test_cost_scaling(self):
        """Different costs scale the QP objective."""
        task1 = ManipulabilityTask(self.frame_name, cost=1.0, lm_damping=0.0)
        task2 = ManipulabilityTask(self.frame_name, cost=2.0, lm_damping=0.0)
        H1, c1 = task1.compute_qp_objective(self.configuration)
        H2, c2 = task2.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(2.0 * H1, H2))
        self.assertTrue(np.allclose(2.0 * c1, c2))

    def test_hessian_symmetry_properties(self):
        """Kinematic Hessian has expected structure for linear part."""
        task = ManipulabilityTask(self.frame_name, cost=1.0)
        H = task.compute_kinematic_hessian(self.configuration)
        H_linear = H[:, :3, :]
        for k in range(3):
            H_linear_k = H_linear[:, k, :]
            self.assertTrue(
                np.allclose(H_linear_k, H_linear_k.T),
                f"Linear Hessian not symmetric for component {k}",
            )


if __name__ == "__main__":
    unittest.main()
