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
        robot = load_robot_description(
            "ur3_official_description", root_joint=None
        )
        self.configuration = Configuration(robot.model, robot.data, robot.q0)
        # Use the end-effector frame for manipulability computation
        self.frame_name = "tool0"

    def test_task_repr(self):
        """String representation reports the task parameters."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0
        )
        self.assertTrue("frame=" in repr(task))
        self.assertTrue("cost=" in repr(task))
        self.assertTrue("gain=" in repr(task))
        self.assertTrue("lm_damping=" in repr(task))
        self.assertTrue("manipulability_rate=" in repr(task))

    def test_compute_manipulability(self):
        """Manipulability should be a non-negative scalar."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0
        )
        m = task.compute_manipulability(self.configuration)
        self.assertIsInstance(m, float)
        self.assertGreaterEqual(m, 0.0)

    def test_compute_jacobian_shape(self):
        """Jacobian should have shape (1, nv)."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0
        )
        J = task.compute_jacobian(self.configuration)
        nv = self.configuration.model.nv
        self.assertEqual(J.shape, (1, nv))

    def test_compute_error_shape(self):
        """Error should have shape (1,)."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0
        )
        e = task.compute_error(self.configuration)
        self.assertEqual(e.shape, (1,))

    def test_compute_error_value(self):
        """Error should be the negative of the manipulability rate."""
        manipulability_rate = 0.5
        task = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            manipulability_rate=manipulability_rate,
        )
        e = task.compute_error(self.configuration)
        self.assertAlmostEqual(e[0], -manipulability_rate)

    def test_compute_kinematic_hessian_shape(self):
        """Kinematic Hessian should have shape (nv, 6, nv)."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0
        )
        H = task.compute_kinematic_hessian(self.configuration)
        nv = self.configuration.model.nv
        self.assertEqual(H.shape, (nv, 6, nv))

    def test_qp_objective_shapes(self):
        """QP objective matrices should have correct shapes."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0
        )
        H, c = task.compute_qp_objective(self.configuration)
        nv = self.configuration.model.nv
        self.assertEqual(H.shape, (nv, nv))
        self.assertEqual(c.shape, (nv,))

    def test_unit_cost_qp_objective(self):
        """Unit cost means the QP objective is exactly (J^T J, -e^T J)."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0, lm_damping=0.0
        )
        J = task.compute_jacobian(self.configuration)
        e = task.compute_error(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(e.T @ J, c))

    def test_zero_cost_disables_task(self):
        """The task has no effect when its cost is zero."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=0.0
        )
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
                self.configuration.model,
                cost=1.0,
                reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

    def test_mask_position(self):
        """Position mask selects only linear velocity components."""
        task = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            mask="position",
        )
        self.assertTrue(
            np.array_equal(task.mask, np.array([1, 1, 1, 0, 0, 0]))
        )
        m = task.compute_manipulability(self.configuration)
        self.assertGreaterEqual(m, 0.0)

    def test_mask_orientation(self):
        """Orientation mask selects only angular velocity components."""
        task = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            mask="orientation",
        )
        self.assertTrue(
            np.array_equal(task.mask, np.array([0, 0, 0, 1, 1, 1]))
        )
        m = task.compute_manipulability(self.configuration)
        self.assertGreaterEqual(m, 0.0)

    def test_mask_planar_xy(self):
        """Planar XY mask selects only x and y linear velocity components."""
        task = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            mask="planar_xy",
        )
        self.assertTrue(
            np.array_equal(task.mask, np.array([1, 1, 0, 0, 0, 0]))
        )
        m = task.compute_manipulability(self.configuration)
        self.assertGreaterEqual(m, 0.0)

    def test_custom_mask(self):
        """Custom binary mask is applied correctly."""
        custom_mask = np.array([1, 0, 1, 0, 1, 0])
        task = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            mask=custom_mask,
        )
        self.assertTrue(np.array_equal(task.mask, custom_mask))
        m = task.compute_manipulability(self.configuration)
        self.assertGreaterEqual(m, 0.0)

    def test_invalid_mask_string_raises(self):
        """Invalid mask string should raise ValueError."""
        with self.assertRaises(ValueError):
            ManipulabilityTask(
                self.frame_name,
                self.configuration.model,
                cost=1.0,
                mask="invalid_mask",
            )

    def test_invalid_mask_shape_raises(self):
        """Custom mask with wrong shape should raise ValueError."""
        with self.assertRaises(ValueError):
            ManipulabilityTask(
                self.frame_name,
                self.configuration.model,
                cost=1.0,
                mask=np.array([1, 1, 1]),
            )

    def test_invalid_mask_values_raises(self):
        """Custom mask with non-binary values should raise ValueError."""
        with self.assertRaises(ValueError):
            ManipulabilityTask(
                self.frame_name,
                self.configuration.model,
                cost=1.0,
                mask=np.array([1, 2, 1, 0, 0, 0]),
            )

    def test_different_manipulability_rates(self):
        """Different manipulability rates produce different errors."""
        task1 = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            manipulability_rate=0.1,
        )
        task2 = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            manipulability_rate=0.5,
        )
        e1 = task1.compute_error(self.configuration)
        e2 = task2.compute_error(self.configuration)
        self.assertNotAlmostEqual(e1[0], e2[0])

    def test_lm_damping_effect(self):
        """LM damping adds regularization to QP objective."""
        task_no_damp = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0, lm_damping=0.0
        )
        task_with_damp = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            lm_damping=1e-3,
        )
        H1, _ = task_no_damp.compute_qp_objective(self.configuration)
        H2, _ = task_with_damp.compute_qp_objective(self.configuration)
        self.assertGreater(np.trace(H2), np.trace(H1))

    def test_gain_effect(self):
        """Task gain affects the QP objective linear term."""
        task_full = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            gain=1.0,
            manipulability_rate=0.1,
        )
        task_half = ManipulabilityTask(
            self.frame_name,
            self.configuration.model,
            cost=1.0,
            gain=0.5,
            manipulability_rate=0.1,
        )
        _, c1 = task_full.compute_qp_objective(self.configuration)
        _, c2 = task_half.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(c2, 0.5 * c1))

    def test_cost_scaling(self):
        """Different costs scale the QP objective."""
        task1 = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0, lm_damping=0.0
        )
        task2 = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=2.0, lm_damping=0.0
        )
        H1, c1 = task1.compute_qp_objective(self.configuration)
        H2, c2 = task2.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(2.0 * H1, H2))
        self.assertTrue(np.allclose(2.0 * c1, c2))

    def test_hessian_symmetry_properties(self):
        """Kinematic Hessian has expected structure for linear part."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0
        )
        H = task.compute_kinematic_hessian(self.configuration)
        H_linear = H[:, :3, :]
        for k in range(3):
            H_linear_k = H_linear[:, k, :]
            self.assertTrue(
                np.allclose(H_linear_k, H_linear_k.T),
                f"Linear Hessian not symmetric for component {k}",
            )

    def test_prismatic_joint_raises(self):
        """Prismatic joint on kinematic path should raise ValueError."""
        urdf_string = """\
<?xml version="1.0"?>
<robot name="prismatic_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="ee"/>
  <joint name="revolute_joint" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>
  <joint name="prismatic_joint" type="prismatic">
    <parent link="link1"/>
    <child link="ee"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="1.0" effort="100" velocity="1.0"/>
  </joint>
</robot>"""
        model = pin.buildModelFromXML(urdf_string)
        with self.assertRaises(ValueError):
            ManipulabilityTask("ee", model, cost=1.0)

    def test_continuous_joint_does_not_raise(self):
        """Continuous (axis-aligned unbounded revolute) joint is supported.

        Regression test: such joints (e.g. the wrist_3_joint of the
        official UR descriptions) report their pinocchio shortname as
        ``JointModelRUBX``/``RUBY``/``RUBZ``, not
        ``JointModelRevoluteUnbounded``.
        """
        urdf_string = """\
<?xml version="1.0"?>
<robot name="continuous_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="ee"/>
  <joint name="continuous_joint" type="continuous">
    <parent link="base_link"/>
    <child link="ee"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>"""
        model = pin.buildModelFromXML(urdf_string)
        ManipulabilityTask("ee", model, cost=1.0)

    def test_manipulability_jacobian_vs_finite_differences(self):
        """Manipulability Jacobian should match finite differences."""
        task = ManipulabilityTask(
            self.frame_name, self.configuration.model, cost=1.0
        )
        # q0 is singular for UR3; use a non-singular configuration
        model = self.configuration.model
        q = pin.integrate(
            model,
            pin.neutral(model),
            np.array(
                [0.0, -np.pi / 4, np.pi / 2, -np.pi / 4, -np.pi / 2, 0.0]
            ),
        )
        configuration = Configuration(model, self.configuration.data, q)
        Jm = task.compute_jacobian(configuration)

        eps = 1e-6
        nv = model.nv
        grad_fd = np.zeros(nv)
        for i in range(nv):
            e_i = np.eye(nv)[i]
            q_plus = pin.integrate(model, q, eps * e_i)
            config_plus = Configuration(
                model,
                self.configuration.data,
                q_plus,
            )
            q_minus = pin.integrate(model, q, -eps * e_i)
            config_minus = Configuration(
                model,
                self.configuration.data,
                q_minus,
            )
            m_plus = task.compute_manipulability(config_plus)
            m_minus = task.compute_manipulability(config_minus)
            grad_fd[i] = (m_plus - m_minus) / (2 * eps)

        np.testing.assert_allclose(Jm.flatten(), grad_fd, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
