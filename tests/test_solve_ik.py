#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Test inverse kinematics."""

import unittest

import numpy as np
import pinocchio as pin
import qpsolvers
from numpy.linalg import norm
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration, build_ik, solve_ik
from pink.barriers import PositionBarrier
from pink.exceptions import NotWithinConfigurationLimits
from pink.tasks import ComTask, FrameTask


class TestSolveIK(unittest.TestCase):
    """Test fixture for the solve_ik function."""

    def test_checks_configuration_limits(self):
        """IK checks for configuration limits."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        q = robot.q0
        q[7] = 20  # above limit for Upkie's first joint
        configuration = Configuration(robot.model, robot.data, q)
        with self.assertRaises(NotWithinConfigurationLimits):
            solve_ik(configuration, [], dt=1.0, solver="daqp")

    def test_ignore_configuration_limits(self):
        """If flag is set, do not check configuration limits."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        q = robot.q0
        q[7] = 20  # above limit for Upkie's first joint
        configuration = Configuration(robot.model, robot.data, q)

        solve_ik(
            configuration,
            [],
            dt=1.0,
            solver="daqp",
            safety_break=False,
        )

    def test_model_with_no_joint_limit(self):
        """Model with no joint limit has no inequality constraints."""
        model = pin.Model()
        model.addJoint(
            0, pin.JointModelSpherical(), pin.SE3.Identity(), "spherical"
        )
        robot = pin.RobotWrapper(model=model)
        configuration = Configuration(robot.model, robot.data, robot.q0)
        problem = build_ik(configuration, [], dt=1.0)
        self.assertIsNone(problem.G)
        self.assertIsNone(problem.h)

    def test_no_task(self):
        """Raise an error when the robot body is not found."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        tasks = []
        v = solve_ik(configuration, tasks, dt=1e-3, solver="daqp")
        self.assertTrue(np.allclose(v, np.zeros(robot.nv)))

    def test_single_task_fulfilled(self):
        """Velocity is zero when the only task is already fulfilled."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        task = FrameTask(
            "left_contact", position_cost=1.0, orientation_cost=1.0
        )
        task.set_target(
            configuration.get_transform_frame_to_world("left_contact")
        )
        velocity = solve_ik(configuration, [task], dt=5e-3, solver="daqp")
        self.assertTrue(np.allclose(velocity, 0.0))

    def test_barrier_fullfilled(self):
        """Velocity is zero when the only task is already fulfilled."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        task = FrameTask(
            "left_contact", position_cost=1.0, orientation_cost=1.0
        )
        barrier = PositionBarrier(
            "left_contact",
            p_min=configuration.get_transform_frame_to_world(
                "left_contact"
            ).translation
            - 0.1 * np.ones(3),
        )
        task.set_target(
            configuration.get_transform_frame_to_world("left_contact")
        )
        velocity = solve_ik(
            configuration,
            [task],
            dt=5e-3,
            solver="daqp",
            barriers=[barrier],
        )
        self.assertTrue(np.allclose(velocity, 0.0))

    def test_barrier_violated(self):
        """Velocity is zero when the only task is already fulfilled."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        task = FrameTask(
            "left_contact", position_cost=1.0, orientation_cost=1.0
        )
        barrier = PositionBarrier(
            "left_contact",
            p_min=configuration.get_transform_frame_to_world(
                "left_contact"
            ).translation
            + 1.0 * np.ones(3),
        )
        task.set_target(
            configuration.get_transform_frame_to_world("left_contact")
        )
        velocity = solve_ik(
            configuration,
            [task],
            dt=5e-3,
            solver="daqp",
            barriers=[barrier],
        )
        self.assertFalse(np.allclose(velocity, 0.0))

    def test_single_task_convergence(self):
        """Integrating velocities makes a task converge to its target."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        task = FrameTask(
            "left_contact", position_cost=1.0, orientation_cost=1.0
        )
        transform_init_to_world = configuration.get_transform_frame_to_world(
            "left_contact"
        )
        transform_target_to_init = pin.SE3(
            np.eye(3), np.array([0.0, 0.0, 0.1])
        )
        transform_target_to_world = (
            transform_init_to_world * transform_target_to_init
        )
        task.set_target(transform_target_to_world)
        dt = 5e-3  # [s]
        velocity = solve_ik(configuration, [task], dt, solver="daqp")

        # Initially we are nowhere near the target and moving
        self.assertFalse(np.allclose(velocity, 0.0))
        self.assertAlmostEqual(norm(task.compute_error(configuration)), 0.1)
        self.assertFalse(
            configuration.get_transform_frame_to_world(
                "left_contact"
            ).isApprox(transform_target_to_world, prec=1e-4)
        )

        last_error = 1e6
        for nb_steps in range(42):
            error = norm(task.compute_error(configuration))
            if error < 1e-6 and np.allclose(velocity, 0.0):
                break
            self.assertLess(error, last_error)  # error stictly decreases
            last_error = error
            q = configuration.integrate(velocity, dt)
            configuration = Configuration(robot.model, robot.data, q)
            velocity = solve_ik(configuration, [task], dt, solver="daqp")

        # After nb_steps we are at the target and not moving
        self.assertTrue(np.allclose(velocity, 0.0))
        self.assertAlmostEqual(norm(task.compute_error(configuration)), 0.0)
        self.assertTrue(
            configuration.get_transform_frame_to_world(
                "left_contact"
            ).isApprox(transform_target_to_world, prec=1e-8)
        )
        self.assertLess(nb_steps, 3)

    def test_single_task_translation(self):
        """Translating a target yields a pure linear velocity."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        contact_task = FrameTask(
            "right_contact", position_cost=1.0, orientation_cost=1.0
        )
        transform_target_to_world = configuration.get_transform_frame_to_world(
            "right_contact"
        ).copy()
        self.assertTrue(
            np.allclose(transform_target_to_world.rotation, np.eye(3)),
            "default orientation should be aligned with the world frame",
        )
        transform_target_to_world.translation[1] -= 0.1
        contact_task.set_target(transform_target_to_world)
        contact_task.lm_damping = 0.0  # only Tikhonov damping for this test
        velocity = solve_ik(
            configuration,
            [contact_task],
            dt=1e-3,
            damping=1e-12,
            solver="daqp",
        )
        jacobian_contact_in_contact = configuration.get_frame_jacobian(
            "right_contact"
        )
        velocity_contact_in_contact = jacobian_contact_in_contact @ velocity
        linear_velocity_contact_in_contact = velocity_contact_in_contact[0:3]
        angular_velocity_contact_in_contact = velocity_contact_in_contact[3:6]
        self.assertTrue(np.allclose(angular_velocity_contact_in_contact, 0.0))
        self.assertAlmostEqual(linear_velocity_contact_in_contact[0], 0.0)
        self.assertLess(linear_velocity_contact_in_contact[1], 0.0)
        self.assertAlmostEqual(linear_velocity_contact_in_contact[2], 0.0)

    def test_three_tasks_fulfilled(self):
        """No motion when all targets are reached."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        left_ankle_task = FrameTask(
            "l_ankle", position_cost=1.0, orientation_cost=3.0
        )
        right_ankle_task = FrameTask(
            "r_ankle", position_cost=1.0, orientation_cost=3.0
        )
        pelvis_task = FrameTask(
            "PELVIS_S", position_cost=1.0, orientation_cost=3.0
        )

        left_ankle_task.set_target(
            configuration.get_transform_frame_to_world("l_ankle")
        )
        right_ankle_task.set_target(
            configuration.get_transform_frame_to_world("r_ankle")
        )
        pelvis_task.set_target(
            configuration.get_transform_frame_to_world("PELVIS_S")
        )

        tasks = [pelvis_task, left_ankle_task, right_ankle_task]
        velocity = solve_ik(configuration, tasks, dt=5e-3, solver="daqp")
        self.assertTrue(np.allclose(velocity, 0.0))

    def test_three_tasks_convergence(self):
        """Three simultaneously feasible tasks on the JVRC model converge."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)

        # Define tasks
        left_ankle_task = FrameTask(
            "l_ankle", position_cost=1.0, orientation_cost=3.0
        )
        right_ankle_task = FrameTask(
            "r_ankle", position_cost=1.0, orientation_cost=3.0
        )
        pelvis_task = FrameTask(
            "PELVIS_S", position_cost=1.0, orientation_cost=0.0
        )
        tasks = [pelvis_task, left_ankle_task, right_ankle_task]

        # Set task targets
        transform_l_ankle_target_to_init = pin.SE3(
            np.eye(3), np.array([0.1, 0.0, 0.0])
        )
        transform_r_ankle_target_to_init = pin.SE3(
            np.eye(3), np.array([-0.1, 0.0, 0.0])
        )
        left_ankle_task.set_target(
            configuration.get_transform_frame_to_world("l_ankle")
            * transform_l_ankle_target_to_init
        )
        right_ankle_task.set_target(
            configuration.get_transform_frame_to_world("r_ankle")
            * transform_r_ankle_target_to_init
        )
        pelvis_task.set_target(
            configuration.get_transform_frame_to_world("PELVIS_S")
        )

        # Run IK in closed loop
        dt = 4e-3  # [s]
        max_iter = 42
        conv_velocity_norm = 1e-6
        for nb_iter in range(max_iter):
            velocity = solve_ik(
                configuration,
                tasks,
                dt,
                solver="proxqp",
                check_duality_gap=True,
                eps_abs=1e-6,
            )
            if norm(velocity) < conv_velocity_norm:
                break
            q = configuration.integrate(velocity, dt)
            configuration = Configuration(robot.model, robot.data, q)
        self.assertLess(nb_iter, max_iter)
        self.assertLess(norm(velocity), conv_velocity_norm)
        self.assertLess(
            max(norm(task.compute_error(configuration)) for task in tasks),
            0.5,
        )

    def test_com_task_fulfilled(self):
        """No motion when all targets, including the CoM, are reached."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        left_ankle_task = FrameTask(
            "l_ankle", position_cost=1.0, orientation_cost=3.0
        )
        right_ankle_task = FrameTask(
            "r_ankle", position_cost=1.0, orientation_cost=3.0
        )
        com_task = ComTask(cost=2.0)

        left_ankle_task.set_target(
            configuration.get_transform_frame_to_world("l_ankle")
        )
        right_ankle_task.set_target(
            configuration.get_transform_frame_to_world("r_ankle")
        )
        com_task.set_target_from_configuration(configuration)

        tasks = [com_task, left_ankle_task, right_ankle_task]
        velocity = solve_ik(configuration, tasks, dt=5e-3, solver="daqp")
        self.assertTrue(np.allclose(velocity, 0.0))

    def test_com_task_convergence(self):
        """Feasible CoM and ankle tasks on the JVRC model converge."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)

        # Define tasks
        left_ankle_task = FrameTask(
            "l_ankle", position_cost=1.0, orientation_cost=3.0
        )
        right_ankle_task = FrameTask(
            "r_ankle", position_cost=1.0, orientation_cost=3.0
        )
        com_task = ComTask(cost=2.0)
        tasks = [com_task, left_ankle_task, right_ankle_task]

        # Set task targets
        transform_l_ankle_target_to_init = pin.SE3(
            np.eye(3), np.array([0.1, 0.0, 0.0])
        )
        transform_r_ankle_target_to_init = pin.SE3(
            np.eye(3), np.array([-0.1, 0.0, 0.0])
        )
        left_ankle_task.set_target(
            configuration.get_transform_frame_to_world("l_ankle")
            * transform_l_ankle_target_to_init
        )
        right_ankle_task.set_target(
            configuration.get_transform_frame_to_world("r_ankle")
            * transform_r_ankle_target_to_init
        )

        # Set the desired CoM target 0.05 above the initial CoM position
        initial_com = pin.centerOfMass(
            robot.model, robot.data, configuration.q
        )
        desired_com = initial_com.copy()
        desired_com[2] += 0.05
        com_task.set_target(desired_com)

        # Run IK in closed loop
        dt = 4e-3  # [s]
        max_iter = 42
        conv_velocity_norm = 1e-5
        for nb_iter in range(max_iter):
            velocity = solve_ik(
                configuration,
                tasks,
                dt,
                solver="osqp",
                eps_abs=1e-8,
            )
            if np.linalg.norm(velocity) < conv_velocity_norm:
                break
            configuration.integrate_inplace(velocity, dt)

        self.assertLess(nb_iter, max_iter)
        self.assertLess(np.linalg.norm(velocity), conv_velocity_norm)
        self.assertLess(
            max(
                np.linalg.norm(task.compute_error(configuration))
                for task in tasks
            ),
            0.5,
        )

    def get_jvrc_problem(self):
        """Get an IK problem with three tasks on a humanoid model."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        left_ankle_task = FrameTask(
            "l_ankle", position_cost=1.0, orientation_cost=3.0
        )
        right_ankle_task = FrameTask(
            "r_ankle", position_cost=1.0, orientation_cost=3.0
        )
        pelvis_task = FrameTask(
            "PELVIS_S", position_cost=1.0, orientation_cost=3.0
        )
        left_ankle_task.set_target(
            configuration.get_transform_frame_to_world("l_ankle")
        )
        right_ankle_task.set_target(
            configuration.get_transform_frame_to_world("r_ankle")
        )
        pelvis_task.set_target(
            configuration.get_transform_frame_to_world("PELVIS_S")
        )
        tasks = [pelvis_task, left_ankle_task, right_ankle_task]
        dt = 5e-3
        return configuration, tasks, dt

    @staticmethod
    def get_solver_test(solver: str):
        """Get test function for a given QP solver.

        Args:
            solver: Name of the QP solver to use as backend.

        Returns:
            Test function for that solver.
        """

        def test(self):
            configuration, tasks, dt = self.get_jvrc_problem()
            solve_ik(configuration, tasks, dt, solver=solver)

        return test


# Generate test fixtures for each solver
for solver in qpsolvers.available_solvers:
    if solver != "highs":
        setattr(
            TestSolveIK, f"test_{solver}", TestSolveIK.get_solver_test(solver)
        )
