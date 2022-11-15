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
Function to solve inverse kinematics.
"""

from typing import Iterable, Optional, Tuple

import numpy as np
from qpsolvers import solve_qp

from .configuration import Configuration
from .limits import compute_velocity_limits
from .tasks import Task


def __compute_qp_objective(
    configuration: Configuration, tasks: Iterable[Task], damping: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Hessian matrix :math:`H` and linear vector :math:`c` of the
    QP objective function:

    .. math::

        \\frac{1}{2} \\Delta q^T H \\Delta q + c^T q

    The configuration displacement :math:`\\Delta q` is the output of inverse
    kinematics (we divide it by :math:`\\Delta t` to get a commanded velocity).

    Args:
        configuration: Robot configuration to read kinematics from.
        tasks: List of kinematic tasks to fulfill at (weighted) best.
        damping: weight of Tikhonov (everywhere) regularization. Its unit is
            `[cost]^2 / [tangent]` where `[tangent]` is "the" unit of robot
            velocities. Improves numerical stability, but larger values slow
            down all tasks.

    Returns:
        Pair :math:`(H, c)` of Hessian matrix and linear vector of the QP
        objective.
    """
    H = damping * configuration.tangent.eye
    c = configuration.tangent.zeros
    for task in tasks:
        H_task, c_task = task.compute_qp_objective(configuration)
        H += H_task
        c += c_task
    return (H, c)


def __compute_qp_inequalities(
    configuration, dt
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute inequality constraints for the quadratic program.

    Args:
        configuration: Robot configuration to read kinematics from.
        dt: Integration timestep in [s].

    Returns:
        Pair :math:`(A, b)` of inequality matrix and vector.

    Notes:
        We trim comparisons to infinity (equivalently: big floats) because some
        solvers don't support it. See for instance
        https://github.com/tasts-robots/pink/issues/10.
    """
    v_max, v_min = compute_velocity_limits(configuration, dt)
    bounded_proj = configuration.bounded.tangent.projection_matrix
    A = np.vstack([bounded_proj, -bounded_proj])
    b = np.hstack([dt * v_max, -dt * v_min])
    if b.size < 1:
        return None, None
    return A, b


def build_ik(
    configuration: Configuration,
    tasks: Iterable[Task],
    dt: float,
    damping: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build quadratic program from current configuration and tasks.

    Args:
        configuration: Robot configuration to read kinematics from.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: weight of Tikhonov (everywhere) regularization. Its unit is
            :math:`[\\mathrm{cost}]^2 / [\\mathrm{tangent}]` where
            :math:`[\\mathrm{tangent}]` is "the" unit of robot velocities.
            Improves numerical stability, but larger values slow down all
            tasks.

    Returns:
        QP matrices :math:`(H, c)` for the cost and :math:`(A, b)` for linear
        inequalities.
    """
    configuration.check_limits()
    H, c = __compute_qp_objective(configuration, tasks, damping)
    A, b = __compute_qp_inequalities(configuration, dt)
    return H, c, A, b


def solve_ik(
    configuration: Configuration,
    tasks: Iterable[Task],
    dt: float,
    solver: str,
    damping: float = 1e-12,
) -> np.ndarray:
    """
    Compute a velocity tangent to the current robot configuration that
    satisfies at (weighted) best a given set of kinematic tasks.

    Args:
        configuration: Robot configuration to read kinematics from.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: weight of Tikhonov (everywhere) regularization. Its unit is
            :math:`[\\mathrm{cost}]^2 / [\\mathrm{tangent}]` where
            :math:`[\\mathrm{tangent}]` is "the" unit of robot velocities.
            Improves numerical stability, but larger values slow down all
            tasks.
        solver: Backend quadratic programming solver.

    Returns:
        Velocity :math:`v` in tangent space.

    Raises:
        NotWithinConfigurationLimits: if the current configuration is not
            within limits.

    Note:
        Our Tikhonov damping is isotropic despite tangent velocities not being
        homogeneous. If it helps we can add a tangent-space scaling to damp the
        floating base differently from joint angular velocities.
    """
    configuration.check_limits()
    H, c, A, b = build_ik(configuration, tasks, dt, damping)
    Delta_q = solve_qp(H, c, A, b, solver=solver)
    assert Delta_q is not None
    v: np.ndarray = Delta_q / dt
    return v
