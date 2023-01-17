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

"""Build and solve the inverse kinematics problem."""

from typing import Iterable, Optional, Tuple

import numpy as np
import qpsolvers

from .configuration import Configuration
from .limits import compute_velocity_limits
from .tasks import Task


def __compute_qp_objective(
    configuration: Configuration, tasks: Iterable[Task], damping: float
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute the QP objective function.

    The Hessian matrix :math:`H` and linear vector :math:`c` define the QP
    objective function as:

    .. math::

        \frac{1}{2} \Delta q^T H \Delta q + c^T q

    The configuration displacement :math:`\Delta q` is the output of inverse
    kinematics (we divide it by :math:`\Delta t` to get a commanded velocity).

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
    c = configuration.tangent.zeros.copy()
    for task in tasks:
        H_task, c_task = task.compute_qp_objective(configuration)
        H += H_task
        c += c_task
    return (H, c)


def __compute_qp_inequalities(
    configuration, dt
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute inequality constraints for the quadratic program.

    Args:
        configuration: Robot configuration to read kinematics from.
        dt: Integration timestep in [s].

    Returns:
        Pair :math:`(G, h)` of inequality matrix and vector.

    Notes:
        We trim comparisons to infinity (equivalently: big floats) because some
        solvers don't support it. See for instance
        https://github.com/tasts-robots/pink/issues/10.
    """
    bounded_tangent = configuration.model.bounded_tangent
    if bounded_tangent.dim < 1:
        return None, None

    v_max, v_min = compute_velocity_limits(configuration, dt)
    bounded_proj = bounded_tangent.projection_matrix
    G = np.vstack([bounded_proj, -bounded_proj])
    h = np.hstack([dt * v_max, -dt * v_min])
    return G, h


def build_ik(
    configuration: Configuration,
    tasks: Iterable[Task],
    dt: float,
    damping: float = 1e-12,
) -> qpsolvers.Problem:
    r"""Build quadratic program from current configuration and tasks.

    This quadratic program is, in standard form:

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{\Delta q}{\mbox{minimize}} &
                \frac{1}{2} {\Delta q}^T H {\Delta q} + c^T {\Delta q} \\
            \mbox{subject to}
                & G {\Delta q} \leq h
        \end{array}\end{split}

    where :math:`\Delta q` is a vector of joint displacements corresponding to
    the joint velocities :math:`v = {\Delta q} / {\mathrm{d}t}`.

    Args:
        configuration: Robot configuration to read kinematics from.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: weight of Tikhonov (everywhere) regularization. Its unit is
            :math:`[\mathrm{cost}]^2 / [\mathrm{tangent}]` where
            :math:`[\mathrm{tangent}]` is "the" unit of robot velocities.
            Improves numerical stability, but larger values slow down all
            tasks.

    Returns:
        QP matrices :math:`(H, c)` for the cost and :math:`(G, h)` for linear
        inequalities.
    """
    configuration.check_limits()
    P, q = __compute_qp_objective(configuration, tasks, damping)
    G, h = __compute_qp_inequalities(configuration, dt)
    problem = qpsolvers.Problem(P, q, G, h)
    return problem


def solve_ik(
    configuration: Configuration,
    tasks: Iterable[Task],
    dt: float,
    solver: str,
    damping: float = 1e-12,
) -> np.ndarray:
    r"""Compute a velocity tangent to the current robot configuration.

    The computed velocity satisfies at (weighted) best the set of kinematic
    tasks given in argument.

    Args:
        configuration: Robot configuration to read kinematics from.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: weight of Tikhonov (everywhere) regularization. Its unit is
            :math:`[\mathrm{cost}]^2 / [\mathrm{tangent}]` where
            :math:`[\mathrm{tangent}]` is "the" unit of robot velocities.
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
    problem = build_ik(configuration, tasks, dt, damping)
    result = qpsolvers.solve_problem(problem, solver=solver)
    Delta_q = result.x
    assert Delta_q is not None
    v: np.ndarray = Delta_q / dt
    return v
