#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Build and solve the inverse kinematics problem."""

from typing import Iterable, Optional, Tuple

import numpy as np
import qpsolvers

from .barriers import Barrier
from .configuration import Configuration
from .exceptions import NoSolutionFound
from .limits import Limit
from .tasks import Task


def __compute_qp_objective(
    configuration: Configuration,
    tasks: Iterable[Task],
    damping: float,
    barriers: Optional[Iterable[Barrier]] = None,
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
            :math:`[\mathrm{cost}]^2 / [\mathrm{tangent}]` where
            :math:`[\mathrm{tangent}]` is "the" unit of robot velocities.
            Improves numerical stability, but larger values slow down all
            tasks. The same weight is applied to all joints, including if
            applicable the root joint (floating base, ...).
        dt: Integration timestep in [s].
        barriers: Collection of barriers.

    Returns:
        Pair :math:`(H, c)` of Hessian matrix and linear vector of the QP
        objective.
    """
    barriers = barriers if barriers is not None else []
    H = damping * configuration.tangent.eye
    c = configuration.tangent.zeros.copy()
    for task in tasks:
        H_task, c_task = task.compute_qp_objective(configuration)
        H += H_task
        c += c_task

    for barrier in barriers:
        H_cbf, c_cbf = barrier.compute_qp_objective(configuration)
        H += H_cbf
        c += c_cbf

    return (H, c)


def __compute_qp_inequalities(
    configuration: Configuration,
    limits: Optional[Iterable[Limit]],
    dt: float,
    barriers: Optional[Iterable[Barrier]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    r"""Compute inequality constraints for the quadratic program.

    Args:
        configuration: Robot configuration to read kinematics from.
        limits: Collection of limits.
        dt: Integration timestep in [s].
        barriers: Collection of barriers.

    Returns:
        Pair :math:`(G, h)` of inequality matrix and vector representing the
        inequality :math:`G \Delta q \leq h`, or ``(None, None)`` if there is
        no inequality.

    Notes:
        We trim comparisons to infinity (equivalently: big floats) because some
        solvers don't support it. See for instance
        https://github.com/stephane-caron/pink/issues/10.
    """
    if limits is None:
        configuration_limit = configuration.model.configuration_limit
        velocity_limit = configuration.model.velocity_limit
        limits = (configuration_limit, velocity_limit)
    barriers = barriers if barriers is not None else []
    G_list = []
    h_list = []
    for limit in limits:
        matvec = limit.compute_qp_inequalities(configuration.q, dt)
        if matvec is not None:
            G_list.append(matvec[0])
            h_list.append(matvec[1])
    for barrier in barriers:
        G_barrier, h_barrier = barrier.compute_qp_inequalities(
            configuration, dt
        )
        G_list.append(G_barrier)
        h_list.append(h_barrier)
    if not G_list:
        return None, None
    return np.vstack(G_list), np.hstack(h_list)


def __compute_qp_equalities(
    configuration: Configuration,
    constraints: Optional[Iterable[Task]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    r"""Compute equality constraints for the quadratic program.

    Args:
        configuration: Robot configuration to read kinematics from.
        constraints: List of tasks to enforce via equality constraints.

    Returns:
        Pair :math:`(A, b)` of equality matrix and vector representing the
        equation :math:`A \Delta q = b`, or ``(None, None)`` if there is no
        equality constraint.
    """
    if not constraints:
        return None, None
    A_list = []
    b_list = []
    for task in constraints:
        jacobian = task.compute_jacobian(configuration)
        feedback = -task.gain * task.compute_error(configuration)
        A_list.append(jacobian)
        b_list.append(feedback)
    return np.vstack(A_list), np.hstack(b_list)


def build_ik(
    configuration: Configuration,
    tasks: Iterable[Task],
    dt: float,
    damping: float = 1e-12,
    limits: Optional[Iterable[Limit]] = None,
    barriers: Optional[Iterable[Barrier]] = None,
    constraints: Optional[Iterable[Task]] = None,
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
            tasks. The same weight is applied to all joints, including if
            applicable the root joint (floating base, ...).
        limits: Collection of limits to enforce. By default, consists of
            configuration and velocity limits (set to the empty list ``[]`` to
            disable limits).
        barriers: Collection of barriers.
        constraints: List of kinematic tasks to be enforced strictly, as hard
            equality constraints in the underlying QP rather than in its cost
            function.

    Returns:
        Quadratic program of the inverse kinematics problem.
    """
    P, q = __compute_qp_objective(configuration, tasks, damping, barriers)
    G, h = __compute_qp_inequalities(configuration, limits, dt, barriers)
    A, b = __compute_qp_equalities(configuration, constraints)
    problem = qpsolvers.Problem(P, q, G, h, A, b)
    return problem


def solve_ik(
    configuration: Configuration,
    tasks: Iterable[Task],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    limits: Optional[Iterable[Limit]] = None,
    barriers: Optional[Iterable[Barrier]] = None,
    constraints: Optional[Iterable[Task]] = None,
    safety_break: bool = True,
    **kwargs,
) -> np.ndarray:
    r"""Compute a velocity tangent to the current robot configuration.

    The computed velocity satisfies at (weighted) best the set of kinematic
    tasks given in argument.

    Args:
        configuration: Robot configuration to read kinematics from.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        solver: Backend quadratic programming (QP) solver.
        damping: weight of Tikhonov (everywhere) regularization. Its unit is
            :math:`[\mathrm{cost}]^2 / [\mathrm{tangent}]` where
            :math:`[\mathrm{tangent}]` is "the" unit of robot velocities.
            Improves numerical stability, but larger values slow down all
            tasks. The same weight is applied to all joints, including if
            applicable the root joint (floating base, ...).
        limits: Collection of limits to enforce. By default, consists of
            configuration and velocity limits (set to the empty list ``[]`` to
            disable limits).
        barriers: Collection of barrier functions.
        constraints: List of kinematic tasks to be enforced strictly, as hard
            equality constraints in the underlying QP rather than in its cost
            function.
        safety_break: If True, stop execution and raise an exception if
            the current configuration is outside limits. If False, print a
            warning and continue execution.
        kwargs: Keyword arguments to forward to the backend QP solver.

    Returns:
        Velocity :math:`v` in tangent space.

    Raises:
        NoSolutionFound: if the backend QP solver does not find a solution to
            the differential IK problem.
        NotWithinConfigurationLimits: if the current configuration is not
            within limits.

    Note:
        Our Tikhonov damping is isotropic despite tangent velocities not being
        homogeneous. If it helps we can add a tangent-space scaling to damp the
        floating base differently from joint angular velocities.
    """
    configuration.check_limits(safety_break=safety_break)
    problem = build_ik(
        configuration,
        tasks,
        dt,
        damping,
        limits,
        barriers,
        constraints,
    )
    result = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
    Delta_q = result.x
    if not result.found or Delta_q is None:
        raise NoSolutionFound(problem, result)
    v: np.ndarray = Delta_q / dt
    return v
