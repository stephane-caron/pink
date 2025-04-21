#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

r"""Linear holonomic task :math:`A (q \ominus q_0) = b`."""

from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError, TaskJacobianNotSet
from .task import Task


class LinearHolonomicTask(Task):
    r"""Linear holonomic task :math:`A (q \ominus q_0) = b`.

    Attributes:
        A: Matrix that defines the task:

            .. raw:: latex html

                \begin{align}
                e(q) & = A (q \ominus^\ell q_0) - b \\
                \dot{e}(q) & := A \dot{q}
                \end{align}

            where :math:`e(q) \in \mathbb{R}^{k}` is the quantity that the task
            aims to drive to zero (:math:`k` is the dimension of the task).
        b: Vector that defines the affine part of the task
        q_0: Element for which we work in the Lie algebra. If set to None, it
            will use the neutral configuration of the robot.

    Notes:
        To be fully explicit, our quantities in the task equation :math:`e(q) =
        A (q \ominus^\ell q_0) - b` belong to the following sets:

            .. raw:: latex html

                \begin{align}
                e(q) & \in \mathbb{R}^p &
                A & \in \mathcal{L}(\mathfrak{g}^\ell_{q_0}, \mathbb{R}^p) &
                b & \in \mathbb{R}^p
                \end{align}

        where :math:`\mathfrak{g}^\ell_{q_0}` is the Lie algebra associated
        with the Lie group :math:`\mathcal{C}` (our configuration space), taken
        in the local frame (:math:`\ell`, *a.k.a.* body frame) at the reference
        configuration :math:`q_0`.

        We take the *local* difference :math:`q \ominus^\ell q_0` between our
        configurations :math:`q \in \mathcal{C}` and :math:`q_0 \in
        \mathcal{C}`, which is an element of the local Lie algebra in
        :math:`\mathfrak{g}^\ell_{q_0}` at :math:`q_0`. In Pinocchio the
        convention is to use local differences, unless specified otherwise
        (e.g. world-aligned frames). It should not be confused with the general
        Lie difference :math:`\ominus` (that one is never used in practice),
        nor the spatial difference :math:`\ominus^s`, which is an element of
        the spatial Lie algebra :math:`\mathfrak{g}^s_{q_0}` associated with
        :math:`q_0`:

            .. raw:: latex html

                \begin{align}
                q \ominus q_0 & \in \mathcal{T}_{q_0} \mathcal{C} \\
                q \ominus^\ell q_0 & \in \mathfrak{g}^\ell_{q_0} \\
                q \ominus^s q_0 & \in \mathfrak{g}^s_{q_0}
                \end{align}

        On a side note, here's a look at the dimensions of these elements:

            .. raw:: latex html

                \begin{align}
                \mathrm{dim}(q) = \mathrm{dim}(q_0) = n_q \\
                \mathrm{dim}(q \ominus q_0) = n_q \\
                \mathrm{dim}(q \ominus^\ell q_0)
                = \mathrm{dim}(q \ominus^s q_0)
                = n_v
                \end{align}

        Finally, we derive the Jacobian of our error :math:`e(q)` as follows,
        denoting by :math:`a = \log(q_0^{-1} q)` so that :math:`q \ominus^\ell
        q_0 = \log(a)`:

            .. raw:: latex html

                \begin{align}
                \left.\frac{\partial e(q)}{\partial q} \right|_{\ell}
                = A \left. \frac{\partial \log}{\partial q}
                \right|_{\ell}^{\ell} \left.
                \frac{\partial a}{\partial q} \right|_{\ell} = A \left.
                \frac{\partial \log}{\partial q}
                \right|_{\ell}
                \end{align}

        By virtue of the identity :math:`\left.{}^\ell \frac{\partial
        a}{\partial q}\right|_{\ell} = I_{\mathfrak{g}}`. (See Yann de
        Mont-Marin's future blog for details :p)
    """

    A: np.ndarray
    b: np.ndarray
    q_0: np.ndarray

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        q_0: np.ndarray,
        cost: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        r"""Create task.

        Args:
            A: Jacobian matrix of the task, of size :math:`p \times n_v`.
            b: Target vector of the task, of dimension :math:`p`.
            q_0: Stationary configuration :math:`q_0 \in \mathcal{C}`
                associated with the task, of dimension :math:`n_q`.
            cost: Cost vector of dimension :math:`p`.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
        """
        super().__init__(
            cost=cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        if b.shape[0] != A.shape[0]:
            raise TaskDefinitionError(
                f"Shape mismatch between {A.shape=} and {b.shape=}"
            )
        self.A = A
        self.b = b
        self.q_0 = q_0

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task error function.

        The error function :math:`e(q) \in \mathbb{R}^{p}` is the quantity that
        the task aims to drive to zero (:math:`p` is the dimension of the
        task). See the documentation of
        :func:`pink.tasks.task.Task.compute_error` for details.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error vector :math:`e(q)`.
        """
        q_ref = (
            pin.neutral(configuration.model) if self.q_0 is None else self.q_0
        )
        if not self.A.shape[1] == configuration.model.nv:
            raise TaskJacobianNotSet
        return (
            self.A
            @ pin.difference(configuration.model, q_ref, configuration.q)
            - self.b
        )

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task Jacobian at a given configuration.

        See the documentation of :func:`pink.tasks.task.Task.compute_jacobian`
        for details on the task Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task Jacobian :math:`J(q)`.
        """
        q_ref = (
            pin.neutral(configuration.model) if self.q_0 is None else self.q_0
        )
        if not self.A.shape[1] == configuration.model.nv:
            raise TaskJacobianNotSet
        return self.A @ pin.dDifference(
            configuration.model, q_ref, configuration.q, pin.ARG1
        )

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "LinearHolonomicTask("
            f"cost={self.cost}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )
