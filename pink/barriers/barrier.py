#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron, Ivan Domrachev, Simeon Nedelchev

"""All barriers derive from the :class:`Barrier` base class.

The formalism used in this implementation is written down in
examples/barriers/NOTES.md
"""

import abc
from typing import Callable, Optional, Tuple, Union

import numpy as np

from ..configuration import Configuration


class Barrier(abc.ABC):
    r"""Abstract base class for barrier.

    A barrier is a function :math:`h(q)` that
    satisfies the following condition:

    .. math::

        \frac{\partial h_j}{\partial q}
        \dot{q} +\alpha_j(h_j(q))
        \geq 0, \quad \forall j

    where :math:`\frac{\partial h_j}{\partial q}`
    are the Jacobians of the constraint functions, :math:`\dot{q}`
    is the joint velocity vector, and :math:`\alpha_j` are extended
    `class kappa <https://en.wikipedia.org/wiki/Class_kappa_function>`__ functions.

    Attributes:
        dim: Dimension of the barrier.
        gain: linear barrier gain.
        gain_function: Extended class K function.
            Defines stabilization term as nonlinear function of barrier.
            Defaults to the (linear) identity function.
        safe_policy: Safe backup control policy.
        r: Weighting factor for the safe backup policy regularization term.
    """

    gain: np.ndarray
    gain_function: Callable[[float], float]
    safe_policy: np.ndarray
    r: float

    def __init__(
        self,
        dim: int,
        gain: Union[float, np.ndarray] = 1.0,
        gain_function: Optional[Callable[[float], float]] = None,
        r: float = 3.0,
    ):
        """Initialize the barrier.

        Args:
            dim: Dimension of the barrier.
            gain: barrier gain. Defaults to 1.0.
            class_k_fn: Extended class K function.
                Defaults to the identity function.
            r: Weighting factor for the safe backup policy regularization term.
                Defaults to 3.0.
        """
        self.dim = dim
        self.gain = (
            gain if isinstance(gain, np.ndarray) else np.ones(dim) * gain
        )

        self.gain_function = (
            gain_function if gain_function is not None else lambda x: x
        )
        self.r = r
        self.safe_policy = np.zeros(self.dim)

    @abc.abstractmethod
    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the value of the barrier function.

        The barrier function :math:`h(q)`
        is a vector-valued function that represents the safety constraints.
        It should be designed such that the set
        :math:`\{q : h(q) \geq 0\}`
        represents the safe region of the configuration space.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Value of the barrier function :math:`h(q)`.
        """

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the Jacobian matrix of the barrier function.

        The Jacobian matrix
        :math:`\frac{\partial h}{\partial q}(q)`
        of the barrier function with respect to the configuration variables is
        required for the computation of the barrier condition.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix
            :math:`\frac{\partial h}{\partial q}(q)`.
        """

    def compute_safe_policy(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the safe backup control policy.

        The safe backup control policy
        :math:`\dot{q}_{safe}(q)` is a joint
        velocity vector that can be used as a regularization term in the
        optimization problem to ensure safety.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Safe backup joint velocities
                :math:`\dot{q}_{safe}(q)`.
        """
        return np.zeros(configuration.model.nv)

    def compute_qp_objective(
        self,
        configuration: Configuration,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the quadratic objective function for the barrier-based QP.

        The quadratic objective function includes a regularization term
        based on the safe backup policy:

        .. math::

            \gamma(q)\left\| \dot{q}-
            \dot{q}_{safe}(q)\right\|^{2}

        where :math:`\gamma(q)` is a configuration-dependent
        weight and :math:`\dot{q}_{safe}(q)`
        is the safe backup policy.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Time step for discrete-time implementation. Defaults to 1e-3.

        Returns:
            Tuple containing the quadratic objective matrix (H) and linear
                objective vector (c).
        """
        jac = self.compute_jacobian(configuration)
        H = np.zeros((configuration.model.nv, configuration.model.nv))
        # c = 1e-3 * np.linalg.norm(jac, axis=0) / dt
        c = np.zeros(configuration.model.nv)

        if self.r > 1e-6:
            self.safe_policy = self.compute_safe_policy(configuration)

            H += (
                self.r
                / (np.linalg.norm(jac) ** 2)
                * np.eye(configuration.model.nv)
            )
            c += -2 * self.r / (np.linalg.norm(jac) ** 2) * self.safe_policy

        return (H, c)

    def compute_qp_inequality(
        self,
        configuration: Configuration,
        dt: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the linear inequality constraints for the barrier-based QP.

        The linear inequality constraints enforce the barrier conditions:

        .. math::

            \frac{\partial h_j}
            {\partial q} \dot{q} +
            \alpha_j(h_j(q)) \geq 0, \quad \forall j

        where :math:`\frac{\partial h_j}{\partial q}`
        are the Jacobians of the constraint functions,
        :math:`\dot{q}` is the joint velocity vector,
        and :math:`\alpha_j` are extended class K functions.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Time step for discrete-time implementation. Defaults to 1e-3.

        Returns:
            Tuple containing the inequality constraint matrix (G)
                and vector (h).
        """
        G = -self.compute_jacobian(configuration) / dt
        barrier_value = self.compute_barrier(configuration)
        h = np.array(
            [
                self.gain[i] * self.gain_function(barrier_value[i])
                for i in range(self.dim)
            ]
        )

        return (G, h)

    def __repr__(self) -> str:
        """Return a string representation of the barrier.

        Returns:
            str: String representation.
        """
        return (
            f"Barrier("
            f"gain={self.gain}, "
            f"safety_policy={'no' if np.allclose(self.r, 0) else self.safe_policy}, "  # noqa: E501
            f"r={'no' if np.allclose(self.r, 0) else self.r})"
        )
