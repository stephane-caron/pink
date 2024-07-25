#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""All barriers derive from the :class:`Barrier` base class.

The formalism used in this implementation is written down in
https://simeon-ned.com/blog/2024/cbf/
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
    `class kappa <https://en.wikipedia.org/wiki/Class_kappa_function>`__
    functions.

    On top of that, following `this article
    <https://arxiv.org/pdf/2404.12329>`__ barriers utilize safe displacement
    term is added to the cost of the optimization problem:

    .. math::

        \frac{r}{2\|J_h\|^{2}}\|dq-dq_{\text{safe}}(q)\|^{2},

    where :math:`J_h` is the Jacobian of the barrier function, dq is the
    joint displacement vector, and :math:`dq_{\text{safe}}(q)` is the safe
    displacement vector.

    Attributes:
        dim: Dimension of the barrier.
        gain: linear barrier gain.
        gain_function: function, that defines stabilization term as nonlinear
            function of barrier. Defaults to the (linear) identity function.
        safe_displacement: Safe backup displacement.
        safe_displacement_gain: positive gain for safe backup displacement.
    """

    gain: np.ndarray
    gain_function: Callable[[float], float]
    safe_displacement: np.ndarray
    safe_displacement_gain: float

    def __init__(
        self,
        dim: int,
        gain: Union[float, np.ndarray] = 1.0,
        gain_function: Optional[Callable[[float], float]] = None,
        safe_displacement_gain: float = 0.0,
    ):
        """Initialize the barrier.

        Args:
            dim: Dimension of the barrier.
            gain: barrier gain. Defaults to 1.0.
            gain_function: gain function.
                Defaults to the identity function.
            safe_displacement_gain: gain for the safe backup displacement
                cost term. Defaults to 3.0.
        """
        self.dim = dim
        self.gain = (
            gain if isinstance(gain, np.ndarray) else np.ones(dim) * gain
        )

        self.gain_function = (
            gain_function if gain_function is not None else lambda x: x
        )
        self.safe_displacement = np.zeros(self.dim)
        self.safe_displacement_gain = safe_displacement_gain

        # Cached values to avoid recomputation
        self.__q_cache: Optional[np.ndarray] = None
        self.__jac_cache: Optional[np.ndarray] = None
        self.__barrier_cache: Optional[np.ndarray] = None

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

    def compute_safe_displacement(
        self, configuration: Configuration
    ) -> np.ndarray:
        r"""Compute the safe backup displacement.

        The safe backup control displacement :math:`dq_{safe}(q)`
        is a joint displacement vector that can guarantee that system
        would stay in safety set.

        By default, it is set to zero, since it could not violate safety set.

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

        where :math:`\gamma(q)` is a configuration-dependent weight and
        :math:`\dot{q}_{safe}(q)` is the safe backup policy.

        Note:
            If `safe_displacement_gain` is set to zero, the regularization
            term is not included. Jacobian and barrier values are cached
            to avoid recomputation.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Time step for discrete-time implementation. Defaults to 1e-3.

        Returns:
            Tuple containing the quadratic objective matrix (H) and linear
                objective vector (c).
        """
        if (
            self.__q_cache is None
            or self.__jac_cache is None
            or self.__barrier_cache is None
        ) or not np.allclose(self.__q_cache, configuration.q):
            self.__q_cache = configuration.q
            self.__jac_cache = self.compute_jacobian(configuration)
            self.__barrier_cache = self.compute_barrier(configuration)

        H = np.zeros((configuration.model.nv, configuration.model.nv))
        c = np.zeros(configuration.model.nv)

        if self.safe_displacement_gain > 1e-6:
            self.safe_displacement = self.compute_safe_displacement(
                configuration
            )
            jac_squared_norm = np.linalg.norm(self.__jac_cache) ** 2
            gain_over_jacobian = self.safe_displacement_gain / jac_squared_norm

            H += gain_over_jacobian * np.eye(configuration.model.nv)
            c += -gain_over_jacobian * self.safe_displacement

        return (H, c)

    def compute_qp_inequalities(
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

        Note:
            Jacobian and barrier values are cached to avoid recomputation.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Time step for discrete-time implementation. Defaults to 1e-3.

        Returns:
            Tuple containing the inequality constraint matrix (G)
                and vector (h).
        """
        if (
            self.__q_cache is None
            or self.__jac_cache is None
            or self.__barrier_cache is None
            or not np.allclose(self.__q_cache, configuration.q)
        ):
            self.__q_cache = configuration.q
            self.__jac_cache = self.compute_jacobian(configuration)
            self.__barrier_cache = self.compute_barrier(configuration)

        G = -self.__jac_cache / dt
        h = np.array(
            [
                self.gain[i] * self.gain_function(self.__barrier_cache[i])
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
            f"safe_displacement={'no' if np.allclose(self.safe_displacement_gain, 0) else self.safe_displacement}, "  # noqa: E501
            f"safe_displacement_gain={'no' if np.allclose(self.safe_displacement_gain, 0) else self.safe_displacement_gain})"  # noqa: E501
            f"dim={self.dim})"
        )
