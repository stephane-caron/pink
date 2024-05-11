#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""All barriers derive from the :class:`CBF` base class.

The formalism used in this implementation is written down in examples/cbf/NOTES.md
"""

import abc
from typing import Callable, Optional, Tuple, Union

import numpy as np

from ..configuration import Configuration


class Barrier(abc.ABC):
    r"""Abstract base class for Control Barrier Functions (CBFs).

    A CBF is a function :math:`\boldsymbol{h}(\boldsymbol{q})` that satisfies the following condition:

    .. math::

        \frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{q}} \dot{\boldsymbol{q}} +\alpha_j(\boldsymbol{h}_j(\boldsymbol{q})) \geq 0, \quad \forall j

    where :math:`\frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{q}}` are the Jacobians of the constraint functions,
    :math:`\dot{\boldsymbol{q}}` is the joint velocity vector, and :math:`\alpha_j` are extended class K functions.

    Attributes:
        dim (int): Dimension of the CBF.
        gain (Union[float, np.ndarray]): CBF gain.
        class_k_fn (Callable[[np.ndarray], float]): Extended class K function.
        safe_policy (Optional[np.ndarray]): Safe backup control policy.
        r (float): Weighting factor for the safe backup policy regularization term.
    """

    gain: Union[float, np.ndarray]
    class_k_fns: Callable[[np.ndarray], float]
    safe_policy: Optional[np.ndarray]
    r: float

    def __init__(
        self,
        dim: int,
        gain: Union[float, np.ndarray] = 1.0,
        class_k_fn: Optional[Callable[[np.ndarray], float]] = None,
        r: float = 3.0,
    ):
        """Initialize the CBF.

        Args:
            dim (int): Dimension of the CBF.
            gain (Union[float, np.ndarray]): CBF gain. Defaults to 1.0.
            class_k_fn (Optional[Callable[[np.ndarray], float]]): Extended class K function.
                Defaults to the identity function.
            r (float): Weighting factor for the safe backup policy regularization term.
                Defaults to 3.0.
        """

        self.dim = dim
        self.gain = gain if isinstance(gain, np.ndarray) else np.ones(dim) * gain
        self.class_k_fn = class_k_fn if class_k_fn is not None else lambda x: x
        self.r = r

    @abc.abstractmethod
    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the value of the barrier function.

        The barrier function :math:`\boldsymbol{h}(\boldsymbol{q})` is a vector-valued function
        that represents the safety constraints. It should be designed such that
        the set :math:`\{\boldsymbol{q} : \boldsymbol{h}(\boldsymbol{q}) \geq \boldsymbol{0}\}` represents the safe region of the
        configuration space.

        Args:
            configuration: Robot configuration :math:`\boldsymbol{q}`.

        Returns:
            Value of the barrier function :math:`\boldsymbol{h}(\boldsymbol{q})`.
        """
        pass

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the Jacobian matrix of the barrier function.

        The Jacobian matrix :math:`\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{q}}(\boldsymbol{q})` of the
        barrier function with respect to the configuration variables is required
        for the computation of the CBF condition.

        Args:
            configuration: Robot configuration :math:`\boldsymbol{q}`.

        Returns:
            Jacobian matrix :math:`\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{q}}(\boldsymbol{q})`.
        """
        pass

    def compute_safe_policy(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the safe backup control policy.

        The safe backup control policy :math:`\dot{\boldsymbol{q}}_{safe}(\boldsymbol{q})` is a joint
        velocity vector that can be used as a regularization term in the
        optimization problem to ensure safety.

        Args:
            configuration: Robot configuration :math:`\boldsymbol{q}`.

        Returns:
            Safe backup joint velocities :math:`\dot{\boldsymbol{q}}_{safe}(\boldsymbol{q})`.
        """
        return np.zeros(configuration.model.nq)

    def compute_qp_objective(
        self,
        configuration: Configuration,
        dt: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the quadratic objective function for the CBF-based QP.

        The quadratic objective function includes a regularization term based on the safe backup policy:

        .. math::

            \gamma(\boldsymbol{q})\left\| \dot{\boldsymbol{q}}-\dot{\boldsymbol{q}}_{safe}(\boldsymbol{q})\right\|^{2}

        where :math:`\gamma(\boldsymbol{q})` is a configuration-dependent weight and
        :math:`\dot{\boldsymbol{q}}_{safe}(\boldsymbol{q})` is the safe backup policy.

        Args:
            configuration: Robot configuration :math:`\boldsymbol{q}`.
            dt: Time step for discrete-time implementation. Defaults to 1e-3.

        Returns:
            Tuple containing the quadratic objective matrix (H) and linear
            objective vector (c).
        """
        jac = self.compute_jacobian(configuration)
        H = np.zeros((configuration.model.nq, configuration.model.nq))
        # c = 1e-3 * np.linalg.norm(jac, axis=0) / dt
        c = np.zeros(configuration.model.nq)

        if self.r > 1e-6:
            safe_policy = self.compute_safe_policy(configuration)

            H += self.r / (np.linalg.norm(jac) ** 2) * np.eye(configuration.model.nq)
            c += -2 * self.r / (np.linalg.norm(jac) ** 2) * safe_policy

        return (H, c)

    def compute_qp_inequality(
        self,
        configuration: Configuration,
        dt: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the linear inequality constraints for the CBF-based QP.

        The linear inequality constraints enforce the CBF conditions:

        .. math::

            \frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{q}} \dot{\boldsymbol{q}} +\alpha_j(\boldsymbol{h}_j(\boldsymbol{q})) \geq 0, \quad \forall j

        where :math:`\frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{q}}` are the Jacobians of the constraint functions,
        :math:`\dot{\boldsymbol{q}}` is the joint velocity vector, and :math:`\alpha_j` are extended class K functions.

        Args:
            configuration: Robot configuration :math:`\boldsymbol{q}`.
            dt: Time step for discrete-time implementation. Defaults to 1e-3.

        Returns:
            Tuple containing the inequality constraint matrix (G) and vector (h).
        """
        G = -self.compute_jacobian(configuration) / dt
        barrier_value = self.compute_barrier(configuration)
        h = np.array([self.gain[i] * self.class_k_fn(barrier_value[i]) for i in range(self.dim)])

        return (G, h)

    def __repr__(self) -> str:
        """
        Return a string representation of the CBF.

        Returns:
            str: String representation.
        """
        return (
            f"Barrier("
            f"gain={self.gain}, "
            f"safety_policy={'no' if self.safe_policy is None else self.safe_policy}, "  # noqa: E501
            f"r={'no' if self.safe_policy is None else self.r})"
        )
