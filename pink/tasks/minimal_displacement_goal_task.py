#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025

"""Minimal displacement goal task implementation."""

from typing import Optional, Union, Sequence

import numpy as np

from ..configuration import Configuration
from .task import Task


class MinimalDisplacementGoalTask(Task):
    """Minimize displacement from initial configuration guess.

    This task minimizes the weighted sum of squared displacements of all
    configuration variables from their initial guess values. It's typically
    used as a secondary objective to prefer solutions that are closer to
    the initial configuration when multiple solutions exist.

    The cost function is:
    .. math::
        \\sum_{i} w_i (q_i - q_{i,initial})^2

    where q_i are the configuration variables, q_{i,initial} are the initial
    guess values, and w_i are the variable weights.

    Attributes:
        initial_configuration: Reference configuration to minimize displacement from.
        variable_weights: Weights for each configuration variable.
        is_secondary: Whether this is a secondary objective (lower priority).
    """

    initial_configuration: Optional[Configuration]
    variable_weights: Optional[np.ndarray]
    is_secondary: bool

    def __init__(
        self,
        cost: float = 1.0,
        is_secondary: bool = True,
        lm_damping: float = 0.0,
        gain: float = 1.0,
        secondary_scale: float = 0.1, 
    ) -> None:
        """Define a new minimal displacement goal task.

        Args:
            cost: Overall cost weight for this task.
            is_secondary: Whether this is a secondary objective. Secondary
                objectives have lower priority and are typically used for
                regularization or optimization preferences rather than
                hard constraints.
            lm_damping: Levenberg-Marquardt damping (see class attributes). The
                default value is conservatively low.
            gain: Task gain α ∈ [0, 1] for additional low-pass filtering.
                Defaults to 1.0 (no filtering) for dead-beat control.
        """
        # Apply reduced cost if this is a secondary objective
        effective_cost = cost * (0.1 if is_secondary else 1.0)
        
        super().__init__(
            cost=effective_cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        self.variable_weights = None
        self.is_secondary = is_secondary
        self.base_cost = float(cost)
        self.secondary_scale = float(secondary_scale)
        self.cost = self.base_cost * (self.secondary_scale if self.is_secondary else 1.0)

        self.q0: Optional[np.ndarray] = None
        self.variable_weights: Optional[np.ndarray] = None


    def set_initial_configuration(self, configuration: Configuration) -> None:
        q = np.asarray(configuration.q, dtype=float).reshape(-1)
        self.q0 = q.copy()
        if self.variable_weights is None:
            self.variable_weights = np.ones_like(self.q0)

    def set_variable_weights(self, weights: Union[np.ndarray, Sequence[float]]) -> None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if np.any(w < 0):
            raise ValueError("variable_weights must be non-negative")
        self.variable_weights = w


    def get_initial_configuration(self) -> np.ndarray:
        if self.q0 is None:
            raise ValueError("No initial configuration set")
        return self.q0

    def _check_dims(self, n_vars: int) -> None:
        if self.q0 is None:
            raise ValueError("No initial configuration set")
        if self.q0.shape[0] != n_vars:
            raise ValueError(f"Current q dim {n_vars} != initial q0 dim {self.q0.shape[0]}")
        if self.variable_weights is not None and self.variable_weights.shape[0] != n_vars:
            raise ValueError(f"Weight dim {self.variable_weights.shape[0]} != q dim {n_vars}")

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        q = np.asarray(configuration.q, dtype=float).reshape(-1)
        n = q.shape[0]
        self._check_dims(n)

        # Compute displacement: d = q - q₀
        disp = q - self.q0
        
        # Apply variable weights: weighted_disp = w ⊙ (q - q₀)
        if self.variable_weights is not None:
            weighted_disp = self.variable_weights * disp
        else:
            weighted_disp = disp
        
        # Apply task cost scaling: error = √c × weighted_displacement
        # This ensures ||error||² = c × Σᵢ [wᵢ(qᵢ - q₀ᵢ)]²
        sqrt_cost = np.sqrt(self.cost)
        return sqrt_cost * weighted_disp


    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        q = np.asarray(configuration.q, dtype=float).reshape(-1)
        n = q.shape[0]
        self._check_dims(n)

        if self.variable_weights is None:
            J = np.eye(n)
        else:
            J = np.diag(self.variable_weights)

        sqrt_cost = np.sqrt(self.cost)
        return sqrt_cost * J

    def set_secondary(self, is_secondary: bool) -> None:
        self.is_secondary = bool(is_secondary)
        self.cost = self.base_cost * (self.secondary_scale if self.is_secondary else 1.0)


    def is_secondary_objective(self) -> bool:
        """Check if this is a secondary objective.

        Returns:
            True if this is a secondary objective, False otherwise.
        """
        return self.is_secondary

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "MinimalDisplacementGoalTask("
            f"base_cost={self.base_cost}, cost={self.cost}, "
            f"is_secondary={self.is_secondary}, "
            f"has_q0={self.q0 is not None}, "
            f"has_variable_weights={self.variable_weights is not None}, "
            f"lm_damping={self.lm_damping}, "
            f"gain={self.gain})"
        )
