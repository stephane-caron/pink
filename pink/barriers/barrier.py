#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""General description"""

import abc
from typing import Callable, Optional, Tuple, Union

import numpy as np

from ..configuration import Configuration


class CBF(abc.ABC):
    r"""Abstract class description.

    Attributes:
        ...
    """

    gain: Union[float, np.ndarray]
    class_k_fns: Callable[[np.ndarray], float]
    safe_policy: Optional[np.ndarray]
    r: float

    def __init__(
        self,
        dim: int,
        gain: Union[float, np.ndarray] = 1.0,
        class_k_fn: Optional[Callable[..., float]] = None,
        r: float = 3.0,
    ):
        """..."""
        self.dim = dim
        self.gain = (
            gain if isinstance(gain, np.ndarray) else np.ones(dim) * gain
        )
        self.class_k_fn = class_k_fn if class_k_fn is not None else lambda x: x
        self.r = r

    @abc.abstractmethod
    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        """..."""
        pass

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""..."""
        pass

    def compute_safe_policy(self, configuration: Configuration) -> np.ndarray:
        r"""..."""
        return np.zeros(configuration.model.nq)

    def compute_qp_objective(
        self,
        configuration: Configuration,
        dt: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""..."""
        jac = self.compute_jacobian(configuration)
        H = np.zeros((configuration.model.nq, configuration.model.nq))
        # c = 1e-3 * np.linalg.norm(jac, axis=0) / dt
        c = np.zeros(configuration.model.nq)

        if self.r > 1e-6:
            safe_policy = self.compute_safe_policy(configuration)

            H += (
                self.r
                / (np.linalg.norm(jac) ** 2)
                * np.eye(configuration.model.nq)
            )
            c += -2 * self.r / (np.linalg.norm(jac) ** 2) * safe_policy

        return (H, c)

    def compute_qp_inequality(
        self,
        configuration: Configuration,
        dt: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """..."""
        G = -self.compute_jacobian(configuration) / dt
        barrier_value = self.compute_barrier(configuration)
        h = np.array(
            [
                self.gain[i] * self.class_k_fn(barrier_value[i])
                for i in range(self.dim)
            ]
        )

        return (G, h)

    def __repr__(self):
        """..."""
        return (
            f"Barrier("
            f"gain={self.gain}, "
            f"safety_policy={'no' if self.safe_policy is None else self.safe_policy}, "  # noqa: E501
            f"r={'no' if self.safe_policy is None else self.r})"
        )
