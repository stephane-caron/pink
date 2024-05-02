#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""General description"""

import abc
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from ..configuration import Configuration


class CBF(abc.ABC):
    r"""Abstract class description.

    Attributes:
        ...
    """

    gain: Union[float, np.ndarray]
    class_k_fns: Optional[Callable[[np.ndarray], np.ndarray]]
    safe_policy: Optional[np.ndarray]
    r: float

    def __init__(
        self,
        gain: float = 1.0,
        class_k_fn: Optional[Callable[..., float]] = None,
        safe_control: Optional[np.ndarray] = None,
        r: float = 1.0,
    ):
        """..."""
        self.gain = gain
        self.class_k_fn = class_k_fn if class_k_fn is not None else class_k_fn
        self.safe_policy = safe_control
        self.r = r

    @abc.abstractmethod
    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        """..."""
        pass

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""..."""
        pass

    def compute_qp_objective(self, configuration: Configuration) -> Tuple[np.ndarray, np.ndarray]:
        r"""..."""
        if self.safe_policy is None:
            return (
                np.zeros((configuration.model.nq, configuration.model.nq)),
                np.zeros(configuration.model.nq),
            )

        jac = self.compute_jacobian(configuration)
        H = self.r / np.linalg.norm(jac) ** 2 * np.eye(configuration.model.nq)
        c = 2 * self.r / np.linalg.norm(jac) ** 2 * self.safe_policy
        return (H, c)

    def compute_qp_inequality(self, configuration: Configuration) -> Tuple[np.ndarray, np.ndarray]:
        """..."""
        G = -self.compute_jacobian(configuration)
        barrier_value = self.compute_barrier(configuration)
        h = self.gain * self.class_k_fns(barrier_value)

        return (G, h)

    def __repr__(self):
        """..."""
        return (
            f"Barrier("
            f"gain={self.gain}, "
            f"safety_policy={'no' if self.safe_policy is None else self.safe_policy}, "  # noqa: E501
            f"r={'no' if self.safe_policy is None else self.r})"
        )
