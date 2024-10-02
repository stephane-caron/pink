#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Rolling task implementation."""

from pink import Task


class RollingTask(Task):
    hub_frame: str
    floor_frame: str
    wheel_radius: float

    def __init__(
        self,
        hub_frame: str,
        floor_frame: str,
        wheel_radius: float,
        cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(
            cost=cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        self.floor_frame = floor_frame
        self.hub_frame = hub_frame
        self.wheel_radius = wheel_radius
