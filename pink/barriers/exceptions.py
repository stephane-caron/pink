#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

from ..exceptions import PinkError


class NoPositionLimitProvided(PinkError):
    """If neither minimum nor maximum position limits are provided."""


class NegativeMinimumDistance(PinkError):
    """If the minimum distance in body spherical barrier is negative."""
