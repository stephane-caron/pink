#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron, Ivan Domrachev, Simeon Nedelchev

from ..exceptions import PinkError


class NoPositionLimitProvided(PinkError):
    """If neither minimum nor maximum position limits are provided."""
