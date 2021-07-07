// SPDX-FileCopyrightText: 2021 Guillaume DIDIER
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: MIT

use crate::FFPrimitives;
use basic_timing_cache_channel::naive::NaiveTimingChannel;

pub type NaiveFlushAndFlush = NaiveTimingChannel<FFPrimitives>;
