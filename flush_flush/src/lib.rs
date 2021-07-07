#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

// SPDX-FileCopyrightText: 2021 Guillaume DIDIER
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: MIT

pub mod naive;

use basic_timing_cache_channel::{
    SingleChannel, TimingChannelPrimitives, TopologyAwareTimingChannel,
};

use cache_side_channel::MultipleAddrCacheSideChannel;
use cache_utils::calibration::only_flush;

#[derive(Debug)]
pub struct FFPrimitives {}

impl TimingChannelPrimitives for FFPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { only_flush(addr) }
    }
}

pub type FlushAndFlush = TopologyAwareTimingChannel<FFPrimitives>;

pub type FFHandle = <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle;

pub type SingleFlushAndFlush = SingleChannel<FlushAndFlush>;
