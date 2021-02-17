#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod naive;

use basic_timing_cache_channel::{
    SingleChannel, TimingChannelPrimitives, TopologyAwareTimingChannel,
};

use cache_utils::calibration::only_flush;

#[derive(Debug)]
pub struct FFPrimitives {}

impl TimingChannelPrimitives for FFPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { only_flush(addr) }
    }
}

pub type FlushAndFlush = TopologyAwareTimingChannel<FFPrimitives>;

pub type SingleFlushAndFlush = SingleChannel<FlushAndFlush>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
