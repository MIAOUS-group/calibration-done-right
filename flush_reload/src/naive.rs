use basic_timing_cache_channel::naive::NaiveTimingChannel;
use basic_timing_cache_channel::TimingChannelPrimitives;

use cache_utils::calibration::only_reload;

#[derive(Debug)]
pub struct NaiveFRPrimitives {}

impl TimingChannelPrimitives for NaiveFRPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { only_reload(addr) }
    }
}

pub type NaiveFlushAndReload = NaiveTimingChannel<NaiveFRPrimitives>;
