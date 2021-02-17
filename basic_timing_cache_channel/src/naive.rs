use crate::TimingChannelPrimitives;
use cache_side_channel::{
    CacheStatus, ChannelFatalError, ChannelHandle, CoreSpec, SideChannelError,
    SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{get_vpn, only_flush, only_reload, HashMap, Threshold, VPN};
use cache_utils::flush;
use covert_channels_evaluation::{BitIterator, CovertChannel};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;
use std::fmt::Debug;

// Parameters required : The threshold.
#[derive(Debug)]
pub struct NaiveTimingChannel<T: TimingChannelPrimitives> {
    threshold: Threshold,
    current: HashMap<VPN, *const u8>,
    main_core: CpuSet,
    helper_core: CpuSet,
    channel_primitive: T,
}

pub struct NaiveTimingChannelHandle {
    vpn: VPN,
    addr: *const u8,
}

impl ChannelHandle for NaiveTimingChannelHandle {
    fn to_const_u8_pointer(&self) -> *const u8 {
        self.addr
    }
}

unsafe impl<T: TimingChannelPrimitives + Send> Send for NaiveTimingChannel<T> {}
unsafe impl<T: TimingChannelPrimitives + Sync> Sync for NaiveTimingChannel<T> {}

impl<T: TimingChannelPrimitives> NaiveTimingChannel<T> {
    pub fn new(threshold: Threshold, t: T) -> Self {
        Self {
            threshold,
            current: Default::default(),
            main_core: sched_getaffinity(Pid::from_raw(0)).unwrap(),
            helper_core: sched_getaffinity(Pid::from_raw(0)).unwrap(),
            channel_primitive: t,
        }
    }

    pub fn set_cores(&mut self, main_core: usize, helper_core: usize) {
        self.main_core = CpuSet::new();
        self.main_core.set(main_core).unwrap();

        self.helper_core = CpuSet::new();
        self.helper_core.set(helper_core).unwrap();
    }

    pub fn unready_page(
        &mut self,
        handle: NaiveTimingChannelHandle,
    ) -> Result<*const u8, ChannelFatalError> {
        if let Some(addr) = self.current.remove(&handle.vpn) {
            Ok(addr)
        } else {
            Err(ChannelFatalError::Oops)
        }
    }

    unsafe fn test_impl(
        &self,
        handle: &mut NaiveTimingChannelHandle,
    ) -> Result<CacheStatus, SideChannelError> {
        // This should be handled in prepare / unprepare
        let t = unsafe { self.channel_primitive.attack(handle.addr) };
        if self.threshold.is_hit(t) {
            Ok(CacheStatus::Hit)
        } else {
            Ok(CacheStatus::Miss)
        }
    }

    unsafe fn calibrate_impl(
        &mut self,
        addr: *const u8,
    ) -> Result<NaiveTimingChannelHandle, ChannelFatalError> {
        let vpn = get_vpn(addr);
        if self.current.get(&vpn).is_some() {
            return Err(ChannelFatalError::Oops);
        } else {
            self.current.insert(vpn, addr);
            Ok(NaiveTimingChannelHandle { vpn, addr })
        }
    }
}

impl<T: TimingChannelPrimitives> CoreSpec for NaiveTimingChannel<T> {
    fn main_core(&self) -> CpuSet {
        self.main_core
    }

    fn helper_core(&self) -> CpuSet {
        self.helper_core
    }
}

impl<T: TimingChannelPrimitives + Send + Sync> CovertChannel for NaiveTimingChannel<T> {
    type Handle = NaiveTimingChannelHandle;
    const BIT_PER_PAGE: usize = 1;

    unsafe fn transmit<'a>(&self, handle: &mut Self::Handle, bits: &mut BitIterator<'a>) {
        if let Some(b) = bits.next() {
            if b {
                unsafe { only_reload(handle.addr) };
            } else {
                unsafe { only_flush(handle.addr) };
            }
        }
    }

    unsafe fn receive(&self, handle: &mut Self::Handle) -> Vec<bool> {
        let r = unsafe { self.test_impl(handle) };
        match r {
            Err(e) => panic!(),
            Ok(status) => match status {
                CacheStatus::Hit => vec![true],
                CacheStatus::Miss => vec![false],
            },
        }
    }

    unsafe fn ready_page(&mut self, page: *const u8) -> Result<Self::Handle, ()> {
        unsafe { self.calibrate_impl(page) }.map_err(|_| ())
    }
}
impl<T: TimingChannelPrimitives> SingleAddrCacheSideChannel for NaiveTimingChannel<T> {
    type Handle = NaiveTimingChannelHandle;

    unsafe fn test_single(
        &mut self,
        handle: &mut Self::Handle,
    ) -> Result<CacheStatus, SideChannelError> {
        unsafe { self.test_impl(handle) }
    }

    unsafe fn prepare_single(&mut self, handle: &mut Self::Handle) -> Result<(), SideChannelError> {
        unsafe { flush(handle.addr) };
        Ok(())
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        operation()
    }

    /// # Safety
    ///
    /// addr needs to be a valid pointer
    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        let mut result = vec![];
        for addr in addresses {
            match unsafe { self.calibrate_impl(addr) } {
                Ok(handle) => result.push(handle),
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(result)
    }
}

// Include a helper code to get global threshold model ?

// TODO
