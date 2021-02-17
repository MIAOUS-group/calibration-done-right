use crate::{
    CacheStatus, ChannelFatalError, ChannelHandle, CoreSpec, MultipleAddrCacheSideChannel,
    SideChannelError, SingleAddrCacheSideChannel,
};

use std::collections::HashMap;
use std::fmt::Debug;

pub struct TableAttackResult {
    pub addr: *const u8,
    hit: u32,
    miss: u32,
}

impl TableAttackResult {
    pub fn get(&self, cache_status: CacheStatus) -> u32 {
        match cache_status {
            CacheStatus::Hit => self.hit,
            CacheStatus::Miss => self.miss,
        }
    }
}

pub trait TableCacheSideChannel<Handle: ChannelHandle>: CoreSpec + Debug {
    //type ChannelFatalError: Debug;
    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Handle>, ChannelFatalError>;
    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn attack<'a, 'b, 'c, 'd>(
        &'a mut self,
        addresses: &'b mut Vec<&'c mut Handle>,
        victim: &'d dyn Fn(),
        num_iteration: u32,
    ) -> Result<Vec<TableAttackResult>, ChannelFatalError>
    where
        Handle: 'c;
}

impl<T: SingleAddrCacheSideChannel> TableCacheSideChannel<T::Handle> for T {
    default unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<T::Handle>, ChannelFatalError> {
        unsafe { self.calibrate_single(addresses) }
    }
    //type ChannelFatalError = T::SingleChannelFatalError;

    default unsafe fn attack<'a, 'b, 'c, 'd>(
        &'a mut self,
        addresses: &'b mut Vec<&'c mut T::Handle>,
        victim: &'d dyn Fn(),
        num_iteration: u32,
    ) -> Result<Vec<TableAttackResult>, ChannelFatalError>
    where
        T::Handle: 'c,
    {
        let mut result = Vec::new();

        for addr in addresses {
            let mut hit = 0;
            let mut miss = 0;
            for iteration in 0..100 {
                match unsafe { self.prepare_single(addr) } {
                    Ok(_) => {}
                    Err(e) => match e {
                        SideChannelError::NeedRecalibration => unimplemented!(),
                        SideChannelError::FatalError(e) => return Err(e),
                        SideChannelError::AddressNotReady(_addr) => panic!(),
                        SideChannelError::AddressNotCalibrated(_addr) => unimplemented!(),
                    },
                }
                self.victim_single(victim);
                let r = unsafe { self.test_single(addr) };
                match r {
                    Ok(status) => {}
                    Err(e) => match e {
                        SideChannelError::NeedRecalibration => panic!(),
                        SideChannelError::FatalError(e) => {
                            return Err(e);
                        }
                        _ => panic!(),
                    },
                }
            }
            for _iteration in 0..num_iteration {
                match unsafe { self.prepare_single(addr) } {
                    Ok(_) => {}
                    Err(e) => match e {
                        SideChannelError::NeedRecalibration => unimplemented!(),
                        SideChannelError::FatalError(e) => return Err(e),
                        SideChannelError::AddressNotReady(_addr) => panic!(),
                        SideChannelError::AddressNotCalibrated(_addr) => unimplemented!(),
                    },
                }
                self.victim_single(victim);
                let r = unsafe { self.test_single(addr) };
                match r {
                    Ok(status) => match status {
                        CacheStatus::Hit => {
                            hit += 1;
                        }
                        CacheStatus::Miss => {
                            miss += 1;
                        }
                    },
                    Err(e) => match e {
                        SideChannelError::NeedRecalibration => panic!(),
                        SideChannelError::FatalError(e) => {
                            return Err(e);
                        }
                        _ => panic!(),
                    },
                }
            }
            result.push(TableAttackResult {
                addr: addr.to_const_u8_pointer(),
                hit,
                miss,
            });
        }
        Ok(result)
    }
}

// TODO limit number of simultaneous tested address + randomise order ?

impl<T: MultipleAddrCacheSideChannel> TableCacheSideChannel<T::Handle> for T {
    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<T::Handle>, ChannelFatalError> {
        unsafe { self.calibrate(addresses) }
    }
    //type ChannelFatalError = T::MultipleChannelFatalError;

    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn attack<'a, 'b, 'c, 'd>(
        &'a mut self,
        mut addresses: &'b mut Vec<&'c mut T::Handle>,
        victim: &'d dyn Fn(),
        num_iteration: u32,
    ) -> Result<Vec<TableAttackResult>, ChannelFatalError>
    where
        T::Handle: 'c,
    {
        let mut addr_iter = addresses.iter_mut();
        let mut v = Vec::new();
        while let Some(addr) = addr_iter.next() {
            let mut batch = Vec::new();
            batch.push(&mut **addr);
            let mut hits: HashMap<*const u8, u32> = HashMap::new();
            let mut misses: HashMap<*const u8, u32> = HashMap::new();
            for i in 1..T::MAX_ADDR {
                if let Some(addr) = addr_iter.next() {
                    batch.push(&mut **addr);
                } else {
                    break;
                }
            }
            for i in 0..100 {
                // TODO Warmup
            }
            for i in 0..num_iteration {
                match unsafe { MultipleAddrCacheSideChannel::prepare(self, &mut batch) } {
                    Ok(_) => {}
                    Err(e) => match e {
                        SideChannelError::NeedRecalibration => unimplemented!(),
                        SideChannelError::FatalError(e) => return Err(e),
                        SideChannelError::AddressNotReady(_addr) => panic!(),
                        SideChannelError::AddressNotCalibrated(addr) => {
                            eprintln!(
                                "Addr: {:p}\n\
                            {:#?}",
                                addr, self
                            );
                            unimplemented!()
                        }
                    },
                }
                MultipleAddrCacheSideChannel::victim(self, victim);

                let r = unsafe { MultipleAddrCacheSideChannel::test(self, &mut batch) }; // Fixme error handling
                match r {
                    Err(e) => match e {
                        SideChannelError::NeedRecalibration => {
                            panic!();
                        }
                        SideChannelError::FatalError(e) => {
                            return Err(e);
                        }
                        _ => {
                            panic!();
                        }
                    },
                    Ok(vector) => {
                        for (addr, status) in vector {
                            match status {
                                CacheStatus::Hit => {
                                    *hits.entry(addr).or_default() += 1;
                                }
                                CacheStatus::Miss => {
                                    *misses.entry(addr).or_default() += 1;
                                }
                            }
                        }
                    }
                }
            }

            for addr in batch {
                v.push(TableAttackResult {
                    addr: addr.to_const_u8_pointer(),
                    hit: *hits.get(&addr.to_const_u8_pointer()).unwrap_or(&0u32),
                    miss: *misses.get(&addr.to_const_u8_pointer()).unwrap_or(&0u32),
                })
            }
        }
        Ok(v)
    }
}
