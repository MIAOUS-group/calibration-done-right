use x86_64::registers::model_specific::Msr;

use crate::calibration::only_flush;

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

const MSR_MISC_FEATURE8CONTROL: u32 = 0x1a4;

const N: i32 = 10;

pub fn prefetcher_status() -> bool {
    let msr = Msr::new(MSR_MISC_FEATURE8CONTROL);
    let value = unsafe { msr.read() };

    value & 0xf != 0xf
}

pub fn enable_prefetchers(status: bool) {
    let mut msr = Msr::new(MSR_MISC_FEATURE8CONTROL);
    let mut value = unsafe { msr.read() } & !0xf;
    if !status {
        value |= 0xf;
    }
    unsafe { msr.write(value) };
}

pub unsafe fn prefetcher_fun(
    victim_4k_addr: *mut u8,
    #[allow(non_snake_case)] _victim_2M_addr: *mut u8,
    threshold_ff: u64,
) -> Vec<i32> {
    fn implementation(
        victim_4k_addr: *mut u8,
        #[allow(non_snake_case)] _victim_2M_addr: *mut u8,
        threshold_ff: u64,
    ) -> Vec<i32> {
        let mut results = vec![0; 4096 / 64];

        if false {
            for _ in 0..N {
                //unsafe { maccess(victim4kaddr) };
                for j in (0..4096).step_by(64).rev() {
                    let t = unsafe { only_flush(victim_4k_addr.offset(j)) };
                    if threshold_ff < t {
                        // hit
                        results[(j / 64) as usize] += 1;
                    } else if threshold_ff > t {
                        results[(j / 64) as usize] -= 1;
                    }
                }
            }
        }
        results
    }
    implementation(victim_4k_addr, _victim_2M_addr, threshold_ff)
}
