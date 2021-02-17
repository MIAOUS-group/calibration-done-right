#![cfg_attr(feature = "no_std", no_std)]
#![feature(ptr_internals)]
#![allow(clippy::missing_safety_doc)]

use static_assertions::assert_cfg;

assert_cfg!(
    all(
        not(all(feature = "use_std", feature = "no_std")),
        any(feature = "use_std", feature = "no_std")
    ),
    "Choose std or no-std but not both"
);

pub mod cache_info;
pub mod calibration;
pub mod complex_addressing;
#[cfg(feature = "use_std")]
pub mod mmap;
pub mod prefetcher;

pub mod frequency;

#[cfg(feature = "use_std")]
mod calibrate_2t;

use core::arch::x86_64 as arch_x86;
use core::ptr;

// rdtsc no fence
pub unsafe fn rdtsc_nofence() -> u64 {
    arch_x86::_rdtsc()
}
// rdtsc (has mfence before and after)
pub unsafe fn rdtsc_fence() -> u64 {
    arch_x86::_mm_mfence();
    let tsc: u64 = arch_x86::_rdtsc();
    arch_x86::_mm_mfence();
    tsc
}

pub unsafe fn maccess<T>(p: *const T) {
    ptr::read_volatile(p);
}

// flush (cflush)
pub unsafe fn flush(p: *const u8) {
    arch_x86::_mm_clflush(p);
}

pub fn noop<T>(_: *const T) {}

#[cfg(feature = "use_std")]
pub fn find_core_per_socket() -> u8 {
    use std::process::Command;
    use std::str::from_utf8;

    let core_per_socket_out = Command::new("sh")
        .arg("-c")
        .arg("lscpu | grep socket | cut -b 22-")
        .output()
        .expect("Failed to detect cpu count");
    //println!("{:#?}", core_per_socket_str);

    let core_per_socket_str = from_utf8(&core_per_socket_out.stdout).unwrap();

    //println!("Number of cores per socket: {}", cps_str);

    let core_per_socket: u8 = core_per_socket_str[0..(core_per_socket_str.len() - 1)]
        .parse()
        .unwrap_or(0);
    core_per_socket
}

// future enhancements
// prefetch
// long nop (64 nops)
