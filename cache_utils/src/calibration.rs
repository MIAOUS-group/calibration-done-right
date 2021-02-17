#![allow(clippy::missing_safety_doc)]

use crate::complex_addressing::{cache_slicing, CacheSlicing};
use crate::{flush, maccess, rdtsc_fence};

use cpuid::MicroArchitecture;

use core::arch::x86_64 as arch_x86;
#[cfg(feature = "no_std")]
use polling_serial::{serial_print as print, serial_println as println};

#[cfg(feature = "use_std")]
use nix::unistd::Pid;
//#[cfg(feature = "use_std")]
//use nix::Error::Sys;
#[cfg(feature = "use_std")]
use nix::Error;
#[cfg(feature = "use_std")]
use std::sync::Arc;
#[cfg(feature = "use_std")]
use std::thread;

#[cfg(feature = "use_std")]
pub use crate::calibrate_2t::*;

extern crate alloc;
use crate::calibration::Verbosity::*;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;
use core::ptr::null_mut;
use core::sync::atomic::{spin_loop_hint, AtomicBool, AtomicPtr, Ordering};
use itertools::Itertools;

use atomic::Atomic;
use core::hash::Hash;
use core::ops::{Add, AddAssign};
#[cfg(feature = "no_std")]
pub use hashbrown::HashMap;
#[cfg(feature = "use_std")]
pub use std::collections::HashMap;

#[derive(Ord, PartialOrd, Eq, PartialEq)]
pub enum Verbosity {
    NoOutput,
    Thresholds,
    RawResult,
    Debug,
}

pub struct HistParams {
    pub iterations: u32,
    pub bucket_size: usize,
    pub bucket_number: usize,
}

pub struct CalibrationOptions {
    pub hist_params: HistParams,
    pub verbosity: Verbosity,
    pub optimised_addresses: bool,
}

impl CalibrationOptions {
    pub fn new(hist_params: HistParams, verbosity: Verbosity) -> CalibrationOptions {
        CalibrationOptions {
            hist_params,
            verbosity,
            optimised_addresses: false,
        }
    }
}

pub unsafe fn only_reload(p: *const u8) -> u64 {
    let t = rdtsc_fence();
    maccess(p);
    rdtsc_fence() - t
}

pub unsafe fn flush_and_reload(p: *const u8) -> u64 {
    flush(p);
    only_reload(p)
}

pub unsafe fn reload_and_flush(p: *const u8) -> u64 {
    let r = only_reload(p);
    flush(p);
    r
}

pub unsafe fn only_flush(p: *const u8) -> u64 {
    let t = rdtsc_fence();
    flush(p);
    rdtsc_fence() - t
}

pub unsafe fn load_and_flush(p: *const u8) -> u64 {
    maccess(p);
    only_flush(p)
}

pub unsafe fn flush_and_flush(p: *const u8) -> u64 {
    flush(p);
    only_flush(p)
}

pub unsafe fn l3_and_reload(p: *const u8) -> u64 {
    flush(p);
    arch_x86::_mm_mfence();
    arch_x86::_mm_prefetch(p as *const i8, arch_x86::_MM_HINT_T2);
    arch_x86::__cpuid_count(0, 0);
    only_reload(p)
}

pub const PAGE_SHIFT: usize = 12;
pub const PAGE_LEN: usize = 1 << PAGE_SHIFT;

pub fn get_vpn<T>(p: *const T) -> usize {
    (p as usize) & (!(PAGE_LEN - 1)) // FIXME
}

const BUCKET_SIZE: usize = 5;
const BUCKET_NUMBER: usize = 250;

// TODO same as below, also add the whole page calibration

pub fn calibrate_access(array: &[u8; 4096]) -> u64 {
    println!("Calibrating...");

    // Allocate a target array
    // TBD why size, why the position in the array, why the type (usize)
    //    let mut array = Vec::<usize>::with_capacity(5 << 10);
    //    array.resize(5 << 10, 1);

    //    let array = array.into_boxed_slice();

    // Histograms bucket of 5 and max at 400 cycles
    // Magic numbers to be justified
    // 80 is a size of screen
    let mut hit_histogram = vec![0; BUCKET_NUMBER]; //Vec::<u32>::with_capacity(BUCKET_NUMBER);
                                                    //hit_histogram.resize(BUCKET_NUMBER, 0);

    let mut miss_histogram = hit_histogram.clone();

    // the address in memory we are going to target
    let pointer = &array[0] as *const u8;

    println!("buffer start {:p}", pointer);

    if pointer as usize & 0x3f != 0 {
        panic!("not aligned nicely");
    }

    // do a large sample of accesses to a cached line
    unsafe { maccess(pointer) };
    for i in 0..(4 << 10) {
        for _ in 0..(1 << 10) {
            let d = unsafe { only_reload(pointer.offset(i & (!0x3f))) } as usize;
            hit_histogram[min(BUCKET_NUMBER - 1, d / BUCKET_SIZE) as usize] += 1;
        }
    }

    // do a large numer of accesses to uncached line
    unsafe { flush(pointer) };
    for i in 0..(4 << 10) {
        for _ in 0..(1 << 10) {
            let d = unsafe { flush_and_reload(pointer.offset(i & (!0x3f))) } as usize;
            miss_histogram[min(BUCKET_NUMBER - 1, d / BUCKET_SIZE) as usize] += 1;
        }
    }

    let mut hit_max = 0;
    let mut hit_max_i = 0;
    let mut miss_min_i = 0;
    for i in 0..hit_histogram.len() {
        println!(
            "{:3}: {:10} {:10}",
            i * BUCKET_SIZE,
            hit_histogram[i],
            miss_histogram[i]
        );
        if hit_max < hit_histogram[i] {
            hit_max = hit_histogram[i];
            hit_max_i = i;
        }
        if miss_histogram[i] > 3 /* Magic */ && miss_min_i == 0 {
            miss_min_i = i
        }
    }
    println!("Miss min {}", miss_min_i * BUCKET_SIZE);
    println!("Max hit {}", hit_max_i * BUCKET_SIZE);

    let mut min = u32::max_value();
    let mut min_i = 0;
    for i in hit_max_i..miss_min_i {
        if min > hit_histogram[i] + miss_histogram[i] {
            min = hit_histogram[i] + miss_histogram[i];
            min_i = i;
        }
    }

    println!("Threshold {}", min_i * BUCKET_SIZE);
    println!("Calibration done.");
    (min_i * BUCKET_SIZE) as u64
}

pub const CFLUSH_BUCKET_SIZE: usize = 1;
pub const CFLUSH_BUCKET_NUMBER: usize = 500;

pub const CFLUSH_NUM_ITER: u32 = 1 << 11;

pub fn calibrate_flush(
    array: &[u8],
    cache_line_size: usize,
    verbose_level: Verbosity,
) -> Vec<CalibrateResult> {
    let pointer = (&array[0]) as *const u8;

    if pointer as usize & (cache_line_size - 1) != 0 {
        panic!("not aligned nicely");
    }

    calibrate_impl_fixed_freq(
        pointer,
        cache_line_size,
        array.len() as isize,
        &[
            CalibrateOperation {
                op: load_and_flush,
                name: "clflush_hit",
                display_name: "clflush hit",
            },
            CalibrateOperation {
                op: flush_and_flush,
                name: "clflush_miss",
                display_name: "clflush miss",
            },
        ],
        HistParams {
            bucket_number: CFLUSH_BUCKET_NUMBER,
            bucket_size: CFLUSH_BUCKET_SIZE,
            iterations: CFLUSH_NUM_ITER,
        },
        verbose_level,
    )
}

#[derive(Debug)]
pub struct CalibrateResult {
    pub page: VPN,
    pub offset: isize,
    pub histogram: Vec<Vec<u32>>,
    pub median: Vec<u64>,
    pub min: Vec<u64>,
    pub max: Vec<u64>,
}

pub struct CalibrateOperation<'a> {
    pub op: unsafe fn(*const u8) -> u64,
    pub name: &'a str,
    pub display_name: &'a str,
}

pub unsafe fn calibrate(
    p: *const u8,
    increment: usize,
    len: isize,
    operations: &[CalibrateOperation],
    buckets_num: usize,
    bucket_size: usize,
    num_iterations: u32,
    verbosity_level: Verbosity,
) -> Vec<CalibrateResult> {
    calibrate_impl_fixed_freq(
        p,
        increment,
        len,
        operations,
        HistParams {
            bucket_number: buckets_num,
            bucket_size,
            iterations: num_iterations,
        },
        verbosity_level,
    )
}

pub const SPURIOUS_THRESHOLD: u32 = 1;
fn calibrate_impl_fixed_freq(
    p: *const u8,
    increment: usize,
    len: isize,
    operations: &[CalibrateOperation],
    hist_params: HistParams,
    verbosity_level: Verbosity,
) -> Vec<CalibrateResult> {
    if verbosity_level >= Thresholds {
        println!(
            "Calibrating {}...",
            operations
                .iter()
                .map(|operation| { operation.display_name })
                .format(", ")
        );
    }

    let to_bucket = |time: u64| -> usize { time as usize / hist_params.bucket_size };
    let from_bucket = |bucket: usize| -> u64 { (bucket * hist_params.bucket_size) as u64 };

    // FIXME : Core per socket
    let slicing = if let Some(uarch) = MicroArchitecture::get_micro_architecture() {
        if let Some(vendor_family_model_stepping) = MicroArchitecture::get_family_model_stepping() {
            Some(cache_slicing(
                uarch,
                8,
                vendor_family_model_stepping.0,
                vendor_family_model_stepping.1,
                vendor_family_model_stepping.2,
            ))
        } else {
            None
        }
    } else {
        None
    };

    let h = if let Some(s) = slicing {
        if s.can_hash() {
            Some(|addr: usize| -> u8 { slicing.unwrap().hash(addr).unwrap() })
        } else {
            None
        }
    } else {
        None
    };
    // TODO fix the GROSS hack of using max cpu core supported

    let mut ret = Vec::new();
    if verbosity_level >= Thresholds {
        print!("CSV: address, ");
        if h.is_some() {
            print!("hash, ");
        }
        println!(
            "{} min, {} median, {} max",
            operations
                .iter()
                .map(|operation| operation.name)
                .format(" min, "),
            operations
                .iter()
                .map(|operation| operation.name)
                .format(" median, "),
            operations
                .iter()
                .map(|operation| operation.name)
                .format(" max, ")
        );
    }
    if verbosity_level >= RawResult {
        print!("RESULT:address,");
        if h.is_some() {
            print!("hash,");
        }
        println!(
            "time,{}",
            operations
                .iter()
                .map(|operation| operation.name)
                .format(",")
        );
    }

    for i in (0..len).step_by(increment) {
        let pointer = unsafe { p.offset(i) };
        let hash = h.map(|h| h(pointer as usize));

        if verbosity_level >= Thresholds {
            print!("Calibration for {:p}", pointer);
            if let Some(h) = hash {
                print!(" (hash: {:x})", h)
            }
            println!();
        }

        // TODO add some useful impl to CalibrateResults
        let mut calibrate_result = CalibrateResult {
            page: get_vpn(pointer),
            offset: i,
            histogram: Vec::new(),
            median: vec![0; operations.len()],
            min: vec![0; operations.len()],
            max: vec![0; operations.len()],
        };
        calibrate_result.histogram.reserve(operations.len());

        for op in operations {
            let mut hist = vec![0; hist_params.bucket_number];
            for _ in 0..hist_params.iterations {
                let time = unsafe { (op.op)(pointer) };
                let bucket = min(hist_params.bucket_number - 1, to_bucket(time));
                hist[bucket] += 1;
            }
            calibrate_result.histogram.push(hist);
        }

        let mut sums = vec![0; operations.len()];

        let median_thresholds: Vec<u32> = calibrate_result
            .histogram
            .iter()
            .map(|h| (hist_params.iterations - h[hist_params.bucket_number - 1]) / 2)
            .collect();

        for j in 0..hist_params.bucket_number - 1 {
            if verbosity_level >= RawResult {
                print!("RESULT:{:p},", pointer);
                if let Some(h) = hash {
                    print!("{:x},", h);
                }
                print!("{}", from_bucket(j));
            }
            // ignore the last bucket : spurious context switches etc.
            for op in 0..operations.len() {
                let hist = &calibrate_result.histogram[op][j];
                let min = &mut calibrate_result.min[op];
                let max = &mut calibrate_result.max[op];
                let med = &mut calibrate_result.median[op];
                let sum = &mut sums[op];
                if verbosity_level >= RawResult {
                    print!(",{}", hist);
                }

                if *min == 0 {
                    // looking for min
                    if *hist > SPURIOUS_THRESHOLD {
                        *min = from_bucket(j);
                    }
                } else if *hist > SPURIOUS_THRESHOLD {
                    *max = from_bucket(j);
                }

                if *med == 0 {
                    *sum += *hist;
                    if *sum >= median_thresholds[op] {
                        *med = from_bucket(j);
                    }
                }
            }
            if verbosity_level >= RawResult {
                println!();
            }
        }
        if verbosity_level >= Thresholds {
            for (j, op) in operations.iter().enumerate() {
                println!(
                    "{}: min {}, median {}, max {}",
                    op.display_name,
                    calibrate_result.min[j],
                    calibrate_result.median[j],
                    calibrate_result.max[j]
                );
            }
            print!("CSV: {:p}, ", pointer);
            if let Some(h) = hash {
                print!("{:x}, ", h)
            }
            println!(
                "{}, {}, {}",
                calibrate_result.min.iter().format(", "),
                calibrate_result.median.iter().format(", "),
                calibrate_result.max.iter().format(", ")
            );
        }
        ret.push(calibrate_result);
    }
    ret
}

pub fn get_cache_slicing(core_per_socket: u8) -> Option<CacheSlicing> {
    if let Some(uarch) = MicroArchitecture::get_micro_architecture() {
        if let Some(vendor_family_model_stepping) = MicroArchitecture::get_family_model_stepping() {
            Some(cache_slicing(
                uarch,
                core_per_socket,
                vendor_family_model_stepping.0,
                vendor_family_model_stepping.1,
                vendor_family_model_stepping.2,
            ))
        } else {
            None
        }
    } else {
        None
    }
}

#[allow(non_snake_case)]
pub fn calibrate_L3_miss_hit(
    array: &[u8],
    cache_line_size: usize,
    verbose_level: Verbosity,
) -> CalibrateResult {
    if verbose_level > NoOutput {
        println!("Calibrating L3 access...");
    }
    let pointer = (&array[0]) as *const u8;

    let r = calibrate_impl_fixed_freq(
        pointer,
        cache_line_size,
        array.len() as isize,
        &[CalibrateOperation {
            op: l3_and_reload,
            name: "l3_hit",
            display_name: "L3 hit",
        }],
        HistParams {
            bucket_number: 512,
            bucket_size: 2,
            iterations: 1 << 11,
        },
        verbose_level,
    );

    r.into_iter().next().unwrap()
}

/*
   ASVP trait ?
   Easily put any combination, use None to signal Any possible value, Some to signal fixed value.
*/

pub type VPN = usize;
pub type Slice = u8;

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct ASVP {
    pub attacker: usize,
    pub slice: Slice,
    pub victim: usize,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct CSP {
    pub core: usize,
    pub slice: Slice,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct ASP {
    pub attacker: usize,
    pub slice: Slice,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct SVP {
    pub slice: Slice,
    pub victim: usize,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct SP {
    pub slice: Slice,
    pub page: VPN,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Default)]
pub struct AV {
    pub attacker: usize,
    pub victim: usize,
}

#[derive(Debug, Clone)]
pub struct RawHistogram {
    pub hit: Vec<u32>,
    pub miss: Vec<u32>,
}

// ALL Histogram deal in buckets : FIXME we should clearly distinguish bucket vs time.
// Thresholds are less than equal.
impl RawHistogram {
    pub fn from(
        mut calibrate_result: CalibrateResult,
        hit_index: usize,
        miss_index: usize,
    ) -> Self {
        calibrate_result.histogram.push(Vec::default());
        let hit = calibrate_result.histogram.swap_remove(hit_index);
        calibrate_result.histogram.push(Vec::default());
        let miss = calibrate_result.histogram.swap_remove(miss_index);
        RawHistogram { hit, miss }
    }

    pub fn empty(len: usize) -> Self {
        Self {
            hit: vec![0; len],
            miss: vec![0; len],
        }
    }
}

// Addition logic

// Tough case, both references.

impl Add for &RawHistogram {
    type Output = RawHistogram;

    fn add(self, rhs: &RawHistogram) -> Self::Output {
        assert_eq!(self.hit.len(), rhs.hit.len());
        assert_eq!(self.miss.len(), rhs.miss.len());
        assert_eq!(self.hit.len(), self.miss.len());
        let len = self.hit.len();
        let mut r = RawHistogram {
            hit: vec![0; len],
            miss: vec![0; len],
        };
        for i in 0..len {
            r.hit[i] = self.hit[i] + rhs.hit[i];
            r.miss[i] = self.miss[i] + rhs.miss[i];
        }
        r
    }
}

// most common case re-use of self is possible. (Or a reduction to such a case)

impl AddAssign<&RawHistogram> for RawHistogram {
    //type Rhs = &RawHistogram;
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.hit.len(), rhs.hit.len());
        assert_eq!(self.miss.len(), rhs.miss.len());
        assert_eq!(self.hit.len(), self.miss.len());

        for i in 0..self.hit.len() {
            self.hit[i] += rhs.hit[i];
            self.miss[i] += rhs.miss[i];
        }
    }
}

// Fallback to most common case

impl Add for RawHistogram {
    type Output = RawHistogram;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for RawHistogram {
    type Output = RawHistogram;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<RawHistogram> for &RawHistogram {
    type Output = RawHistogram;

    fn add(self, mut rhs: RawHistogram) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl AddAssign<Self> for RawHistogram {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

pub fn cum_sum(vector: &[u32]) -> Vec<u32> {
    let len = vector.len();
    let mut res = vec![0; len];
    res[0] = vector[0];
    for i in 1..len {
        res[i] = res[i - 1] + vector[i];
    }
    assert_eq!(len, res.len());
    assert_eq!(len, vector.len());
    res
}

#[derive(Debug, Clone)]
pub struct HistogramCumSum {
    pub num_hit: u32,
    pub num_miss: u32,
    pub hit: Vec<u32>,
    pub miss: Vec<u32>,
    pub hit_cum_sum: Vec<u32>,
    pub miss_cum_sum: Vec<u32>,
}

impl HistogramCumSum {
    pub fn from(raw_histogram: RawHistogram) -> Self {
        let len = raw_histogram.miss.len();

        assert_eq!(raw_histogram.hit.len(), len);

        // Cum Sums
        let miss_cum_sum = cum_sum(&raw_histogram.miss);
        let hit_cum_sum = cum_sum(&raw_histogram.hit);
        let miss_total = miss_cum_sum[len - 1];
        let hit_total = hit_cum_sum[len - 1];
        Self {
            num_hit: hit_total,
            num_miss: miss_total,
            hit: raw_histogram.hit,
            miss: raw_histogram.miss,
            hit_cum_sum,
            miss_cum_sum,
        }
    }

    pub fn from_calibrate(
        calibrate_result: CalibrateResult,
        hit_index: usize,
        miss_index: usize,
    ) -> Self {
        Self::from(RawHistogram::from(calibrate_result, hit_index, miss_index))
    }

    pub fn error_for_threshold(&self, threshold: Threshold) -> ErrorPrediction {
        if threshold.miss_faster_than_hit {
            ErrorPrediction {
                true_hit: self.num_hit - self.hit_cum_sum[threshold.bucket_index],
                true_miss: self.miss_cum_sum[threshold.bucket_index],
                false_hit: self.num_miss - self.miss_cum_sum[threshold.bucket_index],
                false_miss: self.hit_cum_sum[threshold.bucket_index],
            }
        } else {
            ErrorPrediction {
                true_hit: self.hit_cum_sum[threshold.bucket_index],
                true_miss: self.num_miss - self.miss_cum_sum[threshold.bucket_index],
                false_hit: self.miss_cum_sum[threshold.bucket_index],
                false_miss: self.num_hit - self.hit_cum_sum[threshold.bucket_index],
            }
        }
    }

    pub fn len(&self) -> usize {
        self.hit.len()
    }

    pub fn empty(len: usize) -> Self {
        Self {
            num_hit: 0,
            num_miss: 0,
            hit: vec![0; len],
            miss: vec![0; len],
            hit_cum_sum: vec![0; len],
            miss_cum_sum: vec![0; len],
        }
    }
}

// Addition logic

// Tough case, both references.

impl Add for &HistogramCumSum {
    type Output = HistogramCumSum;

    fn add(self, rhs: &HistogramCumSum) -> Self::Output {
        assert_eq!(self.hit.len(), self.miss.len());
        assert_eq!(self.hit.len(), self.hit_cum_sum.len());
        assert_eq!(self.hit.len(), self.miss_cum_sum.len());
        assert_eq!(self.hit.len(), rhs.hit.len());
        assert_eq!(self.hit.len(), rhs.miss.len());
        assert_eq!(self.hit.len(), rhs.hit_cum_sum.len());
        assert_eq!(self.hit.len(), rhs.miss_cum_sum.len());
        let len = self.len();
        let mut r = HistogramCumSum {
            num_hit: self.num_hit + rhs.num_hit,
            num_miss: self.num_miss + rhs.num_miss,
            hit: vec![0; len],
            miss: vec![0; len],
            hit_cum_sum: vec![0; len],
            miss_cum_sum: vec![0; len],
        };
        for i in 0..len {
            r.hit[i] = self.hit[i] + rhs.hit[i];
            r.miss[i] = self.miss[i] + rhs.miss[i];
            r.hit_cum_sum[i] = self.hit_cum_sum[i] + rhs.hit_cum_sum[i];
            r.miss_cum_sum[i] = self.miss_cum_sum[i] + rhs.miss_cum_sum[i];
        }
        r
    }
}

// most common case re-use of self is possible. (Or a reduction to such a case)

impl AddAssign<&Self> for HistogramCumSum {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.hit.len(), self.miss.len());
        assert_eq!(self.hit.len(), self.hit_cum_sum.len());
        assert_eq!(self.hit.len(), self.miss_cum_sum.len());
        assert_eq!(self.hit.len(), rhs.hit.len());
        assert_eq!(self.hit.len(), rhs.miss.len());
        assert_eq!(self.hit.len(), rhs.hit_cum_sum.len());
        assert_eq!(self.hit.len(), rhs.miss_cum_sum.len());
        self.num_hit += rhs.num_hit;
        self.num_miss += rhs.num_miss;
        let len = self.len();
        for i in 0..len {
            self.hit[i] += rhs.hit[i];
            self.miss[i] += rhs.miss[i];
            self.hit_cum_sum[i] += rhs.hit_cum_sum[i];
            self.miss_cum_sum[i] += rhs.miss_cum_sum[i];
        }
    }
}

// Fallback to most common case

impl Add for HistogramCumSum {
    type Output = HistogramCumSum;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for HistogramCumSum {
    type Output = HistogramCumSum;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<HistogramCumSum> for &HistogramCumSum {
    type Output = HistogramCumSum;

    fn add(self, mut rhs: HistogramCumSum) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl AddAssign<Self> for HistogramCumSum {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ErrorPrediction {
    pub true_hit: u32,
    pub true_miss: u32,
    pub false_hit: u32,
    pub false_miss: u32,
}

impl ErrorPrediction {
    pub fn total_error(&self) -> u32 {
        self.false_hit + self.false_miss
    }
    pub fn total(&self) -> u32 {
        self.false_hit + self.false_miss + self.true_hit + self.true_miss
    }
    pub fn error_rate(&self) -> f32 {
        (self.false_miss + self.false_hit) as f32 / (self.total() as f32)
    }
}

impl Add for &ErrorPrediction {
    type Output = ErrorPrediction;

    fn add(self, rhs: &ErrorPrediction) -> Self::Output {
        ErrorPrediction {
            true_hit: self.true_hit + rhs.true_hit,
            true_miss: self.true_miss + rhs.true_miss,
            false_hit: self.false_hit + rhs.false_hit,
            false_miss: self.true_miss + rhs.false_miss,
        }
    }
}

// most common case re-use of self is possible. (Or a reduction to such a case)

impl AddAssign<&Self> for ErrorPrediction {
    fn add_assign(&mut self, rhs: &Self) {
        self.true_hit += rhs.true_hit;
        self.true_miss += rhs.true_miss;
        self.false_hit += rhs.false_hit;
        self.false_miss += rhs.false_miss;
    }
}

// Fallback to most common case

impl Add for ErrorPrediction {
    type Output = ErrorPrediction;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for ErrorPrediction {
    type Output = ErrorPrediction;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<ErrorPrediction> for &ErrorPrediction {
    type Output = ErrorPrediction;

    fn add(self, mut rhs: ErrorPrediction) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl AddAssign<Self> for ErrorPrediction {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

#[derive(Debug, Clone)]
pub struct ErrorPredictions {
    pub histogram: HistogramCumSum,
    pub error_miss_less_than_hit: Vec<u32>,
    pub error_hit_less_than_miss: Vec<u32>,
}

impl ErrorPredictions {
    // BUGGY TODO
    pub fn predict_errors(hist: HistogramCumSum) -> Self {
        let mut error_miss_less_than_hit = vec![0; hist.len() - 1];
        let mut error_hit_less_than_miss = vec![0; hist.len() - 1];
        for threshold_bucket_index in 0..(hist.len() - 1) {
            error_miss_less_than_hit[threshold_bucket_index] = hist
                .error_for_threshold(Threshold {
                    bucket_index: threshold_bucket_index,
                    miss_faster_than_hit: true,
                })
                .total_error();

            error_hit_less_than_miss[threshold_bucket_index] = hist
                .error_for_threshold(Threshold {
                    bucket_index: threshold_bucket_index,
                    miss_faster_than_hit: false,
                })
                .total_error();
        }
        Self {
            histogram: hist,
            error_miss_less_than_hit,
            error_hit_less_than_miss,
        }
    }

    pub fn empty(len: usize) -> Self {
        Self::predict_errors(HistogramCumSum::empty(len))
    }

    pub fn debug(&self) {
        println!("Debug:HEADER TBD");
        for i in 0..(self.histogram.len() - 1) {
            println!(
                "Debug:{:5},{:5},{:6},{:6},{:6}, {:6}",
                self.histogram.hit[i],
                self.histogram.miss[i],
                self.histogram.hit_cum_sum[i],
                self.histogram.miss_cum_sum[i],
                self.error_miss_less_than_hit[i],
                self.error_hit_less_than_miss[i]
            );
        }
        let i = self.histogram.len() - 1;
        println!(
            "Debug:{:5},{:5},{:6},{:6}",
            self.histogram.hit[i],
            self.histogram.miss[i],
            self.histogram.hit_cum_sum[i],
            self.histogram.miss_cum_sum[i]
        );
    }
}

// Addition logic

// Tough case, both references.

impl Add for &ErrorPredictions {
    type Output = ErrorPredictions;

    fn add(self, rhs: &ErrorPredictions) -> Self::Output {
        assert_eq!(
            self.error_hit_less_than_miss.len(),
            rhs.error_hit_less_than_miss.len()
        );
        assert_eq!(
            self.error_hit_less_than_miss.len(),
            self.error_miss_less_than_hit.len()
        );
        assert_eq!(
            self.error_miss_less_than_hit.len(),
            rhs.error_miss_less_than_hit.len()
        );
        let len = self.error_miss_less_than_hit.len();
        let mut r = ErrorPredictions {
            histogram: &self.histogram + &rhs.histogram,
            error_miss_less_than_hit: vec![0; len],
            error_hit_less_than_miss: vec![0; len],
        };
        for i in 0..len {
            r.error_miss_less_than_hit[i] =
                self.error_miss_less_than_hit[i] + rhs.error_miss_less_than_hit[i];
            r.error_hit_less_than_miss[i] =
                self.error_hit_less_than_miss[i] + rhs.error_hit_less_than_miss[i];
        }
        r
    }
}

// most common case re-use of self is possible. (Or a reduction to such a case)

impl AddAssign<&Self> for ErrorPredictions {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(
            self.error_hit_less_than_miss.len(),
            rhs.error_hit_less_than_miss.len()
        );
        assert_eq!(
            self.error_hit_less_than_miss.len(),
            self.error_miss_less_than_hit.len()
        );
        assert_eq!(
            self.error_miss_less_than_hit.len(),
            rhs.error_miss_less_than_hit.len()
        );
        self.histogram += &rhs.histogram;
        for i in 0..self.error_hit_less_than_miss.len() {
            self.error_hit_less_than_miss[i] += rhs.error_hit_less_than_miss[i];
            self.error_miss_less_than_hit[i] += rhs.error_miss_less_than_hit[i];
        }
    }
}

// Fallback to most common case

impl Add for ErrorPredictions {
    type Output = ErrorPredictions;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for ErrorPredictions {
    type Output = ErrorPredictions;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<ErrorPredictions> for &ErrorPredictions {
    type Output = ErrorPredictions;

    fn add(self, mut rhs: ErrorPredictions) -> Self::Output {
        rhs += self;
        rhs
    }
}

impl AddAssign<Self> for ErrorPredictions {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ThresholdError {
    pub threshold: Threshold,
    pub error: ErrorPrediction,
}

#[derive(Debug, Clone)]
pub struct PotentialThresholds {
    pub threshold_errors: Vec<ThresholdError>,
}

impl PotentialThresholds {
    pub fn median(mut self) -> Option<ThresholdError> {
        if self.threshold_errors.len() > 0 {
            let index = (self.threshold_errors.len() - 1) / 2;
            self.threshold_errors.push(Default::default());
            Some(self.threshold_errors.swap_remove(index))
        } else {
            None
        }
    }

    pub fn minimizing_total_error(error_pred: ErrorPredictions) -> Self {
        let mut min_error = u32::max_value();
        let mut threshold_errors = Vec::new();
        for i in 0..error_pred.error_miss_less_than_hit.len() {
            if error_pred.error_miss_less_than_hit[i] < min_error {
                min_error = error_pred.error_miss_less_than_hit[i];
                threshold_errors = Vec::new();
            }
            if error_pred.error_hit_less_than_miss[i] < min_error {
                min_error = error_pred.error_hit_less_than_miss[i];
                threshold_errors = Vec::new();
            }
            if error_pred.error_miss_less_than_hit[i] == min_error {
                let threshold = Threshold {
                    bucket_index: i,
                    miss_faster_than_hit: true,
                };
                let error = error_pred.histogram.error_for_threshold(threshold);
                threshold_errors.push(ThresholdError { threshold, error })
            }
            if error_pred.error_hit_less_than_miss[i] == min_error {
                let threshold = Threshold {
                    bucket_index: i,
                    miss_faster_than_hit: false,
                };
                let error = error_pred.histogram.error_for_threshold(threshold);
                threshold_errors.push(ThresholdError { threshold, error })
            }
        }
        Self { threshold_errors }
    }
}

// Thresholds are less than equal.
// usize for bucket, u64 for time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Threshold {
    pub bucket_index: usize,
    pub miss_faster_than_hit: bool,
}

impl Threshold {
    // FIXME !!!!
    fn to_time(&self, bucket: usize) -> u64 {
        bucket as u64
    }

    pub fn is_hit(&self, time: u64) -> bool {
        self.miss_faster_than_hit && time >= self.to_time(self.bucket_index)
            || !self.miss_faster_than_hit && time < self.to_time(self.bucket_index)
    }
}

pub fn map_values<K, U, V, F>(input: HashMap<K, U>, f: F) -> HashMap<K, V>
where
    K: Hash + Eq,
    F: Fn(U, &K) -> V,
{
    let mut results = HashMap::new();
    for (k, u) in input {
        let f_u = f(u, &k);
        results.insert(k, f_u);
    }
    results
}

pub fn accumulate<K, V, RK, Reduction, Accumulator, Accumulation, AccumulatorDefault>(
    input: HashMap<K, V>,
    reduction: Reduction,
    accumulator_default: AccumulatorDefault,
    aggregation: Accumulation,
) -> HashMap<RK, Accumulator>
where
    K: Hash + Eq + Copy,
    RK: Hash + Eq + Copy,
    Reduction: Fn(K) -> RK,
    Accumulation: Fn(&mut Accumulator, V, K, RK) -> (),
    AccumulatorDefault: Fn() -> Accumulator,
{
    let mut accumulators = HashMap::new();
    for (k, v) in input {
        let rk = reduction(k);
        aggregation(
            accumulators
                .entry(rk)
                .or_insert_with(|| accumulator_default()),
            v,
            k,
            rk,
        );
    }
    accumulators
}

pub fn reduce<K, V, RK, RV, Reduction, Accumulator, Accumulation, AccumulatorDefault, Extract>(
    input: HashMap<K, V>,
    reduction: Reduction,
    accumulator_default: AccumulatorDefault,
    aggregation: Accumulation,
    extraction: Extract,
) -> HashMap<RK, RV>
where
    K: Hash + Eq + Copy,
    RK: Hash + Eq + Copy,
    Reduction: Fn(K) -> RK,
    AccumulatorDefault: Fn() -> Accumulator,
    Accumulation: Fn(&mut Accumulator, V, K, RK) -> (),
    Extract: Fn(Accumulator, &RK) -> RV,
{
    let accumulators = accumulate(input, reduction, accumulator_default, aggregation);
    let result = map_values(accumulators, extraction);
    result
}

/*
pub fn compute_threshold_error() -> (Threshold, ()) {
    unimplemented!();
} // TODO
*/

#[cfg(test)]
mod tests {

    use crate::calibration::map_values;
    #[cfg(feature = "no_std")]
    use hashbrown::HashMap;
    #[cfg(feature = "use_std")]
    use std::collections::HashMap;

    #[test]
    fn test_map_values() {
        let mut input = HashMap::new();
        input.insert(0, "a");
        input.insert(1, "b");
        let output = map_values(input, |c| c.to_uppercase());
        assert_eq!(output[&0], "A");
        assert_eq!(output[&1], "B");
    }
}
