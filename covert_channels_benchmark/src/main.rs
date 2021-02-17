#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::io::{stdout, Write};

//use basic_timing_cache_channel::{naive::NaiveTimingChannel, TopologyAwareTimingChannel};
use basic_timing_cache_channel::{TopologyAwareError, TopologyAwareTimingChannel};
use cache_utils::calibration::Threshold;
use covert_channels_evaluation::{benchmark_channel, CovertChannel, CovertChannelBenchmarkResult};
use flush_flush::naive::NaiveFlushAndFlush;
use flush_flush::{FFPrimitives, FlushAndFlush, SingleFlushAndFlush};
use flush_reload::naive::{NaiveFRPrimitives, NaiveFlushAndReload};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;

const NUM_BYTES: usize = 1 << 12;

const NUM_PAGES: usize = 1;

const NUM_PAGES_2: usize = 4;

const NUM_PAGE_MAX: usize = 32;

const NUM_ITER: usize = 16;

struct BenchmarkStats {
    raw_res: Vec<(CovertChannelBenchmarkResult, usize, usize)>,
    average_p: f64,
    var_p: f64,
    average_C: f64,
    var_C: f64,
    average_T: f64,
    var_T: f64,
}

fn run_benchmark<T: CovertChannel + 'static>(
    name: &str,
    constructor: impl Fn(usize, usize) -> (T, usize, usize),
    num_iter: usize,
    num_pages: usize,
    old: CpuSet,
) -> BenchmarkStats {
    let mut results = Vec::new();
    print!("Benchmarking {} with {} pages", name, num_pages);
    let mut count = 0;
    for i in 0..CpuSet::count() {
        for j in 0..CpuSet::count() {
            if old.is_set(i).unwrap() && old.is_set(j).unwrap() && i != j {
                for _ in 0..num_iter {
                    count += 1;
                    print!(".");
                    stdout().flush().expect("Failed to flush");
                    let (channel, main_core, helper_core) = constructor(i, j);
                    let r = benchmark_channel(channel, num_pages, NUM_BYTES);
                    results.push((r, main_core, helper_core));
                }
            }
        }
    }
    println!();
    let mut average_p = 0.0;
    let mut average_C = 0.0;
    let mut average_T = 0.0;
    for result in results.iter() {
        println!(
            "main: {} helper: {} result: {:?}",
            result.1, result.2, result.0
        );
        println!(
            "C: {}, T: {}",
            result.0.capacity(),
            result.0.true_capacity()
        );
        println!(
            "Detailed:\"{}\",{},{},{},{},{},{}",
            name,
            num_pages,
            result.1,
            result.2,
            result.0.csv(),
            result.0.capacity(),
            result.0.true_capacity()
        );
        average_p += result.0.error_rate;
        average_C += result.0.capacity();
        average_T += result.0.true_capacity()
    }
    average_p /= count as f64;
    average_C /= count as f64;
    average_T /= count as f64;
    println!(
        "{} - {} Average p: {} C: {}, T: {}",
        name, num_pages, average_p, average_C, average_T
    );
    let mut var_p = 0.0;
    let mut var_C = 0.0;
    let mut var_T = 0.0;
    for result in results.iter() {
        let p = result.0.error_rate - average_p;
        var_p += p * p;
        let C = result.0.capacity() - average_C;
        var_C += C * C;
        let T = result.0.true_capacity() - average_T;
        var_T += T * T;
    }
    var_p /= count as f64;
    var_C /= count as f64;
    var_T /= count as f64;
    println!(
        "{} - {} Variance of p: {}, C: {}, T:{}",
        name, num_pages, var_p, var_C, var_T
    );
    println!(
        "CSV:\"{}\",{},{},{},{},{},{},{}",
        name, num_pages, average_p, average_C, average_T, var_p, var_C, var_T
    );

    BenchmarkStats {
        raw_res: results,
        average_p,
        var_p,
        average_C,
        var_C,
        average_T,
        var_T,
    }
}

fn main() {
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();
    println!(
        "Detailed:Benchmark,Pages,main_core,helper_core,{},C,T",
        CovertChannelBenchmarkResult::csv_header()
    );
    println!("CSV:Benchmark,Pages,main_core,helper_core,p,C,T,var_p,var_C,var_T");

    for num_pages in 1..=32 {
        let naive_ff = run_benchmark(
            "Naive F+F",
            |i, j| {
                let mut r = NaiveFlushAndFlush::new(
                    Threshold {
                        bucket_index: 202,
                        miss_faster_than_hit: true,
                    },
                    FFPrimitives {},
                );
                r.set_cores(i, j);
                (r, i, j)
            },
            NUM_ITER,
            num_pages,
            old,
        );

        let fr = run_benchmark(
            "F+R",
            |i, j| {
                let mut r = NaiveFlushAndReload::new(
                    Threshold {
                        bucket_index: 250,
                        miss_faster_than_hit: false,
                    },
                    NaiveFRPrimitives {},
                );
                r.set_cores(i, j);
                (r, i, j)
            },
            NUM_ITER,
            num_pages,
            old,
        );

        let ff = run_benchmark(
            "Better F+F",
            |i, j| {
                let (mut r, i, j) = match FlushAndFlush::new_any_two_core(true, FFPrimitives {}) {
                    Ok((channel, _old, main_core, helper_core)) => {
                        (channel, main_core, helper_core)
                    }
                    Err(e) => {
                        panic!("{:?}", e);
                    }
                };
                (r, i, j)
            },
            1,
            num_pages,
            old,
        );
    }
}
/*
fn main() {
      for num_pages in 1..=32 {
        /*println!("Benchmarking F+F");
        for _ in 0..16 {
            // TODO Use the best possible ASV, not best possible AV
            let (channel, old, receiver, sender) = match SingleFlushAndFlush::new_any_two_core(true) {
                Err(e) => {
                    panic!("{:?}", e);
                }
                Ok(r) => r,
            };

            let r = benchmark_channel(channel, NUM_PAGES, NUM_BYTES);
            println!("{:?}", r);
                    println!("C: {}, T: {}", r.capacity(), r.true_capacity());

        }*/

        let naive_ff = run_benchmark(
            "Naive F+F",
            || NaiveFlushAndFlush::from_threshold(202),
            NUM_ITER << 4,
            num_pages,
        );

        let better_ff = run_benchmark(
            "Better F+F",
            || {
                match FlushAndFlush::new_any_two_core(true) {
                    Err(e) => {
                        panic!("{:?}", e);
                    }
                    Ok(r) => r,
                }
                .0
            },
            NUM_ITER,
            num_pages,
        );

        let fr = run_benchmark(
            "F+R",
            || NaiveFlushAndReload::from_threshold(230),
            NUM_ITER,
            num_pages,
        );
    }
}
*/
