#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]
/*
use std::io::{stdout, Write};

use covert_channels_evaluation::{benchmark_channel, CovertChannel, CovertChannelBenchmarkResult};
use flush_flush::naive::NaiveFlushAndFlush;
use flush_flush::{FlushAndFlush, SingleFlushAndFlush};
use flush_reload::naive::NaiveFlushAndReload;
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;

const NUM_BYTES: usize = 1 << 14; //20

const NUM_PAGES: usize = 1;

const NUM_PAGES_2: usize = 4;

const NUM_PAGE_MAX: usize = 32;

const NUM_ITER: usize = 32;

struct BenchmarkStats {
    raw_res: Vec<CovertChannelBenchmarkResult>,
    average_p: f64,
    var_p: f64,
    average_C: f64,
    var_C: f64,
    average_T: f64,
    var_T: f64,
}

fn run_benchmark<T: CovertChannel + 'static>(
    name: &str,
    constructor: impl Fn(usize, usize) -> T,
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
                    let channel = constructor(i, j);
                    let r = benchmark_channel(channel, num_pages, NUM_BYTES);
                    results.push(r);
                }
            }
        }
    }
    println!();
    let mut average_p = 0.0;
    let mut average_C = 0.0;
    let mut average_T = 0.0;
    for result in results.iter() {
        println!("{:?}", result);
        println!("C: {}, T: {}", result.capacity(), result.true_capacity());
        println!(
            "Detailed:\"{}\",{},{},{},{}",
            name,
            num_pages,
            result.csv(),
            result.capacity(),
            result.true_capacity()
        );
        average_p += result.error_rate;
        average_C += result.capacity();
        average_T += result.true_capacity()
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
        let p = result.error_rate - average_p;
        var_p += p * p;
        let C = result.capacity() - average_C;
        var_C += C * C;
        let T = result.true_capacity() - average_T;
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
*/
fn main() {
    /*
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();
    println!(
        "Detailed:Benchmark,Pages,{},C,T",
        CovertChannelBenchmarkResult::csv_header()
    );
    println!("CSV:Benchmark,Pages,p,C,T,var_p,var_C,var_T");

    for num_pages in 1..=32 {
        let naive_ff = run_benchmark(
            "Naive F+F",
            |i, j| {
                let mut r = NaiveFlushAndFlush::from_threshold(202);
                r.set_cores(i, j);
                r
            },
            NUM_ITER,
            num_pages,
            old,
        );

        let fr = run_benchmark(
            "F+R",
            |i, j| {
                let mut r = NaiveFlushAndReload::from_threshold(250);
                r.set_cores(i, j);
                r
            },
            NUM_ITER,
            num_pages,
            old,
        );
    }

     */
}
