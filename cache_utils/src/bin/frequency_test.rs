use cache_utils::frequency::get_freq_cpufreq_kernel;
use cache_utils::rdtsc_fence;
use libc::sched_getcpu;
use nix::sched::{sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use static_assertions::_core::time::Duration;
use std::thread::sleep;
use std::time::Instant;

const NUM_SAMPLE: i32 = 1000000;
const NUM_SAMPLE_SLEEP: i32 = 10000;

pub fn main() {
    let mut core = CpuSet::new();
    core.set(unsafe { sched_getcpu() } as usize).unwrap();

    sched_setaffinity(Pid::from_raw(0), &core).unwrap();
    let t0_pre = unsafe { rdtsc_fence() };
    let start = Instant::now();
    let t0_post = unsafe { rdtsc_fence() };

    let mut tsc = t0_post;

    println!("TSC,Freq");
    for _ in 0..NUM_SAMPLE {
        //let t1 = unsafe { rdtsc_fence() };
        let frequency = get_freq_cpufreq_kernel();
        let t2 = unsafe { rdtsc_fence() };
        let delta = t2 - tsc;
        tsc = t2;
        if let Ok(freq) = frequency {
            println!("{},{}", delta, freq);
        }
    }
    println!("Idling");
    for _ in 0..NUM_SAMPLE_SLEEP {
        sleep(Duration::from_micros(1000));
        let frequency = get_freq_cpufreq_kernel();
        let t2 = unsafe { rdtsc_fence() };
        let delta = t2 - tsc;
        tsc = t2;
        if let Ok(freq) = frequency {
            println!("{},{}", delta, freq);
        }
    }

    let tf_pre = unsafe { rdtsc_fence() };
    let elapsed = start.elapsed();
    let tf_post = unsafe { rdtsc_fence() };
    println!(
        "Time elapsed: {} us, number of tsc tick: {} - {} - {}",
        elapsed.as_micros(),
        tf_pre - t0_pre,
        (tf_pre - t0_pre + tf_post - t0_post) / 2,
        tf_post - t0_post
    );
    eprintln!(
        "Time elapsed: {} us, number of tsc tick: {} - {} - {}",
        elapsed.as_micros(),
        tf_pre - t0_pre,
        (tf_pre - t0_pre + tf_post - t0_post) / 2,
        tf_post - t0_post
    );
}
