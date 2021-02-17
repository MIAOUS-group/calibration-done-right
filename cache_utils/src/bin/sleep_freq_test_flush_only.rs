use cache_utils::calibration::only_flush;
use cache_utils::frequency::get_freq_cpufreq_kernel;
use cache_utils::rdtsc_fence;
use static_assertions::_core::time::Duration;
use std::thread::sleep;

const SAMPLE_BETWEEN_SLEEP: usize = 100;
const NUM_ITERATION: usize = 10;

fn main() {
    let sleep_time: Duration = Duration::new(1, 0);
    let p = Box::new(42u8);
    let pointer = p.as_ref() as *const u8;
    // preheat
    for _ in 0..SAMPLE_BETWEEN_SLEEP {
        unsafe { only_flush(pointer) };
    }

    let mut results = vec![(0u64, 0u64); SAMPLE_BETWEEN_SLEEP * NUM_ITERATION];
    let mut frequency_info: Vec<(u64, u64)> = vec![(0, 0); NUM_ITERATION];

    println!("CSV:Freq_Sample,Iteration,Sample,time,duration");
    println!("FREQ:Freq_Sample,Iteration,time,freq");

    for frequency_sample in 0..SAMPLE_BETWEEN_SLEEP {
        for i in 0..NUM_ITERATION {
            sleep(sleep_time);
            for j in 0..SAMPLE_BETWEEN_SLEEP {
                if j == frequency_sample {
                    let t = unsafe { rdtsc_fence() };
                    let f = get_freq_cpufreq_kernel();
                    frequency_info[i] = (t, f.ok().unwrap());
                }
                let t = unsafe { rdtsc_fence() };
                let d = unsafe { only_flush(pointer) };
                results[i * SAMPLE_BETWEEN_SLEEP + j] = (t, d);
            }
        }
        for i in 0..NUM_ITERATION {
            for j in 0..SAMPLE_BETWEEN_SLEEP {
                if j == frequency_sample {
                    println!(
                        "FREQ:{},{},{},{}",
                        frequency_sample, i, frequency_info[i].0, frequency_info[i].1
                    );
                }
                println!(
                    "CSV:{},{},{},{},{}",
                    frequency_sample,
                    i,
                    j,
                    results[i * SAMPLE_BETWEEN_SLEEP + j].0,
                    results[i * SAMPLE_BETWEEN_SLEEP + j].1
                );
            }
        }
    }
}
