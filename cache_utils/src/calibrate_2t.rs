use crate::calibration::Verbosity::{RawResult, Thresholds};
use crate::calibration::{
    get_cache_slicing, get_vpn, CalibrateResult, CalibrationOptions, HashMap, ASVP,
    SPURIOUS_THRESHOLD,
};
use core::arch::x86_64 as arch_x86;
use itertools::Itertools;
use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use nix::Error;
use std::cmp::min;
use std::ptr::null_mut;
//use std::sync::atomic::Ordering;
use std::mem::forget;
use std::sync::{Arc, Mutex};
use std::thread;
use turn_lock::TurnHandle;

pub struct CalibrateOperation2T<'a, T> {
    pub prepare: unsafe fn(*const u8) -> (),
    pub op: unsafe fn(&T, *const u8) -> u64,
    pub name: &'a str,
    pub display_name: &'a str,
    pub t: &'a T,
}

pub struct CalibrateResult2T {
    pub main_core: usize,
    pub helper_core: usize,
    pub res: Result<Vec<CalibrateResult>, nix::Error>, // TODO

                                                       // TODO
}

pub unsafe fn calibrate_fixed_freq_2_thread<I: Iterator<Item = (usize, usize)>, T>(
    p: *const u8,
    increment: usize,
    len: isize,
    cores: &mut I,
    operations: &[CalibrateOperation2T<T>],
    options: CalibrationOptions,
    core_per_socket: u8,
) -> Vec<CalibrateResult2T> {
    calibrate_fixed_freq_2_thread_impl(
        p,
        increment,
        len,
        cores,
        operations,
        options,
        core_per_socket,
    )
}

const OPTIMISED_ADDR_ITER_FACTOR: u32 = 16;

struct HelperThreadParams {
    stop: bool,
    op: unsafe fn(*const u8),
    address: *const u8,
}

// TODO : Add the optimised address support
// TODO : Modularisation / factorisation of some of the common code with the single threaded no_std version ?

#[cfg(feature = "use_std")]
fn calibrate_fixed_freq_2_thread_impl<I: Iterator<Item = (usize, usize)>, T>(
    p: *const u8,
    increment: usize,
    len: isize,
    cores: &mut I,
    operations: &[CalibrateOperation2T<T>],
    mut options: CalibrationOptions,
    core_per_socket: u8,
) -> Vec<CalibrateResult2T> {
    if options.verbosity >= Thresholds {
        println!(
            "Calibrating {}...",
            operations
                .iter()
                .map(|operation| { operation.display_name })
                .format(", ")
        );
    }

    let bucket_size = options.hist_params.bucket_size;

    let to_bucket = |time: u64| -> usize { time as usize / bucket_size };
    let from_bucket = |bucket: usize| -> u64 { (bucket * bucket_size) as u64 };

    let slicing = get_cache_slicing(core_per_socket);

    let h = if let Some(s) = slicing {
        if s.can_hash() {
            Some(|addr: usize| -> u8 { slicing.unwrap().hash(addr).unwrap() })
        } else {
            None
        }
    } else {
        None
    };

    let mut ret = Vec::new();

    let mut turn_handles = TurnHandle::new(
        2,
        HelperThreadParams {
            stop: true,
            op: operations[0].prepare,
            address: null_mut(),
        },
    );

    let mut helper_turn_handle = Arc::new(Mutex::new(turn_handles.pop().unwrap()));
    let mut main_turn_handle = turn_handles.pop().unwrap();

    let mut params = main_turn_handle.wait();

    if options.verbosity >= Thresholds {
        print!("CSV: main_core, helper_core, address, ");
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

    if options.verbosity >= RawResult {
        print!("RESULT:main_core,helper_core,address,");
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

    let image_antecedent = match slicing {
        Some(s) => s.image_antecedent(len as usize - 1),
        None => None,
    };
    if image_antecedent.is_some() {
        options.hist_params.iterations *= OPTIMISED_ADDR_ITER_FACTOR;
    }

    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    for (main_core, helper_core) in cores {
        // set main thread affinity

        if options.verbosity >= Thresholds {
            println!(
                "Calibration for main_core {}, helper {}.",
                main_core, helper_core
            );

            eprintln!(
                "Calibration for main_core {}, helper {}.",
                main_core, helper_core
            );
        }

        let mut core = CpuSet::new();
        match core.set(main_core) {
            Ok(_) => {}
            Err(e) => {
                ret.push(CalibrateResult2T {
                    main_core,
                    helper_core,
                    res: Err(e),
                });
                continue;
            }
        }

        match sched_setaffinity(Pid::from_raw(0), &core) {
            Ok(_) => {}
            Err(e) => {
                ret.push(CalibrateResult2T {
                    main_core,
                    helper_core,
                    res: Err(e),
                });
                continue;
            }
        }

        let helper_thread = if helper_core != main_core {
            params.stop = false;
            // set up the helper thread

            let hc = helper_core;
            let th = helper_turn_handle.clone();
            Some(thread::spawn(move || {
                calibrate_fixed_freq_2_thread_helper(th, hc)
            }))
        } else {
            None
        };
        // do the calibration
        let mut calibrate_result_vec = Vec::new();

        let offsets: Box<dyn Iterator<Item = isize>> = match image_antecedent {
            Some(ref ima) => Box::new(ima.values().copied()),
            None => Box::new((0..len as isize).step_by(increment)),
        };

        for i in offsets {
            let pointer = unsafe { p.offset(i) };
            params.address = pointer;

            let hash = h.map(|h| h(pointer as usize));

            if options.verbosity >= Thresholds {
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

            if helper_core != main_core {
                for op in operations {
                    params = main_turn_handle.wait();
                    params.op = op.prepare;
                    let mut hist = vec![0; options.hist_params.bucket_number];
                    for _ in 0..options.hist_params.iterations {
                        main_turn_handle.next();
                        params = main_turn_handle.wait();
                        let _time = unsafe { (op.op)(op.t, pointer) };
                    }
                    for _ in 0..options.hist_params.iterations {
                        main_turn_handle.next();
                        params = main_turn_handle.wait();
                        let time = unsafe { (op.op)(op.t, pointer) };
                        let bucket = min(options.hist_params.bucket_number - 1, to_bucket(time));
                        hist[bucket] += 1;
                    }
                    calibrate_result.histogram.push(hist);
                }
            } else {
                for op in operations {
                    let mut hist = vec![0; options.hist_params.bucket_number];
                    for _ in 0..options.hist_params.iterations {
                        unsafe { (op.prepare)(pointer) };
                        unsafe { arch_x86::_mm_mfence() }; // Test with this ?
                        let _time = unsafe { (op.op)(op.t, pointer) };
                    }
                    for _ in 0..options.hist_params.iterations {
                        unsafe { (op.prepare)(pointer) };
                        unsafe { arch_x86::_mm_mfence() }; // Test with this ?
                        let time = unsafe { (op.op)(op.t, pointer) };
                        let bucket = min(options.hist_params.bucket_number - 1, to_bucket(time));
                        hist[bucket] += 1;
                    }
                    calibrate_result.histogram.push(hist);
                }
            }
            let mut sums = vec![0; operations.len()];

            let median_thresholds: Vec<u32> = calibrate_result
                .histogram
                .iter()
                .map(|h| {
                    (options.hist_params.iterations - h[options.hist_params.bucket_number - 1]) / 2
                })
                .collect();

            for j in 0..options.hist_params.bucket_number - 1 {
                if options.verbosity >= RawResult {
                    print!("RESULT:{},{},{:p},", main_core, helper_core, pointer);
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
                    if options.verbosity >= RawResult {
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
                if options.verbosity >= RawResult {
                    println!();
                }
            }
            if options.verbosity >= Thresholds {
                for (j, op) in operations.iter().enumerate() {
                    println!(
                        "{}: min {}, median {}, max {}",
                        op.display_name,
                        calibrate_result.min[j],
                        calibrate_result.median[j],
                        calibrate_result.max[j]
                    );
                }
                print!("CSV: {},{},{:p}, ", main_core, helper_core, pointer);
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
            calibrate_result_vec.push(calibrate_result);
        }

        ret.push(CalibrateResult2T {
            main_core,
            helper_core,
            res: Ok(calibrate_result_vec),
        });

        if helper_core != main_core {
            // terminate the thread
            params.stop = true;
            main_turn_handle.next();
            params = main_turn_handle.wait();
            // join thread.
            helper_thread.unwrap().join();
            // FIXME error handling
        }
    }

    sched_setaffinity(Pid::from_raw(0), &old).unwrap();

    ret
    // return the result
    // TODO
}

fn calibrate_fixed_freq_2_thread_helper(
    turn_handle: Arc<Mutex<TurnHandle<HelperThreadParams>>>,
    helper_core: usize,
) -> Result<(), Error> {
    let mut turn_handle = turn_handle.lock().unwrap();
    // set thread affinity
    let mut core = CpuSet::new();
    match core.set(helper_core) {
        Ok(_) => {}
        Err(e) => {
            let mut params = turn_handle.wait();
            params.stop = true;
            turn_handle.next();
            return Err(e);
        }
    }

    match sched_setaffinity(Pid::from_raw(0), &core) {
        Ok(_) => {}
        Err(e) => {
            let mut params = turn_handle.wait();
            params.stop = true;
            turn_handle.next();
            return Err(e);
        }
    }

    loop {
        // grab lock
        let params = turn_handle.wait();
        if params.stop {
            turn_handle.next();
            return Ok(());
        }
        // get the relevant parameters
        let addr: *const u8 = params.address;
        let op = params.op;
        unsafe { op(addr) };
        // release lock
        turn_handle.next();
    }
}

// ------------------- Analysis ------------------

pub fn calibration_result_to_ASVP<T, Analysis: Fn(CalibrateResult) -> T>(
    results: Vec<CalibrateResult2T>,
    analysis: Analysis,
    slicing: &impl Fn(usize) -> u8,
) -> Result<HashMap<ASVP, T>, nix::Error> {
    let mut analysis_result: HashMap<ASVP, T> = HashMap::new();
    for calibrate_2t_result in results {
        let attacker = calibrate_2t_result.main_core;
        let victim = calibrate_2t_result.helper_core;
        match calibrate_2t_result.res {
            Err(e) => return Err(e),
            Ok(calibrate_1t_results) => {
                for result_1t in calibrate_1t_results {
                    let offset = result_1t.offset;
                    let page = result_1t.page;
                    let addr = page + offset as usize;
                    let slice = slicing(addr as usize);
                    let analysed = analysis(result_1t);
                    let asvp = ASVP {
                        attacker,
                        slice,
                        victim,
                        page,
                    };
                    analysis_result.insert(asvp, analysed);
                }
            }
        }
    }
    Ok(analysis_result)
}
