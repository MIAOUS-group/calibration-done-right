use cache_utils::calibration::{
    accumulate, calibrate_fixed_freq_2_thread, calibration_result_to_ASVP, flush_and_reload,
    get_cache_slicing, load_and_flush, map_values, only_flush, only_reload, reduce,
    reload_and_flush, CalibrateOperation2T, CalibrateResult2T, CalibrationOptions, ErrorPrediction,
    ErrorPredictions, HistParams, HistogramCumSum, PotentialThresholds, ThresholdError, Verbosity,
    ASP, ASVP, AV, CFLUSH_BUCKET_NUMBER, CFLUSH_BUCKET_SIZE, CFLUSH_NUM_ITER, SP, SVP,
};
use cache_utils::mmap::MMappedMemory;
use cache_utils::{flush, maccess, noop};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;

use core::arch::x86_64 as arch_x86;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::process::Command;
use std::str::from_utf8;

unsafe fn multiple_access(p: *const u8) {
    maccess::<u8>(p);
    maccess::<u8>(p);
    arch_x86::_mm_mfence();
    maccess::<u8>(p);
    arch_x86::_mm_mfence();
    maccess::<u8>(p);
    arch_x86::_mm_mfence();
    maccess::<u8>(p);
    maccess::<u8>(p);
}

const SIZE: usize = 2 << 20;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
struct ASV {
    pub attacker: u8,
    pub slice: u8,
    pub victim: u8,
}

struct ResultAnalysis {
    // indexed by bucket size
    pub miss: Vec<u32>,
    pub miss_cum_sum: Vec<u32>,
    pub miss_total: u32,
    pub hit: Vec<u32>,
    pub hit_cum_sum: Vec<u32>,
    pub hit_total: u32,
    pub error_miss_less_than_hit: Vec<u32>,
    pub error_hit_less_than_miss: Vec<u32>,
    pub min_error_hlm: u32,
    pub min_error_mlh: u32,
}

// Split the threshold and error in two separate structs ?

#[derive(Debug, Clone, Copy)]
struct Threshold {
    pub error_rate: f32,
    pub threshold: usize,
    // extend with other possible algorithm ?
    pub is_hlm: bool,
    pub num_true_hit: u32,
    pub num_false_hit: u32,
    pub num_true_miss: u32,
    pub num_false_miss: u32,
}

unsafe fn only_flush_wrap(_: &(), addr: *const u8) -> u64 {
    unsafe { only_flush(addr) }
}

unsafe fn only_reload_wrap(_: &(), addr: *const u8) -> u64 {
    unsafe { only_reload(addr) }
}

unsafe fn load_and_flush_wrap(_: &(), addr: *const u8) -> u64 {
    unsafe { load_and_flush(addr) }
}
unsafe fn flush_and_reload_wrap(_: &(), addr: *const u8) -> u64 {
    unsafe { flush_and_reload(addr) }
}

unsafe fn reload_and_flush_wrap(_: &(), addr: *const u8) -> u64 {
    unsafe { reload_and_flush(addr) }
}

fn main() {
    // Grab a slice of memory

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

    println!("Number of cores per socket: {}", core_per_socket);

    let m = MMappedMemory::new(SIZE, true);
    let array = m.slice();

    let cache_line_size = 64;

    // Generate core iterator
    let mut core_pairs: Vec<(usize, usize)> = Vec::new();
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    for i in 0..CpuSet::count() {
        for j in 0..CpuSet::count() {
            if old.is_set(i).unwrap() && old.is_set(j).unwrap() {
                core_pairs.push((i, j));
                println!("{},{}", i, j);
            }
        }
    }

    // operations
    // Call calibrate 2T \o/

    let verbose_level = Verbosity::RawResult;

    let pointer = (&array[0]) as *const u8;
    if pointer as usize & (cache_line_size - 1) != 0 {
        panic!("not aligned nicely");
    }

    let operations = [
        CalibrateOperation2T {
            prepare: maccess::<u8>,
            op: only_flush_wrap,
            name: "clflush_remote_hit",
            display_name: "clflush remote hit",
            t: &(),
        },
        CalibrateOperation2T {
            prepare: maccess::<u8>,
            op: load_and_flush_wrap,
            name: "clflush_shared_hit",
            display_name: "clflush shared hit",
            t: &(),
        },
        CalibrateOperation2T {
            prepare: flush,
            op: only_flush_wrap,
            name: "clflush_miss_f",
            display_name: "clflush miss - f",
            t: &(),
        },
        CalibrateOperation2T {
            prepare: flush,
            op: load_and_flush_wrap,
            name: "clflush_local_hit_f",
            display_name: "clflush local hit - f",
            t: &(),
        },
        CalibrateOperation2T {
            prepare: noop::<u8>,
            op: only_flush_wrap,
            name: "clflush_miss_n",
            display_name: "clflush miss - n",
            t: &(),
        },
        CalibrateOperation2T {
            prepare: noop::<u8>,
            op: load_and_flush_wrap,
            name: "clflush_local_hit_n",
            display_name: "clflush local hit - n",
            t: &(),
        },
        CalibrateOperation2T {
            prepare: noop::<u8>,
            op: flush_and_reload_wrap,
            name: "reload_miss",
            display_name: "reload miss",
            t: &(),
        },
        CalibrateOperation2T {
            prepare: maccess::<u8>,
            op: reload_and_flush_wrap,
            name: "reload_remote_hit",
            display_name: "reload remote hit",
            t: &(),
        },
        CalibrateOperation2T {
            prepare: maccess::<u8>,
            op: only_reload_wrap,
            name: "reload_shared_hit",
            display_name: "reload shared hit",
            t: &(),
        },
        CalibrateOperation2T {
            prepare: noop::<u8>,
            op: only_reload_wrap,
            name: "reload_local_hit",
            display_name: "reload local hit",
            t: &(),
        },
    ];

    let r = unsafe {
        calibrate_fixed_freq_2_thread(
            pointer,
            64,                        // FIXME : MAGIC
            array.len() as isize >> 3, // MAGIC
            &mut core_pairs.into_iter(),
            &operations,
            CalibrationOptions {
                hist_params: HistParams {
                    bucket_number: CFLUSH_BUCKET_NUMBER,
                    bucket_size: CFLUSH_BUCKET_SIZE,
                    iterations: CFLUSH_NUM_ITER << 4,
                },
                verbosity: verbose_level,
                optimised_addresses: true,
            },
            core_per_socket,
        )
    };

    //let mut analysis = HashMap::<ASV, ResultAnalysis>::new();

    let miss_name = "clflush_miss_n";
    let hit_name = "clflush_remote_hit";

    let miss_index = operations
        .iter()
        .position(|op| op.name == miss_name)
        .unwrap();
    let hit_index = operations
        .iter()
        .position(|op| op.name == hit_name)
        .unwrap();

    let slicing = get_cache_slicing(core_per_socket);

    let h = if let Some(s) = slicing {
        if s.can_hash() {
            |addr: usize| -> u8 { slicing.unwrap().hash(addr).unwrap() }
        } else {
            panic!("No slicing function known");
        }
    } else {
        panic!("No slicing function known");
    };

    /* Analysis Flow
        Vec<CalibrationResult2T> (or Vec<CalibrationResult>) -> Corresponding ASVP + Analysis (use the type from two_thread_cal, or similar)
        ASVP,Analysis -> ASVP,Thresholds,Error
        ASVP,Analysis -> ASP,Analysis (mobile victim) -> ASP, Threshold, Error -> ASVP detailed Threshold,Error in ASP model
        ASVP,Analysis -> SP, Analysis (mobile A and V) -> SP, Threshold, Error -> ASVP detailed Threshold,Error in SP model
        ASVP,Analysis -> AV, Analysis (legacy attack)  -> AV, Threshold, Error -> ASVP detailed Threshold,Error in AV model
        ASVP,Analysis -> Global Analysis            -> Global Threshold, Error -> ASVP detailed Threshold,Error in Global Model
        The last step is done as a apply operation on original ASVP Analysis using the new Thresholds.

        This model correspond to an attacker that can chose its core and its victim core, and has slice knowledge
        ASVP,Thresholds,Error -> Best AV selection for average error. HashMap<AV,(ErrorPrediction,HashMap<ASVP,Threshold,Error>)>

        This model corresponds to an attacker that can chose its own core, measure victim location, and has slice knowledge.
        ASVP,Thresholds,Error -> Best A  selection for average error. HashMap<AV,(ErrorPrediction,HashMap<ASVP,Threshold,Error>)>

        Also compute best AV pair for AV model

        What about chosing A but no knowing V at all, from ASP detiled analysis ?




        Compute for each model averages, worst and best cases ?

    */

    let new_analysis: Result<HashMap<ASVP, ErrorPredictions>, nix::Error> =
        calibration_result_to_ASVP(
            r,
            |cal_1t_res| {
                ErrorPredictions::predict_errors(HistogramCumSum::from_calibrate(
                    cal_1t_res, hit_index, miss_index,
                ))
            },
            &h,
        );

    // Analysis aka HashMap<subset of ASVP, ErrorPredictions> --------------------------------------

    let asvp_analysis = match new_analysis {
        Ok(a) => a,
        Err(e) => panic!("Error: {}", e),
    };

    asvp_analysis[&ASVP {
        attacker: 0,
        slice: 0,
        victim: 0,
        page: pointer as usize,
    }]
        .debug();

    let asp_analysis = accumulate(
        asvp_analysis.clone(),
        |asvp: ASVP| ASP {
            attacker: asvp.attacker,
            slice: asvp.slice,
            page: asvp.page,
        },
        || ErrorPredictions::empty(CFLUSH_BUCKET_NUMBER),
        |accumulator: &mut ErrorPredictions, error_preds: ErrorPredictions, _key, _rkey| {
            *accumulator += error_preds;
        },
    );

    let sp_analysis = accumulate(
        asp_analysis.clone(),
        |asp: ASP| SP {
            slice: asp.slice,
            page: asp.page,
        },
        || ErrorPredictions::empty(CFLUSH_BUCKET_NUMBER),
        |accumulator: &mut ErrorPredictions, error_preds: ErrorPredictions, _key, _rkey| {
            *accumulator += error_preds;
        },
    );

    // This one is the what would happen if you ignored slices
    let av_analysis = accumulate(
        asvp_analysis.clone(),
        |asvp: ASVP| AV {
            attacker: asvp.attacker,
            victim: asvp.victim,
        },
        || ErrorPredictions::empty(CFLUSH_BUCKET_NUMBER),
        |accumulator: &mut ErrorPredictions, error_preds: ErrorPredictions, _key, _rkey| {
            *accumulator += error_preds;
        },
    );

    let global_analysis = accumulate(
        av_analysis.clone(),
        |_av: AV| (),
        || ErrorPredictions::empty(CFLUSH_BUCKET_NUMBER),
        |accumulator: &mut ErrorPredictions, error_preds: ErrorPredictions, _key, _rkey| {
            *accumulator += error_preds;
        },
    )
    .remove(&())
    .unwrap();

    // Thresholds aka HashMap<subset of ASVP,ThresholdError> ---------------------------------------

    let asvp_threshold_errors: HashMap<ASVP, ThresholdError> = map_values(
        asvp_analysis.clone(),
        |error_predictions: ErrorPredictions, _| {
            PotentialThresholds::minimizing_total_error(error_predictions)
                .median()
                .unwrap()
        },
    );

    let asp_threshold_errors =
        map_values(asp_analysis, |error_predictions: ErrorPredictions, _| {
            PotentialThresholds::minimizing_total_error(error_predictions)
                .median()
                .unwrap()
        });

    let sp_threshold_errors = map_values(sp_analysis, |error_predictions: ErrorPredictions, _| {
        PotentialThresholds::minimizing_total_error(error_predictions)
            .median()
            .unwrap()
    });

    let av_threshold_errors = map_values(av_analysis, |error_predictions: ErrorPredictions, _| {
        PotentialThresholds::minimizing_total_error(error_predictions)
            .median()
            .unwrap()
    });

    let gt_threshold_error = PotentialThresholds::minimizing_total_error(global_analysis)
        .median()
        .unwrap();

    // ASVP detailed Threshold,Error in strict subset of ASVP model --------------------------------
    // HashMap<ASVP, (Thershold ?)Error>,
    // with the same threshold for all the ASVP sharing the same value of an ASVP subset.

    let asp_detailed_errors: HashMap<ASVP, ThresholdError> = map_values(
        asvp_analysis.clone(),
        |error_pred: ErrorPredictions, asvp: &ASVP| {
            let asp = ASP {
                attacker: asvp.attacker,
                slice: asvp.slice,
                page: asvp.page,
            };
            let threshold = asp_threshold_errors[&asp].threshold;
            let error = error_pred.histogram.error_for_threshold(threshold);
            ThresholdError { threshold, error }
        },
    );

    let sp_detailed_errors: HashMap<ASVP, ThresholdError> = map_values(
        asvp_analysis.clone(),
        |error_pred: ErrorPredictions, asvp: &ASVP| {
            let sp = SP {
                slice: asvp.slice,
                page: asvp.page,
            };
            let threshold = sp_threshold_errors[&sp].threshold;
            let error = error_pred.histogram.error_for_threshold(threshold);
            ThresholdError { threshold, error }
        },
    );

    let av_detailed_errors: HashMap<ASVP, ThresholdError> = map_values(
        asvp_analysis.clone(),
        |error_pred: ErrorPredictions, asvp: &ASVP| {
            let av = AV {
                attacker: asvp.attacker,
                victim: asvp.victim,
            };
            let threshold = av_threshold_errors[&av].threshold;
            let error = error_pred.histogram.error_for_threshold(threshold);
            ThresholdError { threshold, error }
        },
    );

    let gt_detailed_errors: HashMap<ASVP, ThresholdError> =
        map_values(asvp_analysis.clone(), |error_pred: ErrorPredictions, _| {
            let threshold = gt_threshold_error.threshold;
            let error = error_pred.histogram.error_for_threshold(threshold);
            ThresholdError { threshold, error }
        });

    // Best core selections

    let asvp_best_av_errors: HashMap<AV, (ErrorPrediction, HashMap<SP, ThresholdError>)> =
        accumulate(
            asvp_threshold_errors.clone(),
            |asvp: ASVP| AV {
                attacker: asvp.attacker,
                victim: asvp.victim,
            },
            || (ErrorPrediction::default(), HashMap::new()),
            |acc: &mut (ErrorPrediction, HashMap<SP, ThresholdError>),
             threshold_error,
             asvp: ASVP,
             av| {
                assert_eq!(av.attacker, asvp.attacker);
                assert_eq!(av.victim, asvp.victim);
                let sp = SP {
                    slice: asvp.slice,
                    page: asvp.page,
                };
                acc.0 += threshold_error.error;
                acc.1.insert(sp, threshold_error);
            },
        );

    let asvp_best_a_errors: HashMap<usize, (ErrorPrediction, HashMap<SVP, ThresholdError>)> =
        accumulate(
            asvp_threshold_errors.clone(),
            |asvp: ASVP| asvp.attacker,
            || (ErrorPrediction::default(), HashMap::new()),
            |acc: &mut (ErrorPrediction, HashMap<SVP, ThresholdError>),
             threshold_error,
             asvp: ASVP,
             attacker| {
                assert_eq!(attacker, asvp.attacker);
                let svp = SVP {
                    slice: asvp.slice,
                    page: asvp.page,
                    victim: asvp.victim,
                };
                acc.0 += threshold_error.error;
                acc.1.insert(svp, threshold_error);
            },
        );

    let asp_best_a_errors: HashMap<usize, (ErrorPrediction, HashMap<SVP, ThresholdError>)> =
        accumulate(
            asp_detailed_errors.clone(),
            |asvp: ASVP| asvp.attacker,
            || (ErrorPrediction::default(), HashMap::new()),
            |acc: &mut (ErrorPrediction, HashMap<SVP, ThresholdError>),
             threshold_error,
             asvp: ASVP,
             attacker| {
                assert_eq!(attacker, asvp.attacker);
                let svp = SVP {
                    slice: asvp.slice,
                    page: asvp.page,
                    victim: asvp.victim,
                };
                acc.0 += threshold_error.error;
                acc.1.insert(svp, threshold_error);
            },
        );

    //let av_best_av_errors
    let av_best_a_erros: HashMap<usize, (ErrorPrediction, HashMap<SVP, ThresholdError>)> =
        accumulate(
            av_detailed_errors.clone(),
            |asvp: ASVP| asvp.attacker,
            || (ErrorPrediction::default(), HashMap::new()),
            |acc: &mut (ErrorPrediction, HashMap<SVP, ThresholdError>),
             threshold_error,
             asvp: ASVP,
             attacker| {
                assert_eq!(attacker, asvp.attacker);
                let svp = SVP {
                    slice: asvp.slice,
                    page: asvp.page,
                    victim: asvp.victim,
                };
                acc.0 += threshold_error.error;
                acc.1.insert(svp, threshold_error);
            },
        );

    // Find best index in each model...

    // CSV output logic

    /* moving parts :
       - order of lines
       - columns and columns header.
       - Probably should be a macro ?
       Or something taking a Vec of Column and getter, plus a vec (or iterator) of 'Keys'
    */

    let mut keys = asvp_threshold_errors.keys().collect::<Vec<&ASVP>>();
    keys.sort_unstable_by(|a: &&ASVP, b: &&ASVP| {
        if a.page > b.page {
            Ordering::Greater
        } else if a.page < b.page {
            Ordering::Less
        } else if a.slice > b.slice {
            Ordering::Greater
        } else if a.slice < b.slice {
            Ordering::Less
        } else if a.attacker > b.attacker {
            Ordering::Greater
        } else if a.attacker < b.attacker {
            Ordering::Less
        } else if a.victim > b.victim {
            Ordering::Greater
        } else if a.victim < b.victim {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    });

    // In theory there should be a way of making such code much more modular.

    let error_header = |name: &str| {
        format!(
            "{}ErrorRate,{}Errors,{}Measures,{}TrueHit,{}TrueMiss,{}FalseHit,{}FalseMiss",
            name, name, name, name, name, name, name
        )
    };

    let header = |name: &str| {
        format!(
            "{}_Threshold,{}_MFH,{}_GlobalErrorRate,{}",
            name,
            name,
            name,
            error_header(&format!("{}_ASVP", name))
        )
    };

    println!(
        "Analysis:Page,Slice,Attacker,Victim,ASVP_Threshold,ASVP_MFH,{},{},{},{},{}",
        error_header("ASVP_"),
        header("ASP"),
        header("SP"),
        header("AV"),
        header("GT")
    );

    let format_error = |error_pred: &ErrorPrediction| {
        format!(
            "{},{},{},{},{},{},{}",
            error_pred.error_rate(),
            error_pred.total_error(),
            error_pred.total(),
            error_pred.true_hit,
            error_pred.true_miss,
            error_pred.false_hit,
            error_pred.false_miss
        )
    };

    let format_detailed_model = |global: &ThresholdError, detailed: &ThresholdError| {
        assert_eq!(global.threshold, detailed.threshold);
        format!(
            "{},{},{},{}",
            global.threshold.bucket_index,
            global.threshold.miss_faster_than_hit,
            global.error.error_rate(),
            format_error(&detailed.error)
        )
    };

    for key in keys {
        print!(
            "Analysis:{},{},{},{},",
            key.page, key.slice, key.attacker, key.victim
        );
        let threshold_error = asvp_threshold_errors[key];
        print!(
            "{},{},{},",
            threshold_error.threshold.bucket_index,
            threshold_error.threshold.miss_faster_than_hit,
            format_error(&threshold_error.error)
        );

        let asp_global = &asp_threshold_errors[&ASP {
            attacker: key.attacker,
            slice: key.slice,
            page: key.page,
        }];
        let asp_detailed = &asp_detailed_errors[key];
        print!("{},", format_detailed_model(asp_global, asp_detailed));

        let sp_global = &sp_threshold_errors[&SP {
            slice: key.slice,
            page: key.page,
        }];
        let sp_detailed = &sp_detailed_errors[key];
        print!("{},", format_detailed_model(sp_global, sp_detailed));

        let av_global = &av_threshold_errors[&AV {
            attacker: key.attacker,
            victim: key.victim,
        }];
        let av_detailed = &av_detailed_errors[key];
        print!("{},", format_detailed_model(av_global, av_detailed));

        let gt_global = &gt_threshold_error;
        let gt_detailed = &gt_detailed_errors[key];
        print!("{},", format_detailed_model(gt_global, gt_detailed));
        println!();
    }

    //The two other CSV are summaries that allowdetermining the best case. Index in the first CSV for the detailed info.
    // Second CSV output logic:

    // Build keys
    let mut keys = asvp_best_av_errors.keys().collect::<Vec<&AV>>();
    keys.sort_unstable_by(|a: &&AV, b: &&AV| {
        if a.attacker > b.attacker {
            Ordering::Greater
        } else if a.attacker < b.attacker {
            Ordering::Less
        } else if a.victim > b.victim {
            Ordering::Greater
        } else if a.victim < b.victim {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    });

    // Print header
    println!(
        "AVAnalysis:Attacker,Victim,{},{}",
        error_header("AVSP_AVAverage_"),
        error_header("AV_AVAverage_")
    );
    //print lines

    for av in keys {
        println!(
            "AVAnalysis:{attacker},{victim},{AVSP},{AV}",
            attacker = av.attacker,
            victim = av.victim,
            AVSP = format_error(&asvp_best_av_errors[av].0),
            AV = format_error(&av_threshold_errors[av].error),
        );
    }

    // Third CSV output logic:

    // Build keys
    let mut keys = asvp_best_a_errors.keys().collect::<Vec<&usize>>();
    keys.sort_unstable();

    println!(
        "AttackerAnalysis:Attacker,{},{},{}",
        error_header("AVSP_AAverage_"),
        error_header("ASP_AAverage_"),
        error_header("AV_AAverage_"),
    );

    for attacker in keys {
        println!(
            "AttackerAnalysis:{attacker},{AVSP},{ASP},{AV}",
            attacker = attacker,
            AVSP = format_error(&asvp_best_a_errors[&attacker].0),
            ASP = format_error(&asp_best_a_errors[&attacker].0),
            AV = format_error(&av_best_a_erros[&attacker].0)
        );
    }
}
