# calibration-done-right
Public code for the calibration done right paper

The code base is written entirely in rust.

There are three crates with binaries providing results :
- **aes-t-tables** runs the T-table attack using the 3 side channels
- **cache_utils** `two_thread_cal` runs a full calibration on all core pair for Flush+Flush
and analyses the results to provide error rate predictions in various attacker models
- **covert_channel_beanchmark** is the crate that runs covert channel benchmarks on the various covert channels

The code presented runs under Fedora 30, and can also be made run on Ubuntu 18.04 LTS with minor tweaks

(Notably lib cpupower may also be called libcpufreq)

# Usage

## General set-up

Requires rust nightly features. Install rust nightly using rustup

One should disable turbo boost and other source of idle frequency scaling

Depending on the experiment you may be interested in disabling prefetchers.

## Two thread calibration set-up and usage

In addition to the general set-up you need to enable 2MB hugepage and ensure at least one is available.

Then you can run `cargo run --release --bin two_thread_cal > result.log`

Various scripts are also included that have been used to parse the log.

`analyse.sh` -> `analyse_csv.py` -> `analyse_median.py` Is used to analyse the timing histograms
`extract_analysis_csv.sh` Is used to extract the attacker model results.

The python scripts requires an environment (such as a virtual env) with the packages in `cache_utils/requirements.txt`

## AES T-table set-up and usage

One needs an OpenSSL built with the no-asm and the no-hw flags install in ~/openssl (the path is in aes-t-tables/cargo.sh and can be changed).

You the need to extract the T-table addresses, this can be done using `nm libcrypto.so | "grep Te[0-4]"`, and update those in aes-t-tables/src/main.rs

You'll also want to update the thresholds in main.rs using the results from the calibration.

You can then run `./cargo.sh run --release > result.log`


## Covert Channel benchmark

Do the general set-up, update the thresholds for Naive channels in main.rs and then run `cargo run --release | tee results.log`


# Crate documentation

- `cpuid` is a small crate that handles CPU microarchitecture indentification and provides info about what is known about it
- `cache_utils` contains utilities related to cache attacks
- `cache_side_channel` defines the interface cache side channels have to implement
- `basic_timing_cache_channel` contains generic implementations of Naive and Optimised cache side channels, that just require providing the actual operation used
- `flush_flush` and `flush_reload` are tiny crates that use `basic_timing_cache_channel` to export Flush+Flush and Flush+Reload primitives
- `turn_lock` is the synchronisation primitive used by `cache_utils` and the `covert_channel_evaluation`.
- `covert_channel_evaluation` is a generic implementation of a `covert_channel` benchmark
- `covert_channel_benchmark` calls the previous implementation over the 3 channels.
