#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]
use turn_lock::TurnHandle;

const PAGE_SIZE: usize = 1 << 12; // FIXME Magic

// design docs

// Build a channel using x pages + one synchronisation primitive.

// F+R only use one line per page
// F+F should use several line per page
// Each page has 1<<12 bytes / 1<<6 bytes per line, hence 64 lines (or 6 bits of info).

// General structure : two threads, a transmitter and a reciever. Transmitter generates bytes, Reciever reads bytes, then on join compare results for accuracy.
// Alos time in order to determine duration, in rdtsc and seconds.

use bit_field::BitField;
use cache_side_channel::{restore_affinity, set_affinity, CoreSpec};
use cache_utils::mmap::MMappedMemory;
use cache_utils::rdtsc_fence;
use nix::sched::sched_getaffinity;
use nix::unistd::Pid;
use std::any::Any;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::Arc;
use std::thread;

/*  TODO : replace page with a handle type,
    require exclusive handle access,
    Handle protected by the turn lock
*/
/**
 * Safety considerations : Not ensure thread safety, need proper locking as needed.
 */
pub trait CovertChannel: Send + Sync + CoreSpec + Debug {
    type Handle;
    const BIT_PER_PAGE: usize;
    unsafe fn transmit(&self, handle: &mut Self::Handle, bits: &mut BitIterator);
    unsafe fn receive(&self, handle: &mut Self::Handle) -> Vec<bool>;
    unsafe fn ready_page(&mut self, page: *const u8) -> Result<Self::Handle, ()>; // TODO Error Type
}

#[derive(Debug)]
pub struct CovertChannelBenchmarkResult {
    pub num_bytes_transmitted: usize,
    pub num_bit_errors: usize,
    pub error_rate: f64,
    pub time_rdtsc: u64,
    pub time_seconds: std::time::Duration,
}

impl CovertChannelBenchmarkResult {
    pub fn capacity(&self) -> f64 {
        (self.num_bytes_transmitted * 8) as f64 / self.time_seconds.as_secs_f64()
    }

    pub fn true_capacity(&self) -> f64 {
        let p = self.error_rate;
        if p == 0.0 || p == 0.0 {
            self.capacity()
        } else {
            self.capacity() * (1.0 + ((1.0 - p) * f64::log2(1.0 - p) + p * f64::log2(p)))
        }
    }

    pub fn csv(&self) -> String {
        format!(
            "{},{},{},{},{}",
            self.num_bytes_transmitted,
            self.num_bit_errors,
            self.error_rate,
            self.time_rdtsc,
            self.time_seconds.as_nanos()
        )
    }

    pub fn csv_header() -> String {
        format!("bytes_transmitted,bits_error,error_rate,time_rdtsc,time_nanosec")
    }
}

pub struct BitIterator<'a> {
    bytes: &'a Vec<u8>,
    byte_index: usize,
    bit_index: u8,
}

impl<'a> BitIterator<'a> {
    pub fn new(bytes: &'a Vec<u8>) -> BitIterator<'a> {
        BitIterator {
            bytes,
            byte_index: 0,
            bit_index: 0,
        }
    }

    pub fn atEnd(&self) -> bool {
        self.byte_index >= self.bytes.len()
    }
}

impl Iterator for BitIterator<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(b) = self.bytes.get(self.byte_index) {
            let r = (b >> (u8::BIT_LENGTH - 1 - self.bit_index as usize)) & 1 != 0;
            self.bit_index += 1;
            self.byte_index += self.bit_index as usize / u8::BIT_LENGTH;
            self.bit_index = self.bit_index % u8::BIT_LENGTH as u8;
            Some(r)
        } else {
            None
        }
    }
}

struct CovertChannelParams<T: CovertChannel + Send> {
    handles: Vec<TurnHandle<T::Handle>>,
    covert_channel: Arc<T>,
}

unsafe impl<T: 'static + CovertChannel + Send> Send for CovertChannelParams<T> {}

fn transmit_thread<T: CovertChannel>(
    num_bytes: usize,
    mut params: CovertChannelParams<T>,
) -> (u64, std::time::Instant, Vec<u8>) {
    let old_affinity = set_affinity(&(*params.covert_channel).helper_core());

    let mut result = Vec::new();
    result.reserve(num_bytes);
    for _ in 0..num_bytes {
        let byte = rand::random();
        result.push(byte);
    }

    let mut bit_sent = 0;
    let mut bit_iter = BitIterator::new(&result);
    let start_time = std::time::Instant::now();
    let start = unsafe { rdtsc_fence() };
    while !bit_iter.atEnd() {
        for page in params.handles.iter_mut() {
            let mut handle = page.wait();
            unsafe { params.covert_channel.transmit(&mut *handle, &mut bit_iter) };
            bit_sent += T::BIT_PER_PAGE;
            page.next();
            if bit_iter.atEnd() {
                break;
            }
        }
    }
    (start, start_time, result)
}

pub fn benchmark_channel<T: 'static + Send + CovertChannel>(
    mut channel: T,
    num_pages: usize,
    num_bytes: usize,
) -> CovertChannelBenchmarkResult {
    // Allocate pages

    let old_affinity = set_affinity(&channel.main_core());

    let size = num_pages * PAGE_SIZE;
    let mut m = MMappedMemory::new(size, false);
    let mut receiver_turn_handles = Vec::new();
    let mut transmit_turn_handles = Vec::new();

    for i in 0..num_pages {
        m.slice_mut()[i * PAGE_SIZE] = i as u8;
    }
    let array: &[u8] = m.slice();
    for i in 0..num_pages {
        let addr = &array[i * PAGE_SIZE] as *const u8;
        let handle = unsafe { channel.ready_page(addr) }.unwrap();
        let mut turns = TurnHandle::new(2, handle);
        let mut t_iter = turns.drain(0..);
        let transmit_lock = t_iter.next().unwrap();
        let receive_lock = t_iter.next().unwrap();

        assert!(t_iter.next().is_none());

        transmit_turn_handles.push(transmit_lock);
        receiver_turn_handles.push(receive_lock);
    }

    let covert_channel_arc = Arc::new(channel);
    let params = CovertChannelParams {
        handles: transmit_turn_handles,
        covert_channel: covert_channel_arc.clone(),
    };

    let helper = thread::spawn(move || transmit_thread(num_bytes, params));
    // Create the thread parameters
    let mut received_bytes: Vec<u8> = Vec::new();
    let mut received_bits = VecDeque::<bool>::new();
    while received_bytes.len() < num_bytes {
        for handle in receiver_turn_handles.iter_mut() {
            let mut page = handle.wait();
            let mut bits = unsafe { covert_channel_arc.receive(&mut *page) };
            handle.next();
            received_bits.extend(&mut bits.iter());
            while received_bits.len() >= u8::BIT_LENGTH {
                let mut byte = 0;
                for i in 0..u8::BIT_LENGTH {
                    byte <<= 1;
                    let bit = received_bits.pop_front().unwrap();
                    byte |= bit as u8;
                }
                received_bytes.push(byte);
            }
            if received_bytes.len() >= num_bytes {
                break;
            }
        }
        // TODO
        // receiver thread
    }

    let stop = unsafe { rdtsc_fence() };
    let stop_time = std::time::Instant::now();
    let r = helper.join();
    let (start, start_time, sent_bytes) = match r {
        Ok(r) => r,
        Err(e) => panic!("Join Error: {:?#}"),
    };
    assert_eq!(sent_bytes.len(), received_bytes.len());
    assert_eq!(num_bytes, received_bytes.len());

    restore_affinity(&old_affinity);

    let mut num_bit_error = 0;
    for i in 0..num_bytes {
        num_bit_error += (sent_bytes[i] ^ received_bytes[i]).count_ones() as usize;
    }

    let error_rate = (num_bit_error as f64) / ((num_bytes * u8::BIT_LENGTH) as f64);

    CovertChannelBenchmarkResult {
        num_bytes_transmitted: num_bytes,
        num_bit_errors: num_bit_error,
        error_rate,
        time_rdtsc: stop - start,
        time_seconds: stop_time - start_time,
    }
}

#[cfg(test)]
mod tests {
    use crate::BitIterator;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_bit_vec() {
        let bits = vec![0x55, 0xf];
        let bit_iter = BitIterator::new(&bits);
        let results = vec![
            false, true, false, true, false, true, false, true, false, false, false, false, true,
            true, true, true,
        ];
        for (i, bit) in bit_iter.enumerate() {
            assert_eq!(results[i], bit);
        }
    }
}
