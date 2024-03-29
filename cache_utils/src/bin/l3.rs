// SPDX-FileCopyrightText: 2021 Guillaume DIDIER
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: MIT

use cache_utils::flush;
use cache_utils::mmap::MMappedMemory;

pub fn main() {
    let m = MMappedMemory::new(2 << 20, true);
    let array = m.slice();
    loop {
        unsafe {
            flush(&array[0]);
            flush(&array[(1 << 8) ^ (1 << 12) ^ (1 << 10)]);
        }
    }
}
