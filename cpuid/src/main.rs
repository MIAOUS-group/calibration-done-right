// SPDX-FileCopyrightText: 2021 Guillaume DIDIER
//
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: MIT

use cpuid::MicroArchitecture;

fn main() {
    println!("{:?}", MicroArchitecture::get_micro_architecture());
}
