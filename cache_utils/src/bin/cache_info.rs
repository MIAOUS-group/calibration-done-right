use cache_utils::cache_info::get_cache_info;
use cache_utils::complex_addressing::cache_slicing;
use cpuid::MicroArchitecture;

use std::process::Command;
use std::str::from_utf8;

pub fn main() {
    println!("{:#?}", get_cache_info());

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

    if let Some(uarch) = MicroArchitecture::get_micro_architecture() {
        if let Some(vendor_family_model_stepping) = MicroArchitecture::get_family_model_stepping() {
            println!("{:?}", uarch);
            let slicing = cache_slicing(
                uarch,
                core_per_socket,
                vendor_family_model_stepping.0,
                vendor_family_model_stepping.1,
                vendor_family_model_stepping.2,
            );
            println!("{:?}", slicing.image((1 << 12) - 1));
            println!("{:?}", slicing.kernel_compl_basis((1 << 12) - 1));
            println!("{:?}", slicing.image_antecedent((1 << 12) - 1));
        }
    }
}
