#![cfg_attr(not(feature = "use_std"), no_std)]

// TODO import x86 or x86_64
// TODO no_std

extern crate alloc;

use alloc::vec::Vec;
use core::arch::x86_64;

//#[cfg(feature = "no_std")]
//use cstr_core::{CStr, CString};

use crate::CPUVendor::{Intel, Unknown};
use crate::MicroArchitecture::{
    Airmont, Bonnell, Broadwell, CannonLake, CascadeLake, CoffeeLake, CooperLake, Core, Goldmont,
    GoldmontPlus, Haswell, HaswellE, IceLake, IvyBridge, IvyBridgeE, KabyLake, KnightsLanding,
    KnightsMill, Nehalem, NetBurst, Penryn, PentiumM, Saltwell, SandyBridge, Silvermont, Skylake,
    SkylakeServer, Tremont, Westmere, Yonah, P5, P6,
};
//#[cfg(feature = "std")]
//use std::ffi::{CStr, CString};

#[derive(Debug, Eq, PartialEq)]
pub enum CPUVendor {
    None,
    Intel,
    AMD,
    Unknown,
}

impl CPUVendor {
    pub fn get_cpu_vendor() -> CPUVendor {
        // if has cpuid if  x86_64::__cp
        if true {
            let r = unsafe { x86_64::__cpuid(0) };
            CPUVendor::decode_cpu_vendor(r)

        //let feature_string = (r.ebx, r.ecx, r.edx);
        //CPUVendor::Unknown
        } else {
            CPUVendor::None
        }
        // else
    }
    pub fn decode_cpu_vendor(cpuid_result: x86_64::CpuidResult) -> CPUVendor {
        let feature_string = [cpuid_result.ebx, cpuid_result.edx, cpuid_result.ecx]
            .iter()
            .map(|&u| u.to_le_bytes())
            .collect::<Vec<_>>()
            .concat();
        match feature_string.as_slice() {
            b"GenuineIntel" => Intel,
            // TODO add more vendors
            _ => Unknown,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum MicroArchitecture {
    P5,
    P6, // Models:
    // P6 family processors are IA-32 processors based on the P6 family microarchitecture.
    // This includes the Pentium ®Pro, Pentium® II, Pentium® III, and Pentium® III Xeon® processors.
    NetBurst,
    // The Pentium® 4, Pentium® D, and Pentium® processor Extreme Editions are based on the Intel
    // NetBurst® microarchitecture. Most early Intel® Xeon® processors are based on the Intel
    // NetBurst® microarchitecture. Intel Xeon processor 5000, 7100 series are based on the Intel
    // NetBurst ® microarchitecture.
    PentiumM,
    Yonah,
    // The Intel® CoreTM Duo, Intel® CoreTM Solo and dual-core Intel® Xeon® processor LV are based
    // on an improved Pentium® M processor microarchitecture.
    Core,
    // The Intel® Xeon® processor 3000, 3200, 5100, 5300, 7200, and 7300 series, Intel® Pentium®
    // dual-core, Intel® CoreTM2 Duo, Intel® CoreTM2 Quad, and Intel® CoreTM2 Extreme processors
    // are based on Intel® CoreTM microarchitecture.
    Penryn, // aka enhanced core
    // The Intel® Xeon® processor 5200, 5400, 7400 series, Intel® CoreTM2 Quad processor Q9000
    // series, and Intel® CoreTM2 Extreme processors QX9000, X9000 series, Intel® CoreTM2 processor
    // E8000 series are based on Enhanced Intel® CoreTM microarchitecture.
    Saltwell,
    Bonnell, // split Atom early microarch in 2.
    // The Intel® AtomTM processors 200, 300, D400, D500, D2000, N200, N400, N2000, E2000, Z500, Z600, Z2000,
    // C1000 series are based on the Intel® AtomTM microarchitecture and supports Intel 64 architecture.

    /* P6 family, Pentium® M, Intel® CoreTM Solo, Intel® CoreTM Duo processors, dual-core Intel® Xeon® processor LV,
    and early generations of Pentium 4 and Intel Xeon processors support IA-32 architecture. The Intel® Atom TM
    processor Z5xx series support IA-32 architecture.

    The Intel® Xeon® processor 3000, 3200, 5000, 5100, 5200, 5300, 5400, 7100, 7200, 7300, 7400 series, Intel ®
    CoreTM2 Duo, Intel® CoreTM2 Extreme, Intel® CoreTM2 Quad processors, Pentium® D processors, Pentium® Dual-
    Core processor, newer generations of Pentium 4 and Intel Xeon processor family support Intel® 64 architecture.
    */
    Nehalem,
    Westmere,
    // The Intel® CoreTM i7 processor and Intel® Xeon® processor 3400, 5500, 7500 series are based on 45 nm Nehalem
    // microarchitecture. Westmere microarchitecture is a 32 nm version of the Nehalem microarchitecture. Intel®
    // Xeon® processor 5600 series, Intel Xeon processor E7 and various Intel Core i7, i5, i3 processors are based on the
    // Westmere microarchitecture. These processors support Intel 64 architecture.
    SandyBridge,
    // The Intel® Xeon® processor E5 family, Intel® Xeon® processor E3-1200 family, Intel® Xeon® processor E7-
    // 8800/4800/2800 product families, Intel ® CoreTM i7-3930K processor, and 2nd generation Intel® CoreTM i7-2xxx,
    // Intel® CoreTM i5-2xxx, Intel® CoreTM i3-2xxx processor series are based on the Sandy Bridge microarchitecture and
    // support Intel 64 architecture.
    IvyBridge,
    // The Intel® Xeon® processor E7-8800/4800/2800 v2 product families, Intel® Xeon® processor E3-1200 v2 product
    // family and 3rd generation Intel ® CoreTM processors are based on the Ivy Bridge microarchitecture and support
    // Intel 64 architecture.
    IvyBridgeE,
    // The Intel® Xeon® processor E5-4600/2600/1600 v2 product families, Intel® Xeon® processor E5-2400/1400 v2
    // product families and Intel® CoreTM i7-49xx Processor Extreme Edition are based on the Ivy Bridge-E microarchitec-
    // ture and support Intel 64 architecture.
    Haswell,
    // The Intel® Xeon® processor E3-1200 v3 product family and 4th Generation Intel® CoreTM processors are based on
    // the Haswell microarchitecture and support Intel 64 architecture.
    HaswellE,
    // The Intel® Xeon® processor E5-2600/1600 v3 product families and the Intel® CoreTM i7-59xx Processor Extreme
    // Edition are based on the Haswell-E microarchitecture and support Intel 64 architecture.
    Airmont,
    // The Intel® AtomTM processor Z8000 series is based on the Airmont microarchitecture.
    Silvermont,
    // The Intel® AtomTM processor Z3400 series and the Intel® AtomTM processor Z3500 series are based on the Silver-
    // mont microarchitecture.
    Broadwell,
    // The Intel® CoreTM M processor family, 5th generation Intel® CoreTM processors, Intel® Xeon® processor D-1500
    // product family and the Intel® Xeon® processor E5 v4 family are based on the Broadwell microarchitecture and
    // support Intel 64 architecture.
    Skylake,
    // The Intel® Xeon® Processor Scalable Family, Intel® Xeon® processor E3-1500m v5 product family and 6th gener-
    // ation Intel® CoreTM processors are based on the Skylake microarchitecture and support Intel 64 architecture.
    KabyLake,
    // The 7th generation Intel® CoreTM processors are based on the Kaby Lake microarchitecture and support Intel 64
    // architecture.
    Goldmont,
    // The Intel® AtomTM processor C series, the Intel® AtomTM processor X series, the Intel® Pentium® processor J
    // series, the Intel® Celeron® processor J series, and the Intel® Celeron® processor N series are based on the Gold-
    // mont microarchitecture.
    KnightsLanding,
    // The Intel® Xeon PhiTM Processor 3200, 5200, 7200 Series is based on the Knights Landing microarchitecture and
    // supports Intel 64 architecture.
    GoldmontPlus,
    // The Intel® Pentium® Silver processor series, the Intel® Celeron® processor J series, and the Intel® Celeron®
    // processor N series are based on the Goldmont Plus microarchitecture.
    Tremont, // Atom ?
    CoffeeLake,
    // The 8th generation Intel® CoreTM processors, 9th generation Intel® CoreTM processors, and Intel® Xeon® E proces-
    // sors are based on the Coffee Lake microarchitecture and support Intel 64 architecture.
    KnightsMill,
    // The Intel® Xeon PhiTM Processor 7215, 7285, 7295 Series is based on the Knights Mill microarchitecture and
    // supports Intel 64 architecture.
    CascadeLake,
    CannonLake, // Only in volume 4 ??
    // The 2nd generation Intel® Xeon® Processor Scalable Family is based on the Cascade Lake product and supports
    // Intel 64 architecture.
    IceLake,
    // The 10th generation Intel® CoreTM processors are based on the Ice Lake microarchitecture and support Intel 64
    // architecture.
    SkylakeServer,

    // The Intel® Xeon® Processor Scalable Family is based on the Skylake Server microarchitecture. Proces-
    // sors based on the Skylake microarchitecture can be identified using CPUID’s DisplayFamily_DisplayModel
    // signature, which can be found in Table 2-1 of CHAPTER 2 of Intel® 64 and IA-32 Architectures Software
    // Developer’s Manual, Volume 4.
    CooperLake, // Future Xeon Scalable ?? (To be checked)
                // TODO: Check server architecture post Skylake (wikichip ?), add AMD
}

impl MicroArchitecture {
    pub fn from_family_model(
        vendor: CPUVendor,
        family_model_display: u32,
        stepping: u32,
    ) -> Option<MicroArchitecture> {
        match vendor {
            Intel => Some(match family_model_display {
                0x06_85 => KnightsMill, // Intel® Xeon PhiTM Processor 7215, 7285, 7295 Series based on Knights Mill microarchitecture
                0x06_57 => KnightsLanding, // Intel® Xeon PhiTM Processor 3200, 5200, 7200 Series based on Knights Landing microarchitecture
                0x06_7D | 0x06_7E => IceLake, // 10th generation Intel® CoreTM processors based on Ice Lake microarchitecture
                0x06_66 => CannonLake, // Intel® CoreTM processors based on Cannon Lake microarchitecture

                // 7th generation Intel® CoreTM processors based on Kaby Lake microarchitecture, 8th and 9th generation
                // Intel® CoreTM processors based on Coffee Lake microarchitecture, Intel® Xeon® E processors based on
                // Coffee Lake microarchitecture
                0x06_8E => {
                    if stepping <= 9 {
                        KabyLake
                    } else {
                        CoffeeLake
                    }
                }
                0x06_9E => {
                    if stepping <= 9 {
                        KabyLake
                    } else {
                        CoffeeLake
                    }
                }
                // Future Intel® Xeon® processors based on Ice Lake microarchitecture
                0x06_6A | 0x06_6C => IceLake, // Check Server Ahem

                // Intel® Xeon® Processor Scalable Family based on Skylake microarchitecture, 2nd generation Intel®
                // Xeon® Processor Scalable Family based on Cascade Lake product, and future Cooper Lake product
                0x06_55 => {
                    if stepping <= 4 {
                        SkylakeServer
                    } else if stepping <= 7 {
                        CascadeLake
                    } else {
                        CooperLake
                    }
                }
                // 6th generation Intel Core processors and Intel Xeon processor E3-1500m v5 product family and E3-
                // 1200 v5 product family based on Skylake microarchitecture
                0x06_4E | 0x06_5E => Skylake,
                0x06_56 => Broadwell, // Intel Xeon processor D-1500 product family based on Broadwell microarchitecture
                // Intel Xeon processor E5 v4 Family based on Broadwell microarchitecture, Intel Xeon processor E7 v4
                // Family, Intel Core i7-69xx Processor Extreme Edition
                0x06_4F => Broadwell,
                // 5th generation Intel Core processors, Intel Xeon processor E3-1200 v4 product family based on
                // Broadwell microarchitecture
                0x06_47 => Broadwell,
                // Intel Core M-5xxx Processor, 5th generation Intel Core processors based on Broadwell
                // microarchitecture
                0x06_3D => Broadwell,
                // Intel Xeon processor E5-4600/2600/1600 v3 product families, Intel Xeon processor E7 v3 product
                // families based on Haswell-E microarchitecture, Intel Core i7-59xx Processor Extreme Edition
                0x06_3F => HaswellE,
                // 4th Generation Intel Core processor and Intel Xeon processor E3-1200 v3 product family based on
                // Haswell microarchitecture
                0x06_3C | 0x06_45 | 0x06_46 => Haswell,
                // Intel Xeon processor E7-8800/4800/2800 v2 product families based on Ivy Bridge-E
                // microarchitecture
                // Intel Xeon processor E5-2600/1600 v2 product families and Intel Xeon processor E5-2400 v2
                // product family based on Ivy Bridge-E microarchitecture, Intel Core i7-49xx Processor Extreme Edition
                0x06_3E => IvyBridgeE,
                // 3rd Generation Intel Core Processor and Intel Xeon processor E3-1200 v2 product family based on Ivy
                // Bridge microarchitecture
                0x06_3A => IvyBridge,
                // Intel Xeon processor E5 Family based on Intel microarchitecture code name Sandy Bridge, Intel Core
                // i7-39xx Processor Extreme Edition
                0x06_2D => SandyBridge,
                0x06_2F => Westmere, //Intel Xeon Processor E7 Family
                // Intel Xeon processor E3-1200 product family; 2nd Generation Intel Core i7, i5, i3 Processors 2xxx
                // Series
                0x06_2A => SandyBridge,
                0x06_2E => Nehalem, // Intel Xeon processor 7500, 6500 series
                0x06_25 | 0x06_2C => Westmere, // Intel Xeon processors 3600, 5600 series, Intel Core i7, i5 and i3 Processors
                0x06_1E | 0x06_1F => Nehalem,  // Intel Core i7 and i5 Processors
                0x06_1A => Nehalem, // Intel Core i7 Processor, Intel Xeon processor 3400, 3500, 5500 series
                0x06_1D => Penryn,  // Intel Xeon processor MP 7400 series
                0x06_17 => Penryn, // Intel Xeon processor 3100, 3300, 5200, 5400 series, Intel Core 2 Quad processors 8000, 9000 series
                // Intel Xeon processor 3000, 3200, 5100, 5300, 7300 series, Intel Core 2 Quad processor 6000 series,
                // Intel Core 2 Extreme 6000 series, Intel Core 2 Duo 4000, 5000, 6000, 7000 series processors, Intel
                // Pentium dual-core processors
                0x06_0F => Core,               // Merom
                0x06_0E => Yonah,              // Intel Core Duo, Intel Core Solo processors - Yonah
                0x06_0D => PentiumM,           // Intel Pentium M processor
                0x06_86 => Tremont, // Intel® AtomTM processors based on Tremont Microarchitecture
                0x06_7A => GoldmontPlus, // Intel Atom processors based on Goldmont Plus Microarchitecture
                0x06_5F => Goldmont, // Intel Atom processors based on Goldmont Microarchitecture (code name Denverton)
                0x06_5C => Goldmont, // Intel Atom processors based on Goldmont Microarchitecture
                0x06_4C => Airmont, // Intel Atom processor X7-Z8000 and X5-Z8000 series based on Airmont Microarchitecture
                0x06_5D => Silvermont, // Intel Atom processor X3-C3000 based on Silvermont Microarchitecture
                0x06_5A => Silvermont, // Intel Atom processor Z3500 series
                0x06_4A => Silvermont, // Intel Atom processor Z3400 series
                0x06_37 => Silvermont, //Intel Atom processor E3000 series, Z3600 series, Z3700 series
                0x06_4D => Silvermont, // Intel Atom processor C2000 series
                0x06_36 => Saltwell,   // Intel Atom processor S1000 Series
                0x06_27 | 0x06_35 => Saltwell, // Intel Atom processor family, Intel Atom processor D2000, N2000, E2000, Z2000, C1000 series
                0x06_1C | 0x06_26 => Bonnell,
                0x0F_06 => NetBurst, // Intel Xeon processor 7100, 5000 Series, Intel Xeon Processor MP, Intel Pentium 4, Pentium D processors
                0x0F_03 | 0x0F_04 => NetBurst, // Intel Xeon processor, Intel Xeon processor MP, Intel Pentium 4, Pentium D processors
                0x06_09 => PentiumM,           // Intel Pentium M processor
                0x0F_02 => NetBurst, // Intel Xeon Processor, Intel Xeon processor MP, Intel Pentium 4 processors
                0x0F_00 | 0x0F_01 => NetBurst, // Intel Xeon Processor, Intel Xeon processor MP, Intel Pentium 4 processors
                0x06_07 | 0x06_08 | 0x06_0A | 0x06_0B => P6, //  Intel Pentium III Xeon processor, Intel Pentium III processor
                0x06_03 | 0x06_05 => P6, // Intel Pentium II Xeon processor, Intel Pentium II processor
                0x06_01 => P6,           // Intel Pentium Pro processor
                0x05_01 | 0x05_02 | 0x05_04 => P5, // Intel Pentium processor, Intel Pentium processor with MMX Technology

                // TODO: Keep adding stuff in here
                _ => {
                    return None;
                }
            }),
            _ => None,
        }
    }
    pub fn get_family_model_stepping() -> Option<(CPUVendor, u32, u32)> {
        // Warning this might not support AMD
        if true {
            // TODO refactor some of this into a separate function.
            // has cpuid
            let vendor = CPUVendor::get_cpu_vendor();
            let eax = unsafe { x86_64::__cpuid(1) }.eax;
            let stepping = eax & 0xf;
            let mut model = (eax >> 4) & 0xf;
            let mut family = (eax >> 8) & 0xf;
            if family == 0xf {
                family += (eax >> 20) & 0xff
            }
            if family == 0xf || family == 0x6 {
                model += (eax >> 12) & 0xf0
            }
            let family_model_display = family << 8 | model;
            Some((vendor, family_model_display, stepping))
        } else {
            None
        }
    }
    pub fn get_micro_architecture() -> Option<MicroArchitecture> {
        if let Some((vendor, family_model_display, stepping)) =
            MicroArchitecture::get_family_model_stepping()
        {
            MicroArchitecture::from_family_model(vendor, family_model_display, stepping)
        } else {
            None
        }
    }
}

// Family Model Stepping Processor Type -> leaf 0x1, in EAX, plus brand index in EBX,
// sdm 3-226 2A or Brand string

// 3B 16-1, 16-3,

// 4 2-1 huge table to be cross referenced with 1-2
