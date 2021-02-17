/// Stuff to do in here :
/// This module is meant to compute and return info about the caching structure
/// Should include if needed the work for reverse engineering L3 complex addressing
/// May also have a module to deal with prefetchers
extern crate alloc;

use alloc::vec::Vec;
use core::arch::x86_64 as arch_x86;
const CACHE_INFO_CPUID_LEAF: u32 = 0x4;

pub fn get_cache_info() -> Vec<CacheInfo> {
    let mut ret = Vec::new();
    let mut i = 0;

    while let Some(cache_info) =
        CacheInfo::from_cpuid_result(&unsafe { arch_x86::__cpuid_count(CACHE_INFO_CPUID_LEAF, i) })
    {
        ret.push(cache_info);
        i += 1;
    }
    ret
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CacheType {
    Null = 0,
    Data = 1,
    Instruction = 2,
    Unified = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_snake_case)]
pub struct CacheInfo {
    pub cache_type: CacheType,
    pub level: u8,
    pub self_init: bool,
    pub fully_assoc: bool,
    pub max_ID_for_cache: u16,
    pub core_in_package: u16,
    pub cache_line_size: u16,
    pub physical_line_partition: u16,
    pub associativity: u16,
    pub sets: u32,
    pub wbinvd_no_guarantee: bool,
    pub inclusive: bool,
    pub complex_cache_indexing: bool,
}

impl CacheInfo {
    pub fn from_cpuid_result(cr: &arch_x86::CpuidResult) -> Option<CacheInfo> {
        let ctype = cr.eax & 0x1f;
        let cache_type = match ctype {
            0 => {
                return None;
            }
            1 => CacheType::Data,
            2 => CacheType::Instruction,
            3 => CacheType::Unified,
            _ => {
                return None;
            }
        };
        let level: u8 = (cr.eax >> 5 & 0x7) as u8;
        let self_init = (cr.eax >> 8 & 0x1) != 0;
        let fully_assoc = (cr.eax >> 9 & 0x1) != 0;
        #[allow(non_snake_case)]
        let max_ID_for_cache = (cr.eax >> 14 & 0xfff) as u16 + 1;
        let core_in_package = (cr.eax >> 26 & 0x3f) as u16 + 1;
        let cache_line_size = (cr.ebx & 0xfff) as u16 + 1;
        let physical_line_partition = (cr.ebx >> 12 & 0x3ff) as u16 + 1;
        let associativity = (cr.ebx >> 22 & 0x3ff) as u16 + 1;
        let sets = cr.ecx + 1;
        let wbinvd_no_guarantee = (cr.edx & 0x1) != 0;
        let inclusive = (cr.edx & 0x2) != 0;
        let complex_cache_indexing = (cr.edx & 0x4) != 0;

        Some(CacheInfo {
            cache_type,
            level,
            self_init,
            fully_assoc,
            max_ID_for_cache,
            core_in_package,
            cache_line_size,
            physical_line_partition,
            associativity,
            sets,
            wbinvd_no_guarantee,
            inclusive,
            complex_cache_indexing,
        })
    }
}
