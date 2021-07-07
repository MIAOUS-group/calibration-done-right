#![cfg(feature = "use_std")]

use core::borrow::{Borrow, BorrowMut};
use core::ffi::c_void;
use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use core::ptr::null_mut;
use core::ptr::Unique;
use core::slice::{from_raw_parts, from_raw_parts_mut};
use nix::errno::Errno::EINVAL;
use nix::sys::mman;

/* from linux kernel headers.
#define HUGETLB_FLAG_ENCODE_SHIFT       26
#define HUGETLB_FLAG_ENCODE_MASK        0x3f

#define HUGETLB_FLAG_ENCODE_64KB        (16 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_512KB       (19 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_1MB         (20 << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_2MB         (21 << HUGETLB_FLAG_ENCODE_SHIFT)
*/

pub struct MMappedMemory<T> {
    pointer: Unique<T>,
    size: usize,
}

impl<T> MMappedMemory<T> {
    pub fn try_new(size: usize, huge: bool) -> Result<MMappedMemory<T>, nix::Error> {
        assert_ne!(size_of::<T>(), 0);
        if let Some(p) = unsafe {
            let p = mman::mmap(
                null_mut(),
                size * size_of::<T>(),
                mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE,
                mman::MapFlags::MAP_PRIVATE
                    | mman::MapFlags::MAP_ANONYMOUS
                    | if huge {
                        mman::MapFlags::MAP_HUGETLB
                    } else {
                        mman::MapFlags::MAP_ANONYMOUS
                    },
                -1,
                0,
            )?;
            let pointer_T = p as *mut T;
            Unique::new(pointer_T)
        } {
            Ok(MMappedMemory { pointer: p, size })
        } else {
            Err(nix::Error::Sys(EINVAL))
        }
    }

    pub fn new(size: usize, huge: bool) -> MMappedMemory<T> {
        Self::try_new(size, huge).unwrap()
    }

    pub fn slice(&self) -> &[T] {
        unsafe { from_raw_parts(self.pointer.as_ptr(), self.size) }
    }

    pub fn slice_mut(&mut self) -> &mut [T] {
        unsafe { from_raw_parts_mut(self.pointer.as_ptr(), self.size) }
    }
}

impl<T> Drop for MMappedMemory<T> {
    fn drop(&mut self) {
        unsafe {
            mman::munmap(self.pointer.as_ptr() as *mut c_void, self.size).unwrap();
        }
    }
}

impl<T> Deref for MMappedMemory<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.slice()
    }
}

impl<T> DerefMut for MMappedMemory<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice_mut()
    }
}

impl<T> AsRef<[T]> for MMappedMemory<T> {
    fn as_ref(&self) -> &[T] {
        self.slice()
    }
}

impl<T> AsMut<[T]> for MMappedMemory<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.slice_mut()
    }
}

impl<T> Borrow<[T]> for MMappedMemory<T> {
    fn borrow(&self) -> &[T] {
        self.slice()
    }
}

impl<T> BorrowMut<[T]> for MMappedMemory<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.slice_mut()
    }
}
