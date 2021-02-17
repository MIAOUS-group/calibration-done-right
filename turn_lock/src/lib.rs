use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{spin_loop_hint, AtomicUsize, Ordering};
use std::sync::Arc;

// FIXME There may be significant unsafety if wait is called twice ?
// Add some extra mutual exclusion ?

pub struct RawTurnLock {
    turn: AtomicUsize,
    num_turns: usize,
}

impl RawTurnLock {
    pub fn new(num_turns: usize) -> Self {
        Self {
            turn: AtomicUsize::new(0),
            num_turns,
        }
    }

    pub fn is_poisoned(&self) -> bool {
        let current = self.turn.load(Ordering::Relaxed);
        current >= self.num_turns
    }

    pub unsafe fn try_wait(&self, turn: usize) -> bool {
        let current = self.turn.load(Ordering::Acquire);
        current == turn
    }

    pub unsafe fn wait(&self, turn: usize) {
        let mut current = self.turn.load(Ordering::Acquire);
        while current < self.num_turns && current != turn {
            spin_loop_hint();
            current = self.turn.load(Ordering::Acquire);
        }
        if current >= self.num_turns {
            panic!("Waiting on a poisoned turn lock");
        }
        if self.turn.load(Ordering::Relaxed) != turn {
            panic!("Someone stole the turn");
        }
    }

    pub unsafe fn next(&self, turn: usize) {
        if self.is_poisoned() {
            panic!("Using poisoned turn lock");
        }
        let current = self.turn.load(Ordering::Relaxed);
        if current != turn {
            panic!("Releasing turn lock out of turn");
        }

        let r = self.turn.compare_exchange(
            turn,
            (turn + 1) % self.num_turns,
            Ordering::Release,
            Ordering::Relaxed,
        );
        if r.expect("Failed to release turn lock") != turn {
            panic!("Released turn lock out of turn");
        }
    }
}

struct TurnLockData<T> {
    pub lock: RawTurnLock,
    pub data: UnsafeCell<T>,
}

pub struct TurnHandle<T> {
    raw: Arc<TurnLockData<T>>,
    index: usize,
}

impl<T> TurnHandle<T> {
    pub fn new(num_turns: usize, data: T) -> Vec<TurnHandle<T>> {
        let turn_lock = RawTurnLock::new(num_turns);
        let turn_lock_data = TurnLockData {
            lock: turn_lock,
            data: UnsafeCell::new(data),
        };
        let arc = Arc::new(turn_lock_data);
        let mut result = Vec::with_capacity(num_turns);
        for i in 0..num_turns {
            result.push(Self {
                raw: arc.clone(),
                index: i,
            })
        }
        result
    }

    unsafe fn guard(&self) -> TurnLockGuard<T> {
        TurnLockGuard {
            handle: &*self,
            marker: PhantomData,
        }
    }

    pub fn wait(&mut self) -> TurnLockGuard<T> {
        unsafe { self.raw.lock.wait(self.index) };
        // Safety: the turn lock is now held
        unsafe { self.guard() }
    }

    pub fn next(&self) {
        unsafe { self.raw.lock.next(self.index) };
    }
}

#[must_use = "if unused the TurnLock will immediately unlock"]
pub struct TurnLockGuard<'a, T> {
    handle: &'a TurnHandle<T>,
    marker: PhantomData<&'a T>,
}

impl<'a, T> TurnLockGuard<'a, T> {
    pub fn handle(&self) -> &TurnHandle<T> {
        self.handle
    }
}

impl<'a, T> Deref for TurnLockGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.handle.raw.data.get() }
    }
}

impl<'a, T> DerefMut for TurnLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.handle.raw.data.get() }
    }
}

unsafe impl<T> Send for TurnHandle<T> {}
#[cfg(test)]
mod tests {
    use crate::TurnHandle;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn three_turns() {
        let mut v = TurnHandle::<i32>::new(3, 0);
        let t0 = v[0].wait();
        drop(t0);
        v[0].next();
        let t1 = v[1].wait();
        drop(t1);
        v[1].next();
        let t2 = v[2].wait();
        drop(t2);
        v[2].next();
        let t0 = v[0].wait();
    }
}
