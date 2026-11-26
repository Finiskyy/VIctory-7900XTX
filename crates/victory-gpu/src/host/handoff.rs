//! The end of a submission sent ahead given to a thread of its own
//! (`VICTORY_SEND_THREAD=1`; off by default): the wait for the heaps'
//! records, the slots that waited for a texture's host copy, and the calls
//! that put the lists on the queue. The thread that prepares submissions
//! goes on to the next one meanwhile: one submission is prepared while the
//! one before it is sent, by two threads in a row.
//!
//! Why: on the RX 7900 XTX the thread that prepares submissions is the wall
//! of a frame (48 of 55 ms on the water, `docs/findings.md`), and 6 ms of
//! that is the wait for the heaps' records alone: the thread does nothing
//! while another one works. Here the next submission's preparing fills it.
//!
//! What is the same: everything the next submission's preparing reads of
//! this one was settled before the hand-off. Its flight is in the line of
//! flights (`super::Line`) with the fence value that will be signalled
//! behind it, its copies in the card's memory were brought up to date
//! (`Host::card_before`), the ranges its shaders write are declared. Seen
//! from the CPU, a submission handed off and not yet sent is one sent and
//! not yet run: fence values not reached, nothing landed. Every wait for it
//! is a wait for its fence, which this thread signals once the lists are on
//! the queue.
//!
//! What has to wait: whatever goes to the host's queue in order with the
//! lists. Every such call takes the queue through `Host::queue` (or
//! [`behind`] first), which returns once everything handed off is on the
//! queue: the card's copies brought up to date for the next submission,
//! fetches of what shaders wrote, tile mappings, other sends, the present.
//! Only this thread uses the queue without that wait. A call that sends
//! in order with the lists and is not made through `Host::queue` would go
//! ahead of a submission handed off and not sent: there is none (the
//! field is `queue_raw`, and every use of it is read in this module or
//! in a query of the queue).
//!
//! A wait at [`behind`] that takes over two seconds is said once in the log,
//! with the thread's name: this thread waits only for the heaps' records,
//! so that would be a thread doing that work which went to the queue.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::ID3D12CommandList;

/// One submission's end, as handed off.
pub(super) struct Job {
    /// Its lists in the order they go, and where a call of its own begins
    /// (parts of submissions joined, [`super::joins`]).
    pub(super) lists: Vec<Option<ID3D12CommandList>>,
    pub(super) cuts: Vec<usize>,
    /// The value signalled on the host's fence behind them.
    pub(super) value: u64,
    /// The heaps' records, begun for it.
    pub(super) heaps: crate::descriptor::SyncedHeaps,
    /// Diagnostics: its stamps on the card, for the timeline.
    #[cfg(feature = "diagnostics")]
    pub(super) stamp: Option<usize>,
}

// SAFETY: D3D12 objects are free-threaded; closed lists nothing records into
// until their flight has landed, which is after this thread sent them.
unsafe impl Send for Job {}

struct State {
    jobs: VecDeque<Job>,
    /// Jobs handed off, and of them those on the queue now.
    handed: u64,
    sent: u64,
}

static STATE: Mutex<State> = Mutex::new(State {
    jobs: VecDeque::new(),
    handed: 0,
    sent: 0,
});
/// A job was handed off.
static WORK: Condvar = Condvar::new();
/// A job is on the queue.
static SENT: Condvar = Condvar::new();

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

thread_local! {
    /// This is the thread that sends.
    static OURS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether submissions sent ahead end on a thread of their own
/// (`VICTORY_SEND_THREAD=1`). Diagnostics builds: on and off while the title
/// runs (`crate::ab`); [`behind`] keeps the order across a change.
pub(super) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_SEND_THREAD") {
        return on && started();
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_THREAD").is_some_and(|v| v == "1") && started())
}

/// Starts the thread, once: whether there is one.
fn started() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let worker = || {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
            // SAFETY: this thread's own priority.
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) };
            crate::cores::keep_on_fast_cores("victory-send");
            OURS.set(true);
            let ended = std::panic::catch_unwind(serve);
            if ended.is_err() {
                // Nothing handed off would reach the card, and every wait
                // for the queue would stand for good.
                super::fail(format_args!("the thread that sends submissions panicked"));
            }
        };
        match std::thread::Builder::new().name("victory-send".into()).spawn(worker) {
            Ok(_) => {
                tracing::info!(
                    "submissions sent ahead are put on the queue by a thread of their own, after the heaps' records \
                     (VICTORY_SEND_THREAD=1)"
                );
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no thread to send submissions: the thread that prepares them does");
                false
            }
        }
    })
}

/// Hands `job` off; it is on the queue when [`behind`] returns.
pub(super) fn hand(job: Job) {
    let mut s = state();
    s.handed += 1;
    s.jobs.push_back(job);
    let depth = s.jobs.len() as u64;
    drop(s);
    STATS[HANDED].fetch_add(1, Relaxed);
    STATS[DEEPEST].fetch_max(depth, Relaxed);
    WORK.notify_one();
}

/// Waits until everything handed off is on the queue: before anything else
/// goes to it. Nothing on the thread that sends.
pub(super) fn behind() {
    if OURS.get() {
        return;
    }
    let mut s = state();
    if s.sent == s.handed {
        return;
    }
    let ticket = s.handed;
    let began = Instant::now();
    let _waiting = crate::timeline::span("the queue waits for submissions handed off to be sent");
    let mut told = false;
    while s.sent < ticket {
        let (back, out) = SENT.wait_timeout(s, Duration::from_secs(2)).unwrap_or_else(|e| e.into_inner());
        s = back;
        if out.timed_out() && !told {
            told = true;
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let me = std::thread::current();
                tracing::warn!(
                    thread = me.name().unwrap_or("?"),
                    "a call to the queue has waited 2 s for submissions handed off to be sent (VICTORY_SEND_THREAD=1)"
                );
            });
        }
    }
    drop(s);
    STATS[BEHIND].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
    STATS[BEHINDS].fetch_add(1, Relaxed);
}

/// Sends what is handed off, in order, for as long as the process runs.
fn serve() -> ! {
    let Some(host) = super::host() else {
        // (Not reached: a submission was prepared, so there is a host.)
        loop {
            std::thread::park();
        }
    };
    loop {
        let job = {
            let mut s = state();
            loop {
                if let Some(job) = s.jobs.pop_front() {
                    break job;
                }
                s = WORK.wait(s).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Job {
            lists,
            cuts,
            value,
            heaps,
            #[cfg(feature = "diagnostics")]
            stamp,
        } = job;
        let began = Instant::now();
        {
            let _waiting = crate::timeline::span("send: waiting for the heaps' records");
            drop(heaps);
        }
        // The slots that waited for a texture's host copy, before the lists
        // that may read them run.
        crate::descriptor::make_waiting();
        let records = began.elapsed();
        let _sending = crate::timeline::span("send: sending");
        #[cfg(feature = "diagnostics")]
        let sending = crate::timeline::ticks();
        // SAFETY: closed lists; the fence is the host's, and its value was
        // taken under the submission lock for this job, in order.
        let signalled = unsafe {
            let mut from = 0;
            for to in cuts.into_iter().chain([lists.len()]) {
                host.queue_raw.ExecuteCommandLists(&lists[from..to]);
                from = to;
            }
            host.queue_raw.Signal(&host.fence, value)
        };
        super::awake::sent(&host.device);
        #[cfg(feature = "diagnostics")]
        crate::timeline::sent(sending, value, stamp);
        host.card_sent();
        if let Err(e) = signalled {
            super::fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
        }
        drop(lists);
        {
            let mut s = state();
            s.sent += 1;
        }
        SENT.notify_all();
        STATS[RECORDS].fetch_add(records.as_micros() as u64, Relaxed);
        STATS[WORKED].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
    }
}

/// Since the last frame-rate line: jobs handed off, the most waiting at
/// once; microseconds this thread worked, and of that waited for the heaps'
/// records; microseconds other threads waited for it before going to the
/// queue, and how many times.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const HANDED: usize = 0;
const DEEPEST: usize = 1;
const WORKED: usize = 2;
const RECORDS: usize = 3;
const BEHIND: usize = 4;
const BEHINDS: usize = 5;

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Relaxed) as f64).collect();
    if s[HANDED] == 0.0 {
        return;
    }
    let ms = |i: usize| s[i] / 1000.0 / frames;
    tracing::info!(
        "submissions sent by a thread of their own, per frame (VICTORY_SEND_THREAD): {:.1} handed off (at most {:.0} \
         waiting); that thread worked {:.1} ms, {:.1} of it waiting for the heaps' records; other threads waited {:.1} ms \
         in {:.1} calls for it before going to the queue",
        s[HANDED] / frames,
        s[DEEPEST],
        ms(WORKED),
        ms(RECORDS),
        ms(BEHIND),
        s[BEHINDS] / frames,
    );
}
