//! Background rasterisation of pane overlays.
//!
//! The UI thread submits the scene for a pane; a worker renders it and the UI
//! picks the finished frame up on its next update. Only the latest request
//! per pane is kept, so a burst of view changes or plugin updates never
//! queues up stale work. While a frame is rendering, the UI keeps showing the
//! previous one, reprojected to the current view.

use super::raster::{FrameScene, ViewTransform, render};
use parking_lot::{Condvar, Mutex};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::thread::JoinHandle;

pub struct OverlayJob {
    pub pane: u32,
    /// Echoed in the result so the UI can match it to its request.
    pub ticket: u64,
    pub view: ViewTransform,
    pub scene: FrameScene,
}

pub struct OverlayFrame {
    /// The ticket of the request this frame answers.
    pub ticket: u64,
    pub view: ViewTransform,
    pub width: u32,
    pub height: u32,
    /// Premultiplied RGBA8; `None` when nothing is drawn.
    pub pixels: Option<Vec<u8>>,
}

#[derive(Default)]
struct State {
    pending: BTreeMap<u32, OverlayJob>,
    busy: BTreeSet<u32>,
    finished: HashMap<u32, OverlayFrame>,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    work: Condvar,
    notify: Box<dyn Fn() + Send + Sync>,
}

pub struct OverlayWorker {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl OverlayWorker {
    /// Starts `threads` workers; `notify` is called (on a worker thread)
    /// whenever a frame finishes.
    pub fn new(threads: usize, notify: impl Fn() + Send + Sync + 'static) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            work: Condvar::new(),
            notify: Box::new(notify),
        });
        let threads = (0..threads.max(1))
            .map(|i| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("eov-overlay-{i}"))
                    .spawn(move || run(shared))
                    .expect("spawn overlay worker")
            })
            .collect();
        Self { shared, threads }
    }

    /// Queues `job`, replacing any request for the same pane that has not
    /// started yet.
    pub fn submit(&self, job: OverlayJob) {
        let mut state = self.shared.state.lock();
        state.pending.insert(job.pane, job);
        drop(state);
        self.shared.work.notify_one();
    }

    /// Takes the newest finished frame for `pane`.
    pub fn take_finished(&self, pane: u32) -> Option<OverlayFrame> {
        self.shared.state.lock().finished.remove(&pane)
    }

    /// Whether a request for `pane` is queued or rendering.
    pub fn is_busy(&self, pane: u32) -> bool {
        let state = self.shared.state.lock();
        state.pending.contains_key(&pane) || state.busy.contains(&pane)
    }

    /// Drops queued work and results for a pane that no longer exists.
    pub fn forget(&self, pane: u32) {
        let mut state = self.shared.state.lock();
        state.pending.remove(&pane);
        state.finished.remove(&pane);
    }
}

impl Drop for OverlayWorker {
    fn drop(&mut self) {
        self.shared.state.lock().shutdown = true;
        self.shared.work.notify_all();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

fn run(shared: Arc<Shared>) {
    loop {
        let job = {
            let mut state = shared.state.lock();
            loop {
                if state.shutdown {
                    return;
                }
                let next = state
                    .pending
                    .keys()
                    .copied()
                    .find(|pane| !state.busy.contains(pane));
                if let Some(pane) = next {
                    let job = state.pending.remove(&pane).expect("pending job");
                    state.busy.insert(pane);
                    break job;
                }
                shared.work.wait(&mut state);
            }
        };
        let (width, height) = job.view.physical_size();
        let pixels = if job.scene.is_empty() {
            None
        } else {
            // A panic in rendering must not take the worker down.
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                render(&job.view, &job.scene).map(|pixmap| pixmap.take())
            }))
            .unwrap_or_else(|_| {
                tracing::error!("overlay rendering panicked for pane {}", job.pane);
                None
            })
        };
        {
            let mut state = shared.state.lock();
            state.busy.remove(&job.pane);
            state.finished.insert(
                job.pane,
                OverlayFrame {
                    ticket: job.ticket,
                    view: job.view,
                    width,
                    height,
                    pixels,
                },
            );
        }
        // Another job for the same pane may have been waiting on `busy`.
        shared.work.notify_one();
        (shared.notify)();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    fn view() -> ViewTransform {
        ViewTransform {
            left: 0.0,
            top: 0.0,
            zoom: 1.0,
            width: 8.0,
            height: 8.0,
            scale: 1.0,
            mpp: None,
            orientation: Default::default(),
        }
    }

    #[test]
    fn frames_are_delivered_with_their_ticket() {
        let notified = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&notified);
        let worker = OverlayWorker::new(1, move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        worker.submit(OverlayJob {
            pane: 3,
            ticket: 42,
            view: view(),
            scene: FrameScene::default(),
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        let frame = loop {
            if let Some(frame) = worker.take_finished(3) {
                break frame;
            }
            assert!(Instant::now() < deadline, "no frame");
            std::thread::sleep(Duration::from_millis(2));
        };
        assert_eq!(frame.ticket, 42);
        assert!(frame.pixels.is_none(), "empty scenes produce no pixels");
        assert!(notified.load(Ordering::SeqCst) >= 1);
        assert!(!worker.is_busy(3));
    }
}
