//! Running one analysis job: read tiles, skip background, batch, infer,
//! score and deliver.
//!
//! - Reader threads fetch regions in parallel; inference runs on the job
//!   thread in batches. Models with a fixed batch get full batches, padded
//!   only at the end of the job; dynamic models get batches of the configured
//!   size (VAE-9).
//! - Background tiles are detected on every path, recorded as such and never
//!   inferred (VAE-9, VAE-10).
//! - A user cancel and an internal abort are separate: the first real error
//!   is kept (errors outrank each other by severity) and reported as the
//!   reason, never as "cancelled" (VAE-7). A few unreadable tiles are
//!   recorded as failed instead of ending a whole-slide run.
//! - Results are delivered in groups through [`JobSink`], which may block
//!   (backpressure from the disk cache, VAE-12).

use crate::contract::{
    Contract, Layout, PreparedTile, is_background, prepare, score_output, write_input,
};
use crate::engine::{EngineError, InferenceEngine};
use crate::results::{Counts, Need, TilePayload, TileState};
use crate::tiling::{TileCoord, TileGrid};
use eov_plugin_api::abi::RegionRequest;
use eov_plugin_api::protocol::FileId;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Reads slide pixels (the host's `read_region`, or a test double).
pub trait PixelSource: Send + Sync {
    /// Returns `width·height·4` RGBA bytes.
    fn read(&self, request: &RegionRequest) -> Result<Vec<u8>, String>;
}

/// The result for one tile.
#[derive(Debug)]
pub struct TileOutcome {
    pub coord: TileCoord,
    pub state: TileState,
    pub payload: Option<TilePayload>,
    pub embedding: Option<Vec<f32>>,
}

/// Progress of a job.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Progress {
    pub total: usize,
    pub done: usize,
    pub counts: Counts,
    pub elapsed: Duration,
}

impl Progress {
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            1.0
        } else {
            self.done as f32 / self.total as f32
        }
    }

    /// Estimated time left, once enough is done to tell.
    pub fn eta(&self) -> Option<Duration> {
        if self.done < 8 || self.done >= self.total {
            return None;
        }
        let per_tile = self.elapsed.as_secs_f64() / self.done as f64;
        Some(Duration::from_secs_f64(
            per_tile * (self.total - self.done) as f64,
        ))
    }
}

/// Where a job's results go.
pub trait JobSink: Send + Sync {
    /// Takes finished tiles. May block (backpressure).
    fn deliver(&self, outcomes: Vec<TileOutcome>);
    /// Called at most a few times per second.
    fn progress(&self, progress: &Progress);
}

/// How bad an error is; the worst one is reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Plumbing noise (a thread went away because another one failed).
    Internal = 1,
    /// Too many tiles could not be read.
    Read = 2,
    /// The model or runtime failed.
    Inference = 3,
}

/// Cancellation and failure of a running job.
#[derive(Debug, Default)]
pub struct JobControl {
    stop: Arc<AtomicBool>,
    cancelled: AtomicBool,
    error: Mutex<Option<(Severity, String)>>,
}

impl JobControl {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The user (or a newer job) cancelled this one.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub fn stopping(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// The flag engines watch to abandon a run.
    pub fn stop_flag(&self) -> &Arc<AtomicBool> {
        &self.stop
    }

    /// Records a failure and stops the job. The first error of the highest
    /// severity is kept.
    pub fn fail(&self, severity: Severity, message: impl Into<String>) {
        let mut slot = self.error.lock().unwrap_or_else(|e| e.into_inner());
        if slot
            .as_ref()
            .is_none_or(|(existing, _)| severity > *existing)
        {
            *slot = Some((severity, message.into()));
        }
        drop(slot);
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn error(&self) -> Option<String> {
        self.error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|(_, message)| message.clone())
    }
}

/// How a job ended.
#[derive(Clone, Debug, PartialEq)]
pub enum JobOutcome {
    Completed(Counts),
    Cancelled(Counts),
    Failed { error: String, counts: Counts },
}

/// What a job does.
#[derive(Clone, Debug)]
pub struct JobSpec {
    pub file_id: FileId,
    pub grid: TileGrid,
    /// Tiles in processing order.
    pub tiles: Vec<(TileCoord, Need)>,
    pub contract: Contract,
    pub layout: Layout,
    pub skip_background: bool,
    /// Batch size for models without a fixed batch.
    pub batch_size: usize,
    pub readers: usize,
}

enum Item {
    Ready(TileCoord, PreparedTile),
    Background(TileCoord),
    Failed(TileCoord),
}

const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
const DELIVERY_INTERVAL: Duration = Duration::from_millis(300);

/// Runs a job to completion, cancellation or failure.
pub fn run_job(
    spec: &JobSpec,
    source: &dyn PixelSource,
    engine: &mut dyn InferenceEngine,
    sink: &dyn JobSink,
    control: &JobControl,
) -> JobOutcome {
    let started = Instant::now();
    let total = spec.tiles.len();
    let tile = spec.grid.tile;
    let item_len = tile as usize * tile as usize * 3;
    let info = engine.info().clone();
    if info.item_len != item_len {
        return JobOutcome::Failed {
            error: format!(
                "the engine expects {} values per tile but tiles have {item_len}",
                info.item_len
            ),
            counts: Counts::default(),
        };
    }
    let fixed = info.fixed_batch;
    let batch = fixed.unwrap_or(spec.batch_size).max(1);
    let max_failures = (total / 20).max(8);
    let next_index = AtomicUsize::new(0);
    let failure_count = AtomicUsize::new(0);
    let (next, failures) = (&next_index, &failure_count);
    let (tx, rx) = mpsc::sync_channel::<Item>((batch * 2).clamp(4, 256));

    let mut counts = Counts::default();
    let mut done = 0usize;
    let mut outcomes: Vec<TileOutcome> = Vec::new();
    let mut last_delivery = Instant::now();
    let mut last_progress: Option<Instant> = None;
    let mut input: Vec<f32> = Vec::new();

    let deliver = |outcomes: &mut Vec<TileOutcome>, counts: &mut Counts, done: &mut usize| {
        if outcomes.is_empty() {
            return;
        }
        for outcome in outcomes.iter() {
            match outcome.state {
                TileState::Analysed { .. } => counts.analysed += 1,
                TileState::Background => counts.background += 1,
                TileState::Failed => counts.failed += 1,
            }
        }
        *done += outcomes.len();
        sink.deliver(std::mem::take(outcomes));
    };

    let reader = |tx: mpsc::SyncSender<Item>| {
        move || {
            loop {
                if control.stopping() {
                    return;
                }
                let index = next.fetch_add(1, Ordering::SeqCst);
                let Some(&(coord, need)) = spec.tiles.get(index) else {
                    return;
                };
                let (x, y) = spec.grid.origin0(coord);
                let request = RegionRequest {
                    file_id: spec.file_id,
                    level: spec.grid.level,
                    x,
                    y,
                    width: tile,
                    height: tile,
                    background: spec.contract.background_rgba(),
                };
                let item = match source.read(&request) {
                    Ok(rgba) => {
                        match prepare(&rgba, tile, spec.grid.valid_size(coord), &spec.contract) {
                            Some(prepared) => {
                                if need == Need::Score
                                    && spec.skip_background
                                    && is_background(&prepared)
                                {
                                    Item::Background(coord)
                                } else {
                                    Item::Ready(coord, prepared)
                                }
                            }
                            None => {
                                control.fail(
                                    Severity::Read,
                                    format!(
                                        "the host returned {} bytes for a {tile}×{tile} region",
                                        rgba.len()
                                    ),
                                );
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        let failed = failures.fetch_add(1, Ordering::SeqCst) + 1;
                        if failed > max_failures {
                            control.fail(
                                Severity::Read,
                                format!(
                                    "{failed} tiles could not be read (level {}); last error: {error}",
                                    spec.grid.level
                                ),
                            );
                            return;
                        }
                        Item::Failed(coord)
                    }
                };
                if tx.send(item).is_err() {
                    return;
                }
            }
        }
    };

    std::thread::scope(|scope| {
        for _ in 0..spec.readers.clamp(1, 16).min(total.max(1)) {
            scope.spawn(reader(tx.clone()));
        }
        drop(tx);
        let mut pending: Vec<(TileCoord, PreparedTile)> = Vec::with_capacity(batch);
        let mut readers_done = false;
        while !control.stopping() {
            let mut idle = false;
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(Item::Ready(coord, prepared)) => pending.push((coord, prepared)),
                Ok(Item::Background(coord)) => outcomes.push(TileOutcome {
                    coord,
                    state: TileState::Background,
                    payload: None,
                    embedding: None,
                }),
                Ok(Item::Failed(coord)) => outcomes.push(TileOutcome {
                    coord,
                    state: TileState::Failed,
                    payload: None,
                    embedding: None,
                }),
                Err(RecvTimeoutError::Timeout) => idle = true,
                Err(RecvTimeoutError::Disconnected) => readers_done = true,
            }
            // Fixed batches wait until full (or the end); dynamic batches
            // also go when reading stalls, keeping small jobs responsive.
            let run_now = pending.len() >= batch
                || (readers_done && !pending.is_empty())
                || (idle && fixed.is_none() && !pending.is_empty());
            if run_now {
                let items: Vec<(TileCoord, PreparedTile)> = std::mem::take(&mut pending);
                infer(
                    spec,
                    engine,
                    control,
                    &items,
                    fixed,
                    item_len,
                    &mut input,
                    &mut outcomes,
                );
            }
            if !outcomes.is_empty()
                && (outcomes.len() >= batch
                    || readers_done
                    || last_delivery.elapsed() >= DELIVERY_INTERVAL)
            {
                deliver(&mut outcomes, &mut counts, &mut done);
                last_delivery = Instant::now();
            }
            if last_progress.is_none_or(|at| at.elapsed() >= PROGRESS_INTERVAL) {
                last_progress = Some(Instant::now());
                sink.progress(&Progress {
                    total,
                    done,
                    counts,
                    elapsed: started.elapsed(),
                });
            }
            if readers_done && pending.is_empty() {
                break;
            }
        }
        drop(rx);
    });
    // Finished tiles are kept even when the job stops early.
    deliver(&mut outcomes, &mut counts, &mut done);
    sink.progress(&Progress {
        total,
        done,
        counts,
        elapsed: started.elapsed(),
    });
    if let Some(error) = control.error() {
        JobOutcome::Failed { error, counts }
    } else if control.is_cancelled() {
        JobOutcome::Cancelled(counts)
    } else if done < total {
        JobOutcome::Failed {
            error: format!("only {done} of {total} tiles were processed"),
            counts,
        }
    } else {
        JobOutcome::Completed(counts)
    }
}

#[allow(clippy::too_many_arguments)]
fn infer(
    spec: &JobSpec,
    engine: &mut dyn InferenceEngine,
    control: &JobControl,
    items: &[(TileCoord, PreparedTile)],
    fixed: Option<usize>,
    item_len: usize,
    input: &mut Vec<f32>,
    outcomes: &mut Vec<TileOutcome>,
) {
    let count = items.len();
    let tensor_batch = fixed.unwrap_or(count).max(count);
    input.resize(tensor_batch * item_len, 0.0);
    for (slot, (_, tile)) in input.chunks_exact_mut(item_len).zip(items) {
        write_input(tile, &spec.contract, spec.layout, slot);
    }
    // Pad a fixed batch with copies of a real tile (zeros are far outside
    // the training distribution and some runtimes special-case them).
    for slot in count..tensor_batch {
        input.copy_within(0..item_len, slot * item_len);
    }
    let output = match engine.run(input, tensor_batch, control.stop_flag()) {
        Ok(output) => output,
        Err(EngineError::Stopped) => return,
        Err(EngineError::Failed(message)) => {
            control.fail(Severity::Inference, format!("inference failed: {message}"));
            return;
        }
    };
    if output.recon.len() != tensor_batch * item_len {
        control.fail(
            Severity::Inference,
            format!(
                "the model returned {} values for a batch of {tensor_batch} tiles; expected {}",
                output.recon.len(),
                tensor_batch * item_len
            ),
        );
        return;
    }
    let latent_dim = engine.info().latent_dim;
    for (index, (coord, tile)) in items.iter().enumerate() {
        let scored = score_output(
            tile,
            &output.recon[index * item_len..(index + 1) * item_len],
            &spec.contract,
            spec.layout,
        );
        let embedding = match (&output.latent, latent_dim) {
            (Some(latent), Some(dim)) if latent.len() >= tensor_batch * dim => {
                Some(latent[index * dim..(index + 1) * dim].to_vec())
            }
            _ => None,
        };
        outcomes.push(TileOutcome {
            coord: *coord,
            state: TileState::Analysed {
                mae: scored.mae,
                max: scored.max,
            },
            payload: Some(TilePayload {
                size: tile.size,
                valid_w: tile.valid_w,
                valid_h: tile.valid_h,
                recon: scored.recon,
                error: scored.error,
            }),
            embedding,
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::engine::testing::{Behaviour, TestEngine};
    use crate::tiling::SlideGeometry;
    use eov_plugin_api::protocol::FileInfo;
    use std::sync::Mutex;

    /// Tiles at even columns are tissue-like, odd ones are glass; reads of
    /// listed tiles fail.
    pub(crate) struct Slide {
        pub failing: Vec<(i64, i64)>,
        pub reads: AtomicUsize,
    }

    impl PixelSource for Slide {
        fn read(&self, request: &RegionRequest) -> Result<Vec<u8>, String> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if self.failing.contains(&(request.x, request.y)) {
                return Err(format!("corrupt tile at {},{}", request.x, request.y));
            }
            let col = request.x / i64::from(request.width);
            let n = request.width as usize * request.height as usize;
            let mut data = Vec::with_capacity(n * 4);
            for i in 0..n {
                if col % 2 == 0 {
                    let v = if i % 2 == 0 { 90 } else { 200 };
                    data.extend_from_slice(&[v, 60, 150, 255]);
                } else {
                    data.extend_from_slice(&[250, 250, 250, 255]);
                }
            }
            Ok(data)
        }
    }

    #[derive(Default)]
    pub(crate) struct Collect {
        pub outcomes: Mutex<Vec<TileOutcome>>,
        pub deliveries: AtomicUsize,
        pub progress: Mutex<Vec<Progress>>,
    }

    impl JobSink for Collect {
        fn deliver(&self, outcomes: Vec<TileOutcome>) {
            self.deliveries.fetch_add(1, Ordering::SeqCst);
            self.outcomes.lock().unwrap().extend(outcomes);
        }

        fn progress(&self, progress: &Progress) {
            self.progress.lock().unwrap().push(*progress);
        }
    }

    fn spec(cols: u32) -> JobSpec {
        let geometry = SlideGeometry::from_file_info(&FileInfo {
            width: u64::from(cols) * 16,
            height: 16,
            ..Default::default()
        });
        let grid = TileGrid::new(&geometry, &geometry.levels[0], 16);
        JobSpec {
            file_id: 1,
            grid,
            tiles: grid.full_range().iter().map(|c| (c, Need::Score)).collect(),
            contract: Contract::histovae(),
            layout: Layout::Nchw,
            skip_background: true,
            batch_size: 4,
            readers: 2,
        }
    }

    fn slide() -> Slide {
        Slide {
            failing: Vec::new(),
            reads: AtomicUsize::new(0),
        }
    }

    #[test]
    fn static_batches_are_full_and_background_is_never_inferred() {
        let spec = spec(10);
        let mut engine = TestEngine::new(16 * 16 * 3, Some(4), Behaviour::Offset(0.01));
        let batches = Arc::clone(&engine.batches);
        let sink = Collect::default();
        let outcome = run_job(&spec, &slide(), &mut engine, &sink, &JobControl::default());
        let JobOutcome::Completed(counts) = outcome else {
            panic!("{outcome:?}");
        };
        assert_eq!(counts.analysed, 5, "even columns are tissue");
        assert_eq!(counts.background, 5, "glass is recorded, not left pending");
        // 5 tissue tiles with a fixed batch of 4: one full batch, one padded.
        assert_eq!(*batches.lock().unwrap(), vec![4, 4]);
        let outcomes = sink.outcomes.lock().unwrap();
        assert_eq!(outcomes.len(), 10);
        for outcome in outcomes.iter() {
            match outcome.state {
                TileState::Analysed { mae, .. } => {
                    assert!(mae > 0.0 && mae < 0.05, "{mae}");
                    assert!(outcome.payload.is_some());
                    assert_eq!(outcome.embedding.as_ref().map(Vec::len), Some(2));
                }
                TileState::Background => assert_eq!(outcome.coord.col % 2, 1),
                TileState::Failed => panic!("nothing failed"),
            }
        }
    }

    #[test]
    fn dynamic_batches_use_the_configured_size() {
        let mut spec = spec(16);
        spec.skip_background = false;
        let mut engine = TestEngine::new(16 * 16 * 3, None, Behaviour::Identity);
        let batches = Arc::clone(&engine.batches);
        let outcome = run_job(
            &spec,
            &slide(),
            &mut engine,
            &Collect::default(),
            &JobControl::default(),
        );
        assert!(matches!(outcome, JobOutcome::Completed(c) if c.analysed == 16));
        let batches = batches.lock().unwrap();
        assert_eq!(batches.iter().sum::<usize>(), 16);
        assert!(batches.iter().all(|b| *b <= 4));
    }

    #[test]
    fn inference_errors_are_reported_not_called_cancellations() {
        let spec = spec(12);
        let mut engine = TestEngine::new(16 * 16 * 3, Some(2), Behaviour::FailOn(1));
        let outcome = run_job(
            &spec,
            &slide(),
            &mut engine,
            &Collect::default(),
            &JobControl::default(),
        );
        match outcome {
            JobOutcome::Failed { error, counts } => {
                assert!(error.contains("wrong shape"), "{error}");
                // The first batch's results are kept.
                assert_eq!(counts.analysed, 2);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_few_unreadable_tiles_fail_alone_but_many_fail_the_job() {
        let spec = spec(10);
        let mut source = slide();
        source.failing = vec![(0, 0), (32, 0)];
        let mut engine = TestEngine::new(16 * 16 * 3, None, Behaviour::Identity);
        let sink = Collect::default();
        let outcome = run_job(&spec, &source, &mut engine, &sink, &JobControl::default());
        assert!(
            matches!(outcome, JobOutcome::Completed(c) if c.failed == 2),
            "{outcome:?}"
        );

        let mut source = slide();
        source.failing = (0..10).map(|col| (col * 16, 0)).collect();
        let mut engine = TestEngine::new(16 * 16 * 3, None, Behaviour::Identity);
        let outcome = run_job(
            &spec,
            &source,
            &mut engine,
            &Collect::default(),
            &JobControl::default(),
        );
        match outcome {
            JobOutcome::Failed { error, .. } => {
                assert!(error.contains("could not be read"), "{error}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn cancelling_stops_a_running_inference() {
        let spec = spec(8);
        let mut engine = TestEngine::new(16 * 16 * 3, None, Behaviour::Hang);
        let control = JobControl::new();
        let cancel = Arc::clone(&control);
        let started = Instant::now();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            cancel.cancel();
        });
        let outcome = run_job(&spec, &slide(), &mut engine, &Collect::default(), &control);
        canceller.join().unwrap();
        assert!(matches!(outcome, JobOutcome::Cancelled(_)), "{outcome:?}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn errors_rank_by_severity() {
        let control = JobControl::default();
        control.fail(Severity::Internal, "channel closed");
        control.fail(Severity::Inference, "CUDA out of memory");
        control.fail(Severity::Read, "bad tile");
        assert_eq!(control.error().as_deref(), Some("CUDA out of memory"));
        assert!(control.stopping());
        assert!(!control.is_cancelled());
    }

    #[test]
    fn empty_jobs_complete_and_mismatched_engines_fail() {
        let mut empty = spec(1);
        empty.tiles.clear();
        let mut engine = TestEngine::new(16 * 16 * 3, None, Behaviour::Identity);
        assert_eq!(
            run_job(
                &empty,
                &slide(),
                &mut engine,
                &Collect::default(),
                &JobControl::default()
            ),
            JobOutcome::Completed(Counts::default())
        );
        let mut wrong = TestEngine::new(8, None, Behaviour::Identity);
        assert!(matches!(
            run_job(
                &spec(2),
                &slide(),
                &mut wrong,
                &Collect::default(),
                &JobControl::default()
            ),
            JobOutcome::Failed { .. }
        ));
    }
}
