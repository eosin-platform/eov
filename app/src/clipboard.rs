//! The system clipboard, served by one long-lived worker thread so the UI
//! never waits on it (SH-10).

use crate::gpu::SurfaceSlot;
use crate::pane_ui::{with_gpu_renderer, with_pane_render_cache};
use crate::state::PaneId;
use eov_common::{RgbaImageData, crop_transparent_edges};
use std::borrow::Cow;
use std::sync::OnceLock;
use std::sync::mpsc;
use std::thread;
use tracing::warn;

pub(crate) type PaneClipboardImage = RgbaImageData;

/// Called on the UI thread with whether a copy succeeded.
type CopyDone = Box<dyn FnOnce(bool) + Send>;

enum ClipboardCommand {
    SetText {
        text: String,
    },
    SetImage {
        image: PaneClipboardImage,
        done: CopyDone,
    },
}

static CLIPBOARD_WORKER: OnceLock<mpsc::Sender<ClipboardCommand>> = OnceLock::new();

/// Copies text to the clipboard (on the clipboard worker thread).
pub(crate) fn copy_text(text: String) {
    if !send(ClipboardCommand::SetText { text }) {
        warn!("Failed to copy text to clipboard");
    }
}

/// Copies an image to the clipboard without blocking; `done` runs on the UI
/// thread with the outcome.
pub(crate) fn copy_image(image: PaneClipboardImage, done: impl FnOnce(bool) + Send + 'static) {
    let image = crop_transparent_edges(image);
    let done: CopyDone = Box::new(done);
    if let Err(mpsc::SendError(ClipboardCommand::SetImage { done, .. })) =
        clipboard_worker_sender().send(ClipboardCommand::SetImage { image, done })
    {
        let _ = slint::invoke_from_event_loop(move || done(false));
    }
}

/// The pane's frame on screen (with its plugin filters), without overlays.
pub(crate) fn capture_pane_clipboard_image(pane: PaneId) -> Option<PaneClipboardImage> {
    // CPU frames (also filtered GPU frames) are shown from pixel buffers;
    // a GPU surface is read back instead.
    let cached_image = with_pane_render_cache(pane.0 + 1, |cache| {
        let buffer = cache.get(pane.0)?.content.as_ref()?.to_rgba8()?;
        Some(PaneClipboardImage {
            width: buffer.width() as usize,
            height: buffer.height() as usize,
            pixels: buffer.as_bytes().to_vec(),
        })
    });

    if cached_image.is_some() {
        return cached_image;
    }

    with_gpu_renderer(|renderer| renderer.borrow_mut().read_surface_rgba(SurfaceSlot(pane.0)))
        .flatten()
        .map(|(width, height, pixels)| PaneClipboardImage {
            width: width as usize,
            height: height as usize,
            pixels,
        })
}

fn clipboard_worker_sender() -> &'static mpsc::Sender<ClipboardCommand> {
    CLIPBOARD_WORKER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<ClipboardCommand>();
        let worker = tx.clone();
        thread::Builder::new()
            .name("clipboard-worker".to_string())
            .spawn(move || clipboard_worker_loop(rx))
            .expect("failed to spawn clipboard worker");
        worker
    })
}

fn send(command: ClipboardCommand) -> bool {
    clipboard_worker_sender().send(command).is_ok()
}

fn clipboard_worker_loop(rx: mpsc::Receiver<ClipboardCommand>) {
    // The clipboard object lives as long as the worker, so it keeps
    // serving what it owns (X11, Wayland) without waiting per copy.
    let mut clipboard = match arboard::Clipboard::new() {
        Ok(clipboard) => Some(clipboard),
        Err(err) => {
            warn!("Failed to initialize clipboard worker: {}", err);
            None
        }
    };

    while let Ok(command) = rx.recv() {
        match command {
            ClipboardCommand::SetText { text } => {
                if let Some(clipboard) = clipboard.as_mut() {
                    set_text_with_clipboard(clipboard, text);
                }
            }
            ClipboardCommand::SetImage { image, done } => {
                let ok = clipboard
                    .as_mut()
                    .is_some_and(|clipboard| set_image_with_clipboard(clipboard, image));
                let _ = slint::invoke_from_event_loop(move || done(ok));
            }
        }
    }
}

fn set_text_with_clipboard(clipboard: &mut arboard::Clipboard, text: String) -> bool {
    if let Err(err) = clipboard.set_text(text) {
        warn!("Failed to copy text to clipboard: {}", err);
        return false;
    }
    true
}

fn set_image_with_clipboard(clipboard: &mut arboard::Clipboard, image: PaneClipboardImage) -> bool {
    let image_data = arboard::ImageData {
        width: image.width,
        height: image.height,
        bytes: Cow::Owned(image.pixels),
    };
    if let Err(err) = clipboard.set_image(image_data) {
        warn!("Failed to copy image to clipboard: {}", err);
        return false;
    }
    true
}
