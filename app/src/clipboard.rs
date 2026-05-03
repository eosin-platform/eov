use crate::gpu::SurfaceSlot;
use crate::pane_ui::{with_gpu_renderer, with_pane_render_cache};
use crate::state::PaneId;
use common::{
    RgbaImageData, Viewport, crop_image_to_viewport_bounds as crop_to_viewport_bounds,
    crop_transparent_edges,
};
use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tracing::warn;

pub(crate) type PaneClipboardImage = RgbaImageData;

enum ClipboardCommand {
    SetText {
        text: String,
    },
    SetImage {
        image: PaneClipboardImage,
        response: mpsc::Sender<bool>,
    },
}

static CLIPBOARD_WORKER: OnceLock<mpsc::Sender<ClipboardCommand>> = OnceLock::new();

pub(crate) fn copy_text_to_clipboard(
    clipboard: &Rc<RefCell<Option<arboard::Clipboard>>>,
    text: String,
) {
    let _ = clipboard;
    if !send_text_to_clipboard_worker(text) {
        warn!("Failed to copy text to clipboard");
    }
}

pub(crate) fn copy_image_to_clipboard(
    clipboard: &Rc<RefCell<Option<arboard::Clipboard>>>,
    image: PaneClipboardImage,
) -> bool {
    let _ = clipboard;
    let image = crop_transparent_edges(image);
    send_image_to_clipboard_worker(image)
}

pub(crate) fn crop_image_to_viewport_bounds(
    image: PaneClipboardImage,
    viewport: &Viewport,
) -> PaneClipboardImage {
    crop_to_viewport_bounds(image, viewport)
}

pub(crate) fn capture_pane_clipboard_image(pane: PaneId) -> Option<PaneClipboardImage> {
    let cached_image = with_pane_render_cache(pane.0 + 1, |cache| {
        let entry = cache.get(pane.0)?;
        if !entry.preview_buffer.is_empty() && entry.preview_width > 0 && entry.preview_height > 0 {
            return Some(PaneClipboardImage {
                width: entry.preview_width as usize,
                height: entry.preview_height as usize,
                pixels: entry.preview_buffer.clone(),
            });
        }

        let cpu_frame = entry.cpu_frame.as_ref()?;
        Some(PaneClipboardImage {
            width: cpu_frame.width as usize,
            height: cpu_frame.height as usize,
            pixels: cpu_frame.pixels.clone(),
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

fn send_text_to_clipboard_worker(text: String) -> bool {
    if clipboard_worker_sender()
        .send(ClipboardCommand::SetText { text })
        .is_err()
    {
        return false;
    }
    true
}

fn send_image_to_clipboard_worker(image: PaneClipboardImage) -> bool {
    let (response_tx, response_rx) = mpsc::channel();
    if clipboard_worker_sender()
        .send(ClipboardCommand::SetImage {
            image,
            response: response_tx,
        })
        .is_err()
    {
        return false;
    }
    response_rx.recv().unwrap_or(false)
}

fn clipboard_worker_loop(rx: mpsc::Receiver<ClipboardCommand>) {
    let mut clipboard = match arboard::Clipboard::new() {
        Ok(clipboard) => clipboard,
        Err(err) => {
            warn!("Failed to initialize clipboard worker: {}", err);
            while let Ok(command) = rx.recv() {
                match command {
                    ClipboardCommand::SetText { .. } => {}
                    ClipboardCommand::SetImage { response, .. } => {
                        let _ = response.send(false);
                    }
                }
            }
            return;
        }
    };

    while let Ok(command) = rx.recv() {
        match command {
            ClipboardCommand::SetText { text } => {
                let _ = set_text_with_clipboard(&mut clipboard, text);
            }
            ClipboardCommand::SetImage { image, response } => {
                let ok = set_image_with_clipboard(&mut clipboard, image);
                let _ = response.send(ok);
            }
        }
    }
}

fn set_text_with_clipboard(clipboard: &mut arboard::Clipboard, text: String) -> bool {
    #[cfg(all(
        unix,
        not(any(target_os = "macos", target_os = "android", target_os = "emscripten"))
    ))]
    {
        use arboard::SetExtLinux;
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        if let Err(err) = clipboard.set().wait_until(deadline).text(text) {
            warn!("Failed to copy text to clipboard: {}", err);
            return false;
        }
        true
    }

    #[cfg(not(all(
        unix,
        not(any(target_os = "macos", target_os = "android", target_os = "emscripten"))
    )))]
    {
        if let Err(err) = clipboard.set_text(text) {
            warn!("Failed to copy text to clipboard: {}", err);
            return false;
        }
        true
    }
}

fn set_image_with_clipboard(clipboard: &mut arboard::Clipboard, image: PaneClipboardImage) -> bool {
    let image_data = arboard::ImageData {
        width: image.width,
        height: image.height,
        bytes: Cow::Owned(image.pixels),
    };

    #[cfg(all(
        unix,
        not(any(target_os = "macos", target_os = "android", target_os = "emscripten"))
    ))]
    {
        use arboard::SetExtLinux;
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        if let Err(err) = clipboard.set().wait_until(deadline).image(image_data) {
            warn!("Failed to copy image to clipboard: {}", err);
            return false;
        }
        true
    }

    #[cfg(not(all(
        unix,
        not(any(target_os = "macos", target_os = "android", target_os = "emscripten"))
    )))]
    {
        if let Err(err) = clipboard.set_image(image_data) {
            warn!("Failed to copy image to clipboard: {}", err);
            return false;
        }
        true
    }
}
