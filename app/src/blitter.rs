//! Low-level tile blitting functions for CPU rendering
//!
//! This module re-exports the core blitting algorithms from the `common`
//! crate and adds app-specific helpers that depend on Slint or app types.

use slint::{Rgba8Pixel, SharedPixelBuffer};

// Re-export all public blitting primitives from common.
pub use eov_common::blitter::{
    BlitRect, CoarseSrc, TileSrc, blit_tile, blit_tile_lanczos3, blit_tile_trilinear,
    fast_fill_rgba,
};

pub struct FrameSrc<'a> {
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
}

pub fn reproject_frame(
    dest: &mut [u8],
    dest_width: u32,
    dest_height: u32,
    src: FrameSrc<'_>,
    src_viewport: &crate::render_pool::CachedCpuFrame,
    dest_viewport: &eov_common::Viewport,
    clear_rgba: [u8; 4],
) {
    eov_common::blitter::reproject_frame(
        dest,
        dest_width,
        dest_height,
        src.pixels,
        src.width,
        src.height,
        src_viewport.viewport.bounds().left,
        src_viewport.viewport.bounds().top,
        src_viewport.viewport.zoom,
        dest_viewport.bounds().left,
        dest_viewport.bounds().top,
        dest_viewport.zoom,
        clear_rgba,
    );
}

/// Create a SharedPixelBuffer from raw RGBA data
pub fn create_image_buffer(
    data: &[u8],
    width: u32,
    height: u32,
) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
    let expected_len = (width * height * 4) as usize;
    if data.len() < expected_len {
        return None;
    }

    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(width, height);
    buffer
        .make_mut_bytes()
        .copy_from_slice(&data[..expected_len]);
    Some(buffer)
}
