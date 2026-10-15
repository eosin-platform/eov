//! Low-level tile blitting functions for CPU rendering
//!
//! This module re-exports the core blitting algorithms from the `common`
//! crate and adds app-specific helpers that depend on Slint or app types.

use slint::{Rgba8Pixel, SharedPixelBuffer};

// Re-export all public blitting primitives from common.
pub use eov_common::blitter::{
    BlitRect, CoarseSrc, TileSrc, blit_tile, blit_tile_lanczos3, blit_tile_nearest,
    blit_tile_trilinear,
};

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
