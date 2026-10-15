//! Overlay layers drawn over the slides: storage, rendering, labels and
//! picking. Used for the host's own annotations (ROI, measurements, tool
//! previews) and for everything plugins draw.

pub mod geometry;
pub mod labels;
pub mod raster;
pub mod store;
pub mod worker;

pub use store::{ObjectKey, OverlayStore, Owner};
