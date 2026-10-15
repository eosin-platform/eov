//! eovae: reconstruction-error analysis for eov (work in progress).

pub mod contract;
pub mod db;
pub mod engine;
pub mod model;
pub mod onnx_scan;
#[cfg(feature = "onnx")]
pub mod ort_backend;
pub mod pipeline;
pub mod render;
pub mod results;
pub mod run;
pub mod slide;
pub mod stats;
pub mod tiling;
