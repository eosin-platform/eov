//! eovae: reconstruction-error ("VAE anomaly") analysis for eov.
//!
//! The plugin reads slide tiles through the host, runs an ONNX
//! autoencoder on them and shows where the reconstruction differs from the
//! slide: a whole-slide error map, the reconstruction itself and per-pixel
//! differences (as host-drawn overlay rasters), with error statistics, the
//! highest-error regions and CSV export. Research use only.
//!
//! Module map:
//! - [`plugin`]: state, host events, model loading, engines and jobs;
//!   [`presenter`] pushes what is visible; [`sidebar`] is the panel.
//! - [`pipeline`] runs a job; [`engine`] is the inference interface and
//!   `ort_backend` its ONNX Runtime implementation (feature `onnx`).
//! - [`tiling`], [`contract`], [`model`], [`onnx_scan`], [`run`], [`slide`],
//!   [`results`], [`db`], [`render`] and [`stats`] are host-independent.

pub mod config;
pub mod contract;
pub mod db;
pub mod engine;
pub mod model;
pub mod onnx_scan;
#[cfg(feature = "onnx")]
pub mod ort_backend;
pub mod pipeline;
pub mod plugin;
pub mod presenter;
pub mod render;
pub mod results;
pub mod run;
pub mod sidebar;
pub mod slide;
pub mod stats;
pub mod tiling;

#[cfg(test)]
mod tests;

use engine::Backend;
use eov_plugin_api::prelude::*;
use std::sync::Arc;

/// The plugin, generic over its inference backend (tests use a fake one).
pub struct Eovae<B: Backend> {
    core: Arc<plugin::Core<B>>,
}

impl<B: Backend> Plugin for Eovae<B> {
    fn create(host: Host, init: PluginInit) -> Result<Self, String> {
        Ok(Self {
            core: plugin::Core::create(host, init)?,
        })
    }

    fn on_event(&self, event: HostEvent) -> Result<(), String> {
        self.core.on_event(event)
    }

    fn ui_callback(&self, _surface_id: &str, name: &str, args: &[Value]) -> Result<Value, String> {
        sidebar::callback(&self.core, name, args)
    }

    fn shutdown(&self) {
        self.core.shutdown();
    }
}

/// A backend for builds without ONNX Runtime support.
pub struct NoRuntime;

impl Backend for NoRuntime {
    fn new(_plugin_root: &std::path::Path) -> Self {
        NoRuntime
    }

    fn create(
        &self,
        _request: &engine::EngineRequest,
        _status: &dyn Fn(&str),
        _stop: &[Arc<std::sync::atomic::AtomicBool>],
    ) -> Result<Box<dyn engine::InferenceEngine>, String> {
        Err(
            "this eovae build has no ONNX Runtime support (built without the `onnx` feature)"
                .into(),
        )
    }

    fn runtime_label(&self) -> String {
        "no inference runtime in this build".into()
    }
}

#[cfg(feature = "onnx")]
pub type EovaePlugin = Eovae<ort_backend::OrtBackend>;
#[cfg(not(feature = "onnx"))]
pub type EovaePlugin = Eovae<NoRuntime>;

eov_plugin_api::export_plugin!(EovaePlugin);
