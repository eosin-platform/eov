//! Plugin packages on disk.
//!
//! [`discovery`] scans the plugin directory for `.eop` packages, extracts
//! them and parses their manifests. Loading and running plugins is the job of
//! [`crate::plugin_runtime`]; their UI lives in [`crate::host_ui`].

pub mod discovery;
