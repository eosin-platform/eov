//! Application and plugin updates, plugin management and deployment policy.
//!
//! Nothing here contacts the network without a
//! [`network::NetworkPermission`], which only an explicit `--allow-network`
//! or an interactive "yes" can create (and the deployment policy can forbid).
//!
//! # Library API
//!
//! The CLI is a thin layer over functions a GUI can call directly with its own
//! [`prompt::Prompter`] (dialogs instead of a terminal) and its own
//! [`network::ReleaseClient`]:
//!
//! - **Plugins** (`plugins`): [`plugins::inventory`] lists installed packages
//!   with provenance, compatibility, disabled state and which duplicate loads;
//!   [`plugins::list_plugins`] joins that with the official catalog
//!   ([`plugins::fetch_catalog`]); [`plugins::install`], [`plugins::remove`],
//!   [`plugins::set_plugin_enabled`], [`plugins::lock`] and
//!   [`plugins::sync`] change the installation. All take a
//!   [`plugins::PluginContext`].
//! - **Application** (`app`): [`plan_update`] returns an
//!   [`UpdatePlan`] (strategy, signature, per-plugin actions with reasons)
//!   without changing anything; [`execute_update`] applies it;
//!   [`rollback_plan`] / [`rollback`] restore the previous installation; and
//!   [`take_helper_report`] returns what the post-exit update helper reported
//!   (show it once at startup).
//! - **Plugin host integration**: [`disabled_plugins`] /
//!   [`is_plugin_disabled`] (do not load these), [`policy::current`]
//!   (`plugin_dirs`, `allow_third_party_plugins`), and
//!   [`compat::environment_matches`] (bounded, prerelease-aware compatibility).

mod app;
mod appimage;
mod cargo_install;
pub mod compat;
mod flatpak;
mod handoff;
mod homebrew;
pub mod keys;
pub mod manifest;
pub mod network;
pub mod platform;
pub mod plugins;
pub mod policy;
pub mod prompt;
pub mod registry;
mod rollback;
pub mod signature;
mod source;
mod transaction;
pub mod transport;

pub use plugins::OutputFormat as PluginOutputFormat;

/// Library API for a GUI updater (see the module documentation).
#[allow(unused_imports)]
pub use app::{
    PlannedUpdate, RollbackPlan, UpdateContext, UpdateOutcome, UpdatePlan, UpdateRequest,
    execute as execute_update, plan as plan_update, rollback, rollback_plan, take_helper_report,
};
pub use app::{UpdateOptions, run_cli as run_update_cli};
/// Integration points for the plugin host: plugins the user disabled with
/// `eov plugin disable` must not be loaded.
#[allow(unused_imports)]
pub use registry::{disabled_plugins, is_plugin_disabled};
