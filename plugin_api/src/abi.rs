//! The binary interface between eov and plugin libraries.
//!
//! Plugin authors do not use this module directly: implement
//! [`crate::sdk::Plugin`] and export it with [`crate::export_plugin!`].
//!
//! # Compatibility rules
//!
//! The ABI is deliberately small. Almost everything travels as JSON messages
//! defined in [`crate::protocol`]; only pixel data crosses as binary.
//!
//! - abi_stable 0.11's load-time layout check accepts a library whose
//!   prefix type (a module or a `#[sabi_trait]` vtable) has **more** fields
//!   than the host's copy, but rejects one with **fewer** ("too many
//!   fields": a 1.0 library in a host whose copy has one more field).
//! - The root module ([`PluginModule`]) is an `abi_stable` *prefix type*.
//!   The host loads it with `LibHeader::init_root_module`, which checks the
//!   `eov-plugin-api` major version and the layout of every type reachable
//!   from the module before any plugin code runs. Because of the rule above
//!   it checks every library against the root module's base layout
//!   ([`v1_0::PluginModule`]: the fields plugin API 1.0 was first built
//!   with, which every 1.x module has), and then the fields added since (so
//!   far [`PluginModule::slide_source`]) with the full layout, only when the
//!   library's module has them. A library built before a field was added
//!   therefore loads unchanged, and a newer one loads into an older eov,
//!   without its additions.
//! - [`PluginFfi`] and [`HostFfi`] are `#[sabi_trait]` objects. Their
//!   vtables are **frozen for 1.x**: by the rule above, a method appended
//!   to one would make every older plugin fail the check. New capabilities
//!   are JSON methods on [`HostFfi::call`] and new
//!   [`crate::protocol::HostEvent`]s, which older peers answer with
//!   `Unsupported` or receive as `Unknown`; pixel data keeps using the
//!   existing binary calls. The same holds for [`SlideSourceFfi`] and
//!   [`SlideFfi`]: new slide features go through
//!   [`SlideFfi::call`], or through new root-module fields checked the same
//!   way.
//! - [`ABI_VERSION`] is an additional explicit handshake; it only changes
//!   together with the crate's major version.
//!
//! # Slide sources
//!
//! A plugin can read slide formats eov does not read itself (Philips
//! iSyntax, Olympus VSI, Leica LIF, remote sources such as DICOMweb or
//! OMERO). It declares them in its manifest (`[[slide_formats]]`, see
//! [`crate::manifest::SlideFormatSpec`]), so eov knows the extensions and URI
//! schemes without running plugin code, and the SDK exports the root
//! module's [`PluginModule::slide_source`], which hands the host the
//! instance's [`SlideSourceFfi`]. Opening a slide returns a [`SlideFfi`]
//! handle: the metadata as JSON ([`crate::protocol::SlideInfo`]), region
//! reads into a host buffer and associated images.
//!
//! Threading: a slide source and its slides are `Send + Sync` and are
//! called from worker threads (slide openers, render workers, exports,
//! batch runs), several at once and on the same handle. eov holds none of
//! its locks while it calls them. A call that panics fails (the SDK turns
//! the panic into [`FfiErrorKind::Panicked`]) and the plugin is stopped;
//! one that returns an error fails only that read.

// `#[sabi_trait]` expands to impls inside an anonymous const, which trips this
// lint on current compilers; the generated code is correct.
#![allow(non_local_definitions)]

use abi_stable::{
    StableAbi, declare_root_module_statics,
    library::RootModule,
    package_version_strings, sabi_trait,
    sabi_types::VersionStrings,
    std_types::{RBox, RResult, RSliceMut, RStr, RString, RVec},
};

/// Explicit ABI generation, checked by the host in addition to the
/// `abi_stable` layout check.
pub const ABI_VERSION: u32 = 1;

/// The version of `eov-plugin-api` this crate is.
pub const API_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Largest region, in pixels, a single [`HostFfi::read_region`] may request.
pub const MAX_REGION_PIXELS: u64 = 8192 * 8192;

/// Largest raster, in pixels, a single [`HostFfi::put_raster`] may upload.
pub const MAX_RASTER_PIXELS: u64 = 4096 * 4096;

/// Largest region, in pixels, the host asks a [`SlideFfi::read_region`]
/// for at once (larger reads arrive in parts).
pub const MAX_SLIDE_READ_PIXELS: u64 = 4096 * 4096;

/// The index of [`PluginModule::slide_source`] among the root module's
/// fields (the first field added after plugin API 1.0).
pub const SLIDE_SOURCE_FIELD: usize = 3;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Classifies an [`FfiError`].
#[repr(u32)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiErrorKind {
    /// An ordinary failure; the message says what went wrong.
    Failed = 0,
    /// The callee panicked. The host quarantines plugins that panic.
    Panicked = 1,
    /// The method or event is not supported by this peer (feature detection).
    Unsupported = 2,
    /// The arguments could not be decoded or were out of range.
    InvalidArguments = 3,
    /// The plugin was disabled or quarantined; its host handle is revoked.
    Revoked = 4,
}

/// An error crossing the plugin boundary.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct FfiError {
    /// Raw [`FfiErrorKind`] value. Unknown values (from newer peers) must be
    /// treated as [`FfiErrorKind::Failed`]; see [`FfiError::kind`].
    pub kind: u32,
    pub message: RString,
}

impl FfiError {
    pub fn new(kind: FfiErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind: kind as u32,
            message: RString::from(message.into()),
        }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self::new(FfiErrorKind::Failed, message)
    }

    /// The error kind; unknown raw values map to [`FfiErrorKind::Failed`].
    pub fn kind(&self) -> FfiErrorKind {
        match self.kind {
            1 => FfiErrorKind::Panicked,
            2 => FfiErrorKind::Unsupported,
            3 => FfiErrorKind::InvalidArguments,
            4 => FfiErrorKind::Revoked,
            _ => FfiErrorKind::Failed,
        }
    }
}

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message.as_str())
    }
}

impl std::error::Error for FfiError {}

pub type FfiResult<T> = RResult<T, FfiError>;

// ---------------------------------------------------------------------------
// Pixel types
// ---------------------------------------------------------------------------

/// A request for slide pixels.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq)]
pub struct RegionRequest {
    pub file_id: i64,
    /// Pyramid level to read from (see `FileInfo::levels`).
    pub level: u32,
    /// Top-left corner in **level-0** pixels. May be negative or extend past
    /// the slide; pixels outside the scanned area are transparent (or
    /// `background`).
    pub x: i64,
    pub y: i64,
    /// Size in pixels **of `level`**. `width·height` must not exceed
    /// [`MAX_REGION_PIXELS`].
    pub width: u32,
    pub height: u32,
    /// `0xRRGGBBAA`. With alpha 0 the result keeps transparency (straight,
    /// non-premultiplied alpha); otherwise every pixel is composited onto this
    /// colour and the result is opaque.
    pub background: u32,
}

/// An RGBA8 image: straight (non-premultiplied) alpha, row-major, no padding.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct Pixels {
    pub width: u32,
    pub height: u32,
    pub data: RVec<u8>,
}

impl Pixels {
    /// Whether `data` holds exactly `width·height·4` bytes.
    pub fn is_well_formed(&self) -> bool {
        (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(4))
            == Some(self.data.len())
    }
}

/// A raster overlay object (e.g. a heat map) covering a level-0 rectangle.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq)]
pub struct RasterUpload {
    /// A layer previously created with `layer.create`.
    pub layer_id: RString,
    /// Replaces an existing object with the same id.
    pub object_id: RString,
    /// The level-0 rectangle the raster is stretched over.
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    /// At most [`MAX_RASTER_PIXELS`].
    pub pixels: Pixels,
    /// Bilinear (`true`) or nearest-neighbour (`false`) magnification.
    pub smooth: bool,
}

/// The pane a filter is being applied for.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Default)]
pub struct FilterView {
    pub pane: u32,
    pub file_id: i64,
    /// Level-0 coordinates of the pane centre.
    pub center_x: f64,
    pub center_y: f64,
    /// Screen (logical) pixels per level-0 pixel.
    pub zoom: f64,
    /// Pane size in logical pixels.
    pub width: f64,
    pub height: f64,
    /// Physical pixels of the frame per logical pixel.
    pub scale_factor: f64,
    /// Level-0 bounding box of the visible area.
    pub visible_x: f64,
    pub visible_y: f64,
    pub visible_width: f64,
    pub visible_height: f64,
    /// Increases with every rendered frame of this pane.
    pub frame: u64,
}

/// A rendered frame handed to a CPU filter, to be modified in place.
///
/// Pixels are RGBA8 with straight alpha: the slide's pixels, after eov's
/// stain and tone adjustments, are opaque, and the rest of the pane is
/// transparent. eov composites the viewport background and sharpens after
/// the filters, so a filter never sees or changes the background.
/// Rows are `stride` bytes apart (`stride >= width·4`).
#[repr(C)]
#[derive(StableAbi)]
pub struct FilterFrame<'a> {
    pub pixels: RSliceMut<'a, u8>,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub view: FilterView,
}

// ---------------------------------------------------------------------------
// Plugin instance (implemented by plugins, called by the host)
// ---------------------------------------------------------------------------

/// A live plugin instance.
///
/// Created by [`PluginModule::create`]. The host calls `handle_event` on its
/// UI thread with no host locks held, and `apply_filter` on render worker
/// threads (possibly concurrently for different panes).
#[sabi_trait]
pub trait PluginFfi: Send + Sync {
    /// Handles a JSON-encoded [`crate::protocol::HostEvent`] and returns a
    /// JSON response (`null` when the event has none).
    fn handle_event(&self, event_json: RStr<'_>) -> FfiResult<RString>;

    /// Applies the CPU filter `filter_id` to `frame` in place.
    fn apply_filter(&self, filter_id: RStr<'_>, frame: FilterFrame<'_>) -> FfiResult<()>;

    /// Releases resources. No other method is called afterwards.
    #[sabi(last_prefix_field)]
    fn shutdown(&self);
}

/// An owned plugin instance.
pub type PluginBox = PluginFfi_TO<'static, RBox<()>>;

// ---------------------------------------------------------------------------
// Host API (implemented by the host, called by plugins)
// ---------------------------------------------------------------------------

/// The host services available to one plugin.
///
/// Every method may be called from any thread. None of them blocks on the
/// host's UI thread: queries are answered from published state and UI
/// mutations are queued and applied in call order.
#[sabi_trait]
pub trait HostFfi: Send + Sync {
    /// Invokes the JSON method `method` (see [`crate::protocol::methods`]).
    fn call(&self, method: RStr<'_>, args_json: RStr<'_>) -> FfiResult<RString>;

    /// Reads slide pixels on the calling thread through a plugin-private
    /// slide handle. Never holds host state locks while decoding.
    fn read_region(&self, request: RegionRequest) -> FfiResult<Pixels>;

    /// Reads an associated image (`"label"`, `"macro"`, `"thumbnail"`, …).
    fn read_associated_image(&self, file_id: i64, name: RStr<'_>) -> FfiResult<Pixels>;

    /// Adds or replaces a raster overlay object.
    #[sabi(last_prefix_field)]
    fn put_raster(&self, raster: RasterUpload) -> FfiResult<()>;
}

/// An owned host API handle.
pub type HostBox = HostFfi_TO<'static, RBox<()>>;

// ---------------------------------------------------------------------------
// Slide sources (implemented by plugins, called by the host)
// ---------------------------------------------------------------------------

/// A region of one pyramid level, in pixels **of that level** (unlike
/// [`RegionRequest`], whose corner is in level-0 pixels).
///
/// The host only asks for regions inside the level (`x + width` and
/// `y + height` within the level's size, as [`crate::protocol::SlideInfo`]
/// gives it) of at most [`MAX_SLIDE_READ_PIXELS`] pixels; it reads areas
/// past the edges as "no data" itself.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlideRegion {
    /// Index into [`crate::protocol::SlideInfo::levels`].
    pub level: u32,
    /// Top-left corner, in pixels of `level`.
    pub x: u64,
    pub y: u64,
    pub width: u32,
    pub height: u32,
}

/// The slide formats one plugin instance reads (see the
/// [module documentation](self#slide-sources-plugin-api-11)).
///
/// `Send + Sync`: the host opens slides from worker threads, several at
/// once, with none of its locks held.
#[sabi_trait]
pub trait SlideSourceFfi: Send + Sync {
    /// Opens a slide. `request_json` is a JSON
    /// [`crate::protocol::SlideOpenRequest`]: which of the manifest's
    /// formats matched, and the file path or URI.
    #[sabi(last_prefix_field)]
    fn open(&self, request_json: RStr<'_>) -> FfiResult<SlideBox>;
}

/// An owned slide source.
pub type SlideSourceBox = SlideSourceFfi_TO<'static, RBox<()>>;

/// One open slide.
///
/// `Send + Sync`: render workers read regions concurrently, from several
/// threads at once on the same handle, with none of the host's locks held.
/// The host drops the handle (from any thread, possibly after the plugin's
/// `shutdown`) when the slide is closed everywhere.
#[sabi_trait]
pub trait SlideFfi: Send + Sync {
    /// The slide's metadata as a JSON [`crate::protocol::SlideInfo`].
    /// Called once, right after opening; the host validates it.
    fn info(&self) -> FfiResult<RString>;

    /// Writes `region` into `out`: `region.width·region.height·4` bytes of
    /// RGBA with straight alpha, rows top to bottom, no padding. `out`
    /// arrives zeroed, i.e. transparent: pixels without image data may be
    /// left alone (the host shows them as the slide's background).
    fn read_region(&self, region: SlideRegion, out: RSliceMut<'_, u8>) -> FfiResult<()>;

    /// Decodes the associated image `name` (one listed in
    /// [`crate::protocol::SlideInfo::associated_images`], at the size listed
    /// there) as RGBA with straight alpha.
    fn read_associated_image(&self, name: RStr<'_>) -> FfiResult<Pixels>;

    /// A JSON method for slide features added later (the vtable is frozen); unknown methods answer [`FfiErrorKind::Unsupported`].
    #[sabi(last_prefix_field)]
    fn call(&self, method: RStr<'_>, args_json: RStr<'_>) -> FfiResult<RString>;
}

/// An owned slide handle.
pub type SlideBox = SlideFfi_TO<'static, RBox<()>>;

// ---------------------------------------------------------------------------
// Root module
// ---------------------------------------------------------------------------

/// The root module every plugin library exports (via
/// [`crate::export_plugin!`]).
///
/// Fields after `create` were added in later 1.x versions; a module built
/// against an older version does not have them (see the
/// [compatibility rules](self#compatibility-rules)), so their accessors
/// return `Option`.
#[repr(C)]
#[derive(StableAbi)]
#[sabi(kind(Prefix(prefix_ref = PluginModuleRef)))]
#[sabi(missing_field(panic))]
pub struct PluginModule {
    /// Must equal the host's [`ABI_VERSION`].
    pub abi_version: u32,
    /// The `eov-plugin-api` version the plugin was compiled against.
    pub api_version: RStr<'static>,
    /// Creates the plugin instance. `init_json` is a JSON
    /// [`crate::protocol::PluginInit`].
    #[sabi(last_prefix_field)]
    pub create: extern "C" fn(host: HostBox, init_json: RStr<'_>) -> FfiResult<PluginBox>,
    /// The slide source of an instance this module created (missing in
    /// libraries built before slide sources were added)
    /// (the formats of the manifest's `[[slide_formats]]`). The host asks
    /// for it once, after `create`, for plugins that declare formats.
    #[sabi(missing_field(option))]
    pub slide_source: extern "C" fn(instance: &PluginBox) -> FfiResult<SlideSourceBox>,
}

impl RootModule for PluginModuleRef {
    declare_root_module_statics! {PluginModuleRef}
    const BASE_NAME: &'static str = "eov_plugin";
    const NAME: &'static str = "eov plugin";
    const VERSION_STRINGS: VersionStrings = package_version_strings!();
}

/// The root module as plugin API 1.0 defined it, frozen: the layout the
/// host checks every plugin library against first (see the
/// [compatibility rules](self#compatibility-rules)). Plugins never use it.
pub mod v1_0 {
    use super::{FfiResult, HostBox, PluginBox};
    use abi_stable::{
        StableAbi, declare_root_module_statics, library::RootModule, package_version_strings,
        sabi_types::VersionStrings, std_types::RStr,
    };

    /// [`super::PluginModule`] without the fields added after 1.0. Its
    /// name, field names and types must never change.
    #[repr(C)]
    #[derive(StableAbi)]
    #[sabi(kind(Prefix(prefix_ref = PluginModuleRef)))]
    #[sabi(missing_field(panic))]
    pub struct PluginModule {
        pub abi_version: u32,
        pub api_version: RStr<'static>,
        #[sabi(last_prefix_field)]
        pub create: extern "C" fn(host: HostBox, init_json: RStr<'_>) -> FfiResult<PluginBox>,
    }

    impl RootModule for PluginModuleRef {
        declare_root_module_statics! {PluginModuleRef}
        const BASE_NAME: &'static str = "eov_plugin";
        const NAME: &'static str = "eov plugin";
        const VERSION_STRINGS: VersionStrings = package_version_strings!();
    }
}

/// Returns the expected shared-library file name for a plugin id on the
/// current platform.
pub fn plugin_library_filename(id: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{id}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{id}.dll")
    } else {
        format!("lib{id}.so")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_macro_uses_abi_stables_loader_symbol() {
        // `export_plugin!` spells the exported static's name literally so that
        // plugins do not need a direct `abi_stable` dependency.
        assert_eq!(
            abi_stable::library::ROOT_MODULE_LOADER_NAME,
            crate::sdk::ROOT_MODULE_LOADER_SYMBOL
        );
    }

    #[test]
    fn unknown_error_kinds_degrade_to_failed() {
        let error = FfiError {
            kind: 99,
            message: "x".into(),
        };
        assert_eq!(error.kind(), FfiErrorKind::Failed);
        assert_eq!(
            FfiError::new(FfiErrorKind::Unsupported, "y").kind(),
            FfiErrorKind::Unsupported
        );
    }

    extern "C" fn refuse(_host: HostBox, _init_json: RStr<'_>) -> FfiResult<PluginBox> {
        abi_stable::std_types::RErr(FfiError::failed("test module"))
    }

    extern "C" fn no_source(_instance: &PluginBox) -> FfiResult<SlideSourceBox> {
        abi_stable::std_types::RErr(FfiError::failed("test module"))
    }

    /// A root module as a plugin built against plugin API 1.0 exports it.
    fn module_1_0() -> v1_0::PluginModuleRef {
        use abi_stable::prefix_type::PrefixTypeTrait;
        v1_0::PluginModule {
            abi_version: ABI_VERSION,
            api_version: RStr::from_str("1.0.0"),
            create: refuse,
        }
        .leak_into_prefix()
    }

    fn module_1_1() -> PluginModuleRef {
        use abi_stable::prefix_type::PrefixTypeTrait;
        PluginModule {
            abi_version: ABI_VERSION,
            api_version: RStr::from_str(API_VERSION),
            create: refuse,
            slide_source: no_source,
        }
        .leak_into_prefix()
    }

    fn has_slide_source<M: abi_stable::prefix_type::PrefixRefTrait>(module: M) -> bool {
        module
            .to_prefix_ref()
            .field_accessibility()
            .at(SLIDE_SOURCE_FIELD)
            .is_accessible()
    }

    #[test]
    fn hosts_check_plugins_against_the_frozen_1_0_root_module() {
        use abi_stable::library::LibHeader;
        // abi_stable rejects a module with fewer fields than the host's
        // copy of its type ...
        let old = LibHeader::from_module(module_1_0());
        let error = old
            .init_root_module::<PluginModuleRef>()
            .err()
            .expect("a 1.0 module fails the 1.1 layout")
            .to_string();
        assert!(error.contains("too many fields"), "{error}");
        // ... so the host checks the 1.0 layout, which every 1.x module
        // extends, and looks for the newer fields itself.
        let loaded = old.init_root_module::<v1_0::PluginModuleRef>().unwrap();
        assert_eq!(loaded.api_version().as_str(), "1.0.0");
        assert!(!has_slide_source(loaded));

        // A 1.1 module passes both (the first is what a 1.0 eov checks).
        let new = LibHeader::from_module(module_1_1());
        let base = new.init_root_module::<v1_0::PluginModuleRef>().unwrap();
        assert!(has_slide_source(base));
        let full = new.init_root_module::<PluginModuleRef>().unwrap();
        assert!(full.slide_source().is_some());
    }

    #[test]
    fn pixels_well_formedness() {
        let ok = Pixels {
            width: 2,
            height: 3,
            data: vec![0; 24].into(),
        };
        assert!(ok.is_well_formed());
        let bad = Pixels {
            width: 2,
            height: 3,
            data: vec![0; 23].into(),
        };
        assert!(!bad.is_well_formed());
    }
}
