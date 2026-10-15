//! Error types for the common crate

use thiserror::Error;

/// Result type alias using our Error type
pub type Result<T> = std::result::Result<T, Error>;

/// Common crate error types
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("Failed to open WSI file: {0}")]
    OpenFile(String),

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    /// OpenSlide does not recognise the file as a slide.
    #[error("Unsupported slide format: {0}")]
    UnsupportedFormat(String),

    #[error("Failed to read tile at level {level}, x={x}, y={y}: {message}")]
    ReadTile {
        level: u32,
        x: u64,
        y: u64,
        message: String,
    },

    /// The OpenSlide handle is in OpenSlide's terminal ("sticky") error state
    /// and could not be replaced yet.
    ///
    /// This is **transient**: reopening is rate-limited, so retry the read
    /// later. It never means that the requested region itself is bad; see
    /// [`Error::CorruptRegion`] for that.
    #[error("OpenSlide handle is unusable (recovering): {0}")]
    HandlePoisoned(String),

    /// The region failed to decode on a freshly opened handle too, so the
    /// slide data itself is damaged there. The region is remembered and
    /// further reads of it fail fast with this error for a while; draw a
    /// placeholder instead of retrying.
    #[error("Slide data is unreadable at level {level}, x={x}, y={y}, {width}x{height}: {message}")]
    CorruptRegion {
        level: u32,
        x: i64,
        y: i64,
        width: u32,
        height: u32,
        message: String,
    },

    /// A read was refused before allocating because it is larger than the
    /// configured pixel cap.
    #[error("Region {width}x{height} exceeds the limit of {max_pixels} pixels")]
    RegionTooLarge {
        width: u64,
        height: u64,
        max_pixels: u64,
    },

    /// The named associated image does not exist.
    #[error("Associated image not found: {0}")]
    AssociatedImageNotFound(String),

    /// A caller passed an argument that can never succeed.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// An allocation of the given number of bytes failed.
    #[error("Out of memory allocating {0} bytes")]
    OutOfMemory(usize),

    #[error("Invalid level: {0} (max: {1})")]
    InvalidLevel(u32, u32),

    #[error("Invalid coordinates: ({x}, {y}) at level {level}")]
    InvalidCoordinates { x: i64, y: i64, level: u32 },

    #[error("OpenSlide error: {0}")]
    OpenSlide(String),

    #[error("Image encoding error: {0}")]
    ImageEncode(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("File not found: {0}")]
    FileNotFound(String),
}

impl Error {
    /// Whether retrying the same operation later can succeed.
    ///
    /// Tile loaders should retry transient errors after a delay and draw a
    /// placeholder for permanent ones.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Error::HandlePoisoned(_) | Error::OutOfMemory(_) | Error::Io(_)
        )
    }
}
