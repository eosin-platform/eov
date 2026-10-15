//! Minimal, safe binding to the [OpenSlide](https://openslide.org/) C library.
//!
//! eov links OpenSlide directly (see `common/build.rs`) instead of going
//! through a third-party binding crate. **OpenSlide 4.0 or newer is
//! required**: this module uses the shared tile-cache API
//! (`openslide_cache_*`) and the ICC-profile functions that first appeared in
//! OpenSlide 4.0.
//!
//! Most code should use [`WsiFile`](crate::WsiFile), which adds handle-health
//! tracking, recovery, compositing and caching on top of this module. Use
//! [`OpenSlide`] directly only when you need the raw library behaviour (for
//! example to compare against it in tests or diagnostics).
//!
//! # Thread safety
//!
//! All OpenSlide functions except `openslide_close` are thread-safe, so
//! [`OpenSlide`] and [`OpenSlideCache`] are `Send + Sync`. `openslide_close`
//! runs in `Drop`, which Rust only calls once no other reference exists.
//!
//! # Error state
//!
//! OpenSlide errors are **sticky**: after the first decode or I/O error a
//! handle moves permanently into an error state in which every later call
//! fails (reads return zeroed pixels). Every method here checks
//! `openslide_get_error` after the call and reports such a failure as
//! [`Error::HandlePoisoned`]. Once that has happened the handle is useless;
//! drop it and open a new one ([`WsiFile`](crate::WsiFile) does this for you).
//!
//! # Pixel format
//!
//! OpenSlide returns **premultiplied** ARGB as native-endian `u32` values
//! (`0xAARRGGBB`). Regions outside the scanned area are fully transparent
//! (`0x00000000`). See [`crate::wsi::AlphaMode`] for conversions.

use crate::{Error, Result};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;
use std::ptr::NonNull;

#[allow(non_camel_case_types)]
mod ffi {
    use std::ffi::{c_char, c_void};
    use std::marker::{PhantomData, PhantomPinned};

    #[repr(C)]
    pub struct openslide_t {
        _private: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct openslide_cache_t {
        _private: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    // Linked by `common/build.rs` (`cargo:rustc-link-lib=dylib=openslide`).
    unsafe extern "C" {
        pub fn openslide_detect_vendor(filename: *const c_char) -> *const c_char;
        pub fn openslide_open(filename: *const c_char) -> *mut openslide_t;
        pub fn openslide_get_level_count(osr: *mut openslide_t) -> i32;
        pub fn openslide_get_level_dimensions(
            osr: *mut openslide_t,
            level: i32,
            w: *mut i64,
            h: *mut i64,
        );
        pub fn openslide_get_level_downsample(osr: *mut openslide_t, level: i32) -> f64;
        pub fn openslide_get_best_level_for_downsample(
            osr: *mut openslide_t,
            downsample: f64,
        ) -> i32;
        pub fn openslide_read_region(
            osr: *mut openslide_t,
            dest: *mut u32,
            x: i64,
            y: i64,
            level: i32,
            w: i64,
            h: i64,
        );
        pub fn openslide_get_icc_profile_size(osr: *mut openslide_t) -> i64;
        pub fn openslide_read_icc_profile(osr: *mut openslide_t, dest: *mut c_void);
        pub fn openslide_close(osr: *mut openslide_t);
        pub fn openslide_get_error(osr: *mut openslide_t) -> *const c_char;
        pub fn openslide_get_property_names(osr: *mut openslide_t) -> *const *const c_char;
        pub fn openslide_get_property_value(
            osr: *mut openslide_t,
            name: *const c_char,
        ) -> *const c_char;
        pub fn openslide_get_associated_image_names(osr: *mut openslide_t)
        -> *const *const c_char;
        pub fn openslide_get_associated_image_dimensions(
            osr: *mut openslide_t,
            name: *const c_char,
            w: *mut i64,
            h: *mut i64,
        );
        pub fn openslide_read_associated_image(
            osr: *mut openslide_t,
            name: *const c_char,
            dest: *mut u32,
        );
        pub fn openslide_get_associated_image_icc_profile_size(
            osr: *mut openslide_t,
            name: *const c_char,
        ) -> i64;
        pub fn openslide_read_associated_image_icc_profile(
            osr: *mut openslide_t,
            name: *const c_char,
            dest: *mut c_void,
        );
        pub fn openslide_cache_create(capacity: usize) -> *mut openslide_cache_t;
        pub fn openslide_set_cache(osr: *mut openslide_t, cache: *mut openslide_cache_t);
        pub fn openslide_cache_release(cache: *mut openslide_cache_t);
        pub fn openslide_get_version() -> *const c_char;
    }
}

/// Converts a path to the C string OpenSlide expects.
///
/// On Unix the raw OS bytes are passed through unchanged, so paths that are
/// not valid UTF-8 can be opened. On Windows OpenSlide requires UTF-8.
fn path_to_cstring(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    };
    #[cfg(not(unix))]
    let bytes = path
        .to_str()
        .ok_or_else(|| {
            Error::OpenFile(format!(
                "{}: path is not valid Unicode and cannot be passed to OpenSlide",
                path.display()
            ))
        })?
        .as_bytes()
        .to_vec();

    CString::new(bytes)
        .map_err(|_| Error::OpenFile(format!("{}: path contains a NUL byte", path.display())))
}

/// Copies a C string owned by OpenSlide. Returns `None` for NULL.
///
/// # Safety
/// `ptr` must be NULL or point to a NUL-terminated string that stays valid
/// for the duration of the call.
unsafe fn copy_c_str(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        // SAFETY: guaranteed by the caller.
        Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }
}

/// Copies a NULL-terminated array of C strings owned by OpenSlide.
///
/// # Safety
/// `array` must be NULL or point to a NULL-terminated array of valid,
/// NUL-terminated strings that stay valid for the duration of the call.
unsafe fn copy_c_str_array(array: *const *const c_char) -> Vec<String> {
    let mut out = Vec::new();
    if array.is_null() {
        return out;
    }
    let mut cursor = array;
    loop {
        // SAFETY: the array is NULL-terminated, so every element up to and
        // including the terminator is readable.
        let entry = unsafe { *cursor };
        if entry.is_null() {
            break;
        }
        // SAFETY: entries are valid C strings owned by OpenSlide.
        if let Some(value) = unsafe { copy_c_str(entry) } {
            out.push(value);
        }
        // SAFETY: we have not passed the terminator yet.
        cursor = unsafe { cursor.add(1) };
    }
    out
}

fn to_c_name(name: &str) -> Result<CString> {
    CString::new(name)
        .map_err(|_| Error::InvalidArgument(format!("name {name:?} contains a NUL byte")))
}

/// Returns the version string of the linked OpenSlide library, e.g. `"4.0.0"`.
pub fn version() -> String {
    // SAFETY: returns a static string (or NULL on exotic builds).
    unsafe { copy_c_str(ffi::openslide_get_version()) }.unwrap_or_else(|| "unknown".to_string())
}

/// Parses the leading `major.minor` of [`version`].
pub fn version_tuple() -> Option<(u32, u32)> {
    let version = version();
    let mut parts = version.split(|c: char| !c.is_ascii_digit());
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().and_then(|part| part.parse().ok()).unwrap_or(0);
    Some((major, minor))
}

/// Quickly determines whether OpenSlide can open `path`, and with which
/// driver ("vendor"), without fully opening it.
///
/// Returns `Ok(None)` when the file is not a recognised slide format.
pub fn detect_vendor(path: &Path) -> Result<Option<String>> {
    let c_path = path_to_cstring(path)?;
    // SAFETY: `c_path` is a valid C string; the result is a static string.
    Ok(unsafe { copy_c_str(ffi::openslide_detect_vendor(c_path.as_ptr())) })
}

/// An open OpenSlide handle (`openslide_t`).
///
/// See the [module documentation](self) for the error and thread-safety
/// model.
pub struct OpenSlide {
    raw: NonNull<ffi::openslide_t>,
}

// SAFETY: OpenSlide documents every function except `openslide_close` as
// thread-safe, and `openslide_close` is only called from `Drop`.
unsafe impl Send for OpenSlide {}
// SAFETY: see above.
unsafe impl Sync for OpenSlide {}

impl Drop for OpenSlide {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `openslide_open` and is closed exactly once;
        // `&mut self` proves no other thread is using it.
        unsafe { ffi::openslide_close(self.raw.as_ptr()) }
    }
}

impl std::fmt::Debug for OpenSlide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenSlide")
            .field("error", &self.error())
            .finish_non_exhaustive()
    }
}

impl OpenSlide {
    /// Opens a slide.
    ///
    /// Unlike older bindings, a handle that OpenSlide returns already in an
    /// error state is closed again (not leaked) and reported as
    /// [`Error::OpenFile`]. Unrecognised files yield
    /// [`Error::UnsupportedFormat`].
    pub fn open(path: &Path) -> Result<Self> {
        let c_path = path_to_cstring(path)?;
        // SAFETY: `c_path` is a valid C string.
        let raw = unsafe { ffi::openslide_open(c_path.as_ptr()) };
        let Some(raw) = NonNull::new(raw) else {
            return Err(Error::UnsupportedFormat(format!(
                "{}: not a slide format OpenSlide recognises",
                path.display()
            )));
        };
        let slide = Self { raw };
        if let Some(message) = slide.error() {
            // `slide` is dropped here, which closes the handle.
            return Err(Error::OpenFile(format!("{}: {message}", path.display())));
        }
        Ok(slide)
    }

    fn ptr(&self) -> *mut ffi::openslide_t {
        self.raw.as_ptr()
    }

    /// The sticky error message, if the handle is in the error state.
    pub fn error(&self) -> Option<String> {
        // SAFETY: valid handle; the string is owned by the handle.
        unsafe { copy_c_str(ffi::openslide_get_error(self.ptr())) }
    }

    /// Whether the handle is in OpenSlide's (terminal) error state.
    pub fn is_poisoned(&self) -> bool {
        // SAFETY: valid handle.
        !unsafe { ffi::openslide_get_error(self.ptr()) }.is_null()
    }

    fn poisoned_error(&self, context: &str) -> Error {
        Error::HandlePoisoned(match self.error() {
            Some(message) => format!("{context}: {message}"),
            None => context.to_string(),
        })
    }

    fn check(&self, context: &str) -> Result<()> {
        if self.is_poisoned() {
            Err(self.poisoned_error(context))
        } else {
            Ok(())
        }
    }

    /// Error check after a call that fills `dest` and, if *it* fails,
    /// clears all of it (`openslide_read_region`, associated images, ICC).
    ///
    /// A handle can be poisoned by a concurrent call on another thread while
    /// ours succeeds. Any non-zero element proves that our call wrote real
    /// data, so only an all-zero result is reported as a failure. (A
    /// legitimately all-transparent region is reported as a failure too; the
    /// caller's recovery re-reads it on a fresh handle.)
    fn check_filled<T: Copy + PartialEq + Default>(&self, dest: &[T], context: &str) -> Result<()> {
        if !self.is_poisoned() {
            return Ok(());
        }
        let zero = T::default();
        if dest.iter().any(|&value| value != zero) {
            Ok(())
        } else {
            Err(self.poisoned_error(context))
        }
    }

    /// Number of pyramid levels.
    pub fn level_count(&self) -> Result<u32> {
        // SAFETY: valid handle.
        let count = unsafe { ffi::openslide_get_level_count(self.ptr()) };
        if count < 0 {
            return Err(self.poisoned_error("openslide_get_level_count"));
        }
        Ok(count as u32)
    }

    /// Width and height of `level` in pixels.
    pub fn level_dimensions(&self, level: u32) -> Result<(u64, u64)> {
        let level = i32::try_from(level)
            .map_err(|_| Error::InvalidArgument(format!("level {level} out of range")))?;
        let (mut w, mut h) = (-1i64, -1i64);
        // SAFETY: valid handle and out-pointers.
        unsafe { ffi::openslide_get_level_dimensions(self.ptr(), level, &mut w, &mut h) };
        if w < 0 || h < 0 {
            self.check("openslide_get_level_dimensions")?;
            return Err(Error::InvalidLevel(level as u32, 0));
        }
        Ok((w as u64, h as u64))
    }

    /// Downsample factor of `level` relative to level 0.
    pub fn level_downsample(&self, level: u32) -> Result<f64> {
        let level = i32::try_from(level)
            .map_err(|_| Error::InvalidArgument(format!("level {level} out of range")))?;
        // SAFETY: valid handle.
        let downsample = unsafe { ffi::openslide_get_level_downsample(self.ptr(), level) };
        if downsample <= 0.0 || !downsample.is_finite() {
            self.check("openslide_get_level_downsample")?;
            return Err(Error::InvalidLevel(level as u32, 0));
        }
        Ok(downsample)
    }

    /// OpenSlide's own best-level computation. Prefer
    /// [`WsiFile::best_level_for_downsample`](crate::WsiFile::best_level_for_downsample),
    /// which computes the same result in Rust from cached level data and so
    /// keeps working when a handle is poisoned.
    pub fn best_level_for_downsample(&self, downsample: f64) -> Result<u32> {
        // SAFETY: valid handle.
        let level = unsafe { ffi::openslide_get_best_level_for_downsample(self.ptr(), downsample) };
        if level < 0 {
            return Err(self.poisoned_error("openslide_get_best_level_for_downsample"));
        }
        Ok(level as u32)
    }

    /// Reads a region as premultiplied, native-endian ARGB into `dest`.
    ///
    /// `x`/`y` are level-0 coordinates (negative values and areas outside
    /// the slide read as transparent); `width`/`height` are in pixels of
    /// `level`. `dest.len()` must equal `width * height`.
    pub fn read_region_argb(
        &self,
        dest: &mut [u32],
        x: i64,
        y: i64,
        level: u32,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let level = i32::try_from(level)
            .map_err(|_| Error::InvalidArgument(format!("level {level} out of range")))?;
        let pixels = width as usize * height as usize;
        if dest.len() != pixels {
            return Err(Error::InvalidArgument(format!(
                "destination holds {} pixels, region needs {pixels}",
                dest.len()
            )));
        }
        if pixels == 0 {
            return Ok(());
        }
        if let Err(err) = self.check("openslide_read_region") {
            // Same contract as OpenSlide: never leave stale pixels behind.
            dest.fill(0);
            return Err(err);
        }
        // SAFETY: `dest` holds exactly `width * height` u32 values, which is
        // what OpenSlide writes (it also clears the buffer on error).
        unsafe {
            ffi::openslide_read_region(
                self.ptr(),
                dest.as_mut_ptr(),
                x,
                y,
                level,
                i64::from(width),
                i64::from(height),
            );
        }
        self.check_filled(dest, "openslide_read_region")
    }

    /// Names of all properties.
    pub fn property_names(&self) -> Result<Vec<String>> {
        // SAFETY: valid handle; the array is owned by the handle.
        let names = unsafe { copy_c_str_array(ffi::openslide_get_property_names(self.ptr())) };
        self.check("openslide_get_property_names")?;
        Ok(names)
    }

    /// Value of one property, or `None` if it does not exist.
    pub fn property(&self, name: &str) -> Option<String> {
        let c_name = to_c_name(name).ok()?;
        // SAFETY: valid handle and C string; the value is owned by the handle.
        unsafe { copy_c_str(ffi::openslide_get_property_value(self.ptr(), c_name.as_ptr())) }
    }

    /// Every property as an ordered map.
    pub fn properties(&self) -> Result<BTreeMap<String, String>> {
        let mut map = BTreeMap::new();
        for name in self.property_names()? {
            if let Some(value) = self.property(&name) {
                map.insert(name, value);
            }
        }
        self.check("openslide_get_property_value")?;
        Ok(map)
    }

    /// Names of the associated images (e.g. `label`, `macro`, `thumbnail`).
    pub fn associated_image_names(&self) -> Result<Vec<String>> {
        // SAFETY: valid handle; the array is owned by the handle.
        let names =
            unsafe { copy_c_str_array(ffi::openslide_get_associated_image_names(self.ptr())) };
        self.check("openslide_get_associated_image_names")?;
        Ok(names)
    }

    /// Dimensions of an associated image.
    pub fn associated_image_dimensions(&self, name: &str) -> Result<(u64, u64)> {
        let c_name = to_c_name(name)?;
        let (mut w, mut h) = (-1i64, -1i64);
        // SAFETY: valid handle, C string and out-pointers. Unknown names
        // yield -1 without poisoning the handle.
        unsafe {
            ffi::openslide_get_associated_image_dimensions(
                self.ptr(),
                c_name.as_ptr(),
                &mut w,
                &mut h,
            );
        }
        if w < 0 || h < 0 {
            self.check("openslide_get_associated_image_dimensions")?;
            return Err(Error::AssociatedImageNotFound(name.to_string()));
        }
        Ok((w as u64, h as u64))
    }

    /// Reads an associated image as premultiplied ARGB.
    ///
    /// Returns `(width, height, pixels)`. Images larger than `max_pixels`
    /// are rejected with [`Error::RegionTooLarge`] before allocating.
    pub fn read_associated_image_argb(
        &self,
        name: &str,
        max_pixels: u64,
    ) -> Result<(u32, u32, Vec<u32>)> {
        let (w, h) = self.associated_image_dimensions(name)?;
        let pixels = w.saturating_mul(h);
        if pixels > max_pixels || w > u32::MAX as u64 || h > u32::MAX as u64 {
            return Err(Error::RegionTooLarge {
                width: w,
                height: h,
                max_pixels,
            });
        }
        let mut dest = try_alloc_u32(pixels as usize)?;
        let c_name = to_c_name(name)?;
        // SAFETY: `dest` holds `w * h` u32 values, as OpenSlide requires for
        // this image (it writes or clears exactly that many).
        unsafe {
            ffi::openslide_read_associated_image(self.ptr(), c_name.as_ptr(), dest.as_mut_ptr());
        }
        self.check_filled(&dest, "openslide_read_associated_image")?;
        Ok((w as u32, h as u32, dest))
    }

    /// The slide's ICC profile, if it has one.
    pub fn icc_profile(&self) -> Result<Option<Vec<u8>>> {
        // SAFETY: valid handle.
        let size = unsafe { ffi::openslide_get_icc_profile_size(self.ptr()) };
        if size < 0 {
            return Err(self.poisoned_error("openslide_get_icc_profile_size"));
        }
        if size == 0 {
            return Ok(None);
        }
        let mut dest = try_alloc_u8(size as usize)?;
        // SAFETY: `dest` holds exactly the profile size OpenSlide reported.
        unsafe { ffi::openslide_read_icc_profile(self.ptr(), dest.as_mut_ptr().cast::<c_void>()) };
        self.check_filled(&dest, "openslide_read_icc_profile")?;
        Ok(Some(dest))
    }

    /// The ICC profile of an associated image, if it has one.
    pub fn associated_image_icc_profile(&self, name: &str) -> Result<Option<Vec<u8>>> {
        // Validate the name first: OpenSlide requires a valid name here.
        self.associated_image_dimensions(name)?;
        let c_name = to_c_name(name)?;
        // SAFETY: valid handle and C string.
        let size = unsafe {
            ffi::openslide_get_associated_image_icc_profile_size(self.ptr(), c_name.as_ptr())
        };
        if size < 0 {
            self.check("openslide_get_associated_image_icc_profile_size")?;
            return Err(Error::AssociatedImageNotFound(name.to_string()));
        }
        if size == 0 {
            return Ok(None);
        }
        let mut dest = try_alloc_u8(size as usize)?;
        // SAFETY: `dest` holds exactly the profile size OpenSlide reported.
        unsafe {
            ffi::openslide_read_associated_image_icc_profile(
                self.ptr(),
                c_name.as_ptr(),
                dest.as_mut_ptr().cast::<c_void>(),
            );
        }
        self.check_filled(&dest, "openslide_read_associated_image_icc_profile")?;
        Ok(Some(dest))
    }

    /// Attaches a shared tile cache, replacing the handle's private one.
    pub fn set_cache(&self, cache: &OpenSlideCache) {
        // SAFETY: both pointers are valid; OpenSlide takes its own reference
        // to the cache, so it may outlive `cache`.
        unsafe { ffi::openslide_set_cache(self.ptr(), cache.raw.as_ptr()) }
    }
}

/// A tile cache (`openslide_cache_t`) that can be shared by many handles.
///
/// OpenSlide reference-counts caches: a handle keeps its cache alive even
/// after the `OpenSlideCache` that created it is dropped.
pub struct OpenSlideCache {
    raw: NonNull<ffi::openslide_cache_t>,
    capacity: usize,
}

// SAFETY: OpenSlide caches are internally synchronised and documented as
// usable from multiple threads.
unsafe impl Send for OpenSlideCache {}
// SAFETY: see above.
unsafe impl Sync for OpenSlideCache {}

impl OpenSlideCache {
    /// Creates a cache holding up to `capacity_bytes` of decoded tiles.
    pub fn new(capacity_bytes: usize) -> Result<Self> {
        // SAFETY: plain constructor.
        let raw = unsafe { ffi::openslide_cache_create(capacity_bytes) };
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::OpenSlide("openslide_cache_create failed".to_string()))?;
        Ok(Self {
            raw,
            capacity: capacity_bytes,
        })
    }

    /// The capacity this cache was created with.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Drop for OpenSlideCache {
    fn drop(&mut self) {
        // SAFETY: releases the reference taken by `openslide_cache_create`.
        unsafe { ffi::openslide_cache_release(self.raw.as_ptr()) }
    }
}

impl std::fmt::Debug for OpenSlideCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenSlideCache")
            .field("capacity", &self.capacity)
            .finish()
    }
}

fn try_alloc_u32(len: usize) -> Result<Vec<u32>> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|_| Error::OutOfMemory(len.saturating_mul(4)))?;
    buf.resize(len, 0);
    Ok(buf)
}

fn try_alloc_u8(len: usize) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|_| Error::OutOfMemory(len))?;
    buf.resize(len, 0);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{TiffLevel, TiffSpec, write_tiff};

    fn gradient(x: u32, y: u32) -> [u8; 3] {
        [(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]
    }

    #[test]
    fn reports_a_version_of_at_least_four() {
        let (major, _) = version_tuple().expect("parse OpenSlide version");
        assert!(major >= 4, "eov requires OpenSlide 4, found {}", version());
    }

    #[test]
    fn opens_reads_and_describes_a_generic_tiff() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.tiff");
        write_tiff(
            &path,
            &TiffSpec {
                levels: vec![TiffLevel::new(96, 64, 32), TiffLevel::new(48, 32, 32)],
                icc_profile: Some(b"not-really-an-icc-profile".to_vec()),
                ..TiffSpec::default()
            },
            gradient,
        )
        .unwrap();

        assert_eq!(detect_vendor(&path).unwrap().as_deref(), Some("generic-tiff"));
        let slide = OpenSlide::open(&path).unwrap();
        assert_eq!(slide.level_count().unwrap(), 2);
        assert_eq!(slide.level_dimensions(0).unwrap(), (96, 64));
        assert_eq!(slide.level_dimensions(1).unwrap(), (48, 32));
        assert!((slide.level_downsample(1).unwrap() - 2.0).abs() < 1e-9);
        assert_eq!(slide.best_level_for_downsample(3.0).unwrap(), 1);

        let mut argb = vec![0u32; 4];
        slide.read_region_argb(&mut argb, 10, 20, 0, 2, 2).unwrap();
        let [r, g, b] = gradient(10, 20);
        assert_eq!(
            argb[0],
            0xff00_0000 | (r as u32) << 16 | (g as u32) << 8 | b as u32
        );

        let props = slide.properties().unwrap();
        assert_eq!(props.get("openslide.vendor").map(String::as_str), Some("generic-tiff"));
        assert_eq!(slide.property("openslide.level[0].tile-width").as_deref(), Some("32"));
        assert_eq!(slide.property("no.such.property"), None);

        assert_eq!(
            slide.icc_profile().unwrap().as_deref(),
            Some(&b"not-really-an-icc-profile"[..])
        );
        assert!(slide.associated_image_names().unwrap().is_empty());
        assert!(matches!(
            slide.associated_image_dimensions("label"),
            Err(Error::AssociatedImageNotFound(_))
        ));
        // An unknown associated-image name must not poison the handle.
        assert!(!slide.is_poisoned());
    }

    #[test]
    fn reads_associated_images_of_an_aperio_style_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.svs");
        write_tiff(&path, &TiffSpec::aperio(128, 96, 32, 4), gradient).unwrap();

        let slide = OpenSlide::open(&path).unwrap();
        assert_eq!(slide.property("openslide.vendor").as_deref(), Some("aperio"));
        let mut names = slide.associated_image_names().unwrap();
        names.sort();
        assert_eq!(names, ["label", "macro", "thumbnail"]);
        let (w, h, pixels) = slide.read_associated_image_argb("label", u64::MAX).unwrap();
        assert_eq!((w, h), (24, 16));
        assert_eq!(pixels.len(), 24 * 16);
        assert!(pixels.iter().all(|&p| p >> 24 == 0xff));
        assert!(matches!(
            slide.read_associated_image_argb("label", 10),
            Err(Error::RegionTooLarge { .. })
        ));
    }

    #[test]
    fn unrecognised_files_are_reported_without_a_handle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, b"hello").unwrap();
        assert!(matches!(OpenSlide::open(&path), Err(Error::UnsupportedFormat(_))));
        assert_eq!(detect_vendor(&path).unwrap(), None);
    }

    #[cfg(unix)]
    #[test]
    fn opens_paths_that_are_not_valid_utf8() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let dir = tempfile::tempdir().unwrap();
        let name = OsStr::from_bytes(b"slide-\xff\xfe.tiff");
        let path = dir.path().join(name);
        write_tiff(&path, &TiffSpec::single(64, 48, 32), gradient).unwrap();
        let slide = OpenSlide::open(&path).unwrap();
        assert_eq!(slide.level_dimensions(0).unwrap(), (64, 48));
    }

    #[test]
    fn read_errors_poison_the_handle_and_are_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.tiff");
        let spec = TiffSpec {
            levels: vec![
                TiffLevel::new(128, 128, 32).with_broken_tile(1, 1),
                TiffLevel::new(64, 64, 32),
            ],
            ..TiffSpec::default()
        };
        write_tiff(&path, &spec, gradient).unwrap();

        let slide = OpenSlide::open(&path).unwrap();
        let mut good = vec![0u32; 32 * 32];
        slide.read_region_argb(&mut good, 0, 0, 0, 32, 32).unwrap();

        let mut bad = vec![0u32; 32 * 32];
        let err = slide.read_region_argb(&mut bad, 32, 32, 0, 32, 32).unwrap_err();
        assert!(matches!(err, Error::HandlePoisoned(_)), "{err:?}");
        assert!(slide.is_poisoned());

        // The error is sticky: even the previously good tile fails now.
        assert!(slide.read_region_argb(&mut good, 0, 0, 0, 32, 32).is_err());
        assert!(good.iter().all(|&p| p == 0));
        assert!(slide.level_count().is_err());
        assert!(slide.best_level_for_downsample(2.0).is_err());

        // A call that wrote data succeeded even if the handle is poisoned now
        // (by a concurrent call); only an all-cleared result is a failure.
        assert!(slide.check_filled(&[0u32, 0xff00_0000], "test").is_ok());
        assert!(slide.check_filled(&[0u32; 4], "test").is_err());
    }

    #[test]
    fn shared_caches_can_be_attached_and_released() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.tiff");
        write_tiff(&path, &TiffSpec::single(64, 64, 32), gradient).unwrap();

        let cache = OpenSlideCache::new(1 << 20).unwrap();
        assert_eq!(cache.capacity(), 1 << 20);
        let a = OpenSlide::open(&path).unwrap();
        let b = OpenSlide::open(&path).unwrap();
        a.set_cache(&cache);
        b.set_cache(&cache);
        drop(cache); // the handles keep the cache alive
        let mut buf = vec![0u32; 64 * 64];
        a.read_region_argb(&mut buf, 0, 0, 0, 64, 64).unwrap();
        b.read_region_argb(&mut buf, 0, 0, 0, 64, 64).unwrap();
    }

    #[test]
    fn rejects_mismatched_destination_buffers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("slide.tiff");
        write_tiff(&path, &TiffSpec::single(32, 32, 32), gradient).unwrap();
        let slide = OpenSlide::open(&path).unwrap();
        let mut too_small = vec![0u32; 3];
        assert!(matches!(
            slide.read_region_argb(&mut too_small, 0, 0, 0, 2, 2),
            Err(Error::InvalidArgument(_))
        ));
        assert!(!slide.is_poisoned());
    }
}
