//! The HDF5 C library, loaded at run time, for writing HDF5 files
//! (`eov dataset pack --format hdf5`).
//!
//! eov has no build-time HDF5 dependency. [`Hdf5::open`] finds a shared
//! libhdf5 ([`LibrarySearch`]), loads it with `libloading`, calls `H5open`,
//! checks with `H5get_libversion` that it is 1.10 or newer (whose `hid_t`
//! is 64 bits wide, as declared here), and then reads the ids of the
//! predefined types and property-list classes, global variables that are
//! only set after `H5open`. The files are HDF5's own: eov only calls the
//! library.
//!
//! * A library is never unloaded: HDF5 registers exit handlers and
//!   thread-local keys when it initialises, which would point into unmapped
//!   code after a `dlclose`.
//! * HDF5 is not always built thread-safe, so a session ([`Hdf5`]) holds a
//!   process-wide lock: one thread at a time uses the library.
//! * Every id belongs to a [`Handle`] of its kind, which closes it, and
//!   every return value is checked. HDF5 does not print its errors; an
//!   [`Hdf5Error`] carries the error stack's description instead.

mod ffi;
mod search;

pub use search::LIBRARY_ENV;
pub(crate) use search::{LibrarySearch, LoadError};

use ffi::{
    Functions, Globals, H5E_DEFAULT, H5E_WALK_DOWNWARD, H5E_error2_t, H5F_ACC_TRUNC,
    H5F_LIBVER_V18, H5P_DEFAULT, H5S_ALL, H5S_SCALAR, H5S_SELECT_SET, H5T_CSET_UTF8, H5T_VARIABLE,
    H5Z_FILTER_DEFLATE, Optional, Startup, herr_t, hid_t,
};
use libloading::Library;
use std::cell::Cell;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// A loaded and initialised library.
struct Api {
    /// What was loaded: a path, or a name the system's loader found.
    requested: PathBuf,
    version: (u32, u32, u32),
    f: Functions,
    optional: Optional,
    g: Globals,
    #[cfg_attr(not(test), allow(dead_code))]
    library: &'static Library,
    #[cfg(test)]
    read: std::sync::OnceLock<Result<ffi::ReadFunctions, String>>,
}

/// The libraries loaded so far. Holding the lock is holding HDF5.
static LOADED: Mutex<Vec<&'static Api>> = Mutex::new(Vec::new());

thread_local! {
    /// Whether this thread has a session (a second one would deadlock).
    static IN_SESSION: Cell<bool> = const { Cell::new(false) };
}

/// The HDF5 library, in use by this thread until dropped. It cannot be
/// shared with other threads (it is neither `Send` nor `Sync`), and
/// neither can the [`Handle`]s that borrow it.
pub(crate) struct Hdf5 {
    api: &'static Api,
    _lock: MutexGuard<'static, Vec<&'static Api>>,
    _this_thread: PhantomData<Cell<()>>,
}

impl Hdf5 {
    /// Loads the first usable library of `search` (or takes the one loaded
    /// earlier from the same place) and holds it for this thread.
    ///
    /// # Panics
    ///
    /// When this thread already holds a session: pass that one along.
    pub(crate) fn open(search: &LibrarySearch) -> Result<Self, LoadError> {
        assert!(
            !IN_SESSION.get(),
            "HDF5 is already in use on this thread; pass its session along instead of opening another"
        );
        let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
        let mut tried = Vec::new();
        for candidate in search.candidates() {
            let api = match loaded.iter().find(|api| api.requested == candidate.path) {
                Some(api) => Ok(*api),
                None => load(&candidate.path),
            };
            match api.and_then(|api| begin(api).map(|()| api)) {
                Ok(api) => {
                    if !loaded.iter().any(|known| std::ptr::eq(*known, api)) {
                        loaded.push(api);
                    }
                    IN_SESSION.set(true);
                    return Ok(Self {
                        api,
                        _lock: loaded,
                        _this_thread: PhantomData,
                    });
                }
                Err(reason) => tried.push((candidate, reason)),
            }
        }
        Err(LoadError { tried })
    }

    /// The library's version.
    pub(crate) fn version(&self) -> (u32, u32, u32) {
        self.api.version
    }

    /// The library as it was loaded: a path, or a name the system's loader
    /// found.
    pub(crate) fn library(&self) -> &Path {
        &self.api.requested
    }

    /// "HDF5 1.10.10 from libhdf5_serial.so.103", for reports.
    pub(crate) fn describe(&self) -> String {
        let (major, minor, release) = self.version();
        format!(
            "HDF5 {major}.{minor}.{release} from {}",
            self.library().display()
        )
    }

    /// Whether [`Dataset::write_chunk`] works (HDF5 1.10.3 and later).
    pub(crate) fn can_write_chunks(&self) -> bool {
        self.api.optional.H5Dwrite_chunk.is_some()
    }

    fn f(&self) -> &Functions {
        &self.api.f
    }

    fn g(&self) -> &Globals {
        &self.api.g
    }

    /// The error of the call that just failed, with the first and the
    /// innermost descriptions on the error stack, which say what failed
    /// and why.
    fn error(&self, call: &str, what: &str) -> Hdf5Error {
        let mut descriptions: Vec<String> = Vec::new();
        // SAFETY: `collect` only reads the entries and appends to the vector
        // passed along, which outlives the call.
        let walked = unsafe {
            (self.f().H5Ewalk2)(
                H5E_DEFAULT,
                H5E_WALK_DOWNWARD,
                Some(collect),
                (&mut descriptions as *mut Vec<String>).cast(),
            )
        };
        let detail = match descriptions.as_slice() {
            _ if walked < 0 => "HDF5 says nothing more".to_string(),
            [] => "HDF5 says nothing more".to_string(),
            [only] => only.clone(),
            [first, .., last] => format!("{first}: {last}"),
        };
        Hdf5Error(format!("{call} failed for {what}: {detail}"))
    }

    fn check(&self, status: herr_t, call: &str, what: &str) -> Result<()> {
        if status < 0 {
            return Err(self.error(call, what));
        }
        Ok(())
    }

    fn property_list(&self, class: hid_t, what: &str) -> Result<Handle<'_, PropertyListId>> {
        // SAFETY: a property-list class id of this library.
        let id = unsafe { (self.f().H5Pcreate)(class) };
        Handle::new(self, id, "H5Pcreate", what)
    }

    fn dataspace(&self, shape: &[u64], what: &str) -> Result<Handle<'_, DataspaceId>> {
        let rank = c_int::try_from(shape.len())
            .map_err(|_| Hdf5Error(format!("{what} has too many dimensions")))?;
        // SAFETY: `rank` values in `shape`; no maximum dimensions.
        let id = unsafe { (self.f().H5Screate_simple)(rank, shape.as_ptr(), std::ptr::null()) };
        Handle::new(self, id, "H5Screate_simple", what)
    }

    fn scalar_space(&self, what: &str) -> Result<Handle<'_, DataspaceId>> {
        // SAFETY: no pointers.
        let id = unsafe { (self.f().H5Screate)(H5S_SCALAR) };
        Handle::new(self, id, "H5Screate", what)
    }

    /// A variable-length UTF-8 string type.
    fn string_type(&self, what: &str) -> Result<Handle<'_, DatatypeId>> {
        // SAFETY: the predefined C string type, copied.
        let id = unsafe { (self.f().H5Tcopy)(self.g().H5T_C_S1) };
        let string = Handle::new(self, id, "H5Tcopy", what)?;
        // SAFETY: a type this session owns.
        let status = unsafe { (self.f().H5Tset_size)(string.id, H5T_VARIABLE) };
        self.check(status, "H5Tset_size", what)?;
        // SAFETY: as above.
        let status = unsafe { (self.f().H5Tset_cset)(string.id, H5T_CSET_UTF8) };
        self.check(status, "H5Tset_cset", what)?;
        Ok(string)
    }

    /// Creates (or truncates) the file at `path`, in the HDF5 1.8 format,
    /// which HDF5 1.8 and later read and which keeps attributes over 64 KiB
    /// (the slide list of a large cohort) in dense storage. Objects record
    /// no modification times, so a dataset always packs into the same
    /// file.
    pub(crate) fn create(&self, path: &Path) -> Result<File<'_>> {
        let name = path.display().to_string();
        let c_path = c_path(path)?;
        let fcpl = self.property_list(self.g().H5P_CLS_FILE_CREATE_ID, &name)?;
        // SAFETY: a property list this session owns (`hbool_t` false).
        let status = unsafe { (self.f().H5Pset_obj_track_times)(fcpl.id, 0) };
        self.check(status, "H5Pset_obj_track_times", &name)?;
        let fapl = self.property_list(self.g().H5P_CLS_FILE_ACCESS_ID, &name)?;
        // SAFETY: as above.
        let status =
            unsafe { (self.f().H5Pset_libver_bounds)(fapl.id, H5F_LIBVER_V18, H5F_LIBVER_V18) };
        self.check(status, "H5Pset_libver_bounds", &name)?;
        if let Some(set_file_locking) = self.api.optional.H5Pset_file_locking {
            // Nothing else opens the file while it is written, and some
            // network file systems refuse locks.
            // SAFETY: as above (`hbool_t` false, true).
            let status = unsafe { set_file_locking(fapl.id, 0, 1) };
            self.check(status, "H5Pset_file_locking", &name)?;
        }
        // SAFETY: a NUL-terminated path and property lists this session owns.
        let id = unsafe { (self.f().H5Fcreate)(c_path.as_ptr(), H5F_ACC_TRUNC, fcpl.id, fapl.id) };
        Ok(File {
            handle: Handle::new(self, id, "H5Fcreate", &name)?,
            name,
        })
    }

    /// Writes `value` as attribute `name` of the object `location`.
    fn set_attribute(
        &self,
        location: hid_t,
        owner: &str,
        name: &str,
        value: &AttrValue<'_>,
    ) -> Result<()> {
        let what = format!("attribute {name} of {owner}");
        let c_name = c_string(name, &what)?;
        match value {
            AttrValue::Text(text) => {
                let text = c_string(text, &what)?;
                let string = self.string_type(&what)?;
                let space = self.scalar_space(&what)?;
                let attribute =
                    self.create_attribute(location, &c_name, string.id, &space, &what)?;
                let pointer: *const c_char = text.as_ptr();
                // SAFETY: a variable-length string is written from a
                // pointer to its characters; `text` outlives the call.
                let status = unsafe {
                    (self.f().H5Awrite)(
                        attribute.id,
                        string.id,
                        (&pointer as *const *const c_char).cast(),
                    )
                };
                self.check(status, "H5Awrite", &what)?;
                attribute.close(&what)
            }
            AttrValue::Int(value) => {
                self.numeric_attribute(location, &c_name, &what, std::slice::from_ref(value), true)
            }
            AttrValue::Float(value) => {
                self.numeric_attribute(location, &c_name, &what, std::slice::from_ref(value), true)
            }
            AttrValue::Ints(values) => {
                self.numeric_attribute(location, &c_name, &what, values, false)
            }
            AttrValue::Floats(values) => {
                self.numeric_attribute(location, &c_name, &what, values, false)
            }
        }
    }

    /// A numeric attribute: a scalar (`scalar` and one value) or a list.
    fn numeric_attribute<T: Element>(
        &self,
        location: hid_t,
        name: &CStr,
        what: &str,
        values: &[T],
        scalar: bool,
    ) -> Result<()> {
        let space = match values {
            [_] if scalar => self.scalar_space(what)?,
            _ => self.dataspace(&[values.len() as u64], what)?,
        };
        let (memory, stored) = T::types(self.g());
        let attribute = self.create_attribute(location, name, stored, &space, what)?;
        // SAFETY: one value per element of `space`.
        let status = unsafe { (self.f().H5Awrite)(attribute.id, memory, values.as_ptr().cast()) };
        self.check(status, "H5Awrite", what)?;
        attribute.close(what)
    }

    fn create_attribute(
        &self,
        location: hid_t,
        name: &CStr,
        stored: hid_t,
        space: &Handle<'_, DataspaceId>,
        what: &str,
    ) -> Result<Handle<'_, AttributeId>> {
        // SAFETY: valid ids and a NUL-terminated name.
        let id = unsafe {
            (self.f().H5Acreate2)(
                location,
                name.as_ptr(),
                stored,
                space.id,
                H5P_DEFAULT,
                H5P_DEFAULT,
            )
        };
        Handle::new(self, id, "H5Acreate2", what)
    }
}

impl Drop for Hdf5 {
    fn drop(&mut self) {
        IN_SESSION.set(false);
    }
}

/// Loads, initialises and checks the library at `path`.
fn load(path: &Path) -> Result<&'static Api, String> {
    // SAFETY: loading a library runs its initialisers; a libhdf5's are
    // harmless, and so are those of the system libraries a wrong name
    // could find.
    let library = unsafe { open_library(path) }.map_err(|error| error.to_string())?;
    // SAFETY: both functions have had these signatures since HDF5 1.8.
    let Ok(startup) = (unsafe { Startup::resolve(&library) }) else {
        return Err("not an HDF5 library (it has no H5open)".into());
    };
    // Once H5open has run, the library stays loaded.
    let library: &'static Library = Box::leak(Box::new(library));
    // SAFETY: no arguments.
    if unsafe { (startup.H5open)() } < 0 {
        return Err("H5open failed".into());
    }
    let (mut major, mut minor, mut release): (c_uint, c_uint, c_uint) = (0, 0, 0);
    // SAFETY: three valid pointers.
    if unsafe { (startup.H5get_libversion)(&mut major, &mut minor, &mut release) } < 0 {
        return Err("H5get_libversion failed".into());
    }
    let version = format!("HDF5 {major}.{minor}.{release}");
    if (major, minor) < (1, 10) {
        return Err(format!(
            "{version} is too old: eov needs 1.10 or newer (64-bit ids)"
        ));
    }
    // SAFETY: HDF5 1.10 or newer, which is never unloaded.
    let f = unsafe { Functions::resolve(library) }.map_err(|e| format!("{version}: {e}"))?;
    // SAFETY: as above.
    let optional = unsafe { Optional::resolve(library) };
    // SAFETY: as above, initialised by H5open.
    let g = unsafe { Globals::read(library) }.map_err(|e| format!("{version}: {e}"))?;
    // SAFETY: no pointers.
    if unsafe { (f.H5Zfilter_avail)(H5Z_FILTER_DEFLATE) } <= 0 {
        return Err(format!(
            "{version} was built without the deflate filter (zlib)"
        ));
    }
    Ok(Box::leak(Box::new(Api {
        requested: path.to_path_buf(),
        version: (major, minor, release),
        f,
        optional,
        g,
        library,
        #[cfg(test)]
        read: std::sync::OnceLock::new(),
    })))
}

/// Prepares this thread's use of the library: HDF5 does not print errors
/// (a per-thread setting in thread-safe builds); [`Hdf5::error`] reads them.
fn begin(api: &Api) -> Result<(), String> {
    // SAFETY: no callback, no data.
    let status = unsafe { (api.f.H5Eset_auto2)(H5E_DEFAULT, None, std::ptr::null_mut()) };
    if status < 0 {
        return Err("H5Eset_auto2 failed".into());
    }
    Ok(())
}

/// # Safety
///
/// Loading a library runs its initialisers.
#[cfg(not(windows))]
unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    // SAFETY: as the caller promises.
    unsafe { Library::new(path) }
}

/// # Safety
///
/// Loading a library runs its initialisers.
#[cfg(windows)]
unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::windows;
    if path.is_absolute() {
        // A DLL loaded by its path finds the DLLs next to it (zlib, szip).
        // SAFETY: as the caller promises.
        return unsafe {
            windows::Library::load_with_flags(path, windows::LOAD_WITH_ALTERED_SEARCH_PATH)
        }
        .map(Library::from);
    }
    // SAFETY: as the caller promises.
    unsafe { Library::new(path) }
}

/// Appends an error-stack entry's description to the `Vec<String>` at
/// `data`.
unsafe extern "C" fn collect(_: c_uint, entry: *const H5E_error2_t, data: *mut c_void) -> herr_t {
    // SAFETY: HDF5 passes a valid entry or none, and `data` is the vector
    // `Hdf5::error` passed.
    let (Some(entry), Some(descriptions)) = (unsafe { entry.as_ref() }, unsafe {
        data.cast::<Vec<String>>().as_mut()
    }) else {
        return 0;
    };
    if !entry.desc.is_null() {
        // SAFETY: a NUL-terminated string that the stack owns.
        let text = unsafe { CStr::from_ptr(entry.desc) }.to_string_lossy();
        let text = text.trim();
        if !text.is_empty() {
            descriptions.push(text.to_string());
        }
    }
    0
}

/// A file name as HDF5 takes it: bytes on Unix, UTF-8 elsewhere.
fn c_path(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    let bytes = std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str());
    #[cfg(not(unix))]
    let bytes = path
        .to_str()
        .ok_or_else(|| Hdf5Error(format!("{} is not valid Unicode", path.display())))?
        .as_bytes();
    CString::new(bytes)
        .map_err(|_| Hdf5Error(format!("{} contains a NUL character", path.display())))
}

fn c_string(text: &str, what: &str) -> Result<CString> {
    CString::new(text).map_err(|_| Hdf5Error(format!("{what} contains a NUL character")))
}

/// A failed HDF5 call (or a value HDF5 cannot take).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Hdf5Error(String);

impl fmt::Display for Hdf5Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Hdf5Error {}

impl From<Hdf5Error> for std::io::Error {
    fn from(error: Hdf5Error) -> Self {
        std::io::Error::other(error)
    }
}

pub(crate) type Result<T, E = Hdf5Error> = std::result::Result<T, E>;

/// What kind of object an id names, and how it is closed.
pub(crate) trait Kind {
    /// The function that closes ids of this kind, and its name.
    fn close(f: &Functions) -> (unsafe extern "C" fn(hid_t) -> herr_t, &'static str);
}

macro_rules! kinds {
    ($($kind:ident => $close:ident,)*) => {$(
        pub(crate) enum $kind {}

        impl Kind for $kind {
            fn close(f: &Functions) -> (unsafe extern "C" fn(hid_t) -> herr_t, &'static str) {
                (f.$close, stringify!($close))
            }
        }
    )*};
}

kinds! {
    FileId => H5Fclose,
    DatasetId => H5Dclose,
    DataspaceId => H5Sclose,
    DatatypeId => H5Tclose,
    PropertyListId => H5Pclose,
    AttributeId => H5Aclose,
}

/// An id that HDF5 handed out, closed with its kind's function when
/// dropped, or by [`Handle::close`], which reports a failure.
pub(crate) struct Handle<'h, K: Kind> {
    hdf5: &'h Hdf5,
    id: hid_t,
    kind: PhantomData<K>,
}

impl<'h, K: Kind> Handle<'h, K> {
    /// Takes `id` from `call`, or reports why the call failed.
    fn new(hdf5: &'h Hdf5, id: hid_t, call: &str, what: &str) -> Result<Self> {
        if id < 0 {
            return Err(hdf5.error(call, what));
        }
        Ok(Self {
            hdf5,
            id,
            kind: PhantomData,
        })
    }

    /// Closes the id, reporting a failure (closing a file or a dataset
    /// writes what HDF5 still holds).
    fn close(self, what: &str) -> Result<()> {
        let this = ManuallyDrop::new(self);
        let (close, call) = K::close(this.hdf5.f());
        // SAFETY: an id this handle owns; `ManuallyDrop` keeps `Drop` from
        // closing it again.
        let status = unsafe { close(this.id) };
        this.hdf5.check(status, call, what)
    }
}

impl<K: Kind> Drop for Handle<'_, K> {
    fn drop(&mut self) {
        let (close, _) = K::close(self.hdf5.f());
        // SAFETY: an id this handle owns, closed once. Failures cannot be
        // reported here; `close` reports them where it matters.
        unsafe { close(self.id) };
    }
}

/// Element types of datasets and attributes.
pub(crate) trait Element: Copy + 'static {
    /// The type in memory (native) and in the file (little-endian).
    fn types(g: &Globals) -> (hid_t, hid_t);
}

impl Element for u8 {
    fn types(g: &Globals) -> (hid_t, hid_t) {
        (g.H5T_NATIVE_UINT8, g.H5T_STD_U8LE)
    }
}

impl Element for i32 {
    fn types(g: &Globals) -> (hid_t, hid_t) {
        (g.H5T_NATIVE_INT32, g.H5T_STD_I32LE)
    }
}

impl Element for i64 {
    fn types(g: &Globals) -> (hid_t, hid_t) {
        (g.H5T_NATIVE_INT64, g.H5T_STD_I64LE)
    }
}

impl Element for f64 {
    fn types(g: &Globals) -> (hid_t, hid_t) {
        (g.H5T_NATIVE_DOUBLE, g.H5T_IEEE_F64LE)
    }
}

/// An attribute's value: text is a variable-length UTF-8 string (h5py
/// reads it as `str`), numbers are 64-bit, lists one-dimensional.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum AttrValue<'a> {
    Text(&'a str),
    Int(i64),
    Float(f64),
    Ints(&'a [i64]),
    Floats(&'a [f64]),
}

/// How a dataset is stored: chunks of `chunk` elements, with the shuffle
/// filter (which helps elements of more than one byte) and deflate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout<'a> {
    pub(crate) chunk: &'a [u64],
    pub(crate) shuffle: bool,
    /// Deflate level, 0–9.
    pub(crate) deflate: u32,
}

/// An HDF5 file being written.
pub(crate) struct File<'h> {
    handle: Handle<'h, FileId>,
    name: String,
}

impl<'h> File<'h> {
    /// Creates the dataset `name` of `T`s with `shape`, stored as `layout`
    /// says.
    pub(crate) fn create_dataset<T: Element>(
        &self,
        name: &str,
        shape: &[u64],
        layout: &Layout<'_>,
    ) -> Result<Dataset<'h>> {
        let hdf5 = self.handle.hdf5;
        let what = format!("dataset {name} of {}", self.name);
        if layout.chunk.len() != shape.len() {
            return Err(Hdf5Error(format!(
                "{what}: chunks of {} dimensions for a dataset of {}",
                layout.chunk.len(),
                shape.len()
            )));
        }
        let c_name = c_string(name, &what)?;
        // Checks that the rank fits a c_int.
        let space = hdf5.dataspace(shape, &what)?;
        let dcpl = hdf5.property_list(hdf5.g().H5P_CLS_DATASET_CREATE_ID, &what)?;
        let f = hdf5.f();
        // SAFETY: a property list this session owns and one chunk size per
        // dimension.
        let status =
            unsafe { (f.H5Pset_chunk)(dcpl.id, shape.len() as c_int, layout.chunk.as_ptr()) };
        hdf5.check(status, "H5Pset_chunk", &what)?;
        if layout.shuffle {
            // SAFETY: as above.
            let status = unsafe { (f.H5Pset_shuffle)(dcpl.id) };
            hdf5.check(status, "H5Pset_shuffle", &what)?;
        }
        // SAFETY: as above.
        let status = unsafe { (f.H5Pset_deflate)(dcpl.id, layout.deflate) };
        hdf5.check(status, "H5Pset_deflate", &what)?;
        // SAFETY: as above (`hbool_t` false).
        let status = unsafe { (f.H5Pset_obj_track_times)(dcpl.id, 0) };
        hdf5.check(status, "H5Pset_obj_track_times", &what)?;
        let (_, stored) = T::types(hdf5.g());
        // SAFETY: ids this session owns or predefined ones, and a
        // NUL-terminated name.
        let id = unsafe {
            (f.H5Dcreate2)(
                self.handle.id,
                c_name.as_ptr(),
                stored,
                space.id,
                H5P_DEFAULT,
                dcpl.id,
                H5P_DEFAULT,
            )
        };
        Ok(Dataset {
            handle: Handle::new(hdf5, id, "H5Dcreate2", &what)?,
            what,
            shape: shape.to_vec(),
        })
    }

    /// Writes an attribute of the root group.
    pub(crate) fn set_attribute(&self, name: &str, value: &AttrValue<'_>) -> Result<()> {
        self.handle
            .hdf5
            .set_attribute(self.handle.id, &self.name, name, value)
    }

    /// Closes the file, reporting a failure to write what HDF5 still holds.
    /// Every dataset must be closed (or dropped) first.
    pub(crate) fn close(self) -> Result<()> {
        self.handle.close(&self.name)
    }
}

/// A dataset being written.
pub(crate) struct Dataset<'h> {
    handle: Handle<'h, DatasetId>,
    what: String,
    shape: Vec<u64>,
}

impl Dataset<'_> {
    /// Writes the whole dataset from `data`, in row-major order (HDF5
    /// converts `T` to the stored type).
    pub(crate) fn write_all<T: Element>(&self, data: &[T]) -> Result<()> {
        let elements: u64 = self.shape.iter().product();
        if data.len() as u64 != elements {
            return Err(Hdf5Error(format!(
                "{} holds {elements} values, not {}",
                self.what,
                data.len()
            )));
        }
        let hdf5 = self.handle.hdf5;
        let (memory, _) = T::types(hdf5.g());
        // SAFETY: `data` holds one value of the memory type per element.
        let status = unsafe {
            (hdf5.f().H5Dwrite)(
                self.handle.id,
                memory,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                data.as_ptr().cast(),
            )
        };
        hdf5.check(status, "H5Dwrite", &self.what)
    }

    /// Writes whole rows of the first dimension, from row `first` on.
    pub(crate) fn write_rows<T: Element>(&self, first: u64, data: &[T]) -> Result<()> {
        let length = data.len() as u64;
        let (total, row) = match self.shape.split_first() {
            Some((total, rest)) => (*total, rest.iter().product::<u64>()),
            None => (0, 0),
        };
        let rows = length.checked_div(row).unwrap_or(0);
        if row == 0 || rows * row != length || first.checked_add(rows).is_none_or(|end| end > total)
        {
            return Err(Hdf5Error(format!(
                "{}: {length} values are not whole rows from row {first}",
                self.what
            )));
        }
        let hdf5 = self.handle.hdf5;
        let mut start = vec![0; self.shape.len()];
        start[0] = first;
        let mut count = self.shape.clone();
        count[0] = rows;
        let file_space = hdf5.dataspace(&self.shape, &self.what)?;
        // SAFETY: a dataspace this session owns and one start and count per
        // dimension (no stride or block).
        let status = unsafe {
            (hdf5.f().H5Sselect_hyperslab)(
                file_space.id,
                H5S_SELECT_SET,
                start.as_ptr(),
                std::ptr::null(),
                count.as_ptr(),
                std::ptr::null(),
            )
        };
        hdf5.check(status, "H5Sselect_hyperslab", &self.what)?;
        let memory_space = hdf5.dataspace(&count, &self.what)?;
        let (memory, _) = T::types(hdf5.g());
        // SAFETY: `data` holds one value per element of `memory_space`.
        let status = unsafe {
            (hdf5.f().H5Dwrite)(
                self.handle.id,
                memory,
                memory_space.id,
                file_space.id,
                H5P_DEFAULT,
                data.as_ptr().cast(),
            )
        };
        hdf5.check(status, "H5Dwrite", &self.what)
    }

    /// Stores the chunk whose first element is at `offset`, already passed
    /// through the dataset's filters (its stored bytes, deflated): HDF5
    /// 1.10.3 and later ([`Hdf5::can_write_chunks`]).
    pub(crate) fn write_chunk(&self, offset: &[u64], filtered: &[u8]) -> Result<()> {
        let hdf5 = self.handle.hdf5;
        let Some(write_chunk) = hdf5.api.optional.H5Dwrite_chunk else {
            return Err(Hdf5Error(format!(
                "{} cannot write chunks directly (it is older than 1.10.3)",
                hdf5.describe()
            )));
        };
        if offset.len() != self.shape.len() {
            return Err(Hdf5Error(format!(
                "{}: a chunk offset of {} dimensions",
                self.what,
                offset.len()
            )));
        }
        // SAFETY: one offset per dimension (HDF5 checks it is a chunk's
        // first element), and `filtered.len()` bytes at `filtered`.
        let status = unsafe {
            write_chunk(
                self.handle.id,
                H5P_DEFAULT,
                0,
                offset.as_ptr(),
                filtered.len(),
                filtered.as_ptr().cast(),
            )
        };
        hdf5.check(status, "H5Dwrite_chunk", &self.what)
    }

    /// Writes an attribute of the dataset.
    pub(crate) fn set_attribute(&self, name: &str, value: &AttrValue<'_>) -> Result<()> {
        self.handle
            .hdf5
            .set_attribute(self.handle.id, &self.what, name, value)
    }

    /// Closes the dataset, reporting a failure to write what HDF5 still
    /// holds of it.
    pub(crate) fn close(self) -> Result<()> {
        self.handle.close(&self.what)
    }
}

/// Reading files back, for tests.
#[cfg(test)]
pub(crate) mod testing {
    use super::ffi::ReadFunctions;
    use super::*;

    /// A session with the library this machine has, or `None` (and a note
    /// on standard error) without one, for tests that need HDF5.
    pub(crate) fn session(test: &str) -> Option<Hdf5> {
        match Hdf5::open(&LibrarySearch::from_env(None)) {
            Ok(hdf5) => {
                eprintln!("{test}: using {}", hdf5.describe());
                Some(hdf5)
            }
            Err(error) => {
                eprintln!("{test}: skipped, no HDF5 library:\n{error}");
                None
            }
        }
    }

    /// How a dataset is stored.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct Stored {
        pub(crate) shape: Vec<u64>,
        /// `H5T_class_t`: 0 integer, 1 float, 3 string.
        pub(crate) class: c_int,
        pub(crate) size: usize,
        pub(crate) signed: bool,
        pub(crate) little_endian: bool,
        /// Chunk dimensions, when chunked.
        pub(crate) chunk: Option<Vec<u64>>,
        /// Filter ids and their parameters, in pipeline order.
        pub(crate) filters: Vec<(c_int, Vec<c_uint>)>,
    }

    /// An object (`None`: the root group) whose attributes are read.
    enum Object<'h> {
        Root,
        Dataset(Handle<'h, DatasetId>),
    }

    impl Hdf5 {
        fn read_functions(&self) -> &ReadFunctions {
            self.api
                .read
                // SAFETY: the library is HDF5 1.10 or newer and never unloaded.
                .get_or_init(|| unsafe { ReadFunctions::resolve(self.api.library) })
                .as_ref()
                .expect("the read functions of HDF5")
        }

        /// Opens `path` to read.
        pub(crate) fn open_file(&self, path: &Path) -> Result<File<'_>> {
            let name = path.display().to_string();
            let c_path = c_path(path)?;
            // SAFETY: a NUL-terminated path (H5F_ACC_RDONLY is 0).
            let id = unsafe { (self.read_functions().H5Fopen)(c_path.as_ptr(), 0, H5P_DEFAULT) };
            Ok(File {
                handle: Handle::new(self, id, "H5Fopen", &name)?,
                name,
            })
        }

        /// Files, datasets, groups, types and attributes open in any file.
        pub(crate) fn open_objects(&self) -> isize {
            const H5F_OBJ_ALL: c_uint = 0x1F;
            // SAFETY: no pointers.
            unsafe { (self.read_functions().H5Fget_obj_count)(H5F_OBJ_ALL as hid_t, H5F_OBJ_ALL) }
        }
    }

    impl<'h> File<'h> {
        fn r(&self) -> &'h ReadFunctions {
            self.handle.hdf5.read_functions()
        }

        pub(crate) fn exists(&self, name: &str) -> bool {
            let name = c_string(name, name).unwrap();
            // SAFETY: an open file and a NUL-terminated name.
            unsafe { (self.r().H5Lexists)(self.handle.id, name.as_ptr(), H5P_DEFAULT) > 0 }
        }

        fn dataset(&self, name: &str) -> Result<Handle<'h, DatasetId>> {
            let c_name = c_string(name, name)?;
            // SAFETY: as above.
            let id = unsafe { (self.r().H5Dopen2)(self.handle.id, c_name.as_ptr(), H5P_DEFAULT) };
            Handle::new(self.handle.hdf5, id, "H5Dopen2", name)
        }

        fn dims(&self, space: &Handle<'h, DataspaceId>) -> Vec<u64> {
            // SAFETY: a dataspace this session owns.
            let rank = unsafe { (self.r().H5Sget_simple_extent_ndims)(space.id) };
            let mut dims = vec![0; rank.max(0) as usize];
            // SAFETY: room for `rank` dimensions.
            unsafe {
                (self.r().H5Sget_simple_extent_dims)(
                    space.id,
                    dims.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            dims
        }

        /// How dataset `name` is stored.
        pub(crate) fn stored(&self, name: &str) -> Result<Stored> {
            let hdf5 = self.handle.hdf5;
            let r = self.r();
            let dataset = self.dataset(name)?;
            // SAFETY: an open dataset; the ids returned are closed by handles.
            let space: Handle<'_, DataspaceId> = Handle::new(
                hdf5,
                unsafe { (r.H5Dget_space)(dataset.id) },
                "H5Dget_space",
                name,
            )?;
            let shape = self.dims(&space);
            // SAFETY: as above.
            let stored: Handle<'_, DatatypeId> = Handle::new(
                hdf5,
                unsafe { (r.H5Dget_type)(dataset.id) },
                "H5Dget_type",
                name,
            )?;
            // SAFETY: as above.
            let dcpl: Handle<'_, PropertyListId> = Handle::new(
                hdf5,
                unsafe { (r.H5Dget_create_plist)(dataset.id) },
                "H5Dget_create_plist",
                name,
            )?;
            // SAFETY: open ids; buffers as long as their counts say.
            unsafe {
                let chunk = ((r.H5Pget_layout)(dcpl.id) == 2).then(|| {
                    let mut chunk = vec![0; shape.len()];
                    (r.H5Pget_chunk)(dcpl.id, chunk.len() as c_int, chunk.as_mut_ptr());
                    chunk
                });
                let filters = (0..(r.H5Pget_nfilters)(dcpl.id).max(0) as c_uint)
                    .map(|index| {
                        let mut flags = 0;
                        let mut values = [0 as c_uint; 8];
                        let mut count = values.len();
                        let mut config = 0;
                        let mut name = [0 as c_char; 64];
                        let id = (r.H5Pget_filter2)(
                            dcpl.id,
                            index,
                            &mut flags,
                            &mut count,
                            values.as_mut_ptr(),
                            name.len(),
                            name.as_mut_ptr(),
                            &mut config,
                        );
                        (id, values[..count.min(values.len())].to_vec())
                    })
                    .collect();
                Ok(Stored {
                    shape,
                    class: (r.H5Tget_class)(stored.id),
                    size: (r.H5Tget_size)(stored.id),
                    signed: (r.H5Tget_sign)(stored.id) == 1,
                    little_endian: (r.H5Tget_order)(stored.id) == 0,
                    chunk,
                    filters,
                })
            }
        }

        /// The values of dataset `name`, converted to `T`.
        pub(crate) fn read<T: Element + Default>(&self, name: &str) -> Result<Vec<T>> {
            let hdf5 = self.handle.hdf5;
            let dataset = self.dataset(name)?;
            // SAFETY: an open dataset.
            let space: Handle<'_, DataspaceId> = Handle::new(
                hdf5,
                unsafe { (self.r().H5Dget_space)(dataset.id) },
                "H5Dget_space",
                name,
            )?;
            let count: u64 = self.dims(&space).iter().product();
            let mut values = vec![T::default(); count as usize];
            let (memory, _) = T::types(hdf5.g());
            // SAFETY: room for every element.
            let status = unsafe {
                (self.r().H5Dread)(
                    dataset.id,
                    memory,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    values.as_mut_ptr().cast(),
                )
            };
            hdf5.check(status, "H5Dread", name)?;
            Ok(values)
        }

        fn object(&self, object: Option<&str>) -> Result<Object<'h>> {
            Ok(match object {
                None => Object::Root,
                Some(name) => Object::Dataset(self.dataset(name)?),
            })
        }

        fn attribute(&self, object: &Object<'h>, name: &str) -> Result<Handle<'h, AttributeId>> {
            let location = match object {
                Object::Root => self.handle.id,
                Object::Dataset(dataset) => dataset.id,
            };
            let c_name = c_string(name, name)?;
            // SAFETY: an open object and a NUL-terminated name.
            let id = unsafe { (self.r().H5Aopen)(location, c_name.as_ptr(), H5P_DEFAULT) };
            Handle::new(self.handle.hdf5, id, "H5Aopen", name)
        }

        /// A numeric attribute of the root group (`object` `None`) or a
        /// dataset, converted to `T`.
        pub(crate) fn numbers<T: Element + Default>(
            &self,
            object: Option<&str>,
            name: &str,
        ) -> Result<Vec<T>> {
            let hdf5 = self.handle.hdf5;
            let object = self.object(object)?;
            let attribute = self.attribute(&object, name)?;
            // SAFETY: an open attribute.
            let space: Handle<'_, DataspaceId> = Handle::new(
                hdf5,
                unsafe { (self.r().H5Aget_space)(attribute.id) },
                "H5Aget_space",
                name,
            )?;
            let count: u64 = self.dims(&space).iter().product();
            let mut values = vec![T::default(); count as usize];
            let (memory, _) = T::types(hdf5.g());
            // SAFETY: room for every element.
            let status =
                unsafe { (self.r().H5Aread)(attribute.id, memory, values.as_mut_ptr().cast()) };
            hdf5.check(status, "H5Aread", name)?;
            Ok(values)
        }

        /// The rank of an attribute's dataspace (0 for a scalar).
        pub(crate) fn rank(&self, object: Option<&str>, name: &str) -> Result<usize> {
            let object = self.object(object)?;
            let attribute = self.attribute(&object, name)?;
            // SAFETY: an open attribute.
            let space: Handle<'_, DataspaceId> = Handle::new(
                self.handle.hdf5,
                unsafe { (self.r().H5Aget_space)(attribute.id) },
                "H5Aget_space",
                name,
            )?;
            Ok(self.dims(&space).len())
        }

        /// A text attribute, which must be a variable-length UTF-8 string.
        pub(crate) fn text(&self, object: Option<&str>, name: &str) -> Result<String> {
            let hdf5 = self.handle.hdf5;
            let r = self.r();
            let object = self.object(object)?;
            let attribute = self.attribute(&object, name)?;
            // SAFETY: an open attribute.
            let stored: Handle<'_, DatatypeId> = Handle::new(
                hdf5,
                unsafe { (r.H5Aget_type)(attribute.id) },
                "H5Aget_type",
                name,
            )?;
            // SAFETY: an open type.
            let (variable, utf8) = unsafe {
                (
                    (r.H5Tis_variable_str)(stored.id) > 0,
                    (r.H5Tget_cset)(stored.id) == H5T_CSET_UTF8,
                )
            };
            if !(variable && utf8) {
                return Err(Hdf5Error(format!(
                    "{name} is not a variable-length UTF-8 string"
                )));
            }
            let string = hdf5.string_type(name)?;
            let mut pointer: *mut c_char = std::ptr::null_mut();
            // SAFETY: one variable-length string, read into a pointer that
            // HDF5 allocates and `H5free_memory` frees.
            unsafe {
                let status = (r.H5Aread)(
                    attribute.id,
                    string.id,
                    (&mut pointer as *mut *mut c_char).cast(),
                );
                hdf5.check(status, "H5Aread", name)?;
                let text = CStr::from_ptr(pointer).to_string_lossy().into_owned();
                (r.H5free_memory)(pointer.cast());
                Ok(text)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::session;
    use super::*;
    use std::io::Write;

    #[test]
    fn a_file_that_is_not_a_library_is_refused_with_what_to_do() {
        let temp = tempfile::tempdir().unwrap();
        let not_a_library = temp.path().join("libhdf5.so");
        std::fs::write(&not_a_library, "not a library").unwrap();
        let search = LibrarySearch {
            explicit: Some((not_a_library.clone(), LIBRARY_ENV)),
            ..LibrarySearch::default()
        };
        let Err(error) = Hdf5::open(&search) else {
            panic!("a text file loaded as HDF5");
        };
        assert_eq!(error.tried.len(), 1);
        let text = error.to_string();
        assert!(
            text.contains(&not_a_library.display().to_string()) && text.contains(LIBRARY_ENV),
            "{text}"
        );
        assert!(text.contains("--format zarr"), "{text}");
        assert!(text.contains("HDF5 1.10 or newer"), "{text}");

        // Nothing at all: every place is listed.
        let empty = tempfile::tempdir().unwrap();
        let search = LibrarySearch {
            hdf5_dir: Some(empty.path().into()),
            conda_prefix: Some(temp.path().into()),
            ..LibrarySearch::default()
        };
        let Err(error) = Hdf5::open(&search) else {
            panic!("an empty folder held HDF5");
        };
        assert_eq!(error.tried.len(), search.candidates().len());
        let text = error.to_string();
        for (candidate, _) in &error.tried {
            assert!(text.contains(&candidate.path.display().to_string()));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_library_that_is_not_hdf5_is_refused() {
        let search = LibrarySearch {
            explicit: Some(("libm.so.6".into(), "--hdf5-library")),
            ..LibrarySearch::default()
        };
        let Err(error) = Hdf5::open(&search) else {
            panic!("libm loaded as HDF5");
        };
        assert!(error.tried[0].1.contains("not an HDF5 library"), "{error}");
    }

    #[test]
    fn attributes_and_datasets_round_trip() {
        let Some(hdf5) = session("attributes_and_datasets_round_trip") else {
            return;
        };
        let (major, minor, _) = hdf5.version();
        assert!((major, minor) >= (1, 10));
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("test.h5");
        // Over 64 KiB, which the HDF5 1.8 format keeps in dense storage.
        let long = "µm ".repeat(40_000);
        let pixels: Vec<u8> = (0..4 * 8 * 8 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let coords: Vec<i64> = (0..10).map(|i| i * 1_000_000_007).collect();
        {
            let file = hdf5.create(&path).unwrap();
            file.set_attribute("long", &AttrValue::Text(&long)).unwrap();
            file.set_attribute("int", &AttrValue::Int(-5)).unwrap();
            file.set_attribute("float", &AttrValue::Float(0.25))
                .unwrap();
            let images = file
                .create_dataset::<u8>(
                    "images",
                    &[4, 8, 8, 3],
                    &Layout {
                        chunk: &[1, 8, 8, 3],
                        shuffle: false,
                        deflate: 4,
                    },
                )
                .unwrap();
            for (index, patch) in pixels.chunks(8 * 8 * 3).enumerate() {
                if index % 2 == 0 && hdf5.can_write_chunks() {
                    let mut deflated =
                        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(4));
                    deflated.write_all(patch).unwrap();
                    images
                        .write_chunk(&[index as u64, 0, 0, 0], &deflated.finish().unwrap())
                        .unwrap();
                } else {
                    images.write_rows(index as u64, patch).unwrap();
                }
            }
            images.close().unwrap();
            let table = file
                .create_dataset::<i64>(
                    "coords",
                    &[5, 2],
                    &Layout {
                        chunk: &[5, 2],
                        shuffle: true,
                        deflate: 4,
                    },
                )
                .unwrap();
            table.write_all(&coords).unwrap();
            assert!(table.write_all(&coords[..4]).is_err(), "too few values");
            assert!(table.write_rows(4, &coords[..4]).is_err(), "past the end");
            table
                .set_attribute("downsample", &AttrValue::Floats(&[2.0, 2.5]))
                .unwrap();
            table
                .set_attribute("level", &AttrValue::Ints(&[1]))
                .unwrap();
            table.set_attribute("empty", &AttrValue::Ints(&[])).unwrap();
            table.set_attribute("name", &AttrValue::Text("")).unwrap();
            table.close().unwrap();
            file.close().unwrap();
        }
        assert_eq!(hdf5.open_objects(), 0, "every id is closed");

        let file = hdf5.open_file(&path).unwrap();
        assert_eq!(file.text(None, "long").unwrap(), long);
        assert_eq!(file.numbers::<i64>(None, "int").unwrap(), [-5]);
        assert_eq!(file.numbers::<f64>(None, "float").unwrap(), [0.25]);
        assert_eq!(file.read::<u8>("images").unwrap(), pixels);
        let stored = file.stored("images").unwrap();
        assert_eq!(stored.shape, [4, 8, 8, 3]);
        assert_eq!(stored.chunk.as_deref(), Some(&[1, 8, 8, 3][..]));
        assert_eq!(stored.filters, [(1, vec![4])], "deflate at level 4");
        assert_eq!(
            (
                stored.class,
                stored.size,
                stored.signed,
                stored.little_endian
            ),
            (0, 1, false, true)
        );
        assert_eq!(file.read::<i64>("coords").unwrap(), coords);
        let stored = file.stored("coords").unwrap();
        assert_eq!(
            stored.filters.iter().map(|f| f.0).collect::<Vec<_>>(),
            [2, 1]
        );
        assert_eq!((stored.size, stored.signed), (8, true));
        assert_eq!(
            file.numbers::<f64>(Some("coords"), "downsample").unwrap(),
            [2.0, 2.5]
        );
        // A list of one stays a list (h5py reads an array, not a number).
        assert_eq!(file.numbers::<i64>(Some("coords"), "level").unwrap(), [1]);
        assert_eq!(file.rank(Some("coords"), "level").unwrap(), 1);
        assert_eq!(file.rank(None, "int").unwrap(), 0);
        assert!(
            file.numbers::<i64>(Some("coords"), "empty")
                .unwrap()
                .is_empty()
        );
        assert_eq!(file.text(Some("coords"), "name").unwrap(), "");
        assert!(file.exists("coords") && !file.exists("masks"));
        drop(file);
        assert_eq!(hdf5.open_objects(), 0);
    }

    #[test]
    fn failures_say_what_hdf5_reported() {
        let Some(hdf5) = session("failures_say_what_hdf5_reported") else {
            return;
        };
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("missing").join("x.h5");
        let Err(error) = hdf5.create(&path) else {
            panic!("created a file in a missing folder");
        };
        let text = error.to_string();
        assert!(text.starts_with("H5Fcreate failed for "), "{text}");
        assert!(text.contains("unable to"), "{text}");
        let file = hdf5.create(&temp.path().join("y.h5")).unwrap();
        let Err(error) = file.create_dataset::<u8>(
            "bad",
            &[4],
            &Layout {
                chunk: &[8],
                shuffle: false,
                deflate: 4,
            },
        ) else {
            panic!("a chunk larger than its fixed-size dataset");
        };
        assert!(error.to_string().contains("H5Dcreate2"), "{error}");
        file.close().unwrap();
    }
}
