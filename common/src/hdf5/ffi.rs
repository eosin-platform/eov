//! The part of HDF5's C API that eov calls: types, constants, and tables of
//! function pointers and global ids resolved from the loaded library.
//!
//! The signatures are those of HDF5 1.10 and later: `hid_t` has been 64
//! bits wide since 1.10.0, so nothing but `H5open` and `H5get_libversion`
//! (unchanged since 1.8) is resolved before the version is checked.
//! Enumerations are passed as `c_int`, and `hbool_t` as `c_uint`, which is
//! right whether the library was built with C99 `bool` or with `unsigned`.

#![allow(non_camel_case_types, non_snake_case)]

use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void};

pub(crate) type hid_t = i64;
pub(crate) type herr_t = c_int;
pub(crate) type htri_t = c_int;
pub(crate) type hsize_t = u64;
pub(crate) type H5Z_filter_t = c_int;

/// The default property list, also "the whole dataspace" (`H5S_ALL`) and
/// "the current thread's error stack" (`H5E_DEFAULT`).
pub(crate) const H5P_DEFAULT: hid_t = 0;
pub(crate) const H5S_ALL: hid_t = 0;
pub(crate) const H5E_DEFAULT: hid_t = 0;
pub(crate) const H5F_ACC_TRUNC: c_uint = 0x0002;
/// `H5S_class_t`
pub(crate) const H5S_SCALAR: c_int = 0;
/// `H5S_seloper_t`
pub(crate) const H5S_SELECT_SET: c_int = 0;
/// `H5T_cset_t`
pub(crate) const H5T_CSET_UTF8: c_int = 1;
/// The size of a variable-length string type.
pub(crate) const H5T_VARIABLE: usize = usize::MAX;
pub(crate) const H5Z_FILTER_DEFLATE: H5Z_filter_t = 1;
/// `H5E_direction_t`: from the API call down to where the error arose.
pub(crate) const H5E_WALK_DOWNWARD: c_int = 1;
/// `H5F_LIBVER_V18` from HDF5 1.10.2 on. In 1.10.0 and 1.10.1 the same
/// value is `H5F_LIBVER_LATEST`; both allow attributes over 64 KiB.
pub(crate) const H5F_LIBVER_V18: c_int = 1;

/// One entry of an error stack (`H5E_error2_t`).
#[repr(C)]
pub(crate) struct H5E_error2_t {
    pub(crate) cls_id: hid_t,
    pub(crate) maj_num: hid_t,
    pub(crate) min_num: hid_t,
    pub(crate) line: c_uint,
    pub(crate) func_name: *const c_char,
    pub(crate) file_name: *const c_char,
    pub(crate) desc: *const c_char,
}

pub(crate) type H5E_auto2_t = Option<unsafe extern "C" fn(hid_t, *mut c_void) -> herr_t>;
pub(crate) type H5E_walk2_t =
    Option<unsafe extern "C" fn(c_uint, *const H5E_error2_t, *mut c_void) -> herr_t>;

/// A symbol of `library` as a `T` (a function pointer, or the address of a
/// variable).
///
/// # Safety
///
/// `T` must be the symbol's real type, and the library must stay loaded
/// for as long as the result is used.
unsafe fn symbol<T: Copy>(library: &Library, name: &str) -> Result<T, String> {
    // SAFETY: as the caller promises.
    match unsafe { library.get::<T>(name.as_bytes()) } {
        Ok(symbol) => Ok(*symbol),
        Err(_) => Err(format!("it has no {}", name.trim_end_matches('\0'))),
    }
}

/// A struct of function pointers, resolved by name.
macro_rules! functions {
    ($(#[$meta:meta])* struct $table:ident {
        $($name:ident: fn($($arg:ty),* $(,)?) -> $ret:ty,)*
    }) => {
        $(#[$meta])*
        pub(crate) struct $table {
            $(pub(crate) $name: unsafe extern "C" fn($($arg),*) -> $ret,)*
        }

        impl $table {
            /// # Safety
            ///
            /// `library` must be an HDF5 whose functions have these
            /// signatures, and must never be unloaded.
            pub(crate) unsafe fn resolve(library: &Library) -> Result<Self, String> {
                Ok(Self {
                    // SAFETY: as the caller promises.
                    $($name: unsafe { symbol(library, concat!(stringify!($name), "\0"))? },)*
                })
            }
        }
    };
}

functions! {
    /// What is called before the version is known.
    struct Startup {
        H5open: fn() -> herr_t,
        H5get_libversion: fn(*mut c_uint, *mut c_uint, *mut c_uint) -> herr_t,
    }
}

functions! {
    /// What eov writes files with (HDF5 1.10.0 has all of them).
    struct Functions {
        H5Eset_auto2: fn(hid_t, H5E_auto2_t, *mut c_void) -> herr_t,
        H5Ewalk2: fn(hid_t, c_int, H5E_walk2_t, *mut c_void) -> herr_t,
        H5Zfilter_avail: fn(H5Z_filter_t) -> htri_t,
        H5Fcreate: fn(*const c_char, c_uint, hid_t, hid_t) -> hid_t,
        H5Fclose: fn(hid_t) -> herr_t,
        H5Pcreate: fn(hid_t) -> hid_t,
        H5Pclose: fn(hid_t) -> herr_t,
        H5Pset_libver_bounds: fn(hid_t, c_int, c_int) -> herr_t,
        H5Pset_obj_track_times: fn(hid_t, c_uint) -> herr_t,
        H5Pset_chunk: fn(hid_t, c_int, *const hsize_t) -> herr_t,
        H5Pset_shuffle: fn(hid_t) -> herr_t,
        H5Pset_deflate: fn(hid_t, c_uint) -> herr_t,
        H5Screate: fn(c_int) -> hid_t,
        H5Screate_simple: fn(c_int, *const hsize_t, *const hsize_t) -> hid_t,
        H5Sselect_hyperslab: fn(
            hid_t,
            c_int,
            *const hsize_t,
            *const hsize_t,
            *const hsize_t,
            *const hsize_t,
        ) -> herr_t,
        H5Sclose: fn(hid_t) -> herr_t,
        H5Tcopy: fn(hid_t) -> hid_t,
        H5Tset_size: fn(hid_t, usize) -> herr_t,
        H5Tset_cset: fn(hid_t, c_int) -> herr_t,
        H5Tclose: fn(hid_t) -> herr_t,
        H5Dcreate2: fn(hid_t, *const c_char, hid_t, hid_t, hid_t, hid_t, hid_t) -> hid_t,
        H5Dwrite: fn(hid_t, hid_t, hid_t, hid_t, hid_t, *const c_void) -> herr_t,
        H5Dclose: fn(hid_t) -> herr_t,
        H5Acreate2: fn(hid_t, *const c_char, hid_t, hid_t, hid_t, hid_t) -> hid_t,
        H5Awrite: fn(hid_t, hid_t, *const c_void) -> herr_t,
        H5Aclose: fn(hid_t) -> herr_t,
    }
}

/// `H5Dwrite_chunk` (HDF5 1.10.3 and later): stores a chunk that is
/// already filtered.
pub(crate) type WriteChunk =
    unsafe extern "C" fn(hid_t, hid_t, u32, *const hsize_t, usize, *const c_void) -> herr_t;
/// `H5Pset_file_locking` (HDF5 1.10.7 and 1.12.1 and later).
pub(crate) type SetFileLocking = unsafe extern "C" fn(hid_t, c_uint, c_uint) -> herr_t;

/// Functions that newer versions add; eov does without them.
pub(crate) struct Optional {
    pub(crate) H5Dwrite_chunk: Option<WriteChunk>,
    pub(crate) H5Pset_file_locking: Option<SetFileLocking>,
}

impl Optional {
    /// # Safety
    ///
    /// As [`Functions::resolve`].
    pub(crate) unsafe fn resolve(library: &Library) -> Self {
        // SAFETY: as the caller promises.
        unsafe {
            Self {
                H5Dwrite_chunk: symbol(library, "H5Dwrite_chunk\0").ok(),
                H5Pset_file_locking: symbol(library, "H5Pset_file_locking\0").ok(),
            }
        }
    }
}

/// The ids of predefined types and property-list classes. In C they are
/// macros such as `H5T_NATIVE_UINT8` that read a global variable
/// (`H5T_NATIVE_UINT8_g`) after calling `H5open`; the variables are only
/// set once the library is initialised.
macro_rules! globals {
    ($($name:ident,)*) => {
        pub(crate) struct Globals {
            $(pub(crate) $name: hid_t,)*
        }

        impl Globals {
            /// # Safety
            ///
            /// `library` must be an initialised HDF5 1.10 or newer (whose
            /// globals are `hid_t`), and must never be unloaded.
            pub(crate) unsafe fn read(library: &Library) -> Result<Self, String> {
                Ok(Self {
                    // SAFETY: as the caller promises.
                    $($name: unsafe { global(library, concat!(stringify!($name), "_g\0"))? },)*
                })
            }
        }
    };
}

/// # Safety
///
/// As [`Globals::read`].
unsafe fn global(library: &Library, name: &str) -> Result<hid_t, String> {
    // SAFETY: the symbol is a `hid_t` variable, as the caller promises.
    let address: *const hid_t = unsafe { symbol(library, name)? };
    // SAFETY: the address of a variable of the loaded library.
    let id = unsafe { address.read() };
    if id < 0 {
        return Err(format!("{} is not set", name.trim_end_matches('\0')));
    }
    Ok(id)
}

globals! {
    H5T_NATIVE_UINT8,
    H5T_NATIVE_INT32,
    H5T_NATIVE_INT64,
    H5T_NATIVE_DOUBLE,
    H5T_STD_U8LE,
    H5T_STD_I32LE,
    H5T_STD_I64LE,
    H5T_IEEE_F64LE,
    H5T_C_S1,
    H5P_CLS_FILE_CREATE_ID,
    H5P_CLS_FILE_ACCESS_ID,
    H5P_CLS_DATASET_CREATE_ID,
}

#[cfg(test)]
functions! {
    /// What the tests read files back with.
    struct ReadFunctions {
        H5Fopen: fn(*const c_char, c_uint, hid_t) -> hid_t,
        H5Fget_obj_count: fn(hid_t, c_uint) -> isize,
        H5Lexists: fn(hid_t, *const c_char, hid_t) -> htri_t,
        H5Dopen2: fn(hid_t, *const c_char, hid_t) -> hid_t,
        H5Dread: fn(hid_t, hid_t, hid_t, hid_t, hid_t, *mut c_void) -> herr_t,
        H5Dget_space: fn(hid_t) -> hid_t,
        H5Dget_type: fn(hid_t) -> hid_t,
        H5Dget_create_plist: fn(hid_t) -> hid_t,
        H5Pget_layout: fn(hid_t) -> c_int,
        H5Pget_chunk: fn(hid_t, c_int, *mut hsize_t) -> c_int,
        H5Pget_nfilters: fn(hid_t) -> c_int,
        H5Pget_filter2: fn(
            hid_t,
            c_uint,
            *mut c_uint,
            *mut usize,
            *mut c_uint,
            usize,
            *mut c_char,
            *mut c_uint,
        ) -> H5Z_filter_t,
        H5Sget_simple_extent_ndims: fn(hid_t) -> c_int,
        H5Sget_simple_extent_dims: fn(hid_t, *mut hsize_t, *mut hsize_t) -> c_int,
        H5Tget_class: fn(hid_t) -> c_int,
        H5Tget_size: fn(hid_t) -> usize,
        H5Tget_sign: fn(hid_t) -> c_int,
        H5Tget_order: fn(hid_t) -> c_int,
        H5Tis_variable_str: fn(hid_t) -> htri_t,
        H5Tget_cset: fn(hid_t) -> c_int,
        H5Aopen: fn(hid_t, *const c_char, hid_t) -> hid_t,
        H5Aread: fn(hid_t, hid_t, *mut c_void) -> herr_t,
        H5Aget_type: fn(hid_t) -> hid_t,
        H5Aget_space: fn(hid_t) -> hid_t,
        H5free_memory: fn(*mut c_void) -> herr_t,
    }
}
