//! `eov register-file-types` on Windows: carries out the plan of
//! `registry.rs` in `HKEY_CURRENT_USER` and tells Explorer.

use super::registry::{self, Change};
use anyhow::{Context, Result};
use std::io;
use windows_sys::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS};
use windows_sys::Win32::System::Registry::{
    HKEY, HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_EXPAND_SZ,
    REG_OPTION_NON_VOLATILE, REG_SZ, RegCloseKey, RegCreateKeyExW, RegDeleteTreeW, RegDeleteValueW,
    RegOpenKeyExW, RegQueryValueExW, RegSetValueExW,
};
use windows_sys::Win32::UI::Shell::{SHCNE_ASSOCCHANGED, SHCNF_IDLIST, SHChangeNotify};

/// `text` as a NUL-terminated UTF-16 string.
fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

fn check(status: u32) -> io::Result<()> {
    if status == ERROR_SUCCESS {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(status as i32))
    }
}

/// An open key below `HKEY_CURRENT_USER`.
struct Key(HKEY);

impl Drop for Key {
    fn drop(&mut self) {
        // SAFETY: the handle came from RegCreateKeyExW/RegOpenKeyExW.
        unsafe {
            RegCloseKey(self.0);
        }
    }
}

impl Key {
    fn create(path: &str) -> io::Result<Key> {
        let path = wide(path);
        let mut key: HKEY = 0;
        // SAFETY: NUL-terminated string, valid out-pointer, optional
        // arguments null.
        check(unsafe {
            RegCreateKeyExW(
                HKEY_CURRENT_USER,
                path.as_ptr(),
                0,
                std::ptr::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_QUERY_VALUE | KEY_SET_VALUE,
                std::ptr::null(),
                &mut key,
                std::ptr::null_mut(),
            )
        })?;
        Ok(Key(key))
    }

    fn open(path: &str) -> io::Result<Option<Key>> {
        let path = wide(path);
        let mut key: HKEY = 0;
        // SAFETY: NUL-terminated string, valid out-pointer.
        let status = unsafe {
            RegOpenKeyExW(
                HKEY_CURRENT_USER,
                path.as_ptr(),
                0,
                KEY_QUERY_VALUE | KEY_SET_VALUE,
                &mut key,
            )
        };
        if status == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        check(status)?;
        Ok(Some(Key(key)))
    }

    /// A value as text (`Some("?")` for a value that is not text).
    fn get(&self, name: Option<&str>) -> io::Result<Option<String>> {
        let name = name.map(wide);
        let name = name.as_ref().map_or(std::ptr::null(), |name| name.as_ptr());
        let mut kind = 0;
        let mut size = 0u32;
        // SAFETY: size query: no data pointer.
        let status = unsafe {
            RegQueryValueExW(
                self.0,
                name,
                std::ptr::null(),
                &mut kind,
                std::ptr::null_mut(),
                &mut size,
            )
        };
        if status == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        check(status)?;
        if kind != REG_SZ && kind != REG_EXPAND_SZ {
            return Ok(Some("?".to_string()));
        }
        let mut buffer = vec![0u16; (size as usize).div_ceil(2) + 1];
        let mut bytes = (buffer.len() * 2) as u32;
        // SAFETY: the buffer holds `bytes` bytes.
        check(unsafe {
            RegQueryValueExW(
                self.0,
                name,
                std::ptr::null(),
                &mut kind,
                buffer.as_mut_ptr().cast(),
                &mut bytes,
            )
        })?;
        let length = (bytes as usize / 2).min(buffer.len());
        let text = String::from_utf16_lossy(&buffer[..length]);
        Ok(Some(text.trim_end_matches('\0').to_string()))
    }

    fn set(&self, name: Option<&str>, value: &str) -> io::Result<()> {
        let name = name.map(wide);
        let name = name.as_ref().map_or(std::ptr::null(), |name| name.as_ptr());
        let data = wide(value);
        // SAFETY: `data` holds `data.len() * 2` bytes of REG_SZ text.
        check(unsafe {
            RegSetValueExW(
                self.0,
                name,
                0,
                REG_SZ,
                data.as_ptr().cast(),
                (data.len() * 2) as u32,
            )
        })
    }

    fn delete(&self, name: Option<&str>) -> io::Result<()> {
        let name = name.map(wide);
        let name = name.as_ref().map_or(std::ptr::null(), |name| name.as_ptr());
        // SAFETY: NUL-terminated name or null (the default value).
        let status = unsafe { RegDeleteValueW(self.0, name) };
        if status == ERROR_FILE_NOT_FOUND {
            Ok(())
        } else {
            check(status)
        }
    }
}

fn delete_tree(path: &str) -> io::Result<()> {
    let path = wide(path);
    // SAFETY: NUL-terminated string.
    let status = unsafe { RegDeleteTreeW(HKEY_CURRENT_USER, path.as_ptr()) };
    if status == ERROR_FILE_NOT_FOUND {
        Ok(())
    } else {
        check(status)
    }
}

fn apply(change: &Change) -> io::Result<()> {
    match change {
        Change::Set { key, name, value } => Key::create(key)?.set(name.as_deref(), value),
        Change::SetIfUnset { key, name, value } => {
            let key = Key::create(key)?;
            match key.get(name.as_deref())? {
                Some(current) if !current.is_empty() => Ok(()),
                _ => key.set(name.as_deref(), value),
            }
        }
        Change::DeleteValue { key, name } => match Key::open(key)? {
            Some(key) => key.delete(Some(name)),
            None => Ok(()),
        },
        Change::DeleteValueIf { key, name, value } => match Key::open(key)? {
            Some(key) if key.get(name.as_deref())?.as_deref() == Some(value.as_str()) => {
                key.delete(name.as_deref())
            }
            _ => Ok(()),
        },
        Change::DeleteTree { key, if_default } => {
            if let Some(expected) = if_default {
                let ours = match Key::open(key)? {
                    Some(open) => open.get(None)?.as_deref() == Some(expected.as_str()),
                    None => false,
                };
                if !ours {
                    return Ok(());
                }
            }
            delete_tree(key)
        }
    }
}

pub(super) fn run(register: bool, dry_run: bool) -> Result<()> {
    let exe = std::env::current_exe().context("cannot tell where eov.exe is")?;
    let exe = crate::paths::simplified(&exe);
    let changes = if register {
        registry::registration(&exe)
    } else {
        registry::unregistration()
    };
    for change in &changes {
        if dry_run {
            println!("would {change}");
        } else {
            apply(change).with_context(|| format!("could not {change}"))?;
        }
    }
    if dry_run {
        return Ok(());
    }
    // SAFETY: plain Win32 call without items.
    unsafe {
        SHChangeNotify(
            SHCNE_ASSOCCHANGED,
            SHCNF_IDLIST,
            std::ptr::null(),
            std::ptr::null(),
        );
    }
    if register {
        println!(
            "Explorer now offers {} for slide files and eov:// links (for this user). \
             Run `eov register-file-types` again after moving the eov folder.",
            exe.display()
        );
    } else {
        println!("eov is no longer registered for slide files and eov:// links.");
    }
    Ok(())
}
