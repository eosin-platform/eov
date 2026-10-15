//! Slides and links macOS hands to eov (SH-8).
//!
//! Double-clicking a slide in the Finder, dropping one on the Dock icon,
//! "Open With" and `eov://` links reach an app as an Apple event, not on its
//! command line, and AppKit passes them to the application delegate's
//! `application:openURLs:`. winit 0.30, which owns that delegate, does not
//! implement it, so eov adds the method to winit's delegate class at run
//! time (winit patches `NSApplication` the same way) and opens what arrives
//! like a request from a second eov.
//!
//! [`install`] must run after the event loop exists (after the window is
//! created) and before it runs, so that the files of the launch itself
//! arrive too: AppKit delivers them while the event loop starts.

use objc::runtime::{BOOL, Class, Imp, NO, Object, Sel, YES, class_addMethod, object_getClass};
use objc::{class, msg_send, sel, sel_impl};
use std::ffi::{CStr, OsStr, OsString, c_char};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

/// Adds `application:openURLs:` to the application delegate.
#[allow(unexpected_cfgs)]
pub(crate) fn install() {
    // SAFETY: Objective-C calls on the main thread, with the signatures
    // AppKit documents; `open_urls` matches the `v@:@@` type encoding.
    unsafe {
        let application: *mut Object = msg_send![class!(NSApplication), sharedApplication];
        if application.is_null() {
            tracing::warn!("No NSApplication: files opened from the Finder will not arrive");
            return;
        }
        let delegate: *mut Object = msg_send![application, delegate];
        if delegate.is_null() {
            tracing::warn!("No application delegate: files opened from the Finder will not arrive");
            return;
        }
        let class = object_getClass(delegate) as *mut Class;
        let handler = open_urls as extern "C" fn(&Object, Sel, *mut Object, *mut Object);
        let added = class_addMethod(
            class,
            sel!(application:openURLs:),
            std::mem::transmute::<extern "C" fn(&Object, Sel, *mut Object, *mut Object), Imp>(
                handler,
            ),
            c"v@:@@".as_ptr(),
        );
        if added == NO {
            tracing::warn!(
                "The application delegate already handles opened files; eov's handler was not added"
            );
        }
    }
}

/// `-[NSApplicationDelegate application:openURLs:]`.
extern "C" fn open_urls(
    _this: &Object,
    _command: Sel,
    _application: *mut Object,
    urls: *mut Object,
) {
    // Unwinding into AppKit is undefined behaviour.
    let taken = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: AppKit passes an NSArray of NSURL.
        let arguments = unsafe { arguments_of(urls) };
        let request = crate::launch::request_from_arguments(
            &arguments,
            &[],
            crate::launch::ViewRequest::default(),
            Path::new("/"),
            &crate::launch::find_on_disk,
            &crate::config::load_trusted_link_servers(),
        );
        // Opened from the event loop, not inside AppKit's callback.
        let _ = slint::invoke_from_event_loop(move || {
            if let Err(error) = crate::launch::open_handed_over(request) {
                tracing::warn!("Could not open the files from the Finder: {error}");
            }
        });
    }));
    if taken.is_err() {
        tracing::error!("Could not take the files macOS handed over");
    }
}

/// File URLs as paths, other URLs (`eov://…`) as text.
#[allow(unexpected_cfgs)]
unsafe fn arguments_of(urls: *mut Object) -> Vec<OsString> {
    let mut arguments = Vec::new();
    if urls.is_null() {
        return arguments;
    }
    // SAFETY: (all calls) messages NSArray and NSURL document.
    unsafe {
        let count: usize = msg_send![urls, count];
        for index in 0..count {
            let url: *mut Object = msg_send![urls, objectAtIndex: index];
            if url.is_null() {
                continue;
            }
            let is_file: BOOL = msg_send![url, isFileURL];
            if is_file == YES {
                let representation: *const c_char = msg_send![url, fileSystemRepresentation];
                if !representation.is_null() {
                    let bytes = CStr::from_ptr(representation).to_bytes();
                    arguments.push(OsStr::from_bytes(bytes).to_os_string());
                }
            } else {
                let text: *mut Object = msg_send![url, absoluteString];
                if text.is_null() {
                    continue;
                }
                let utf8: *const c_char = msg_send![text, UTF8String];
                if !utf8.is_null() {
                    let text = CStr::from_ptr(utf8).to_string_lossy().into_owned();
                    arguments.push(OsString::from(text));
                }
            }
        }
    }
    arguments
}
