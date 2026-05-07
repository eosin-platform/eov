fn main() {
    unsafe {
        std::env::set_var("SLINT_ENABLE_EXPERIMENTAL_FEATURES", "1");
    }

    slint_build::compile("ui/app-window.slint").expect("Slint build failed");

    if let Ok(dir) = std::env::var("OPENSLIDE_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }

    println!("cargo:rustc-link-lib=dylib=openslide");
}
