fn main() {
    unsafe {
        std::env::set_var("SLINT_ENABLE_EXPERIMENTAL_FEATURES", "1");
    }

    slint_build::compile("ui/app-window.slint").expect("Slint build failed");

    // OpenSlide is linked through eov-common (see common/build.rs); linking it
    // here would also attach it to eov-update-helper, which must stay
    // self-contained so it can run from a temporary directory.
}
