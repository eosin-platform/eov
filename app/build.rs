fn main() {
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"cargo-clippy\"))");
    slint_build::compile("ui/app-window.slint").expect("Slint build failed");
}
