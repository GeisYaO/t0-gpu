//! Build script for t0-gpu.
//!
//! When the `wsl_dxg` feature is enabled, we link against WSL's `libdxcore.so`.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if env::var_os("CARGO_FEATURE_WSL_DXG").is_none() {
        return;
    }

    println!("cargo:rustc-link-search=native=/usr/lib/wsl/lib");
    println!("cargo:rustc-link-lib=dylib=dxcore");
}
