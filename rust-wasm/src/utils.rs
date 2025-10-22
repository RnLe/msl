//! Utility functions for WASM bindings
//!
//! Helper functions for conversion between Rust and JavaScript types

use wasm_bindgen::prelude::*;

/// Set panic hook for better error messages in the browser console
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Log a message to the browser console
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

/// Macro for logging from Rust to browser console
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => {
        $crate::utils::log(&format_args!($($t)*).to_string())
    };
}
