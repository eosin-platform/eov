//! Compiles a plugin .slint file the way eov does and checks property JSON
//! against the component's declared types.
//!
//! usage: eovae-check-ui <file.slint> <component> <include root> [properties.json] [callbacks.json]

use i_slint_compiler::langtype::Type;
use std::collections::HashMap;

fn spin_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Waker};
    let mut cx = Context::from_waker(Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = std::path::PathBuf::from(&args[1]);
    let component = &args[2];
    let root = std::path::PathBuf::from(&args[3]);
    let mut compiler = slint_interpreter::Compiler::default();
    compiler.set_include_paths(vec![root]);
    let result = spin_on(compiler.build_from_path(&path));
    let mut errors = 0;
    for d in result.diagnostics() {
        println!("{:?}: {}", d.level(), d);
        if d.level() == slint_interpreter::DiagnosticLevel::Error {
            errors += 1;
        }
    }
    if errors > 0 {
        println!("FAILED: {errors} errors");
        std::process::exit(1);
    }
    let definition = result.component(component).expect("component exported");
    let mut properties: HashMap<String, Type> = HashMap::new();
    let mut callbacks: HashMap<String, Type> = HashMap::new();
    for (name, (ty, _)) in definition.properties_and_callbacks() {
        match &ty {
            Type::Callback(f) => {
                callbacks.insert(name.clone(), Type::Callback(f.clone()));
            }
            ty if ty.is_property_type() => {
                properties.insert(name, ty.clone());
            }
            _ => {}
        }
    }
    println!(
        "compiled: {} properties, {} callbacks",
        properties.len(),
        callbacks.len()
    );
    let mut problems = 0;
    if let Some(file) = args.get(4) {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
        let map = json.as_object().unwrap();
        for (name, value) in map {
            match properties.get(name) {
                None => {
                    println!("UNKNOWN PROPERTY {name}");
                    problems += 1;
                }
                Some(ty) => {
                    if let Err(e) = slint_interpreter::json::value_from_json(ty, value) {
                        println!("BAD VALUE {name}: {e} (value {value})");
                        problems += 1;
                    }
                }
            }
        }
        for name in properties.keys() {
            if !map.contains_key(name) {
                println!("never set by the plugin: {name}");
            }
        }
        println!("checked {} pushed properties", map.len());
    }
    if let Some(file) = args.get(5) {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
        let handled: Vec<String> = json
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        for name in callbacks.keys() {
            if !handled.contains(name) {
                println!("CALLBACK NOT HANDLED {name}");
                problems += 1;
            }
        }
        for name in &handled {
            if !callbacks.contains_key(name) {
                println!("handled callback missing in UI: {name}");
            }
        }
    }
    for (name, ty) in &callbacks {
        if let Type::Callback(f) = ty {
            println!("callback {name}({:?}) -> {:?}", f.args, f.return_type);
        }
    }
    if problems > 0 {
        println!("FAILED: {problems} problems");
        std::process::exit(2);
    }
    println!("OK");
}
