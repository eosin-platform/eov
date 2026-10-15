//! `eov call METHOD [PARAMS]` and `eov script-endpoint`: the scripting
//! interface from a shell, through the same endpoint and secret as
//! everything else.

use crate::instance::{self, Endpoint, ScriptConnectError};
use base64::Engine;
use serde_json::{Map, Value, json};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// `eov call` exit statuses.
pub(crate) const EXIT_CALL_FAILED: i32 = 1;
pub(crate) const EXIT_USAGE: i32 = 2;
pub(crate) const EXIT_NOT_RUNNING: i32 = 3;

/// How long `eov call` waits for an answer (exports can take minutes).
const ANSWER_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Why `eov call` failed, with its exit status.
#[derive(Debug)]
pub(crate) struct Failure {
    pub(crate) status: i32,
    pub(crate) message: String,
}

impl Failure {
    fn new(status: i32, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

/// The parameters of `eov call`: none, a JSON object, `-` (standard input)
/// or `@FILE`. A relative `path` is made absolute against `cwd`.
pub(crate) fn parse_params(text: Option<&str>, cwd: &Path) -> Result<Value, Failure> {
    let usage = |message: String| Failure::new(EXIT_USAGE, message);
    let text = match text {
        None => return Ok(Value::Object(Map::new())),
        Some("-") => {
            let mut text = String::new();
            std::io::stdin()
                .read_to_string(&mut text)
                .map_err(|error| usage(format!("cannot read the parameters: {error}")))?;
            text
        }
        Some(argument) => match argument.strip_prefix('@') {
            Some(file) => std::fs::read_to_string(file)
                .map_err(|error| usage(format!("cannot read {file}: {error}")))?,
            None => argument.to_string(),
        },
    };
    let mut params: Value = serde_json::from_str(&text)
        .map_err(|error| usage(format!("the parameters are not JSON: {error}")))?;
    let Some(object) = params.as_object_mut() else {
        return Err(usage(
            "the parameters must be a JSON object, such as '{\"slide\": 1}'".to_string(),
        ));
    };
    if let Some(Value::String(path)) = object.get_mut("path")
        && !Path::new(path.as_str()).is_absolute()
    {
        *path = cwd.join(path.as_str()).to_string_lossy().into_owned();
    }
    Ok(params)
}

/// Calls `method` on the running eov; returns the result to print. With
/// `out`, the result's base64 `data` (`region.read`) is written there
/// instead of printed.
pub(crate) fn call(
    endpoint: &Endpoint,
    method: &str,
    params: Value,
    out: Option<&Path>,
) -> Result<Value, Failure> {
    let request = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    }))
    .map_err(|error| Failure::new(EXIT_USAGE, error.to_string()))?;
    let mut client =
        instance::connect_script(endpoint, instance::Timing::default()).map_err(|error| {
            let status = match error {
                ScriptConnectError::NotRunning(_) => EXIT_NOT_RUNNING,
                ScriptConnectError::Failed(_) => EXIT_CALL_FAILED,
            };
            Failure::new(status, error.to_string())
        })?;
    let answer = client.call(&request, ANSWER_TIMEOUT).map_err(|error| {
        Failure::new(
            EXIT_CALL_FAILED,
            format!("eov did not answer the call: {error}"),
        )
    })?;
    let answer: Value = serde_json::from_slice(&answer).map_err(|error| {
        Failure::new(EXIT_CALL_FAILED, format!("an unreadable answer: {error}"))
    })?;
    if let Some(error) = answer.get("error") {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("the call failed");
        let kind = error
            .pointer("/data/kind")
            .and_then(Value::as_str)
            .unwrap_or("error");
        return Err(Failure::new(
            EXIT_CALL_FAILED,
            format!("{message} ({kind})"),
        ));
    }
    let mut result = answer.get("result").cloned().unwrap_or(Value::Null);
    if let Some(out) = out {
        let Some(Value::String(data)) = result.get("data") else {
            return Err(Failure::new(
                EXIT_USAGE,
                "--out takes the data of an answer, and this one has none",
            ));
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|error| Failure::new(EXIT_CALL_FAILED, format!("unreadable data: {error}")))?;
        std::fs::write(out, &bytes).map_err(|error| {
            Failure::new(
                EXIT_CALL_FAILED,
                format!("cannot write {}: {error}", out.display()),
            )
        })?;
        result["data"] = json!(format!(
            "{} bytes written to {}",
            bytes.len(),
            out.display()
        ));
    }
    Ok(result)
}

/// `eov script-endpoint`: where scripts reach this setup's eov.
pub(crate) fn describe(endpoint: &Endpoint, scripting: bool, json: bool) -> String {
    let status = endpoint.status(instance::Timing {
        io: Duration::from_secs(2),
        ..instance::Timing::default()
    });
    let transport = if status.socket.is_some() {
        "unix"
    } else {
        "tcp"
    };
    if json {
        let mut value = serde_json::to_value(&status).unwrap_or_default();
        value["transport"] = json!(transport);
        value["protocol"] = json!(String::from_utf8_lossy(instance::SCRIPT_MAGIC));
        value["scripting"] = json!(scripting);
        return serde_json::to_string_pretty(&value).unwrap_or_default();
    }
    let mut lines = vec![
        format!("Endpoint:   {}", status.name),
        format!(
            "Info file:  {} (port, process id and secret; only you can read it)",
            status.info.display()
        ),
    ];
    match (&status.socket, status.port) {
        (Some(socket), _) => lines.push(format!("Socket:     {}", socket.display())),
        (None, Some(port)) => lines.push(format!("Port:       127.0.0.1:{port}")),
        (None, None) => lines.push("Port:       chosen when eov starts".to_string()),
    }
    lines.push(match (status.running, status.pid) {
        (true, Some(pid)) => format!("Running:    yes (process {pid})"),
        _ => "Running:    no".to_string(),
    });
    lines.push(format!(
        "Scripting:  {}",
        if scripting {
            "on"
        } else {
            "off (scripting = false in config.toml)"
        }
    ));
    lines.join("\n")
}

/// The working folder, for relative paths in `eov call`.
pub(crate) fn current_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_are_objects_and_paths_become_absolute() {
        let cwd = std::env::temp_dir().join("cases");
        assert_eq!(parse_params(None, &cwd).unwrap(), json!({}));
        let params = parse_params(Some(r#"{"path": "a.svs", "x": 1}"#), &cwd).unwrap();
        assert_eq!(params["path"], json!(cwd.join("a.svs").to_string_lossy()));
        assert_eq!(params["x"], json!(1));
        let absolute = cwd.join("b.svs").to_string_lossy().into_owned();
        let params = parse_params(Some(&json!({ "path": absolute }).to_string()), &cwd).unwrap();
        assert_eq!(params["path"], json!(absolute));
        for bad in ["[1, 2]", "not json", "3"] {
            assert_eq!(
                parse_params(Some(bad), &cwd).unwrap_err().status,
                EXIT_USAGE,
                "{bad}"
            );
        }
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("params.json");
        std::fs::write(&file, r#"{"slide": 2}"#).unwrap();
        assert_eq!(
            parse_params(Some(&format!("@{}", file.display())), &cwd).unwrap(),
            json!({"slide": 2})
        );
    }

    #[test]
    fn calls_without_a_running_eov_say_so() {
        let dir = tempfile::tempdir().unwrap();
        let endpoint = Endpoint::new(
            dir.path().join("run"),
            "eov-test".into(),
            crate::instance::Transport::Tcp,
        );
        let failure = call(&endpoint, "version", json!({}), None).unwrap_err();
        assert_eq!(failure.status, EXIT_NOT_RUNNING);
        let text = describe(&endpoint, true, false);
        assert!(text.contains("Running:    no"), "{text}");
        let value: Value = serde_json::from_str(&describe(&endpoint, false, true)).unwrap();
        assert_eq!(value["running"], json!(false));
        assert_eq!(value["scripting"], json!(false));
        assert_eq!(value["protocol"], json!("eov-rpc1"));
    }
}
