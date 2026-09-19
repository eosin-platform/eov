//! Permission-gated HTTPS access for release metadata and artifacts.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::io::{IsTerminal, Read, Write};
use std::time::Duration;
use url::Url;

/// A per-process proof that the user authorized release-network access.
///
/// The constructor is private. Callers can only obtain this value through an
/// explicit flag or the interactive consent prompt.
#[derive(Debug, Clone, Copy)]
pub struct NetworkPermission {
    _private: (),
}

impl NetworkPermission {
    pub fn acquire(allow_network: bool, context: &str) -> Result<Option<Self>> {
        if allow_network {
            return Ok(Some(Self { _private: () }));
        }

        if !std::io::stdin().is_terminal() {
            bail!(
                "This command requires release metadata from GitHub.\n".to_string()
                    + "Re-run interactively or pass --allow-network."
            );
        }

        let mut stdout = std::io::stdout();
        writeln!(
            stdout,
            "In order to determine the latest version(s), this command needs to fetch release metadata from GitHub.\n\nEOV will never make network requests without your permission. It does not collect telemetry.\n\n{context}\n\nDo you wish to proceed? (Y/n)"
        )?;
        stdout.flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        let answer = answer.trim();
        if answer.is_empty()
            || answer.eq_ignore_ascii_case("y")
            || answer.eq_ignore_ascii_case("yes")
        {
            Ok(Some(Self { _private: () }))
        } else {
            Ok(None)
        }
    }

    #[cfg(test)]
    pub fn for_test() -> Self {
        Self { _private: () }
    }
}

pub fn require_permission(allow_network: bool, context: &str) -> Result<NetworkPermission> {
    NetworkPermission::acquire(allow_network, context)?.context(
        "Network access declined; the requested operation cannot continue without release metadata.",
    )
}

/// A synchronous HTTPS client. Every request requires a `NetworkPermission`.
pub struct ReleaseClient {
    agent: ureq::Agent,
}

impl ReleaseClient {
    pub fn new(user_agent: &str) -> Self {
        let config = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(20))
            .timeout_read(Duration::from_secs(60))
            .timeout_write(Duration::from_secs(60))
            .redirects(0)
            .user_agent(user_agent)
            .build();
        Self { agent: config }
    }

    pub fn get_text(&self, _permission: &NetworkPermission, url: &str) -> Result<String> {
        let bytes = self.get_bytes_with_accept(url, "text/plain, application/toml")?;
        String::from_utf8(bytes).context("release response was not valid UTF-8")
    }

    pub fn get_bytes(&self, _permission: &NetworkPermission, url: &str) -> Result<Vec<u8>> {
        self.get_bytes_with_accept(url, "application/octet-stream")
    }

    pub fn get_json(&self, permission: &NetworkPermission, url: &str) -> Result<Value> {
        let _ = permission;
        let text =
            String::from_utf8(self.get_bytes_with_accept(url, "application/vnd.github+json")?)
                .context("GitHub JSON response was not valid UTF-8")?;
        serde_json::from_str(&text).with_context(|| format!("invalid JSON response from {url}"))
    }

    fn get_bytes_with_accept(&self, url: &str, accept: &str) -> Result<Vec<u8>> {
        let mut current = Url::parse(url).with_context(|| format!("invalid HTTPS URL '{url}'"))?;
        for _ in 0..=10 {
            validate_request_url(&current)?;
            let response = match self
                .agent
                .get(current.as_str())
                .set("Accept", accept)
                .call()
            {
                Ok(response) => response,
                Err(ureq::Error::Status(_, response)) => response,
                Err(error) => {
                    return Err(anyhow::Error::new(error))
                        .with_context(|| format!("failed to fetch {current}"));
                }
            };
            let status = response.status();
            if (300..400).contains(&status) {
                let location = response
                    .header("Location")
                    .context("redirect response has no Location header")?;
                current = current
                    .join(location)
                    .with_context(|| format!("invalid redirect target '{location}'"))?;
                continue;
            }
            if !(200..300).contains(&status) {
                bail!("GitHub returned HTTP {status} for {current}");
            }
            validate_request_url(&current)?;
            let mut bytes = Vec::new();
            response
                .into_reader()
                .read_to_end(&mut bytes)
                .with_context(|| format!("failed to read response from {current}"))?;
            return Ok(bytes);
        }
        bail!("too many HTTPS redirects while fetching {url}")
    }
}

fn validate_request_url(url: &Url) -> Result<()> {
    if url.scheme() != "https" {
        bail!("updater requests must use HTTPS: {url}");
    }
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    let allowed = host == "github.com"
        || host == "api.github.com"
        || host == "objects.githubusercontent.com"
        || host == "release-assets.githubusercontent.com"
        || host == "github-production-release-asset.s3.amazonaws.com"
        || host == "github-production-user-asset.s3.amazonaws.com";
    if !allowed {
        bail!("release request redirected to an unexpected host: {host}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_network_constructs_permission_without_prompt() {
        assert!(NetworkPermission::acquire(true, "test").unwrap().is_some());
    }

    #[test]
    fn http_urls_are_rejected_before_transport() {
        let client = ReleaseClient::new("eov/test");
        let permission = NetworkPermission::for_test();
        assert!(
            client
                .get_text(&permission, "http://github.com/eov")
                .is_err()
        );
    }

    #[test]
    fn redirect_hosts_are_limited_to_github_release_domains() {
        assert!(
            validate_request_url(
                &Url::parse("https://objects.githubusercontent.com/release").unwrap()
            )
            .is_ok()
        );
        assert!(
            validate_request_url(
                &Url::parse("https://github-production-release-asset.s3.amazonaws.com/a").unwrap()
            )
            .is_ok()
        );
        assert!(
            validate_request_url(&Url::parse("https://example.invalid/release").unwrap()).is_err()
        );
        assert!(validate_request_url(&Url::parse("http://github.com/release").unwrap()).is_err());
    }
}
