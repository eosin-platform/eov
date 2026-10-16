//! A small HTTP/1.1 server on 127.0.0.1 for tests of model downloads: it
//! serves files (with or without `Range` support), redirects, breaks
//! connections off and trickles bytes, and records every request.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How one path is answered.
#[derive(Clone, Default)]
pub(crate) struct Route {
    pub body: Vec<u8>,
    /// Answers `Range` requests with 206.
    pub ranges: bool,
    /// The first this-many responses stop after `cut_after` bytes of the
    /// body (declaring the whole length), as a dropped connection does.
    pub cut_first: usize,
    pub cut_after: usize,
    /// Waits this long before each 4 KiB of the body.
    pub trickle: Option<Duration>,
    /// Answers 302 to this location instead.
    pub redirect: Option<String>,
    /// Answers this status (with no body) instead.
    pub status: Option<u16>,
}

impl Route {
    pub fn file(body: impl Into<Vec<u8>>) -> Self {
        Self {
            body: body.into(),
            ranges: true,
            ..Self::default()
        }
    }
}

/// A request the server received: the path and its `Range` header.
pub(crate) type Seen = (String, Option<String>);

pub(crate) struct TestServer {
    port: u16,
    routes: Arc<Mutex<HashMap<String, (Route, usize)>>>,
    seen: Arc<Mutex<Vec<Seen>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl TestServer {
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a local port");
        let port = listener.local_addr().unwrap().port();
        let routes: Arc<Mutex<HashMap<String, (Route, usize)>>> = Arc::default();
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (routes, seen, stop) = (Arc::clone(&routes), Arc::clone(&seen), Arc::clone(&stop));
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let (routes, seen) = (Arc::clone(&routes), Arc::clone(&seen));
                    std::thread::spawn(move || {
                        let _ = serve(stream, &routes, &seen);
                    });
                }
            })
        };
        Self {
            port,
            routes,
            seen,
            stop,
            thread: Some(thread),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    pub fn route(&self, path: &str, route: Route) {
        self.routes
            .lock()
            .unwrap()
            .insert(path.to_string(), (route, 0));
    }

    /// Every request so far.
    pub fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Wake the accept loop.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn serve(
    stream: TcpStream,
    routes: &Mutex<HashMap<String, (Route, usize)>>,
    seen: &Mutex<Vec<Seen>>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
    let mut range = None;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':')
            && name.trim().eq_ignore_ascii_case("range")
        {
            range = Some(value.trim().to_string());
        }
    }
    seen.lock().unwrap().push((path.clone(), range.clone()));
    let route_path = path.split('?').next().unwrap_or_default();
    let found = {
        let mut routes = routes.lock().unwrap();
        routes.get_mut(route_path).map(|(route, count)| {
            *count += 1;
            (route.clone(), *count)
        })
    };
    let mut stream = stream;
    let Some((route, count)) = found else {
        write!(
            stream,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )?;
        return Ok(());
    };
    if let Some(location) = &route.redirect {
        write!(
            stream,
            "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )?;
        return Ok(());
    }
    if let Some(status) = route.status {
        write!(
            stream,
            "HTTP/1.1 {status} Nope\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )?;
        return Ok(());
    }
    let total = route.body.len();
    let start = range
        .as_deref()
        .filter(|_| route.ranges)
        .and_then(|value| value.strip_prefix("bytes="))
        .and_then(|value| value.strip_suffix('-'))
        .and_then(|value| value.parse::<usize>().ok());
    let body = match start {
        Some(start) if start >= total => {
            write!(
                stream,
                "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{total}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )?;
            return Ok(());
        }
        Some(start) => {
            write!(
                stream,
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                total - 1,
                total - start
            )?;
            &route.body[start..]
        }
        None => {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
            )?;
            &route.body[..]
        }
    };
    let body = if count <= route.cut_first {
        &body[..route.cut_after.min(body.len())]
    } else {
        body
    };
    for chunk in body.chunks(4096) {
        if let Some(pause) = route.trickle {
            std::thread::sleep(pause);
        }
        stream.write_all(chunk)?;
    }
    stream.flush()?;
    // Closing now ends a cut response early.
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
}
