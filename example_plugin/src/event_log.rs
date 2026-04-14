//! A simple event log accumulator — testable non-UI plugin logic.
//!
//! Records timestamped string events and provides query methods.
//! This demonstrates that plugin crates can ship pure Rust functionality
//! that is unit-testable without a Slint runtime.

#[derive(Debug, Clone)]
pub struct EventEntry {
    pub message: String,
    pub sequence: u64,
}

/// Accumulates string events with a monotonic sequence counter.
#[derive(Debug)]
pub struct EventLog {
    entries: Vec<EventEntry>,
    counter: u64,
}

impl EventLog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            counter: 0,
        }
    }

    /// Record a new event, returning its sequence number.
    pub fn record(&mut self, message: &str) -> u64 {
        let seq = self.counter;
        self.counter += 1;
        self.entries.push(EventEntry {
            message: message.to_string(),
            sequence: seq,
        });
        seq
    }

    /// Total number of recorded events.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current counter value (next sequence number).
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// Get all entries.
    pub fn entries(&self) -> &[EventEntry] {
        &self.entries
    }

    /// Get entries matching a substring.
    pub fn search(&self, substring: &str) -> Vec<&EventEntry> {
        self.entries
            .iter()
            .filter(|e| e.message.contains(substring))
            .collect()
    }

    /// Clear all entries and reset the counter.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.counter = 0;
    }

    /// Format a summary string, e.g. for diagnostics.
    pub fn summary(&self) -> String {
        format!(
            "EventLog: {} event(s), counter={}",
            self.entries.len(),
            self.counter
        )
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_log_is_empty() {
        let log = EventLog::new();
        assert!(log.is_empty());
        assert_eq!(log.counter(), 0);
    }

    #[test]
    fn record_increments_counter() {
        let mut log = EventLog::new();
        assert_eq!(log.record("first"), 0);
        assert_eq!(log.record("second"), 1);
        assert_eq!(log.record("third"), 2);
        assert_eq!(log.len(), 3);
        assert_eq!(log.counter(), 3);
    }

    #[test]
    fn entries_preserved_in_order() {
        let mut log = EventLog::new();
        log.record("alpha");
        log.record("beta");
        log.record("gamma");
        let msgs: Vec<&str> = log.entries().iter().map(|e| e.message.as_str()).collect();
        assert_eq!(msgs, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn search_filters_correctly() {
        let mut log = EventLog::new();
        log.record("user_login");
        log.record("file_open");
        log.record("user_logout");
        log.record("file_close");

        let user_events = log.search("user");
        assert_eq!(user_events.len(), 2);
        assert_eq!(user_events[0].message, "user_login");
        assert_eq!(user_events[1].message, "user_logout");

        let file_events = log.search("file");
        assert_eq!(file_events.len(), 2);
    }

    #[test]
    fn clear_resets_state() {
        let mut log = EventLog::new();
        log.record("a");
        log.record("b");
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.counter(), 0);
        // After clear, next record starts at 0 again
        assert_eq!(log.record("c"), 0);
    }

    #[test]
    fn summary_format() {
        let mut log = EventLog::new();
        log.record("x");
        log.record("y");
        assert_eq!(log.summary(), "EventLog: 2 event(s), counter=2");
    }
}
