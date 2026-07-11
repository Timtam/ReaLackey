//! Minimal, robust SSE framer for the Anthropic Messages stream.
//!
//! Splits a byte stream into `data:` payloads (one event per blank line) and
//! deserializes each into a [`StreamEvent`]. Buffers raw bytes and only decodes
//! complete lines, so a multibyte UTF-8 codepoint split across network chunks is
//! never corrupted. Unknown/`ping` events deserialize into their catch-all
//! variants and are handled by the caller.

use super::types::StreamEvent;

#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,   // raw bytes not yet split into complete lines
    data: String,   // accumulated `data:` payload for the current event
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk; returns any events that completed within it.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<StreamEvent> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                // Blank line: dispatch the accumulated event.
                if !self.data.is_empty() {
                    if let Ok(ev) = serde_json::from_str::<StreamEvent>(&self.data) {
                        out.push(ev);
                    }
                    self.data.clear();
                }
            } else if let Some(rest) = line.strip_prefix("data:") {
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(rest);
            }
            // `event:` name lines and `:` comments are ignored — we dispatch on
            // the JSON payload's own "type" tag.
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::anthropic::types::{Delta, StreamEvent};

    // The verbatim text-streaming example from the Anthropic docs.
    const TEXT_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-opus-4-8\",\"stop_reason\":null,\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: ping\n\
data: {\"type\":\"ping\"}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\", Welt\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":15}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

    fn collect_text(parser: &mut SseParser, chunk: &[u8]) -> (String, Option<String>) {
        let mut text = String::new();
        let mut stop = None;
        for ev in parser.feed(chunk) {
            match ev {
                StreamEvent::ContentBlockDelta {
                    delta: Delta::TextDelta { text: t },
                    ..
                } => text.push_str(&t),
                StreamEvent::MessageDelta { delta, .. } => stop = delta.stop_reason,
                _ => {}
            }
        }
        (text, stop)
    }

    #[test]
    fn parses_full_text_stream() {
        let mut p = SseParser::new();
        let (text, stop) = collect_text(&mut p, TEXT_SSE.as_bytes());
        assert_eq!(text, "Hello, Welt");
        assert_eq!(stop.as_deref(), Some("end_turn"));
    }

    #[test]
    fn tolerates_arbitrary_chunk_boundaries() {
        // Feed one byte at a time — events must still reassemble correctly.
        let mut p = SseParser::new();
        let mut text = String::new();
        let mut stop = None;
        for b in TEXT_SSE.as_bytes() {
            let (t, s) = collect_text(&mut p, &[*b]);
            text.push_str(&t);
            if s.is_some() {
                stop = s;
            }
        }
        assert_eq!(text, "Hello, Welt");
        assert_eq!(stop.as_deref(), Some("end_turn"));
    }

    #[test]
    fn ignores_unknown_event_types() {
        let mut p = SseParser::new();
        let evs = p.feed(b"data: {\"type\":\"some_future_event\",\"x\":1}\n\n");
        assert!(matches!(evs.as_slice(), [StreamEvent::Unknown]));
    }
}
