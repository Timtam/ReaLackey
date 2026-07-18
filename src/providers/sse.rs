//! Minimal Server-Sent Events framer shared by the streaming adapters.
//!
//! Buffers raw bytes and yields each event's `data:` payload once a blank line
//! completes the event. Only complete lines are split, so a multibyte codepoint
//! straddling two network chunks is never corrupted. `event:` / `id:` / comment
//! (`:`) lines are ignored — both the OpenAI-compatible `/chat/completions` stream
//! and the Responses-API `/v1/agent` stream carry their event type inside the
//! JSON `data` payload, so the framer only needs the data.

/// Accumulates SSE bytes and emits complete `data:` payloads.
#[derive(Default)]
pub(crate) struct SseFramer {
    buf: Vec<u8>,
    data: String,
}

impl SseFramer {
    /// Feed a network chunk; return any `data:` payloads that completed within it.
    /// Multi-line `data:` fields are joined with `\n` (per the SSE spec).
    pub(crate) fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                if !self.data.is_empty() {
                    out.push(std::mem::take(&mut self.data));
                }
            } else if let Some(rest) = line.strip_prefix("data:") {
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(rest);
            }
            // `:` comments / keep-alives and other field lines are ignored.
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_framer_splits_data_payloads() {
        let mut s = SseFramer::default();
        let out = s.feed(b"data: {\"a\":1}\n\ndata: [DONE]\n\n");
        assert_eq!(out, vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
    }

    #[test]
    fn multibyte_split_across_chunks_is_not_corrupted() {
        // The bytes of "é" (0xC3 0xA9) split across two feeds must reassemble.
        let mut s = SseFramer::default();
        let mut first = b"data: caf".to_vec();
        first.push(0xC3);
        assert!(s.feed(&first).is_empty());
        let mut second = vec![0xA9];
        second.extend_from_slice(b"\n\n");
        assert_eq!(s.feed(&second), vec!["café".to_string()]);
    }

    #[test]
    fn multi_line_data_fields_join_with_newline() {
        let mut s = SseFramer::default();
        let out = s.feed(b"data: line1\ndata: line2\n\n");
        assert_eq!(out, vec!["line1\nline2".to_string()]);
    }
}
